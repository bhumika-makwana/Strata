import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import decoupler as dc
from data_loader import init_main
import omnipath as op
import mygene


def build_causal_prior_matrix(gene_names):
    """
    Builds an N x N binary prior adjacency matrix aligned with dataset gene list.
    G_matrix -> flat, unweighted, binary adjacency matrix where every connection (whether a transcription factor regulating a gene or a kinase signaling cascade) is represented simply by a 1.0.
    Exact biochemical mechanism of every edge is not guaranteed, but the edges are curated from OmniPath and CollecTRI databases.
    """
    clean_genes = [str(g).split('.')[0].strip().upper() for g in gene_names]
    n_genes = len(clean_genes)
    ensg2idx = {g: i for i, g in enumerate(clean_genes)}
    
    print(f"Dataset sample genes (Ensembl): {clean_genes[:5]}")
    print("1. Mapping Ensembl IDs to HGNC Gene Symbols...")
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        clean_genes, 
        scopes='ensembl.gene', 
        fields='symbol', 
        species='human',
        verbose=False
    )
    
    symbol2idx = {}
    for res in results:
        ens_id = res.get('query')
        sym = res.get('symbol')
        if ens_id and sym and ens_id in ensg2idx:
            symbol2idx[sym.upper().strip()] = ensg2idx[ens_id]
            
    print(f"Mapped {len(symbol2idx)} / {n_genes} Ensembl IDs to HGNC gene symbols.")
    G_prior = np.zeros((n_genes, n_genes), dtype=np.float32)
    print("1. Fetching CollecTRI...")
    try:
        collectri = dc.get_collectri(organism='human', split_complexes=True)
    except Exception:
        collectri = dc.op.collectri(organism='human')
        
    src_col = next((c for c in ['source', 'genesymbol_source', 'source_genesymbol'] if c in collectri.columns), None)
    tgt_col = next((c for c in ['target', 'genesymbol_target', 'target_genesymbol'] if c in collectri.columns), None)
    
    c_df = collectri[[src_col, tgt_col]].dropna().astype(str)
    c_df[src_col] = c_df[src_col].str.upper().str.strip()
    c_df[tgt_col] = c_df[tgt_col].str.upper().str.strip()
    
    c_df = c_df[c_df[src_col].isin(symbol2idx) & c_df[tgt_col].isin(symbol2idx)]
    
    src_idxs = c_df[src_col].map(symbol2idx).values
    tgt_idxs = c_df[tgt_col].map(symbol2idx).values
    G_prior[tgt_idxs, src_idxs] = 1.0
    tf_edges = len(c_df)


    print("2. Fetching OmniPath...")    
    try:
        omni = op.requests.Interactions.get(datasets=['omnipath'])
    except Exception:
        omni = op.interactions.import_intercell_network()
    if 'is_directed' in omni.columns:
        omni = omni[omni['is_directed'].isin([1, 1.0, True])]
    o_df = omni[['source', 'target']].dropna().astype(str)
    # Extract unique UniProt IDs to map them efficiently in bulk using MyGeneInfo
    print("Converting OmniPath UniProt IDs to HGNC Symbols via MyGene...")
    unique_uniprots = pd.concat([o_df['source'], o_df['target']]).astype(str).str.split('-').str[0].unique().tolist()
    
    mg = mygene.MyGeneInfo()
    # Query MyGene in chunks to map uniprot to symbol
    uniprot2sym = {}
    chunk_size = 1000
    for i in range(0, len(unique_uniprots), chunk_size):
        chunk = unique_uniprots[i:i + chunk_size]
        res = mg.querymany(chunk, scopes='uniprot', fields='symbol', species='human', verbose=False)
        for r in res:
            q = r.get('query')
            sym = r.get('symbol')
            if q and sym:
                uniprot2sym[q] = sym.upper().strip()

    # Clean isoform extensions and map to symbols
    clean_src = o_df['source'].astype(str).str.split('-').str[0]
    clean_tgt = o_df['target'].astype(str).str.split('-').str[0]
    
    o_df['source_sym'] = clean_src.map(uniprot2sym)
    o_df['target_sym'] = clean_tgt.map(uniprot2sym)
    
    o_df = o_df[['source_sym', 'target_sym']].dropna().astype(str)
    o_df['source_sym'] = o_df['source_sym'].str.upper().str.strip()
    o_df['target_sym'] = o_df['target_sym'].str.upper().str.strip()
    
    o_df = o_df[o_df['source_sym'].isin(symbol2idx) & o_df['target_sym'].isin(symbol2idx)]
    print(f"Matched signaling interactions: {len(o_df)}")
    
    o_src_idxs = o_df['source_sym'].map(symbol2idx).values
    o_tgt_idxs = o_df['target_sym'].map(symbol2idx).values
    G_prior[o_tgt_idxs, o_src_idxs] = 1.0
    signaling_edges = len(o_df)
    np.fill_diagonal(G_prior, 1.0)
    
    print(f"Graph Built: {tf_edges} TF-target edges, {signaling_edges} signaling edges across {n_genes} genes.")
    return G_prior


class GraphGuidedAttention(nn.Module):
    """Self-Attention modulated by the learnable causal adjacency graph G."""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, G_param, alpha=5.0):
        # x shape: [Batch, N_genes, d_model]
        B, N, D = x.shape
        
        Q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # Standard attention scores: [B, H, N, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Apply Sigmoid to parameter matrix to keep edge probabilities in (0, 1)
        G_prob = torch.sigmoid(G_param)
        
        # Inject graph structure as a log-penalty into pre-softmax logits
        # Non-edges (prob near 0) become large negative values, shutting off attention
        graph_bias = torch.log(G_prob + 1e-6).unsqueeze(0).unsqueeze(0) # [1, 1, N, N]
        
        attn_weights = torch.softmax(scores + alpha * graph_bias, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.out_proj(out)


class CausalGeneTransformer(nn.Module):
    """
        Inputs (forward function):
        - baseline: [Batch, N_genes] tensor of control gene expression values (The unperturbed (control) gene expression values for every gene in your dataset before any genetic modification is applied)
        - pert_idx (Tensor, shape: e.g., [Batch_Size, Max_Perturbations]) - The indices of the genes that are being targeted for perturbation (e.g., a CRISPR knockout or activation).
        - pad_value (Scalar, default: -1)

        Outputs:
        - pred_expression: [Batch, N_genes] tensor of predicted post-perturb
        - torch.sigmoid(self.G_param): The refined, learned causal graph matrix.
    
    """
    def __init__(self, n_genes, G_prior, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.n_genes = n_genes
        
        # Initialize G_param in logit space so sigmoid(G_param) initially matches G_prior
        eps = 1e-3
        G_clamped = np.clip(G_prior, eps, 1.0 - eps)
        G_logits = np.log(G_clamped / (1.0 - G_clamped))
        
        # Learnable Differentiable Graph Parameter - this treats edges as trainable parameters, allowing the model to refine the prior graph structure during training.
        self.G_param = nn.Parameter(torch.tensor(G_logits, dtype=torch.float32))

        # Gene tokenization layers
        self.gene_id_embed = nn.Embedding(n_genes, d_model) #embed each gene with a unique vector representation
        self.baseline_proj = nn.Linear(1, d_model)# project baseline expression values into the same embedding space
        self.pert_embed = nn.Embedding(2, d_model) # 0: Control, 1: CRISPRa perturbed #flag to show if perturbation is applied to a gene or not, allowing the model to learn how perturbations affect gene expression.

        # Transformer Backbone
        self.layers = nn.ModuleList([
            GraphGuidedAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        
        # Readout head to predict scalar gene expression per gene token
        self.head = nn.Linear(d_model, 1)

    def forward(self, baseline, pert_idx, pad_value=-1):
        B, N = baseline.shape
        device = baseline.device

        # 1. Token Embeddings: Gene Identity + Baseline Expression Value
        gene_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        x_gene = self.gene_id_embed(gene_ids)
        x_base = self.baseline_proj(baseline.unsqueeze(-1))

        # 2. Perturbation Injection
        pert_mask = torch.zeros((B, N), dtype=torch.long, device=device)
        if pert_idx is not None:
            valid_mask = pert_idx != pad_value
            if valid_mask.any():
                batch_indices = torch.arange(B, device=device).unsqueeze(1).expand_as(pert_idx)
                valid_b = batch_indices[valid_mask]
                valid_g = pert_idx[valid_mask]
                pert_mask[valid_b, valid_g] = 1

        x_pert = self.pert_embed(pert_mask)
        
        # Step 1: Building Token Embeddings (Identity + Expression) -> Combine embeddings into initial token representations
        x = x_gene + x_base + x_pert

        # 3. Pass through Graph Transformer Blocks
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x = attn(x, self.G_param)
            x = norm(residual + x)

        # 4. Predict post-perturbation gene expression
        pred_expression = self.head(x).squeeze(-1) # [Batch, N_genes]
        
        return pred_expression, torch.sigmoid(self.G_param)

class CausalGraphLoss(nn.Module):
    def __init__(self, G_prior, lambda_prior=1e-3, lambda_sparse=1e-4):
        super().__init__()
        self.register_buffer('G_prior', torch.tensor(G_prior, dtype=torch.float32))
        self.lambda_prior = lambda_prior
        self.lambda_sparse = lambda_sparse
        self.mse = nn.MSELoss()

    def forward(self, pred_expr, target_expr, G_prob):
        # 1. Primary Expression Prediction Loss
        loss_mse = self.mse(pred_expr, target_expr)
        
        # 2. Deviation from OmniPath + CollecTRI prior
        loss_prior = torch.mean(torch.abs(G_prob - self.G_prior))
        
        # 3. Overall graph sparsity penalty
        loss_sparse = torch.mean(torch.abs(G_prob))
        
        total_loss = loss_mse + (self.lambda_prior * loss_prior) + (self.lambda_sparse * loss_sparse)
        return total_loss, loss_mse

def train_causal_transformer(loaders, gene_names, epochs=15, lr=1e-3, device='cuda'):
    # 1. Build prior matrix from databases
    G_prior = build_causal_prior_matrix(gene_names)
    n_genes = len(gene_names)

    # 2. Initialize Model and Loss
    model = CausalGeneTransformer(
        n_genes=n_genes, 
        G_prior=G_prior, 
        d_model=64,   # Scaled for single-cell gene sequence lengths
        n_heads=4, 
        n_layers=2
    ).to(device)
    
    criterion = CausalGraphLoss(G_prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\n--- Starting Training on {device} ---")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_mse = 0.0
        
        for batch in loaders["train_loader"]:
            baseline = batch["baseline"].to(device)
            pert_idx = batch["pert_idx"].to(device)
            target = batch["expression"].to(device)
            
            optimizer.zero_grad()
            
            pred_expr, G_prob = model(baseline, pert_idx, pad_value=-1)
            loss, loss_mse = criterion(pred_expr, target, G_prob)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_mse += loss_mse.item()
            
        avg_train_loss = total_train_loss / len(loaders["train_loader"])
        avg_train_mse = total_train_mse / len(loaders["train_loader"])
        
        # Validation Loop
        model.eval()
        total_val_mse = 0.0
        with torch.no_grad():
            for batch in loaders["val_loader"]:
                baseline = batch["baseline"].to(device)
                pert_idx = batch["pert_idx"].to(device)
                target = batch["expression"].to(device)
                
                pred_expr, _ = model(baseline, pert_idx, pad_value=-1)
                total_val_mse += F.mse_loss(pred_expr, target).item()
                
        avg_val_mse = total_val_mse / len(loaders["val_loader"])
        
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Train MSE: {avg_train_mse:.4f} | Val MSE: {avg_val_mse:.4f}")
        
    return model

# Execution Script
if __name__ == "__main__":
    # Load data using your existing loader function
    pert_data, loaders = init_main()
    gene_names = pert_data.adata.var_names.tolist()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train model
    trained_model = train_causal_transformer(loaders, gene_names, epochs=100, device=device)