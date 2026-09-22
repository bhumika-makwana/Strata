## Strata
Mechanistically-informed model structures for transcriptomics perturbation effect prediction. 

Strata is a research framework for predicting transcriptional responses to genetic perturbations using single-cell RNA-seq data and curated molecular interaction priors.

**Core contribution** - A learnable, biologically guided perturbation-propagation model that uses curated gene regulatory/signaling networks as a prior to model genetic perturbations across the transcriptome and generalize to unseen gene combinations. Instead of treating a biological network as a fixed ground-truth graph which is the case with current state of the art approaches, Strata initializes a learnable graph from curated prior knowledge and allows perturbational data to refine the strength of these relationships during training.

**Prior-knowledge datasets** - 
We use the Norman et al. (2019) Perturb-seq dataset as the primary experimental dataset, containing single-cell transcriptomic profiles of K562 cells under genetic perturbations. To incorporate prior biological knowledge, we construct a gene interaction network by integrating CollecTRI, a curated transcription factor–target regulatory resource, with OmniPath, a curated resource of molecular and causal signaling interactions.

| Resource        | Description                                             | Role                                         |
| --------------- | ------------------------------------------------------- | -------------------------------------------- |
| **CollecTRI**   | TF–target gene regulatory network                       | **Transcriptional regulation**               |
| **OmniPath**    | Curated molecular and signaling interaction network     | **Signaling and protein-level interactions** |
| **Norman 2019** | Single-cell gene expression under genetic perturbations | **Perturbation-response data**               |


## Model Architecture
We propose **CausalGeneTransformer**, a graph-guided transformer that predicts post-perturbation gene expression by combining a curated biological prior network with a learnable, data-driven refinement of that network.

### 1. Causal Prior Graph Construction

Before training, we construct a binary adjacency matrix `G_prior ∈ {0,1}^(N×N)` over the *N* genes in the dataset, encoding known regulatory and signaling relationships:

- **Transcriptional regulation (CollecTRI):** transcription factor → target gene edges, retrieved via `decoupler`/OmniPath's CollecTRI resource.
- **Signaling interactions (OmniPath):** directed protein-protein and signaling edges from the OmniPath database.

Since the dataset genes are indexed by Ensembl gene ID while both databases report HGNC gene symbols (CollecTRI) or UniProt IDs (OmniPath), we map all identifiers to a common Ensembl-indexed space using `mygene`, then populate `G_prior[i,j] = 1` wherever a directed edge `j → i` exists in either resource. Self-loops are added along the diagonal. This yields a sparse, unweighted graph that serves purely as **structural prior knowledge** — it encodes *that* two genes interact, not the strength or exact mechanism of the interaction.

### 2. Gene Tokenization

Each gene in the panel is treated as a token. For an input of *N* genes, three embeddings are summed to form the initial token representation `x ∈ R^(B×N×d)`:

| Component | Purpose |
|---|---|
| **Gene identity embedding** | `nn.Embedding(N, d)` — a unique learned vector per gene, analogous to positional encoding but semantically tied to gene identity |
| **Baseline expression projection** | `nn.Linear(1, d)` — projects the scalar control/baseline expression value for each gene into the model's embedding space |
| **Perturbation embedding** | `nn.Embedding(2, d)` — a binary flag embedding indicating whether a gene is the target of a perturbation (e.g., CRISPRa) in this sample |

### 3. Learnable Differentiable Causal Graph

Rather than treating `G_prior` as fixed, we parameterize a **learnable adjacency matrix** `G_param ∈ R^(N×N)` in logit space, initialized so that `sigmoid(G_param) ≈ G_prior` (via a logit transform of the clamped prior). During training, `G_param` is updated via backpropagation, allowing the model to:

- Reinforce edges supported by the training data
- Down-weight or prune prior edges that don't generalize
- Discover novel gene-gene dependencies not present in OmniPath/CollecTRI

The current edge probabilities are recovered at each forward pass as `G_prob = sigmoid(G_param)`, and converted into an **additive attention bias**:

```
graph_bias = alpha * log(G_prob + eps)
```

where `alpha` (`attn_alpha`) controls how strongly the graph structure constrains attention. This bias is shared across all transformer layers and computed once per forward pass.

### 4. Graph-Guided Attention Backbone

The backbone is a stack of *L* transformer blocks (`GraphGuidedAttention`), each performing standard multi-head self-attention over the *N* gene tokens, but with the causal graph bias injected as an additive mask into attention logits via `torch.nn.functional.scaled_dot_product_attention`:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + graph_bias) V
```

Genes connected in the causal graph (high `G_prob`) receive a boosted attention score toward one another, while unconnected gene pairs are suppressed — biasing the model toward biologically plausible information flow rather than learning unconstrained dense attention. Using SDPA's fused kernel avoids materializing the full `[B, H, N, N]` score tensor in memory, which is important given *N* can be in the thousands for full transcriptome panels.

Each block follows a residual + post-norm pattern:

```
x = LayerNorm(x + GraphGuidedAttention(x, graph_bias))
```

### 5. Readout Head

A final linear layer (`nn.Linear(d_model, 1)`) projects each gene's contextualized embedding down to a single scalar, producing the predicted post-perturbation expression vector `y_hat ∈ R^(B×N)`.

### 6. Loss Function

Training jointly optimizes three terms:

```
L = L_MSE(y_hat, y)
  + lambda_prior * || sigmoid(G_param) - G_prior ||_1
  + lambda_sparse * || sigmoid(G_param) ||_1
```

- **Expression loss** (`L_MSE`): fits predicted to observed post-perturbation expression.
- **Prior regularization**: anchors the learned graph to the OmniPath/CollecTRI prior, preventing it from drifting arbitrarily far from known biology.
- **Sparsity penalty**: discourages the learned graph from collapsing to a dense, uninformative adjacency matrix.
