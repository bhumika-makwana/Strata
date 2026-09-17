import numpy as np
import torch
import pickle
import json
import os
from builtins import Exception
import pandas as pd
import torch
import boto3
import botocore
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

"""
Challenge - 
Semantic distance cannot distinguish between opposite biological functions 
(Text embeddings will group two transcription factors together just because they share phrases like "transcription factor activity.")
"""

GRAPH_DIR = os.path.join(os.getcwd(), "scgenept_gene_graph")
LOCAL_DIR = os.path.join(os.getcwd(), "scgenept_embeddings")
BUCKET_NAME = "czi-scgenept-public"
# 4 modalities 
MODALITIES = {
    "Cellular_Location": "models/gene_embeddings/GO_C_gene_embeddings-gpt3.5-ada-concat.pickle",
    "Molecular_Function": "models/gene_embeddings/GO_F_gene_embeddings-gpt3.5-ada-concat.pickle",
    "Biological_Process": "models/gene_embeddings/GO_P_gene_embeddings-gpt3.5-ada-concat.pickle",
    "NCBI_and_UniProt": "models/gene_embeddings/NCBI+UniProt_embeddings-gpt3.5-ada.pkl"
}

def load_multimodal_registry(dir):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    s3 = boto3.client("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    loaded_dictionaries = {}
    for name, s3_key in MODALITIES.items():
        local_path = os.path.join(LOCAL_DIR, f"{name}.pickle")
        if not os.path.exists(local_path):
            print(f"Downloading {name} prior embeddings from CZI repository...")
            try:
                s3.download_file(BUCKET_NAME, s3_key, local_path)
            except Exception as e:
                print(f"Critical error on key {s3_key}: {e}")
                continue

        data = pd.read_pickle(local_path)
        if isinstance(data, pd.DataFrame):
            loaded_dictionaries[name] = data.to_dict(orient='index')
        else:
            loaded_dictionaries[name] = {str(k).upper(): v for k, v in data.items()}
    
    print(f"Loaded {len(loaded_dictionaries)} of {len(MODALITIES)} modalities.")
    

    # Gene Alignment: Find the absolute intersection of genes across ALL knowledge bases
    #Takes the intersection of gene symbols across all 4 dicts (set.intersection(*all_gene_sets)) a gene only survives if it has an embedding in every single one of the 4 sources.
    all_gene_sets = [set(loaded_dictionaries[name].keys()) for name in loaded_dictionaries.keys()]
    universal_genes = sorted(list(set.intersection(*all_gene_sets)))
    print(f"\nUnified SOTA Registry established with {len(universal_genes)} high-coverage human genes.")

    sota_multimodal_registry = {}
    for gene in universal_genes:
        vector_slices = []
        for name in loaded_dictionaries.keys():
            val = loaded_dictionaries[name][gene]
            if isinstance(val, dict):
                val = list(val.values())
            vector_slices.append(np.array(val))
        sota_multimodal_registry[gene] = np.concatenate(vector_slices)

    # Calculate exact tensor footprint length -- 4 modalities * 1536 dims = 6144
    total_dimension = len(next(iter(sota_multimodal_registry.values())))
    print(f"Total features per gene extended to: {total_dimension} variables.")
    return sota_multimodal_registry, total_dimension


def get_embed_multimodal_tensor(multimodal_registry, total_dimension, target_gene_symbols):
    cleaned_symbols = [g.strip().upper() for g in target_gene_symbols]
    zero_vector = np.zeros(total_dimension)
    vectors = [multimodal_registry.get(gene, zero_vector) for gene in cleaned_symbols]
    missing_count = sum(1 for v in vectors if v is zero_vector)
    if missing_count > 0:
        print(f"[Warning] {missing_count} target genes missing from data. Zero-padded.")
            
    return torch.tensor(np.stack(vectors), dtype=torch.float32)

def build_directed_gene_graph(multimodal_registry, target_genes, top_k=3):
    """
    Builds a directed graph: edge A->B if B is among A's top_k most
    similar genes (by cosine similarity of multimodal embeddings).
    Edge weight = row-normalized similarity (asymmetric "influence" score).
    """
    genes = [g.strip().upper() for g in target_genes]
    genes = [g for g in genes if g in multimodal_registry]
    missing = set(target_genes) - set(genes)
    if missing:
        print(f"[Warning] Skipping genes not in registry: {missing}")

    vectors = np.stack([multimodal_registry[g] for g in genes])
    sim_matrix = cosine_similarity(vectors)          # symmetric, shape (n, n)
    np.fill_diagonal(sim_matrix, -np.inf)             # exclude self-similarity

    # Row-normalize -> asymmetric "influence" weights (denominator differs per row)
    row_shifted = np.where(sim_matrix == -np.inf, 0, sim_matrix)
    row_sums = row_shifted.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div-by-zero
    influence_matrix = row_shifted / row_sums

    G = nx.DiGraph()
    G.add_nodes_from(genes)
    #1. Topological asymmetry (k-NN graph): Draw an edge A→B if B is among A's top-k most similar genes. Since "A's nearest neighbors" and "B's nearest neighbors" don't have to be the same set, this graph is naturally directed even though the underlying metric is symmetric.
    #2. Weight asymmetry (row-normalized "influence"): Normalize each gene's similarity row so it sums to 1 (like a softmax/attention weight). Then W[A→B] = sim(A,B) / Σ_k sim(A,k) while W[B→A] = sim(A,B) / Σ_k sim(B,k)
    for i, gene in enumerate(genes):
        top_indices = np.argsort(sim_matrix[i])[::-1][:top_k]
        for j in top_indices:
            if sim_matrix[i, j] == -np.inf:
                continue
            G.add_edge(gene, genes[j],
                       weight=influence_matrix[i, j],
                       raw_similarity=sim_matrix[i, j])
    return G


# What the graph represents: each node is a gene, and each node's features are its 6144-dim embedding (concatenation of 4 modalities: GO Cellular Component, GO Molecular Function, GO Biological Process, and NCBI+UniProt text embeddings).
# How edges are built, step by step:
# > For every gene, find its 10 nearest neighbors in embedding space using cosine similarity (via NearestNeighbors, done in batches so it never needs a full 16k×16k matrix in memory).
# > Draw a directed edge gene → neighbor for each of those 10. This is what makes it directed at all: gene A's top-10 list and gene B's top-10 list aren't guaranteed to match each other, so A→B doesn't imply B→A automatically.
# > Assign each edge a weight based on rank, not raw similarity — the nearest neighbor (rank 1) gets the highest weight, the 10th-nearest gets the lowest, normalized to sum to 1 per gene.

#End result: 16,293 nodes, 162,930 directed edges (10 outgoing per gene), each edge carrying a rank-based weight that differs depending on which direction you're looking (A→B weight ≠ B→A weight, even when both edges exist), 
#stored as plain arrays (node_features.npy, edge_index.npy, edge_weight.npy, gene2idx.json)
def build_and_store_gnn_graph(multimodal_registry, total_dimension,
                                top_k=10, batch_size=2000, out_dir=GRAPH_DIR,
                                weighting="rank"):  # "rank" or "raw_norm"
    os.makedirs(out_dir, exist_ok=True)

    genes = list(multimodal_registry.keys())
    n = len(genes)
    gene2idx = {g: i for i, g in enumerate(genes)}
    idx2gene = {i: g for g, i in gene2idx.items()}

    print(f"Building GNN graph store for {n} genes, top_k={top_k}, weighting={weighting}...")

    node_features = np.stack([multimodal_registry[g] for g in genes]).astype(np.float32)

    nn_model = NearestNeighbors(n_neighbors=top_k + 1, metric="cosine", algorithm="brute")
    nn_model.fit(node_features)

    src_list, dst_list, weight_list, sim_list = [], [], [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        distances, indices = nn_model.kneighbors(node_features[start:end])

        for row_offset, gene_idx in enumerate(range(start, end)):
            row_dists = distances[row_offset]
            row_idxs = indices[row_offset]

            pairs = [(idx, 1 - dist) for idx, dist in zip(row_idxs, row_dists) if idx != gene_idx][:top_k]
            if not pairs:
                continue

            k_here = len(pairs)
            if weighting == "rank": 
                # nearest neighbor (rank 1) gets highest weight, farthest gets lowest
                # this is asymmetric by construction: A's rank-of-B != B's rank-of-A
                norm_weights = np.array([(k_here - r) / k_here for r in range(k_here)], dtype=np.float32)
                norm_weights = norm_weights / norm_weights.sum()  # sums to 1, like before
            else:  # raw_norm, kept for comparison/backward-compat
                sims = np.array([s for _, s in pairs], dtype=np.float32)
                sim_sum = sims.sum() if sims.sum() > 0 else 1.0
                norm_weights = sims / sim_sum

            for (nbr_idx, sim), w in zip(pairs, norm_weights):
                src_list.append(gene_idx)
                dst_list.append(nbr_idx)
                weight_list.append(w)
                sim_list.append(sim)

        if (start // batch_size) % 5 == 0:
            print(f"  processed {end}/{n} genes -> {len(src_list)} edges so far")

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(weight_list, dtype=np.float32)
    edge_raw_sim = np.array(sim_list, dtype=np.float32)

    print(f"Done. {n} nodes, {edge_index.shape[1]} directed edges, "
          f"{node_features.shape[1]}-dim node features.")

    np.save(os.path.join(out_dir, "node_features.npy"), node_features)
    np.save(os.path.join(out_dir, "edge_index.npy"), edge_index)
    np.save(os.path.join(out_dir, "edge_weight.npy"), edge_weight)
    np.save(os.path.join(out_dir, "edge_raw_similarity.npy"), edge_raw_sim)

    with open(os.path.join(out_dir, "gene2idx.json"), "w") as f:
        json.dump(gene2idx, f)
    with open(os.path.join(out_dir, "idx2gene.json"), "w") as f:
        json.dump(idx2gene, f)

    try:
        from torch_geometric.data import Data
        pyg_data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_weight, dtype=torch.float32).unsqueeze(-1),
        )
        torch.save(pyg_data, os.path.join(out_dir, "gene_graph_pyg.pt"))
        print(f"Saved PyG Data object -> {out_dir}/gene_graph_pyg.pt")
    except ImportError:
        print("[Info] torch_geometric not installed -- skipped .pt export.")

    meta = {"num_nodes": n, "num_edges": int(edge_index.shape[1]),
             "feature_dim": int(node_features.shape[1]), "top_k": top_k,
             "weighting": weighting}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"edge_index": edge_index, "edge_weight": edge_weight,
            "edge_raw_similarity": edge_raw_sim, "node_features": node_features,
            "gene2idx": gene2idx, "idx2gene": idx2gene}

def load_gnn_graph(out_dir=GRAPH_DIR):
    """Reload everything a training script needs, without recomputation."""
    node_features = np.load(os.path.join(out_dir, "node_features.npy"))
    edge_index = np.load(os.path.join(out_dir, "edge_index.npy"))
    edge_weight = np.load(os.path.join(out_dir, "edge_weight.npy"))
    with open(os.path.join(out_dir, "gene2idx.json")) as f:
        gene2idx = json.load(f)
    return node_features, edge_index, edge_weight, gene2idx

if __name__ == "__main__":
    multimodal_registry, total_dimension = load_multimodal_registry(LOCAL_DIR)

    graph_store = build_and_store_gnn_graph(
    multimodal_registry, total_dimension, top_k=10, batch_size=2000, weighting="rank"
)



# from builtins import Exception
# import os
# import pandas as pd
# import numpy as np
# import torch
# import boto3
# import botocore
# import networkx as nx
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity

# LOCAL_DIR = os.path.join(os.getcwd(), "scgenept_embeddings")
# BUCKET_NAME = "czi-scgenept-public"
# # 4 modalities 
# MODALITIES = {
#     "Cellular_Location": "models/gene_embeddings/GO_C_gene_embeddings-gpt3.5-ada-concat.pickle",
#     "Molecular_Function": "models/gene_embeddings/GO_F_gene_embeddings-gpt3.5-ada-concat.pickle",
#     "Biological_Process": "models/gene_embeddings/GO_P_gene_embeddings-gpt3.5-ada-concat.pickle",
#     "NCBI_and_UniProt": "models/gene_embeddings/NCBI+UniProt_embeddings-gpt3.5-ada.pkl"
# }

# def load_multimodal_registry(dir):
#     os.makedirs(LOCAL_DIR, exist_ok=True)
#     s3 = boto3.client("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))
#     loaded_dictionaries = {}
#     for name, s3_key in MODALITIES.items():
#         local_path = os.path.join(LOCAL_DIR, f"{name}.pickle")
#         if not os.path.exists(local_path):
#             print(f"Downloading {name} prior embeddings from CZI repository...")
#             try:
#                 s3.download_file(BUCKET_NAME, s3_key, local_path)
#             except Exception as e:
#                 print(f"Critical error on key {s3_key}: {e}")
#                 continue

#         data = pd.read_pickle(local_path)
#         if isinstance(data, pd.DataFrame):
#             loaded_dictionaries[name] = data.to_dict(orient='index')
#         else:
#             loaded_dictionaries[name] = {str(k).upper(): v for k, v in data.items()}
    
#     print(f"Loaded {len(loaded_dictionaries)} of {len(MODALITIES)} modalities.")
    

#     # Gene Alignment: Find the absolute intersection of genes across ALL knowledge bases
#     #Takes the intersection of gene symbols across all 4 dicts (set.intersection(*all_gene_sets)) a gene only survives if it has an embedding in every single one of the 4 sources.
#     all_gene_sets = [set(loaded_dictionaries[name].keys()) for name in loaded_dictionaries.keys()]
#     universal_genes = sorted(list(set.intersection(*all_gene_sets)))
#     print(f"\nUnified SOTA Registry established with {len(universal_genes)} high-coverage human genes.")

#     sota_multimodal_registry = {}
#     for gene in universal_genes:
#         vector_slices = []
#         for name in loaded_dictionaries.keys():
#             val = loaded_dictionaries[name][gene]
#             if isinstance(val, dict):
#                 val = list(val.values())
#             vector_slices.append(np.array(val))
#         sota_multimodal_registry[gene] = np.concatenate(vector_slices)

#     # Calculate exact tensor footprint length -- 4 modalities * 1536 dims = 6144
#     total_dimension = len(next(iter(sota_multimodal_registry.values())))
#     print(f"Total features per gene extended to: {total_dimension} variables.")
#     return sota_multimodal_registry, total_dimension


# def get_embed_multimodal_tensor(multimodal_registry, total_dimension, target_gene_symbols):
#     cleaned_symbols = [g.strip().upper() for g in target_gene_symbols]
#     zero_vector = np.zeros(total_dimension)
#     vectors = [multimodal_registry.get(gene, zero_vector) for gene in cleaned_symbols]
#     missing_count = sum(1 for v in vectors if v is zero_vector)
#     if missing_count > 0:
#         print(f"[Warning] {missing_count} target genes missing from data. Zero-padded.")
            
#     return torch.tensor(np.stack(vectors), dtype=torch.float32)

# def build_directed_gene_graph(multimodal_registry, target_genes, top_k=3):
#     """
#     Builds a directed graph: edge A->B if B is among A's top_k most
#     similar genes (by cosine similarity of multimodal embeddings).
#     Edge weight = row-normalized similarity (asymmetric "influence" score).
#     """
#     genes = [g.strip().upper() for g in target_genes]
#     genes = [g for g in genes if g in multimodal_registry]
#     missing = set(target_genes) - set(genes)
#     if missing:
#         print(f"[Warning] Skipping genes not in registry: {missing}")

#     vectors = np.stack([multimodal_registry[g] for g in genes])
#     sim_matrix = cosine_similarity(vectors)          # symmetric, shape (n, n)
#     np.fill_diagonal(sim_matrix, -np.inf)             # exclude self-similarity

#     # Row-normalize -> asymmetric "influence" weights (denominator differs per row)
#     row_shifted = np.where(sim_matrix == -np.inf, 0, sim_matrix)
#     row_sums = row_shifted.sum(axis=1, keepdims=True)
#     row_sums[row_sums == 0] = 1  # avoid div-by-zero
#     influence_matrix = row_shifted / row_sums

#     G = nx.DiGraph()
#     G.add_nodes_from(genes)
#     #1. Topological asymmetry (k-NN graph): Draw an edge A→B if B is among A's top-k most similar genes. Since "A's nearest neighbors" and "B's nearest neighbors" don't have to be the same set, this graph is naturally directed even though the underlying metric is symmetric.
#     #2. Weight asymmetry (row-normalized "influence"): Normalize each gene's similarity row so it sums to 1 (like a softmax/attention weight). Then W[A→B] = sim(A,B) / Σ_k sim(A,k) while W[B→A] = sim(A,B) / Σ_k sim(B,k)
#     for i, gene in enumerate(genes):
#         top_indices = np.argsort(sim_matrix[i])[::-1][:top_k]
#         for j in top_indices:
#             if sim_matrix[i, j] == -np.inf:
#                 continue
#             G.add_edge(gene, genes[j],
#                        weight=influence_matrix[i, j],
#                        raw_similarity=sim_matrix[i, j])
#     return G


# def plot_directed_gene_graph(G, title="Gene Multimodal Embedding Graph (Directed)"):
#     pos = nx.spring_layout(G, seed=42, k=0.8)
#     weights = [G[u][v]['weight'] * 15 for u, v in G.edges()]  # scale for visibility

#     plt.figure(figsize=(9, 7))
#     nx.draw_networkx_nodes(G, pos, node_size=1400, node_color="#7fc7ff", edgecolors="black")
#     nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
#     # curved arcs so A->B and B->A (if both exist) don't overlap into one line
#     nx.draw_networkx_edges(
#         G, pos, width=weights, arrowstyle="-|>", arrowsize=18,
#         connectionstyle="arc3,rad=0.12", edge_color="#555555"
#     )
#     plt.title(title)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig("gene_graph.png", dpi=200)
#     plt.show()


# if __name__ == "__main__":
#     my_target_genes = ["TP53", "BRCA1", "MYC", "EGFR", "GAPDH"]
#     multimodal_registry, total_dimension = load_multimodal_registry(LOCAL_DIR)
#     embed_features = get_embed_multimodal_tensor(multimodal_registry, total_dimension, my_target_genes)
#     print(f"\nRetrieved embedding tensor -- shape: {embed_features.shape}")

#     G = build_directed_gene_graph(multimodal_registry, my_target_genes, top_k=3)
#     print(f"\nDirected graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
#     for u, v, d in G.edges(data=True):
#         print(f"  {u} -> {v}  (influence={d['weight']:.3f}, raw_sim={d['raw_similarity']:.3f})")

#     plot_directed_gene_graph(G)
   