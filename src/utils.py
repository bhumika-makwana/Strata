import torch
def test_gene_graph_symmetry():
    data = torch.load("scgenept_gene_graph/gene_graph_pyg.pt", weights_only=False)

    edge_set = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())) 
    edge_weight_lookup = {(s, d): w for (s, d), w in
                        zip(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()),
                            data.edge_attr.squeeze().tolist())}

    diffs = []
    for (a, b) in list(edge_set)[:5000]:
        if (b, a) in edge_weight_lookup:
            diffs.append(abs(edge_weight_lookup[(a, b)] - edge_weight_lookup[(b, a)]))

    if diffs:
        print(f"mutual pairs checked: {len(diffs)}")
        print(f"avg |w(A->B) - w(B->A)|: {sum(diffs)/len(diffs):.5f}")
        print(f"max |w(A->B) - w(B->A)|: {max(diffs):.5f}")
        print(f"min |w(A->B) - w(B->A)|: {min(diffs):.5f}")

