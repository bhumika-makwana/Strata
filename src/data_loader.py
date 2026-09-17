#The Norman 2019 dataset contains ~91,000 K562 cells with 105 distinct genetic targets.
#The data is stored as an AnnData object

"""
The Dataset does not contain unperturbed expression values for every perturbed cell: A single cell can only be sequenced once because the process requires lysing (breaking open) the cell and extracting its genetic material, which completely destroys the physical cell.
Because a cell can only be sequenced once (which destroys it), it is biologically impossible to know what an individual perturbed cell looked like before it was perturbed.

Global baseline strategy (used by GEARS):
> all 7,353 cells labeled as ctrl in dataset, averages their expression profiles together, and creates one single, static baseline vector (mu _{ctrl}).
> Identical Reference Input: Every single time a perturbed cell is processed, the same baseline vector (mu _{ctrl}) is used as the reference input to the model.

Now, each cell might have different starting points (natural "noise" or differences between individual cells in the same group.)?
> The Idea minimizing is that the Mean Squared Error (MSE) across hundreds or thousands of individual cells from the same perturbation group, 
the model naturally learns to predict the average expected shift for that population, 
effectively filtering out the random single-cell noise.
"""

from gears import PertData  # use the GEARS library to load the data and create the train/test split
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# split info: {'train_loader': 49849, 'val_loader': 10754, 'test_loader': 28754}
# === Condition-level counts (unique perturbations) ===
# train :  139 conditions  (99 single-gene, 39 double-gene)
# val   :   31 conditions  (13 single-gene, 18 double-gene)
# test  :  107 conditions  (36 single-gene, 71 double-gene)

# === Cell-level counts ===
# train_loader:  49849 cells
# val_loader  :  10754 cells
# test_loader :  28754 cells

SEED = 1
DATA_ROOT = "./data"
SPLIT_TYPE = "simulation"  # training set - single gene modification and some double gene modifications,
                            # testing set - tested on double-gene combinations where both genes were seen independently
                            # individually during training, but the combination itself is entirely new
save_dir = f"./data/my_norman_split_meta/{SPLIT_TYPE}_seed{SEED}"
os.makedirs(save_dir, exist_ok=True)


#AnnData - handle annotated data matrices in memory [Python and R data structure used to handle large, multi-dimensional data matrices along with metadata for observations and features.]
def load_pert_data(data_root):
    pert_data = PertData(data_root)
    pert_data.load(data_name="norman")
    pert_data.prepare_split(split=SPLIT_TYPE, seed=SEED)
    return pert_data


def export_split_conditions(pert_data):
    set2conditions = pert_data.set2conditions  # {'train': [...], 'val': [...], 'test': [...]}
    split_info = {
        "seed": SEED,
        "split_type": SPLIT_TYPE,
        "train_conditions": set2conditions["train"],
        "val_conditions": set2conditions.get("val", []),
        "test_conditions": set2conditions["test"],
    }
    with open(os.path.join(save_dir, "split_conditions.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    return split_info


#Fetch the baseline expression values for the control cells, grouped by batch 
def compute_ctrl_baseline(adata, group_col="batch"):
    """Group-aware control baseline instead of one pooled global mean.
    Returns dict {group_key: np.array[n_genes]}, plus always includes 'global'."""
    ctrl_mask = adata.obs['condition'] == 'ctrl'
    baselines = {}

    expr_all = adata[ctrl_mask].X
    baselines["global"] = np.asarray(
        (expr_all.mean(axis=0) if hasattr(expr_all, "mean") else expr_all.mean(0))
    ).flatten()

    if group_col in adata.obs.columns:
        for group_id, sub in adata.obs.loc[ctrl_mask].groupby(group_col):
            idx = sub.index
            expr = adata[idx].X
            baselines[str(group_id)] = np.asarray(expr.mean(axis=0)).flatten()

    return baselines


class PerturbationDataset(Dataset):
    """
    Each sample:
        pert_genes  -> list[str], the perturbed gene name(s) for this cell (empty for ctrl)
        pert_idx    -> list[int], indices of those genes into gene_names (for embedding lookup)
        expression  -> torch.FloatTensor [n_genes], observed post-perturbation expression
        baseline    -> torch.FloatTensor [n_genes], matched control baseline for this cell's group
    """

    def __init__(self, adata, conditions, baseline_dict, gene_names, group_col="batch"):
        mask = adata.obs['condition'].isin(conditions)
        self.adata_sub = adata[mask]

        X = self.adata_sub.X
        self.expr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        self.conditions = self.adata_sub.obs['condition'].values
        self.group_col = group_col if group_col in adata.obs.columns else None
        self.groups = self.adata_sub.obs[group_col].values if self.group_col else None

        self.gene_names = gene_names
        self.gene2idx = {g: i for i, g in enumerate(gene_names)}
        self.baseline_dict = baseline_dict  # {group_key: np.array[n_genes]}, includes "global"

    def __len__(self):
        return self.expr.shape[0]

    def __getitem__(self, idx):
        cond = self.conditions[idx]
        pert_genes = [g for g in cond.split('+') if g != 'ctrl']

        baseline_key = str(self.groups[idx]) if self.groups is not None else "global"
        baseline = self.baseline_dict.get(baseline_key, self.baseline_dict["global"])

        return {
            "pert_genes": pert_genes,  # e.g. ["KLF1", "MAP2K6"] or [] for ctrl
            "pert_idx": [self.gene2idx[g] for g in pert_genes if g in self.gene2idx],
            "expression": torch.tensor(self.expr[idx], dtype=torch.float32),
            "baseline": torch.tensor(baseline, dtype=torch.float32),
        }


def perturbation_collate(batch, pad_value=-1):
    """Pads variable-length pert_idx lists (0, 1, or 2+ perturbed genes) within a batch."""
    max_perts = max((len(b["pert_idx"]) for b in batch), default=0)
    max_perts = max(max_perts, 1)  # avoid zero-width tensor if a whole batch is ctrl-only
    pert_idx_padded = torch.full((len(batch), max_perts), pad_value, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["pert_idx"])
        if n > 0:
            pert_idx_padded[i, :n] = torch.tensor(b["pert_idx"], dtype=torch.long)

    return {
        "pert_genes": [b["pert_genes"] for b in batch],  # list of lists, variable length (debug/logging)
        "pert_idx": pert_idx_padded,                       # [batch, max_perts], padded with pad_value
        "expression": torch.stack([b["expression"] for b in batch]),
        "baseline": torch.stack([b["baseline"] for b in batch]),
    }


def build_custom_datasets(pert_data, baseline_dict, split_info, group_col="batch"):
    gene_names = pert_data.adata.var_names.tolist()
    train_ds = PerturbationDataset(pert_data.adata, split_info["train_conditions"], baseline_dict, gene_names, group_col)
    val_ds = PerturbationDataset(pert_data.adata, split_info["val_conditions"], baseline_dict, gene_names, group_col)
    test_ds = PerturbationDataset(pert_data.adata, split_info["test_conditions"], baseline_dict, gene_names, group_col)
    return train_ds, val_ds, test_ds


def build_custom_loaders(train_ds, val_ds, test_ds, batch_size=32, test_batch_size=128):
    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=perturbation_collate),
        "val_loader": DataLoader(val_ds, batch_size=test_batch_size, shuffle=False, collate_fn=perturbation_collate),
        "test_loader": DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, collate_fn=perturbation_collate),
    }

def init_main():
    pert_data = load_pert_data(DATA_ROOT)
    adata = pert_data.adata

    split_info = export_split_conditions(pert_data)
    baseline_dict = compute_ctrl_baseline(adata, group_col="batch")

    # Save metadata + baselines for reproducibility / downstream analysis
    baseline_keys = list(baseline_dict.keys())
    baseline_matrix = np.stack([baseline_dict[k] for k in baseline_keys])
    np.savez_compressed(
        os.path.join(save_dir, f"meta_seed{SEED}.npz"),
        gene_names=np.array(adata.var_names.tolist()),
        train_conditions=np.array(split_info["train_conditions"], dtype=object),
        val_conditions=np.array(split_info["val_conditions"], dtype=object),
        test_conditions=np.array(split_info["test_conditions"], dtype=object),
        baseline_keys=np.array(baseline_keys),
        baseline_matrix=baseline_matrix,
    )

    train_ds, val_ds, test_ds = build_custom_datasets(pert_data, baseline_dict, split_info, group_col="batch")
    loaders = build_custom_loaders(train_ds, val_ds, test_ds)
    return pert_data, loaders


def print_split_summary(pert_data, loaders):
    set2conditions = pert_data.set2conditions

    print("=== Condition-level counts (unique perturbations) ===")
    for split in ["train", "val", "test"]:
        n_conditions = len(set2conditions.get(split, []))
        n_single = sum(1 for c in set2conditions.get(split, []) if len(c.split('+')) == 2 and 'ctrl' in c)
        n_combo = sum(1 for c in set2conditions.get(split, []) if len(c.split('+')) == 2 and 'ctrl' not in c)
        print(f"{split:6s}: {n_conditions:4d} conditions  ({n_single} single-gene, {n_combo} double-gene)")

    print("\n=== Cell-level counts ===")
    for name, loader in loaders.items():
        print(f"{name:12s}: {len(loader.dataset):6d} cells")


if __name__ == "__main__":
    pert_data, loaders = init_main()
    print_split_summary(pert_data, loaders)
    print({k: len(v.dataset) for k, v in loaders.items()})
    sample_batch = next(iter(loaders["train_loader"]))
    print("expression:", sample_batch["expression"].shape)
    print("baseline:", sample_batch["baseline"].shape)
    print("pert_idx:", sample_batch["pert_idx"].shape)
    print("example pert_genes:", sample_batch["pert_genes"][:5])