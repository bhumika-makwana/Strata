import os
import scanpy as sc
import torch
from gears import PertData

data_path = "./data"
if not os.path.exists(data_path):
    os.makedirs(data_path)
    
pert_data = PertData(data_path) #Initialize PertData Loader
pert_data.load(data_name="norman")
pert_data.prepare_split(split='combo_seen2', seed=42) #Create Train/Validation/Test Splits -"combo_seen2" split to ensure 2-gene combinations are completely hidden from training.

pert_data.get_dataloader(batch_size=32, test_batch_size=32) #Create PyTorch Dataloaders
train_loader = pert_data.dataloader['train_loader']
val_loader = pert_data.dataloader['val_loader']
test_loader = pert_data.dataloader['test_loader']

print("\nInspecting Data Structure")
for batch in train_loader:
    # 'x' usually contains the baseline gene expression or cell features
    # 'pert' contains the indices or names of the perturbed transcription factors
    # 'y' contains the true post-perturbation gene expression matrix (the target your model predicts)
    print(f"Batch keys available: {list(batch.keys())}")
    
    # Shape layout: [Batch_Size, Number_of_Genes_Profiled]
    print(f"Target Gene Expression Matrix Shape (y): {batch['y'].shape}")
    break

print("\nSetup complete.")
