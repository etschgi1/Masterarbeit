import torch
print("Using PyTorch version:", torch.__version__)
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if torch.cuda.is_available():
    print("Cuda is available")
    print("With Cuda version:", torch.version.cuda)
import torch.nn.functional as F
from tqdm import tqdm
from scf_guess_tools import Backend, load
from MolGraphNetwork import MolGraphNetwork


BASIS_PATH = "../../../scripts/6-31g_2df_p_custom_nwchem.gbs"
GEOMETRY_Source = "../../../datasets/QM9/xyz_c7h10o2_sorted"

# 1. Instantiate your model and move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MGNN = MolGraphNetwork(xyz_source=GEOMETRY_Source,
                       backend=Backend.PY,
                       basis=BASIS_PATH,
                       batch_size=2,
                       train_val_test_ratio=(0.5, 0.25, 0.25))
MGNN.load_data(max_samples=20) 
MGNN = MGNN.to(device)

# 2. Choose optimizer
optimizer = torch.optim.Adam(MGNN.parameters(), lr=1e-3, weight_decay=1e-5)

# 3. Simple training loop
num_epochs = 5

for epoch in range(1, num_epochs + 1):
    MGNN.train()
    total_train_loss = 0.0

    # Iterate over training batches
    for batch in tqdm(MGNN.train_loader, desc=f"Epoch {epoch} [Train]"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass: batch now contains pred_center_blocks & pred_edge_blocks
        batch = MGNN.forward(batch)

        # 4. Compute loss over all nodes and edges in this batch
        loss_center = 0.0
        for i in range(batch.num_nodes):
            pred = batch.pred_center_blocks[i]                  # Tensor of length = center_sizes[ atom_sym[i] ]
            true = batch.target_center_blocks[i].to(device)      # same length
            loss_center += F.mse_loss(pred, true)

        loss_edge = 0.0
        for k in range(batch.num_edges):
            pred = batch.pred_edge_blocks[k]                    # Tensor of length = edge_sizes[ edge_pair_sym[k] ]
            true = batch.target_edge_blocks[k].to(device)        # same length
            loss_edge += F.mse_loss(pred, true)

        # (Optionally normalize by number of graphs in batch)
        loss = (loss_center + loss_edge) / batch.num_graphs
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(MGNN.train_loader)
    print(f"Epoch {epoch}/{num_epochs} → Avg Train Loss: {avg_train_loss:.6f}")

    # 5. (Optional) Validation pass
    MGNN.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(MGNN.val_loader, desc=f"Epoch {epoch} [Val]"):
            batch = batch.to(device)
            batch = MGNN.forward(batch)

            loss_center = 0.0
            for i in range(batch.num_nodes):
                pred = batch.pred_center_blocks[i]
                true = batch.target_center_blocks[i].to(device)
                loss_center += F.mse_loss(pred, true)

            loss_edge = 0.0
            for k in range(batch.num_edges):
                pred = batch.pred_edge_blocks[k]
                true = batch.target_edge_blocks[k].to(device)
                loss_edge += F.mse_loss(pred, true)

            loss = (loss_center + loss_edge) / batch.num_graphs
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(MGNN.val_loader)
    print(f"Epoch {epoch}/{num_epochs} → Avg Val   Loss: {avg_val_loss:.6f}")
