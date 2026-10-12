import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "best_swin_gnn_improved.pth"
)

print("=" * 75)
print("GNN CHECKPOINT ARCHITECTURE INSPECTOR")
print("=" * 75)

print("\nModel:")
print(MODEL_FILE)

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu"
)

print("\n" + "=" * 75)
print("CHECKPOINT METADATA")
print("=" * 75)

for key, value in checkpoint.items():

    if key != "model_state_dict":

        print(
            f"{key}: {value}"
        )

print("\n" + "=" * 75)
print("MODEL PARAMETERS")
print("=" * 75)

state_dict = checkpoint["model_state_dict"]

for key, tensor in state_dict.items():

    print(
        f"{key:45s} "
        f"shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype}"
    )

print("\n" + "=" * 75)
print("TOTAL PARAMETERS")
print("=" * 75)

total = 0

for tensor in state_dict.values():

    total += tensor.numel()

print(
    "Total parameters:",
    total
)

print("\nInspection complete.")