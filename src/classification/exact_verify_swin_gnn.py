
# ============================================================
# EXACT VERIFICATION OF best_swin_gnn_improved.pth
# Reproduces the graph/model implementation from
# improved_swin_gnn.py exactly.
# ============================================================

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

SEED = 42
K = 5
DROPOUT = 0.40
HIDDEN_DIM = 256
THERMAL_FEATURES = [
    "mean_intensity", "std_intensity", "percentile_90",
    "hotspot_area_ratio", "hotspot_mean_intensity",
    "absolute_asymmetry", "relative_asymmetry"
]
CLASS_NAMES = ["Control", "DM"]
CLASS_TO_LABEL = {"Control": 0, "DM": 1}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "processed" / "standardized"
FEATURE_FILE = ROOT / "data" / "features" / "thermal_features.csv"
SWIN_FILE = ROOT / "models" / "classification" / "best_swin_transformer_thermal.pth"
GNN_FILE = ROOT / "models" / "classification" / "best_swin_gnn_improved.pth"
OUT = ROOT / "results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 75)
print("EXACT SWIN + GNN CHECKPOINT VERIFICATION")
print("=" * 75)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

def collect_images(split):
    records = []
    for cls in CLASS_NAMES:
        d = DATA_ROOT / split / cls
        for p in sorted(d.glob("*.png")):
            records.append({
                "image_path": str(p),
                "filename": p.name,
                "split": split,
                "class": cls,
                "label": CLASS_TO_LABEL[cls]
            })
    return pd.DataFrame(records)

train_df = collect_images("train")
val_df = collect_images("val")
assert len(train_df) == 1444
assert len(val_df) == 422
print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

# Exact transform from original script: no additional resize.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------- Swin --------------------

swin = models.swin_t(weights=None)
swin.head = nn.Linear(swin.head.in_features, 2)

ckpt = torch.load(SWIN_FILE, map_location=DEVICE)
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
state = {
    (k[7:] if k.startswith("module.") else k): v
    for k, v in state.items()
}
swin.load_state_dict(state, strict=True)
swin = swin.to(DEVICE).eval()
print("✓ Swin checkpoint loaded STRICTLY.")

@torch.no_grad()
def extract_embeddings(df):
    emb, labels, names = [], [], []
    for row in df.itertuples(index=False):
        with Image.open(row.image_path) as im:
            x = transform(im.convert("RGB")).unsqueeze(0).to(DEVICE)
        x = swin.features(x)
        x = swin.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        emb.append(x.cpu().numpy()[0])
        labels.append(row.label)
        names.append(row.filename)
    return np.asarray(emb, dtype=np.float32), np.asarray(labels), names

print("Extracting Swin embeddings...")
train_emb, train_y, train_names = extract_embeddings(train_df)
val_emb, val_y, val_names = extract_embeddings(val_df)
assert train_emb.shape == (1444, 768)
assert val_emb.shape == (422, 768)
print("✓ Embeddings:", train_emb.shape, val_emb.shape)

# -------------------- Thermal features --------------------

features = pd.read_csv(FEATURE_FILE)

selected = features[
    (features["noise_type"] == "gaussian") &
    (features["denoising_method"] == "median")
].copy()

assert len(selected) == 1866

def prepare_features(feature_df, filenames, split):
    temp = feature_df[feature_df["split"] == split].copy()

    # EXACT filename normalization from original script.
    temp["filename_key"] = (
        temp["filename"].astype(str)
        .str.replace("_gaussian_median", "", regex=False)
        .str.replace("_gaussian", "", regex=False)
    )

    target = pd.DataFrame({"filename": filenames})
    target["filename_key"] = (
        target["filename"].astype(str)
        .str.replace("_gaussian_median", "", regex=False)
        .str.replace("_gaussian", "", regex=False)
    )

    merged = target.merge(temp, on="filename_key", how="left")
    missing = merged[THERMAL_FEATURES].isna().any(axis=1)

    if missing.any():
        raise RuntimeError(
            f"{missing.sum()} missing feature rows for {split}"
        )

    return merged[THERMAL_FEATURES].values.astype(np.float32)

train_thermal = prepare_features(selected, train_names, "train")
val_thermal = prepare_features(selected, val_names, "val")

assert train_thermal.shape == (1444, 7)
assert val_thermal.shape == (422, 7)
print("✓ Thermal features matched exactly.")

# EXACT train-only standardization.
swin_scaler = StandardScaler()
train_swin = swin_scaler.fit_transform(train_emb)
val_swin = swin_scaler.transform(val_emb)

thermal_scaler = StandardScaler()
train_th = thermal_scaler.fit_transform(train_thermal)
val_th = thermal_scaler.transform(val_thermal)

train_features = np.concatenate([train_swin, train_th], axis=1)
val_features = np.concatenate([val_swin, val_th], axis=1)

assert train_features.shape == (1444, 775)
assert val_features.shape == (422, 775)
print("✓ 775-D features reproduced.")

train_x = torch.tensor(train_features, dtype=torch.float32, device=DEVICE)
val_x = torch.tensor(val_features, dtype=torch.float32, device=DEVICE)

# -------------------- EXACT graph from original --------------------

def build_knn_graph(features, k):
    features_np = features.detach().cpu().numpy()
    n = len(features_np)

    n_neighbors = min(k + 1, n)

    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine"
    )
    knn.fit(features_np)

    distances, indices = knn.kneighbors(features_np)

    adjacency = np.zeros((n, n), dtype=np.float32)

    # EXACT original edge weight: exp(-2*d*d)
    for i in range(n):
        for position in range(1, indices.shape[1]):
            j = indices[i, position]
            distance = distances[i, position]
            weight = np.exp(-2.0 * distance * distance)
            adjacency[i, j] = weight

    # EXACT original symmetric graph.
    adjacency = np.maximum(adjacency, adjacency.T)

    # EXACT original self loops.
    adjacency += np.eye(n, dtype=np.float32)

    # EXACT symmetric degree normalization.
    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0

    normalized = (
        degree_inv_sqrt[:, None]
        * adjacency
        * degree_inv_sqrt[None, :]
    )

    return torch.tensor(
        normalized,
        dtype=torch.float32,
        device=DEVICE
    )

print("Building EXACT K=5 graphs...")
train_graph = build_knn_graph(train_x, K)
val_graph = build_knn_graph(val_x, K)

print("Train non-zero entries:", int((train_graph != 0).sum().item()))
print("Validation non-zero entries:", int((val_graph != 0).sum().item()))

# -------------------- EXACT GNN --------------------

class ResidualGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.4):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        if input_dim == output_dim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, x, adjacency):
        residual = self.skip(x)
        x = torch.matmul(adjacency, x)
        x = self.linear(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual

class ImprovedGNN(nn.Module):
    def __init__(self, input_dim=775, hidden_dim=256,
                 num_classes=2, dropout=0.4):
        super().__init__()

        # EXACT original input projection, including GELU and 0.20 dropout.
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20)
        )

        self.gcn1 = ResidualGraphConv(hidden_dim, hidden_dim, dropout)
        self.gcn2 = ResidualGraphConv(hidden_dim, hidden_dim, dropout)
        self.gcn3 = ResidualGraphConv(hidden_dim, hidden_dim, dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, adjacency):
        x = self.input_projection(x)
        x = self.gcn1(x, adjacency)
        x = self.gcn2(x, adjacency)
        x = self.gcn3(x, adjacency)
        return self.classifier(x)

# -------------------- Load checkpoint --------------------

gnn_ckpt = torch.load(GNN_FILE, map_location=DEVICE)

print("\nCheckpoint metadata:")
for key in ["best_k", "best_epoch", "accuracy", "precision",
            "recall", "f1", "roc_auc"]:
    print(f"{key:12s}: {gnn_ckpt[key]}")

assert int(gnn_ckpt["best_k"]) == 5

model = ImprovedGNN(
    input_dim=775,
    hidden_dim=256,
    num_classes=2,
    dropout=0.4
).to(DEVICE)

gnn_state = gnn_ckpt["model_state_dict"]
gnn_state = {
    (k[7:] if k.startswith("module.") else k): v
    for k, v in gnn_state.items()
}

# This is the decisive architecture check.
model.load_state_dict(gnn_state, strict=True)
model.eval()
print("✓ GNN checkpoint loaded STRICTLY.")
print("✓ Exact original architecture reproduced.")

# -------------------- Exact inference --------------------

with torch.no_grad():
    logits = model(val_x, val_graph)
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    predictions = torch.argmax(logits, dim=1)

pred = predictions.cpu().numpy().astype(int)
prob = probabilities.cpu().numpy()
true = val_y.astype(int)

accuracy = accuracy_score(true, pred)
precision = precision_score(true, pred, zero_division=0)
recall = recall_score(true, pred, zero_division=0)
f1 = f1_score(true, pred, zero_division=0)
auc = roc_auc_score(true, prob)
cm = confusion_matrix(true, pred)

print("\n" + "=" * 75)
print("EXACT RECOMPUTED VALIDATION RESULTS")
print("=" * 75)
print(f"Accuracy : {accuracy:.10f} ({accuracy*100:.4f}%)")
print(f"Precision: {precision:.10f} ({precision*100:.4f}%)")
print(f"Recall   : {recall:.10f} ({recall*100:.4f}%)")
print(f"F1       : {f1:.10f} ({f1*100:.4f}%)")
print(f"ROC-AUC  : {auc:.10f} ({auc*100:.4f}%)")
print("\nConfusion Matrix:")
print(cm)

# -------------------- Save independent predictions --------------------

pred_df = pd.DataFrame({
    "filename": val_names,
    "true_label": true,
    "predicted_label": pred,
    "probability_DM": prob
})

pred_file = OUT / "exact_verified_swin_gnn_predictions.csv"
pred_df.to_csv(pred_file, index=False)

metrics_file = OUT / "exact_verified_swin_gnn_metrics.csv"
pd.DataFrame([{
    "model": "Improved Swin + GNN",
    "k": K,
    "validation_samples": len(true),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": auc
}]).to_csv(metrics_file, index=False)

cm_file = OUT / "exact_verified_swin_gnn_confusion_matrix.csv"
pd.DataFrame(
    cm,
    index=["Actual Control", "Actual DM"],
    columns=["Predicted Control", "Predicted DM"]
).to_csv(cm_file)

# -------------------- Compare with saved metadata --------------------

saved = {
    "accuracy": float(gnn_ckpt["accuracy"]),
    "precision": float(gnn_ckpt["precision"]),
    "recall": float(gnn_ckpt["recall"]),
    "f1": float(gnn_ckpt["f1"]),
    "roc_auc": float(gnn_ckpt["roc_auc"])
}
actual = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": auc
}

print("\n" + "=" * 75)
print("CHECKPOINT VS EXACT RECOMPUTATION")
print("=" * 75)

all_match = True
for name in saved:
    diff = abs(saved[name] - actual[name])
    ok = diff < 1e-6
    if not ok:
        all_match = False
    print(
        f"{name:10s} | "
        f"saved={saved[name]:.10f} | "
        f"recomputed={actual[name]:.10f} | "
        f"diff={diff:.10f} | "
        f"{'MATCH' if ok else 'DIFFER'}"
    )

expected_cm = np.array([[167, 3], [18, 234]])

print("\nExpected saved confusion matrix:")
print(expected_cm)
print("\nExact recomputed confusion matrix:")
print(cm)

if np.array_equal(cm, expected_cm):
    print("✓ Confusion matrix MATCH.")
else:
    print("⚠ Confusion matrix DIFFER.")
    all_match = False

print("\n" + "=" * 75)
print("FINAL VERDICT")
print("=" * 75)

if all_match:
    print("✓ 95.02% RESULT VERIFIED")
    print("✓ The saved checkpoint reproduces the reported validation result.")
else:
    print("⚠ RESULT STILL NOT VERIFIED")
    print("The checkpoint and original metadata do not reproduce exactly.")

print("\nIndependent predictions:")
print(pred_file)
print("\nIndependent metrics:")
print(metrics_file)
print("\nIndependent confusion matrix:")
print(cm_file)
