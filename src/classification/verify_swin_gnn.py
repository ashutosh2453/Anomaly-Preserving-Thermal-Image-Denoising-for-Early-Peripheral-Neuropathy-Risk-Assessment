    # ============================================================
# VERIFY IMPROVED SWIN TRANSFORMER + GNN
# Independent validation audit - NO RETRAINING
# ============================================================

import os
import re
import sys
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import swin_t

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "standardized"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "thermal_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "best_swin_gnn_improved.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 2

SWIN_DIM = 768

THERMAL_DIM = 7

TOTAL_DIM = SWIN_DIM + THERMAL_DIM

K = 5

BATCH_SIZE = 16

SEED = 42


THERMAL_FEATURES = [
    "mean_intensity",
    "std_intensity",
    "percentile_90",
    "hotspot_area_ratio",
    "hotspot_mean_intensity",
    "absolute_asymmetry",
    "relative_asymmetry",
]


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("INDEPENDENT SWIN + GNN VERIFICATION")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDevice:")
print(DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nModel:")
print(MODEL_FILE)


# ============================================================
# 5. CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

required_files = [
    DATASET_ROOT,
    FEATURE_FILE,
    MODEL_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file/folder not found:\n{file_path}"
        )

    print("✓", file_path)


# ============================================================
# 6. DATASET COLLECTION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION DATASET")
print("=" * 70)


def collect_images(split):

    records = []

    split_dir = DATASET_ROOT / split

    class_mapping = {
        "Control": 0,
        "DM": 1,
    }

    for class_name, label in class_mapping.items():

        class_dir = split_dir / class_name

        if not class_dir.exists():
            continue

        for image_path in sorted(class_dir.glob("*")):

            if image_path.suffix.lower() not in [
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
            ]:
                continue

            records.append(
                {
                    "image_path": str(image_path),
                    "filename": image_path.name,
                    "class": class_name,
                    "label": label,
                }
            )

    return pd.DataFrame(records)


train_df = collect_images("train")
val_df = collect_images("val")

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))

print("\nValidation class distribution:")
print(val_df["class"].value_counts())


# ============================================================
# 7. VALIDATION DATASET CHECKS
# ============================================================

print("\nValidation duplicate checks...")

duplicate_paths = val_df["image_path"].duplicated().sum()

duplicate_names = val_df["filename"].duplicated().sum()

print("Duplicate paths:", duplicate_paths)
print("Duplicate filenames:", duplicate_names)

if duplicate_paths != 0:
    raise RuntimeError(
        "Duplicate validation image paths detected."
    )


if len(val_df) != 422:

    print(
        "\nWARNING:"
        f" Expected 422 validation images but found {len(val_df)}."
    )

else:

    print("✓ Exactly 422 validation images found.")


# ============================================================
# 8. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# 9. LOAD SWIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING SWIN TRANSFORMER")
print("=" * 70)


swin_model = swin_t(
    weights=None
)

swin_model.head = nn.Linear(
    swin_model.head.in_features,
    NUM_CLASSES,
)

swin_model = swin_model.to(DEVICE)

swin_model.eval()


# ============================================================
# 10. EXTRACT SWIN EMBEDDINGS
# ============================================================

print("\nExtracting Swin embeddings...")


class SwinEmbeddingModel(nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model

    def forward(self, x):

        x = self.model.features(x)

        x = self.model.norm(x)

        x = self.model.permute(x)

        x = self.model.avgpool(x)

        x = torch.flatten(x, 1)

        return x


# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# We first load the checkpoint.
# The checkpoint may contain either:
#   model_state_dict
#   state_dict
#   swin_state_dict
# ------------------------------------------------------------


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
)

print("\nCheckpoint type:", type(checkpoint))


if isinstance(checkpoint, dict):

    print("\nCheckpoint keys:")

    for key in checkpoint.keys():

        print(" ", key)


# ============================================================
# 11. FIND SWIN WEIGHTS
# ============================================================

def find_state_dict(checkpoint):

    if not isinstance(checkpoint, dict):
        return None

    possible_keys = [
        "swin_state_dict",
        "swin_model_state_dict",
        "model_state_dict",
        "state_dict",
        "model",
    ]

    for key in possible_keys:

        if key in checkpoint:

            value = checkpoint[key]

            if isinstance(value, dict):

                return value

    # If checkpoint itself looks like a state_dict

    if any(
        isinstance(k, str)
        and (
            k.startswith("features.")
            or k.startswith("head.")
            or k.startswith("norm.")
        )
        for k in checkpoint.keys()
    ):

        return checkpoint

    return None


state_dict = find_state_dict(checkpoint)


if state_dict is None:

    raise RuntimeError(
        "\nCould not find a Swin state_dict inside the saved model."
    )


# Remove possible "module." prefix

clean_state_dict = {}

for key, value in state_dict.items():

    new_key = key

    if new_key.startswith("module."):

        new_key = new_key[7:]

    clean_state_dict[new_key] = value


# ============================================================
# 12. LOAD SWIN WEIGHTS
# ============================================================

missing_keys, unexpected_keys = swin_model.load_state_dict(
    clean_state_dict,
    strict=False,
)

print("\nSwin weights loaded.")

print("Missing keys:", len(missing_keys))
print("Unexpected keys:", len(unexpected_keys))

if len(missing_keys) > 0:

    print("\nFirst missing keys:")

    for key in missing_keys[:10]:

        print(" ", key)


# ============================================================
# 13. SWIN EMBEDDING MODEL
# ============================================================

embedding_model = SwinEmbeddingModel(
    swin_model
).to(DEVICE)

embedding_model.eval()


# ============================================================
# 14. EXTRACT EMBEDDINGS FUNCTION
# ============================================================

def extract_embeddings(df):

    embeddings = []

    labels = []

    filenames = []

    with torch.no_grad():

        for start in range(
            0,
            len(df),
            BATCH_SIZE,
        ):

            batch_df = df.iloc[
                start:start + BATCH_SIZE
            ]

            batch_images = []

            for _, row in batch_df.iterrows():

                image = Image.open(
                    row["image_path"]
                ).convert("RGB")

                image = transform(image)

                batch_images.append(image)

            batch_tensor = torch.stack(
                batch_images
            ).to(DEVICE)

            batch_embeddings = embedding_model(
                batch_tensor
            )

            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

            labels.extend(
                batch_df["label"].tolist()
            )

            filenames.extend(
                batch_df["filename"].tolist()
            )

            print(
                f"\rProcessed "
                f"{min(start + BATCH_SIZE, len(df))}"
                f"/{len(df)}",
                end="",
            )

    print()

    embeddings = np.concatenate(
        embeddings,
        axis=0,
    )

    labels = np.asarray(labels)

    return (
        embeddings,
        labels,
        filenames,
    )


val_embeddings, val_labels, val_filenames = (
    extract_embeddings(val_df)
)


print("\nValidation embedding shape:")
print(val_embeddings.shape)

if val_embeddings.shape != (
    len(val_df),
    SWIN_DIM,
):

    raise RuntimeError(
        f"Unexpected embedding shape: "
        f"{val_embeddings.shape}"
    )

print("✓ Swin embeddings verified.")


# ============================================================
# 15. LOAD THERMAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("THERMAL FEATURE VERIFICATION")
print("=" * 70)


features_df = pd.read_csv(
    FEATURE_FILE
)

print(
    "\nTotal thermal feature records:",
    len(features_df),
)

print(
    "Columns:",
    len(features_df.columns),
)


# ============================================================
# 16. SELECT GAUSSIAN + MEDIAN
# ============================================================

# The improved model used the Gaussian + Median
# thermal feature records.

thermal_df = features_df[
    (
        features_df["noise_type"]
        .astype(str)
        .str.lower()
        == "gaussian"
    )
    &
    (
        features_df["denoising_method"]
        .astype(str)
        .str.lower()
        == "median"
    )
].copy()


print(
    "\nGaussian + Median records:",
    len(thermal_df),
)


if len(thermal_df) != 1866:

    print(
        "WARNING: Expected 1866 Gaussian + Median "
        f"records but found {len(thermal_df)}."
    )

else:

    print(
        "✓ Correct 1866 Gaussian + Median records."
    )


# ============================================================
# 17. NORMALIZE FILENAMES FOR MATCHING
# ============================================================

def normalize_filename(filename):

    filename = Path(
        str(filename)
    ).stem

    # Remove Gaussian/denoising suffixes if present

    suffixes = [
        "_gaussian_median",
        "_gaussian_gaussian",
        "_gaussian_bilateral",
        "_gaussian",
    ]

    changed = True

    while changed:

        changed = False

        for suffix in suffixes:

            if filename.endswith(suffix):

                filename = filename[
                    :-len(suffix)
                ]

                changed = True

    return filename


thermal_df["match_key"] = (
    thermal_df["filename"]
    .apply(normalize_filename)
)

val_df["match_key"] = (
    val_df["filename"]
    .apply(normalize_filename)
)


# ============================================================
# 18. CHECK THERMAL DUPLICATES
# ============================================================

thermal_duplicates = (
    thermal_df["match_key"]
    .duplicated()
    .sum()
)

print(
    "\nDuplicate thermal match keys:",
    thermal_duplicates,
)


if thermal_duplicates > 0:

    raise RuntimeError(
        "Duplicate thermal feature matches detected."
    )


# ============================================================
# 19. MATCH VALIDATION FEATURES
# ============================================================

print("\nMatching validation images to thermal features...")


thermal_lookup = thermal_df.set_index(
    "match_key"
)


matched_features = []

matched_labels = []

missing = []

label_mismatch = []


for _, row in val_df.iterrows():

    key = row["match_key"]

    if key not in thermal_lookup.index:

        missing.append(
            row["filename"]
        )

        continue

    feature_row = thermal_lookup.loc[key]

    values = feature_row[
        THERMAL_FEATURES
    ].astype(float).values

    matched_features.append(values)

    matched_labels.append(
        int(feature_row["label"])
    )

    if int(feature_row["label"]) != int(row["label"]):

        label_mismatch.append(
            row["filename"]
        )


print(
    "\nMatched feature rows:",
    len(matched_features),
)

print(
    "Missing feature rows:",
    len(missing),
)

print(
    "Label mismatches:",
    len(label_mismatch),
)


if len(missing) > 0:

    print("\nMissing examples:")

    for item in missing[:10]:

        print(" ", item)


if len(label_mismatch) > 0:

    print("\nLabel mismatch examples:")

    for item in label_mismatch[:10]:

        print(" ", item)


if len(missing) != 0:

    raise RuntimeError(
        "Validation feature matching failed."
    )


if len(label_mismatch) != 0:

    raise RuntimeError(
        "Thermal feature labels do not match image labels."
    )


print(
    "✓ All validation images matched correctly."
)


val_thermal = np.asarray(
    matched_features,
    dtype=np.float32,
)


if val_thermal.shape != (
    len(val_df),
    THERMAL_DIM,
):

    raise RuntimeError(
        f"Unexpected thermal feature shape: "
        f"{val_thermal.shape}"
    )


print(
    "\nThermal feature shape:",
    val_thermal.shape,
)


# ============================================================
# 20. CHECK NaN / INF
# ============================================================

if not np.isfinite(
    val_embeddings
).all():

    raise RuntimeError(
        "NaN/Inf detected in Swin embeddings."
    )


if not np.isfinite(
    val_thermal
).all():

    raise RuntimeError(
        "NaN/Inf detected in thermal features."
    )


print(
    "✓ No NaN/Inf values."
)


# ============================================================
# 21. STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STANDARDIZATION")
print("=" * 70)


# IMPORTANT:
# Fit scalers ONLY on TRAINING data.

train_embeddings, train_labels, train_filenames = (
    extract_embeddings(train_df)
)


# ------------------------------------------------------------
# Match training thermal features
# ------------------------------------------------------------

train_df["match_key"] = (
    train_df["filename"]
    .apply(normalize_filename)
)


train_matched_features = []


for _, row in train_df.iterrows():

    key = row["match_key"]

    if key not in thermal_lookup.index:

        raise RuntimeError(
            "Missing training thermal feature: "
            + row["filename"]
        )

    feature_row = thermal_lookup.loc[key]

    values = feature_row[
        THERMAL_FEATURES
    ].astype(float).values

    train_matched_features.append(
        values
    )


train_thermal = np.asarray(
    train_matched_features,
    dtype=np.float32,
)


print(
    "\nTrain Swin features:",
    train_embeddings.shape,
)

print(
    "Train thermal features:",
    train_thermal.shape,
)


# ------------------------------------------------------------
# Separate train-only scalers
# ------------------------------------------------------------

swin_scaler = StandardScaler()

thermal_scaler = StandardScaler()


train_swin_scaled = (
    swin_scaler.fit_transform(
        train_embeddings
    )
)


val_swin_scaled = (
    swin_scaler.transform(
        val_embeddings
    )
)


train_thermal_scaled = (
    thermal_scaler.fit_transform(
        train_thermal
    )
)


val_thermal_scaled = (
    thermal_scaler.transform(
        val_thermal
    )
)


# ============================================================
# 22. COMBINE FEATURES
# ============================================================

train_features = np.concatenate(
    [
        train_swin_scaled,
        train_thermal_scaled,
    ],
    axis=1,
)


val_features = np.concatenate(
    [
        val_swin_scaled,
        val_thermal_scaled,
    ],
    axis=1,
)


print(
    "\nTrain node features:",
    train_features.shape,
)

print(
    "Validation node features:",
    val_features.shape,
)


if train_features.shape[1] != TOTAL_DIM:

    raise RuntimeError(
        "Unexpected train feature dimension."
    )


if val_features.shape[1] != TOTAL_DIM:

    raise RuntimeError(
        "Unexpected validation feature dimension."
    )


# ============================================================
# 23. KNN GRAPH
# ============================================================

print("\n" + "=" * 70)
print("BUILDING KNN GRAPHS")
print("=" * 70)


def build_knn_graph(features, k):

    # Normalize for cosine distance

    normalized = F.normalize(
        torch.tensor(
            features,
            dtype=torch.float32,
        ),
        p=2,
        dim=1,
    ).numpy()

    knn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
    )

    knn.fit(normalized)

    distances, indices = knn.kneighbors(
        normalized
    )

    edge_src = []
    edge_dst = []
    edge_weight = []

    for i in range(len(features)):

        for j_idx in range(
            1,
            k + 1,
        ):

            j = indices[i, j_idx]

            distance = distances[
                i,
                j_idx,
            ]

            weight = 1.0 - distance

            edge_src.append(i)
            edge_dst.append(j)
            edge_weight.append(
                max(weight, 0.0)
            )

    edge_index = torch.tensor(
        [
            edge_src,
            edge_dst,
        ],
        dtype=torch.long,
    )

    edge_weight = torch.tensor(
        edge_weight,
        dtype=torch.float32,
    )

    return (
        edge_index,
        edge_weight,
    )


train_edge_index, train_edge_weight = (
    build_knn_graph(
        train_features,
        K,
    )
)


val_edge_index, val_edge_weight = (
    build_knn_graph(
        val_features,
        K,
    )
)


print(
    "\nTrain graph edges:",
    train_edge_index.shape[1],
)

print(
    "Validation graph edges:",
    val_edge_index.shape[1],
)


# ============================================================
# 24. GRAPH CONVOLUTION
# ============================================================

class GraphConv(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_weight,
    ):

        src = edge_index[0]

        dst = edge_index[1]

        messages = (
            x[src]
            * edge_weight.unsqueeze(1)
        )

        aggregated = torch.zeros_like(
            x
        )

        aggregated.index_add_(
            0,
            dst,
            messages,
        )

        degree = torch.zeros(
            x.size(0),
            device=x.device,
        )

        degree.index_add_(
            0,
            dst,
            edge_weight,
        )

        degree = degree.clamp(
            min=1e-6
        )

        aggregated = (
            aggregated
            / degree.unsqueeze(1)
        )

        return self.linear(
            aggregated
        )


# ============================================================
# 25. IMPROVED GNN MODEL
# ============================================================

class ImprovedSwinGNN(nn.Module):

    def __init__(
        self,
        input_dim=775,
        hidden_dim=128,
        num_classes=2,
        dropout=0.4,
    ):

        super().__init__()

        self.conv1 = GraphConv(
            input_dim,
            hidden_dim,
        )

        self.norm1 = nn.LayerNorm(
            hidden_dim
        )

        self.conv2 = GraphConv(
            hidden_dim,
            hidden_dim,
        )

        self.norm2 = nn.LayerNorm(
            hidden_dim
        )

        self.conv3 = GraphConv(
            hidden_dim,
            hidden_dim,
        )

        self.norm3 = nn.LayerNorm(
            hidden_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_weight,
    ):

        h1 = self.conv1(
            x,
            edge_index,
            edge_weight,
        )

        h1 = self.norm1(h1)

        h1 = F.gelu(h1)

        h1 = self.dropout(h1)


        h2 = self.conv2(
            h1,
            edge_index,
            edge_weight,
        )

        h2 = self.norm2(h2)

        h2 = F.gelu(h2)

        h2 = self.dropout(h2)

        h2 = h2 + h1


        h3 = self.conv3(
            h2,
            edge_index,
            edge_weight,
        )

        h3 = self.norm3(h3)

        h3 = F.gelu(h3)

        h3 = self.dropout(h3)

        h3 = h3 + h2


        logits = self.classifier(
            h3
        )

        return logits


# ============================================================
# 26. LOAD SAVED GNN CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("LOADING SAVED GNN CHECKPOINT")
print("=" * 70)


gnn_model = ImprovedSwinGNN(
    input_dim=TOTAL_DIM,
    hidden_dim=128,
    num_classes=2,
    dropout=0.4,
).to(DEVICE)


# ============================================================
# 27. FIND GNN STATE DICT
# ============================================================

def find_gnn_state_dict(checkpoint):

    if not isinstance(checkpoint, dict):

        return None

    possible_keys = [
        "gnn_state_dict",
        "model_state_dict",
        "state_dict",
        "model",
    ]

    for key in possible_keys:

        if key in checkpoint:

            value = checkpoint[key]

            if isinstance(value, dict):

                # Check whether this actually contains
                # GNN layers.

                if any(
                    "conv1" in str(k)
                    or "classifier" in str(k)
                    for k in value.keys()
                ):

                    return value

    return None


gnn_state_dict = find_gnn_state_dict(
    checkpoint
)


if gnn_state_dict is None:

    # Sometimes the saved file itself is the GNN state dict.

    if any(
        "conv1" in str(k)
        or "classifier" in str(k)
        for k in checkpoint.keys()
    ):

        gnn_state_dict = checkpoint


if gnn_state_dict is None:

    raise RuntimeError(
        "\nCould not find GNN state_dict "
        "inside checkpoint."
    )


clean_gnn_state = {}

for key, value in gnn_state_dict.items():

    new_key = key

    if new_key.startswith("module."):

        new_key = new_key[7:]

    clean_gnn_state[new_key] = value


missing_gnn, unexpected_gnn = (
    gnn_model.load_state_dict(
        clean_gnn_state,
        strict=False,
    )
)


print(
    "\nGNN weights loaded."
)

print(
    "Missing GNN keys:",
    len(missing_gnn),
)

print(
    "Unexpected GNN keys:",
    len(unexpected_gnn),
)


if len(missing_gnn) > 0:

    print("\nMissing keys:")

    for key in missing_gnn:

        print(" ", key)


# ============================================================
# 28. VALIDATION INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("GENERATING INDEPENDENT VALIDATION PREDICTIONS")
print("=" * 70)


gnn_model.eval()


x_val = torch.tensor(
    val_features,
    dtype=torch.float32,
    device=DEVICE,
)


edge_index_val = (
    val_edge_index.to(DEVICE)
)

edge_weight_val = (
    val_edge_weight.to(DEVICE)
)


with torch.no_grad():

    logits = gnn_model(
        x_val,
        edge_index_val,
        edge_weight_val,
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    predictions = torch.argmax(
        probabilities,
        dim=1,
    )


predictions = (
    predictions
    .cpu()
    .numpy()
)


prob_dm = (
    probabilities[:, 1]
    .cpu()
    .numpy()
)


true_labels = (
    val_labels.astype(int)
)


# ============================================================
# 29. PREDICTION COUNT CHECK
# ============================================================

print(
    "\nTrue labels:",
    len(true_labels),
)

print(
    "Predictions:",
    len(predictions),
)

print(
    "Probabilities:",
    len(prob_dm),
)


if not (
    len(true_labels)
    == len(predictions)
    == len(prob_dm)
    == 422
):

    raise RuntimeError(
        "Prediction count verification failed."
    )


print(
    "✓ Exactly 422 validation predictions."
)


# ============================================================
# 30. SAVE INDIVIDUAL PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "filename": val_filenames,
        "true_label": true_labels,
        "predicted_label": predictions,
        "probability_DM": prob_dm,
    }
)


prediction_file = (
    RESULTS_DIR
    / "verified_swin_gnn_predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False,
)


print(
    "\nSaved:"
)

print(
    prediction_file
)


# ============================================================
# 31. RECOMPUTE METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions,
)

precision = precision_score(
    true_labels,
    predictions,
    zero_division=0,
)

recall = recall_score(
    true_labels,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0,
)

auc = roc_auc_score(
    true_labels,
    prob_dm,
)


cm = confusion_matrix(
    true_labels,
    predictions,
)


# ============================================================
# 32. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("INDEPENDENTLY RECOMPUTED RESULTS")
print("=" * 70)


print(
    f"\nAccuracy : {accuracy:.6f}"
    f"  ({accuracy * 100:.2f}%)"
)

print(
    f"Precision: {precision:.6f}"
    f"  ({precision * 100:.2f}%)"
)

print(
    f"Recall   : {recall:.6f}"
    f"  ({recall * 100:.2f}%)"
)

print(
    f"F1 Score : {f1:.6f}"
    f"  ({f1 * 100:.2f}%)"
)

print(
    f"ROC-AUC  : {auc:.6f}"
    f"  ({auc * 100:.2f}%)"
)


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        target_names=[
            "Control",
            "DM",
        ],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# 33. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Control",
        "Actual_DM",
    ],
    columns=[
        "Predicted_Control",
        "Predicted_DM",
    ],
)


cm_file = (
    RESULTS_DIR
    / "verified_swin_gnn_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_file
)


# ============================================================
# 34. SAVE VERIFIED METRICS
# ============================================================

verified_results = pd.DataFrame(
    [
        {
            "model": "Improved Swin + GNN",
            "k": K,
            "validation_samples": len(true_labels),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
        }
    ]
)


metrics_file = (
    RESULTS_DIR
    / "verified_swin_gnn_metrics.csv"
)


verified_results.to_csv(
    metrics_file,
    index=False,
)


# ============================================================
# 35. COMPARE WITH REPORTED RESULT
# ============================================================

REPORTED_ACCURACY = 0.950237
REPORTED_PRECISION = 0.987342
REPORTED_RECALL = 0.928571
REPORTED_F1 = 0.957055
REPORTED_AUC = 0.978641


print("\n" + "=" * 70)
print("COMPARISON WITH REPORTED 95.02% RESULT")
print("=" * 70)


comparisons = [
    (
        "Accuracy",
        accuracy,
        REPORTED_ACCURACY,
    ),
    (
        "Precision",
        precision,
        REPORTED_PRECISION,
    ),
    (
        "Recall",
        recall,
        REPORTED_RECALL,
    ),
    (
        "F1",
        f1,
        REPORTED_F1,
    ),
    (
        "ROC-AUC",
        auc,
        REPORTED_AUC,
    ),
]


all_match = True


for name, actual, reported in comparisons:

    difference = abs(
        actual - reported
    )

    matches = difference < 0.0001

    if not matches:

        all_match = False

    status = "MATCH" if matches else "DIFFER"

    print(
        f"{name:10s} | "
        f"Verified={actual:.6f} | "
        f"Reported={reported:.6f} | "
        f"Difference={difference:.6f} | "
        f"{status}"
    )


# ============================================================
# 36. FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION VERDICT")
print("=" * 70)


if all_match:

    print(
        "\n✓ VERIFIED"
    )

    print(
        "The independently generated validation "
        "predictions reproduce the reported metrics."
    )

    print(
        "\nVerified headline result:"
    )

    print(
        "Accuracy : 95.02%"
    )

    print(
        "Precision: 98.73%"
    )

    print(
        "Recall   : 92.86%"
    )

    print(
        "F1       : 95.71%"
    )

    print(
        "ROC-AUC  : 97.86%"
    )

else:

    print(
        "\n⚠ METRIC MISMATCH"
    )

    print(
        "The independently recomputed metrics do "
        "not exactly reproduce the reported result."
    )

    print(
        "Do NOT report 95.02% as independently verified yet."
    )


# ============================================================
# 37. FINAL OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    "\nPrediction file:"
)

print(
    prediction_file
)

print(
    "\nMetrics file:"
)

print(
    metrics_file
)

print(
    "\nConfusion matrix:"
)

print(
    cm_file
)

print("\nVerification complete.")