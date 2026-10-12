# ================================================================
# IMPROVED SWIN TRANSFORMER + GNN
# ================================================================
# Extension of Step 9
#
# Original baseline:
# Accuracy : 82.46%
# F1       : 84.65%
# ROC-AUC  : 87.94%
#
# Improvements:
# - Same Swin embeddings
# - Same 7 Spark-selected thermal features
# - Proper 775-D feature construction
# - Distance-weighted KNN graph
# - Tests k = 3, 5, 7, 10
# - Residual GCN layers
# - Layer normalization
# - AdamW
# - Label smoothing
# - Learning-rate scheduler
# - Gradient clipping
# - Early stopping
#
# Existing Step 9 files are NOT overwritten.
# ================================================================

import os
import sys
import time
import copy
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from torchvision import models, transforms

from PIL import Image

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
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

SWIN_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "best_swin_transformer_thermal.pth"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "classification"
)

METRIC_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "results"
    / "plots"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# RANDOM SEED
# ================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# DATA SETTINGS
# ================================================================

CLASS_NAMES = [
    "Control",
    "DM"
]

CLASS_TO_LABEL = {
    "Control": 0,
    "DM": 1
}


THERMAL_FEATURES = [

    "mean_intensity",

    "std_intensity",

    "percentile_90",

    "hotspot_area_ratio",

    "hotspot_mean_intensity",

    "absolute_asymmetry",

    "relative_asymmetry"
]


# ================================================================
# IMPROVEMENT SETTINGS
# ================================================================

# Test different graph neighborhoods
K_VALUES = [
    3,
    5,
    7,
    10
]

# GNN hidden dimension
HIDDEN_DIM = 256

# Dropout
DROPOUT = 0.40

# Training
EPOCHS = 100

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0005

# Early stopping
PATIENCE = 15

MIN_DELTA = 0.0005


# ================================================================
# HEADER
# ================================================================

print("=" * 70)

print(
    "IMPROVED SWIN TRANSFORMER + GNN"
)

print("=" * 70)

print()

print(
    "This experiment does NOT overwrite Step 9."
)

print()

print(
    "Project root:"
)

print(
    PROJECT_ROOT
)

print()

print(
    "PyTorch:"
)

print(
    torch.__version__
)

print()

print(
    "CUDA:"
)

print(
    torch.version.cuda
)

print()

print(
    "Device:"
)

print(
    DEVICE
)

if torch.cuda.is_available():

    print()

    print(
        "GPU:"
    )

    print(
        torch.cuda.get_device_name(0)
    )

print()


# ================================================================
# LOAD IMAGE DATASET
# ================================================================

print("=" * 70)

print(
    "LOADING CLASSIFICATION DATASET"
)

print("=" * 70)

print()


def collect_images(split):

    records = []

    split_dir = (
        DATA_ROOT
        / split
    )

    for class_name in CLASS_NAMES:

        class_dir = (
            split_dir
            / class_name
        )

        if not class_dir.exists():

            continue

        for image_path in sorted(
            class_dir.glob("*.png")
        ):

            records.append({

                "image_path":
                    str(image_path),

                "filename":
                    image_path.name,

                "split":
                    split,

                "class":
                    class_name,

                "label":
                    CLASS_TO_LABEL[
                        class_name
                    ]
            })

    return pd.DataFrame(
        records
    )


train_df = collect_images(
    "train"
)

val_df = collect_images(
    "val"
)


print(
    "Classes:"
)

print(
    CLASS_NAMES
)

print()

print(
    "Class mapping:"
)

print(
    CLASS_TO_LABEL
)

print()

print(
    "Training samples:"
)

print(
    len(train_df)
)

print()

print(
    "Validation samples:"
)

print(
    len(val_df)
)

print()


# ================================================================
# IMAGE TRANSFORM
# ================================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ================================================================
# LOAD TRAINED SWIN
# ================================================================

print("=" * 70)

print(
    "LOADING TRAINED SWIN TRANSFORMER"
)

print("=" * 70)

print()


swin_model = models.swin_t(
    weights=None
)

swin_model.head = nn.Linear(
    swin_model.head.in_features,
    2
)


checkpoint = torch.load(
    SWIN_MODEL_FILE,
    map_location=DEVICE
)


if isinstance(
    checkpoint,
    dict
):

    if (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif (
        "state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith(
        "module."
    ):

        key = key.replace(
            "module.",
            "",
            1
        )

    clean_state_dict[
        key
    ] = value


missing, unexpected = (
    swin_model.load_state_dict(
        clean_state_dict,
        strict=False
    )
)


print(
    "✓ Best Swin Transformer loaded"
)

if missing:

    print(
        "Missing keys:",
        len(missing)
    )

if unexpected:

    print(
        "Unexpected keys:",
        len(unexpected)
    )

print()


swin_model = (
    swin_model.to(DEVICE)
)

swin_model.eval()


# ================================================================
# EXTRACT SWIN EMBEDDINGS
# ================================================================

print("=" * 70)

print(
    "EXTRACTING SWIN EMBEDDINGS"
)

print("=" * 70)

print()


@torch.no_grad()
def extract_embeddings(
    df,
    split_name
):

    embeddings = []

    labels = []

    filenames = []

    paths = []

    total = len(df)

    for i, row in enumerate(
        df.itertuples(
            index=False
        )
    ):

        try:

            image = (
                Image.open(
                    row.image_path
                )
                .convert("RGB")
            )

            tensor = (
                transform(
                    image
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

            x = (
                swin_model
                .features(
                    tensor
                )
            )

            x = (
                swin_model
                .norm(x)
            )

            x = x.permute(
                0,
                3,
                1,
                2
            )

            x = (
                F.adaptive_avg_pool2d(
                    x,
                    (1, 1)
                )
            )

            x = torch.flatten(
                x,
                1
            )

            embeddings.append(
                x.cpu()
                .numpy()[0]
            )

            labels.append(
                row.label
            )

            filenames.append(
                row.filename
            )

            paths.append(
                row.image_path
            )

            if (
                (i + 1) % 20 == 0
                or
                (i + 1) == total
            ):

                print(
                    f"{split_name}: "
                    f"{i + 1}/{total}"
                )

        except Exception as e:

            print(
                "ERROR:",
                row.filename,
                e
            )

    return (

        np.asarray(
            embeddings,
            dtype=np.float32
        ),

        np.asarray(
            labels,
            dtype=np.int64
        ),

        filenames,

        paths
    )


(
    train_embeddings,
    train_labels,
    train_files,
    train_paths
) = extract_embeddings(
    train_df,
    "Train"
)


print()

print(
    "Train embeddings shape:"
)

print(
    train_embeddings.shape
)

print()


(
    val_embeddings,
    val_labels,
    val_files,
    val_paths
) = extract_embeddings(
    val_df,
    "Validation"
)


print()

print(
    "Validation embeddings shape:"
)

print(
    val_embeddings.shape
)

print()


# ================================================================
# LOAD THERMAL FEATURES
# ================================================================

print("=" * 70)

print(
    "LOADING SPARK-SELECTED THERMAL FEATURES"
)

print("=" * 70)

print()


features_df = pd.read_csv(
    FEATURE_FILE
)


print(
    "Total thermal feature records:"
)

print(
    len(features_df)
)

print()


# Use same feature subset as original Step 9
selected_features_df = (
    features_df[
        (
            features_df[
                "noise_type"
            ]
            == "gaussian"
        )
        &
        (
            features_df[
                "denoising_method"
            ]
            == "median"
        )
    ]
    .copy()
)


print(
    "Records after selecting "
    "Gaussian + Median:"
)

print(
    len(selected_features_df)
)

print()


# ================================================================
# MATCH THERMAL FEATURES
# ================================================================

def prepare_features(
    feature_dataframe,
    filenames,
    split_name
):

    temp = feature_dataframe[
        feature_dataframe[
            "split"
        ]
        == split_name
    ].copy()

    temp[
        "filename_key"
    ] = (
        temp[
            "filename"
        ]
        .astype(str)
        .str.replace(
            "_gaussian_median",
            "",
            regex=False
        )
        .str.replace(
            "_gaussian",
            "",
            regex=False
        )
    )

    target = pd.DataFrame({

        "filename":
            filenames
    })

    target[
        "filename_key"
    ] = (
        target[
            "filename"
        ]
        .astype(str)
        .str.replace(
            "_gaussian_median",
            "",
            regex=False
        )
        .str.replace(
            "_gaussian",
            "",
            regex=False
        )
    )

    merged = target.merge(

        temp,

        on="filename_key",

        how="left"
    )

    missing = merged[
        THERMAL_FEATURES
    ].isna().any(
        axis=1
    )

    missing_count = int(missing.sum())

    if missing_count > 0:

        print(
            "ERROR:",
            missing_count,
            "missing feature rows"
        )

    else:

        print(
            f"✓ All {len(target)} "
            f"{split_name.lower()} "
            f"samples matched."
        )

    return merged[
        THERMAL_FEATURES
    ].values.astype(
        np.float32
    )


train_thermal = prepare_features(

    selected_features_df,

    train_files,

    "train"
)


val_thermal = prepare_features(

    selected_features_df,

    val_files,

    "val"
)


print()

print(
    "Training thermal feature shape:"
)

print(
    train_thermal.shape
)

print()

print(
    "Validation thermal feature shape:"
)

print(
    val_thermal.shape
)

print()


# ================================================================
# STANDARDIZATION
# ================================================================

print("=" * 70)

print(
    "STANDARDIZING FEATURES"
)

print("=" * 70)

print()


# IMPORTANT:
# Fit scalers ONLY on training data.

swin_scaler = StandardScaler()

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


thermal_scaler = StandardScaler()

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


# ================================================================
# COMBINE FEATURES
# ================================================================

train_features = np.concatenate(

    [
        train_swin_scaled,
        train_thermal_scaled
    ],

    axis=1
)


val_features = np.concatenate(

    [
        val_swin_scaled,
        val_thermal_scaled
    ],

    axis=1
)


print(
    "Training node feature shape:"
)

print(
    train_features.shape
)

print()

print(
    "Validation node feature shape:"
)

print(
    val_features.shape
)

print()

print(
    "Expected:"
)

print(
    "768 Swin + 7 thermal = 775"
)

print()


# ================================================================
# TORCH TENSORS
# ================================================================

train_x = torch.tensor(

    train_features,

    dtype=torch.float32,

    device=DEVICE
)


val_x = torch.tensor(

    val_features,

    dtype=torch.float32,

    device=DEVICE
)


train_y = torch.tensor(

    train_labels,

    dtype=torch.long,

    device=DEVICE
)


val_y = torch.tensor(

    val_labels,

    dtype=torch.long,

    device=DEVICE
)


# ================================================================
# GRAPH CONSTRUCTION
# ================================================================

def build_knn_graph(
    features,
    k
):

    print(
        f"Building KNN graph "
        f"with k={k}..."
    )

    features_np = (
        features
        .detach()
        .cpu()
        .numpy()
    )

    n = len(
        features_np
    )

    # +1 because each node
    # is its own nearest neighbour.
    n_neighbors = min(
        k + 1,
        n
    )

    knn = NearestNeighbors(

        n_neighbors=n_neighbors,

        metric="cosine"
    )

    knn.fit(
        features_np
    )

    distances, indices = (
        knn.kneighbors(
            features_np
        )
    )

    adjacency = np.zeros(

        (
            n,
            n
        ),

        dtype=np.float32
    )


    # Distance-weighted edges
    for i in range(n):

        for position in range(

            1,

            indices.shape[1]

        ):

            j = indices[
                i,
                position
            ]

            distance = distances[
                i,
                position
            ]

            # Gaussian-like similarity
            weight = np.exp(
                -2.0
                *
                distance
                *
                distance
            )

            adjacency[
                i,
                j
            ] = weight


    # Symmetric graph
    adjacency = np.maximum(

        adjacency,

        adjacency.T
    )


    # Self loops
    adjacency += np.eye(

        n,

        dtype=np.float32
    )


    # Degree normalization
    degree = adjacency.sum(
        axis=1
    )

    degree_inv_sqrt = np.power(

        degree,

        -0.5
    )

    degree_inv_sqrt[
        np.isinf(
            degree_inv_sqrt
        )
    ] = 0


    normalized = (

        degree_inv_sqrt[:, None]

        *

        adjacency

        *

        degree_inv_sqrt[None, :]
    )


    print(
        "Nodes:",
        n
    )

    print(
        "Non-zero graph entries:",
        np.count_nonzero(
            adjacency
        )
    )

    print()


    return torch.tensor(

        normalized,

        dtype=torch.float32,

        device=DEVICE
    )


# ================================================================
# RESIDUAL GRAPH CONVOLUTION
# ================================================================

class ResidualGraphConv(
    nn.Module
):

    def __init__(

        self,

        input_dim,

        output_dim,

        dropout=0.4

    ):

        super().__init__()


        self.linear = nn.Linear(

            input_dim,

            output_dim
        )


        self.norm = nn.LayerNorm(

            output_dim
        )


        self.dropout = nn.Dropout(

            dropout
        )


        if input_dim == output_dim:

            self.skip = nn.Identity()

        else:

            self.skip = nn.Linear(

                input_dim,

                output_dim
            )


    def forward(

        self,

        x,

        adjacency

    ):

        residual = self.skip(
            x
        )


        x = torch.matmul(

            adjacency,

            x
        )


        x = self.linear(
            x
        )


        x = self.norm(
            x
        )


        x = F.gelu(
            x
        )


        x = self.dropout(
            x
        )


        return x + residual


# ================================================================
# IMPROVED GNN MODEL
# ================================================================

class ImprovedGNN(
    nn.Module
):

    def __init__(

        self,

        input_dim=775,

        hidden_dim=256,

        num_classes=2,

        dropout=0.4

    ):

        super().__init__()


        # Initial feature projection
        self.input_projection = nn.Sequential(

            nn.Linear(

                input_dim,

                hidden_dim
            ),

            nn.LayerNorm(

                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(

                0.20
            )
        )


        # Residual graph layers
        self.gcn1 = ResidualGraphConv(

            hidden_dim,

            hidden_dim,

            dropout
        )


        self.gcn2 = ResidualGraphConv(

            hidden_dim,

            hidden_dim,

            dropout
        )


        self.gcn3 = ResidualGraphConv(

            hidden_dim,

            hidden_dim,

            dropout
        )


        # Final classifier
        self.classifier = nn.Sequential(

            nn.LayerNorm(

                hidden_dim
            ),

            nn.Linear(

                hidden_dim,

                128
            ),

            nn.GELU(),

            nn.Dropout(

                dropout
            ),

            nn.Linear(

                128,

                num_classes
            )
        )


    def forward(

        self,

        x,

        adjacency

    ):

        x = self.input_projection(
            x
        )


        x = self.gcn1(

            x,

            adjacency
        )


        x = self.gcn2(

            x,

            adjacency
        )


        x = self.gcn3(

            x,

            adjacency
        )


        return self.classifier(
            x
        )


# ================================================================
# CLASS WEIGHTS
# ================================================================

class_counts = np.bincount(

    train_labels,

    minlength=2
)


class_weights = (

    len(train_labels)

    /

    (
        2
        *
        class_counts
    )
)


class_weights = torch.tensor(

    class_weights,

    dtype=torch.float32,

    device=DEVICE
)


print("=" * 70)

print(
    "CLASS WEIGHTS"
)

print("=" * 70)

print()

print(
    class_weights
    .cpu()
    .numpy()
)

print()


# ================================================================
# TRAIN ONE K CONFIGURATION
# ================================================================

def train_one_k(k):

    print("=" * 70)

    print(
        f"TRAINING CONFIGURATION: k={k}"
    )

    print("=" * 70)

    print()


    # ------------------------------------------------------------
    # Build graphs from fixed 775-D features.
    # ------------------------------------------------------------

    train_graph = build_knn_graph(

        train_x,

        k
    )


    val_graph = build_knn_graph(

        val_x,

        k
    )


    # ------------------------------------------------------------
    # Create model.
    # ------------------------------------------------------------

    model = ImprovedGNN(

        input_dim=775,

        hidden_dim=HIDDEN_DIM,

        num_classes=2,

        dropout=DROPOUT

    ).to(DEVICE)


    # ------------------------------------------------------------
    # Optimizer.
    # ------------------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    # ------------------------------------------------------------
    # Scheduler.
    # ------------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=5,

            min_lr=1e-6
        )
    )


    # ------------------------------------------------------------
    # Loss.
    # ------------------------------------------------------------

    criterion = nn.CrossEntropyLoss(

        weight=class_weights,

        label_smoothing=0.05
    )


    best_f1 = -1

    best_auc = -1

    best_accuracy = -1

    best_epoch = 0

    best_state = None

    patience_counter = 0

    history = []


    start_time = time.time()


    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(

        1,

        EPOCHS + 1

    ):

        epoch_start = time.time()


        # --------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------

        model.train()

        optimizer.zero_grad()


        logits = model(

            train_x,

            train_graph
        )


        loss = criterion(

            logits,

            train_y
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0
        )


        optimizer.step()


        train_pred = (

            torch.argmax(

                logits,

                dim=1
            )

            .detach()

            .cpu()

            .numpy()
        )


        train_accuracy = accuracy_score(

            train_labels,

            train_pred
        )


        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        model.eval()


        with torch.no_grad():

            val_logits = model(

                val_x,

                val_graph
            )


            val_loss = criterion(

                val_logits,

                val_y
            )


            val_prob = (

                torch.softmax(

                    val_logits,

                    dim=1
                )

                [:, 1]

                .cpu()

                .numpy()
            )


            val_pred = (

                torch.argmax(

                    val_logits,

                    dim=1
                )

                .cpu()

                .numpy()
            )


        val_accuracy = accuracy_score(

            val_labels,

            val_pred
        )


        val_precision = precision_score(

            val_labels,

            val_pred,

            zero_division=0
        )


        val_recall = recall_score(

            val_labels,

            val_pred,

            zero_division=0
        )


        val_f1 = f1_score(

            val_labels,

            val_pred,

            zero_division=0
        )


        val_auc = roc_auc_score(

            val_labels,

            val_prob
        )


        scheduler.step(

            val_f1
        )


        current_lr = (

            optimizer.param_groups[0][
                "lr"
            ]
        )


        epoch_time = (

            time.time()

            -

            epoch_start
        )


        history.append({

            "k": k,

            "epoch": epoch,

            "train_loss":
                float(
                    loss.item()
                ),

            "train_accuracy":
                float(
                    train_accuracy
                ),

            "val_loss":
                float(
                    val_loss.item()
                ),

            "val_accuracy":
                float(
                    val_accuracy
                ),

            "val_precision":
                float(
                    val_precision
                ),

            "val_recall":
                float(
                    val_recall
                ),

            "val_f1":
                float(
                    val_f1
                ),

            "val_roc_auc":
                float(
                    val_auc
                ),

            "learning_rate":
                float(
                    current_lr
                )
        })


        # --------------------------------------------------------
        # PRINT
        # --------------------------------------------------------

        if (

            epoch == 1

            or

            epoch % 5 == 0

            or

            epoch == EPOCHS

        ):

            print(
                "-" * 70
            )

            print(
                f"Epoch {epoch}/{EPOCHS}"
            )

            print(
                f"Train Loss     : "
                f"{loss.item():.4f}"
            )

            print(
                f"Train Accuracy : "
                f"{train_accuracy:.4f}"
            )

            print(
                f"Val Loss       : "
                f"{val_loss.item():.4f}"
            )

            print(
                f"Val Accuracy   : "
                f"{val_accuracy:.4f}"
            )

            print(
                f"Val Precision  : "
                f"{val_precision:.4f}"
            )

            print(
                f"Val Recall     : "
                f"{val_recall:.4f}"
            )

            print(
                f"Val F1         : "
                f"{val_f1:.4f}"
            )

            print(
                f"Val ROC-AUC    : "
                f"{val_auc:.4f}"
            )

            print(
                f"Learning Rate  : "
                f"{current_lr:.7f}"
            )

            print(
                f"Epoch Time     : "
                f"{epoch_time:.2f} sec"
            )


        # --------------------------------------------------------
        # BEST MODEL
        # --------------------------------------------------------

        improved = False


        if (

            val_f1
            >
            best_f1 + MIN_DELTA

        ):

            improved = True


        elif (

            abs(
                val_f1 - best_f1
            )
            <= MIN_DELTA

            and

            val_auc > best_auc

        ):

            improved = True


        if improved:

            best_f1 = val_f1

            best_auc = val_auc

            best_accuracy = val_accuracy

            best_epoch = epoch

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            patience_counter = 0

        else:

            patience_counter += 1


        if (

            patience_counter
            >= PATIENCE

        ):

            print()

            print(
                "Early stopping."
            )

            print(
                f"No improvement for "
                f"{PATIENCE} epochs."
            )

            break


    # ============================================================
    # BEST MODEL EVALUATION
    # ============================================================

    model.load_state_dict(
        best_state
    )

    model.eval()


    with torch.no_grad():

        final_logits = model(

            val_x,

            val_graph
        )


        final_prob = (

            torch.softmax(

                final_logits,

                dim=1
            )

            [:, 1]

            .cpu()

            .numpy()
        )


        final_pred = (

            torch.argmax(

                final_logits,

                dim=1
            )

            .cpu()

            .numpy()
        )


    final_accuracy = accuracy_score(

        val_labels,

        final_pred
    )


    final_precision = precision_score(

        val_labels,

        final_pred,

        zero_division=0
    )


    final_recall = recall_score(

        val_labels,

        final_pred,

        zero_division=0
    )


    final_f1 = f1_score(

        val_labels,

        final_pred,

        zero_division=0
    )


    final_auc = roc_auc_score(

        val_labels,

        final_prob
    )


    cm = confusion_matrix(

        val_labels,

        final_pred
    )


    elapsed = (

        time.time()

        -

        start_time
    )


    print()

    print("=" * 70)

    print(
        f"BEST RESULTS - k={k}"
    )

    print("=" * 70)

    print()

    print(
        f"Best Epoch : "
        f"{best_epoch}"
    )

    print(
        f"Accuracy   : "
        f"{final_accuracy:.4f}"
    )

    print(
        f"Precision  : "
        f"{final_precision:.4f}"
    )

    print(
        f"Recall     : "
        f"{final_recall:.4f}"
    )

    print(
        f"F1         : "
        f"{final_f1:.4f}"
    )

    print(
        f"ROC-AUC    : "
        f"{final_auc:.4f}"
    )

    print()

    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )

    print()


    return {

        "k":
            k,

        "best_epoch":
            best_epoch,

        "accuracy":
            final_accuracy,

        "precision":
            final_precision,

        "recall":
            final_recall,

        "f1":
            final_f1,

        "roc_auc":
            final_auc,

        "training_time_seconds":
            elapsed,

        "state":
            best_state,

        "history":
            history,

        "confusion_matrix":
            cm,

        "predictions":
            final_pred,

        "probabilities":
            final_prob
    }


# ================================================================
# RUN ALL K VALUES
# ================================================================

results = []

histories = []

best_result = None


for k in K_VALUES:

    result = train_one_k(
        k
    )


    results.append({

        "k":
            result["k"],

        "best_epoch":
            result["best_epoch"],

        "accuracy":
            result["accuracy"],

        "precision":
            result["precision"],

        "recall":
            result["recall"],

        "f1":
            result["f1"],

        "roc_auc":
            result["roc_auc"],

        "training_time_seconds":
            result[
                "training_time_seconds"
            ]
    })


    histories.extend(
        result["history"]
    )


    if best_result is None:

        best_result = result

    else:

        if (

            result["f1"]
            >
            best_result["f1"]

        ):

            best_result = result

        elif (

            result["f1"]
            ==
            best_result["f1"]

            and

            result["roc_auc"]
            >
            best_result["roc_auc"]

        ):

            best_result = result


# ================================================================
# RESULTS DATAFRAME
# ================================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(

    by=[
        "f1",
        "roc_auc",
        "accuracy"
    ],

    ascending=False
)


results_file = (

    METRIC_DIR

    /

    "improved_swin_gnn_results.csv"
)


results_df.to_csv(

    results_file,

    index=False
)


history_df = pd.DataFrame(
    histories
)


history_file = (

    METRIC_DIR

    /

    "improved_swin_gnn_training_history.csv"
)


history_df.to_csv(

    history_file,

    index=False
)


# ================================================================
# SAVE BEST MODEL
# ================================================================

best_model_file = (

    MODEL_DIR

    /

    "best_swin_gnn_improved.pth"
)


torch.save(

    {

        "model_state_dict":
            best_result[
                "state"
            ],

        "best_k":
            best_result[
                "k"
            ],

        "best_epoch":
            best_result[
                "best_epoch"
            ],

        "accuracy":
            best_result[
                "accuracy"
            ],

        "precision":
            best_result[
                "precision"
            ],

        "recall":
            best_result[
                "recall"
            ],

        "f1":
            best_result[
                "f1"
            ],

        "roc_auc":
            best_result[
                "roc_auc"
            ],

        "input_dimension":
            775,

        "hidden_dimension":
            HIDDEN_DIM,

        "dropout":
            DROPOUT,

        "thermal_features":
            THERMAL_FEATURES,

        "description":
            "Improved Swin + GNN"
    },

    best_model_file
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

cm = best_result[
    "confusion_matrix"
]


cm_df = pd.DataFrame(

    cm,

    index=[
        "Actual Control",
        "Actual DM"
    ],

    columns=[
        "Predicted Control",
        "Predicted DM"
    ]
)


cm_file = (

    METRIC_DIR

    /

    "improved_swin_gnn_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_file
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

report = classification_report(

    val_labels,

    best_result[
        "predictions"
    ],

    target_names=CLASS_NAMES,

    zero_division=0
)


report_file = (

    METRIC_DIR

    /

    "improved_swin_gnn_classification_report.txt"
)


with open(

    report_file,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "IMPROVED SWIN + GNN\n"
    )

    f.write(
        "=" * 60
        +
        "\n\n"
    )

    f.write(
        f"Best K: "
        f"{best_result['k']}\n"
    )

    f.write(
        f"Best Epoch: "
        f"{best_result['best_epoch']}\n\n"
    )

    f.write(
        report
    )


# ================================================================
# BASELINE COMPARISON
# ================================================================

baseline_accuracy = 0.8246

baseline_precision = 0.8870

baseline_recall = 0.8095

baseline_f1 = 0.8465

baseline_auc = 0.8794


comparison_df = pd.DataFrame([

    {

        "model":
            "Original Swin + GNN",

        "accuracy":
            baseline_accuracy,

        "precision":
            baseline_precision,

        "recall":
            baseline_recall,

        "f1":
            baseline_f1,

        "roc_auc":
            baseline_auc
    },

    {

        "model":
            "Improved Swin + GNN",

        "accuracy":
            best_result[
                "accuracy"
            ],

        "precision":
            best_result[
                "precision"
            ],

        "recall":
            best_result[
                "recall"
            ],

        "f1":
            best_result[
                "f1"
            ],

        "roc_auc":
            best_result[
                "roc_auc"
            ]
    }

])


comparison_file = (

    METRIC_DIR

    /

    "swin_gnn_improvement_comparison.csv"
)


comparison_df.to_csv(

    comparison_file,

    index=False
)


# ================================================================
# SUMMARY
# ================================================================

summary_df = pd.DataFrame([

    {

        "model":
            "Improved Swin + GNN",

        "best_k":
            best_result[
                "k"
            ],

        "best_epoch":
            best_result[
                "best_epoch"
            ],

        "accuracy":
            best_result[
                "accuracy"
            ],

        "precision":
            best_result[
                "precision"
            ],

        "recall":
            best_result[
                "recall"
            ],

        "f1":
            best_result[
                "f1"
            ],

        "roc_auc":
            best_result[
                "roc_auc"
            ],

        "baseline_accuracy":
            baseline_accuracy,

        "baseline_f1":
            baseline_f1,

        "baseline_roc_auc":
            baseline_auc,

        "accuracy_change":
            (
                best_result[
                    "accuracy"
                ]
                -
                baseline_accuracy
            ),

        "f1_change":
            (
                best_result[
                    "f1"
                ]
                -
                baseline_f1
            ),

        "roc_auc_change":
            (
                best_result[
                    "roc_auc"
                ]
                -
                baseline_auc
            )
    }

])


summary_file = (

    METRIC_DIR

    /

    "improved_swin_gnn_summary.csv"
)


summary_df.to_csv(

    summary_file,

    index=False
)


# ================================================================
# PLOTS
# ================================================================

try:

    import matplotlib.pyplot as plt


    # ------------------------------------------------------------
    # K comparison
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(

        results_df[
            "k"
        ],

        results_df[
            "accuracy"
        ],

        marker="o",

        label="Accuracy"
    )


    plt.plot(

        results_df[
            "k"
        ],

        results_df[
            "f1"
        ],

        marker="o",

        label="F1"
    )


    plt.plot(

        results_df[
            "k"
        ],

        results_df[
            "roc_auc"
        ],

        marker="o",

        label="ROC-AUC"
    )


    plt.xlabel(
        "KNN k"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Improved Swin + GNN - KNN Comparison"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(

        PLOT_DIR
        /
        "improved_swin_gnn_k_comparison.png",

        dpi=200
    )


    plt.close()


    # ------------------------------------------------------------
    # Baseline vs improved
    # ------------------------------------------------------------

    metric_names = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "ROC-AUC"
    ]


    original_values = [

        baseline_accuracy,

        baseline_precision,

        baseline_recall,

        baseline_f1,

        baseline_auc
    ]


    improved_values = [

        best_result[
            "accuracy"
        ],

        best_result[
            "precision"
        ],

        best_result[
            "recall"
        ],

        best_result[
            "f1"
        ],

        best_result[
            "roc_auc"
        ]
    ]


    x = np.arange(
        len(metric_names)
    )

    width = 0.35


    plt.figure(
        figsize=(10, 5)
    )


    plt.bar(

        x - width / 2,

        original_values,

        width,

        label="Original Swin + GNN"
    )


    plt.bar(

        x + width / 2,

        improved_values,

        width,

        label="Improved Swin + GNN"
    )


    plt.xticks(

        x,

        metric_names
    )


    plt.ylabel(
        "Score"
    )


    plt.title(
        "Original vs Improved Swin + GNN"
    )


    plt.legend()


    plt.ylim(
        0,
        1
    )


    plt.tight_layout()


    plt.savefig(

        PLOT_DIR
        /
        "swin_gnn_original_vs_improved.png",

        dpi=200
    )


    plt.close()


    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )


    plt.imshow(
        cm
    )


    plt.title(
        "Improved Swin + GNN Confusion Matrix"
    )


    plt.xlabel(
        "Predicted"
    )


    plt.ylabel(
        "Actual"
    )


    plt.xticks(

        [0, 1],

        CLASS_NAMES
    )


    plt.yticks(

        [0, 1],

        CLASS_NAMES
    )


    for i in range(2):

        for j in range(2):

            plt.text(

                j,

                i,

                cm[i, j],

                ha="center",

                va="center"
            )


    plt.tight_layout()


    plt.savefig(

        PLOT_DIR
        /
        "improved_swin_gnn_confusion_matrix.png",

        dpi=200
    )


    plt.close()


except Exception as e:

    print()

    print(
        "Plot warning:"
    )

    print(
        e
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print()

print("=" * 70)

print(
    "IMPROVED SWIN + GNN COMPLETED"
)

print("=" * 70)

print()

print(
    "KNN EXPERIMENT RESULTS:"
)

print()

print(
    results_df.to_string(
        index=False
    )
)

print()

print("=" * 70)

print(
    "BEST CONFIGURATION"
)

print("=" * 70)

print()

print(
    "Best K:",
    best_result["k"]
)

print()

print(
    f"Accuracy   : "
    f"{best_result['accuracy']:.4f}"
)

print(
    f"Precision  : "
    f"{best_result['precision']:.4f}"
)

print(
    f"Recall     : "
    f"{best_result['recall']:.4f}"
)

print(
    f"F1         : "
    f"{best_result['f1']:.4f}"
)

print(
    f"ROC-AUC    : "
    f"{best_result['roc_auc']:.4f}"
)

print()

print(
    "ORIGINAL BASELINE:"
)

print(
    "Accuracy   : 82.46%"
)

print(
    "F1         : 84.65%"
)

print(
    "ROC-AUC    : 87.94%"
)

print()

print(
    "IMPROVEMENT:"
)

print(

    f"Accuracy   : "
    f"{(best_result['accuracy'] - baseline_accuracy) * 100:+.2f} percentage points"
)

print(

    f"F1         : "
    f"{(best_result['f1'] - baseline_f1) * 100:+.2f} percentage points"
)

print(

    f"ROC-AUC    : "
    f"{(best_result['roc_auc'] - baseline_auc) * 100:+.2f} percentage points"
)

print()

print("=" * 70)

print(
    "OUTPUT FILES"
)

print("=" * 70)

print()

print(
    "Model:"
)

print(
    best_model_file
)

print()

print(
    "K comparison:"
)

print(
    results_file
)

print()

print(
    "Training history:"
)

print(
    history_file
)

print()

print(
    "Baseline comparison:"
)

print(
    comparison_file
)

print()

print(
    "Summary:"
)

print(
    summary_file
)

print()

print(
    "Confusion matrix:"
)

print(
    cm_file
)

print()

print(
    "Classification report:"
)

print(
    report_file
)

print()

print("=" * 70)

print(
    "✓ IMPROVED SWIN + GNN EXPERIMENT FINISHED"
)

print("=" * 70)