# ============================================================
# VERIFIED SWIN TRANSFORMER + GNN - VERSION 3
# ============================================================
# Independent verification of the saved improved GNN.
#
# IMPORTANT:
#   This script DOES NOT train anything.
#
#   It loads:
#       1. best_swin_transformer_thermal.pth
#       2. best_swin_gnn_improved.pth
#
#   Then:
#       Swin -> 768 features
#       Thermal -> 7 features
#       Combined -> 775 features
#       KNN K=5
#       Saved GNN -> predictions
#
#   The GNN architecture below is reconstructed from the
#   ACTUAL parameter structure in the saved checkpoint.
# ============================================================

import os
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
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "standardized"
)

THERMAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "thermal_features.csv"
)

SWIN_FILE = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "best_swin_transformer_thermal.pth"
)

GNN_FILE = (
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

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIG
# ============================================================

SEED = 42

IMAGE_SIZE = 224

NUM_CLASSES = 2

SWIN_DIM = 768

THERMAL_DIM = 7

INPUT_DIM = 775

HIDDEN_DIM = 256

K = 5

BATCH_SIZE = 16

DROPOUT = 0.4


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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 75)
print("INDEPENDENT SWIN + GNN VERIFICATION V3")
print("=" * 75)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDevice:")
print(DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 5. FILE CHECK
# ============================================================

print("\n" + "=" * 75)
print("CHECKING FILES")
print("=" * 75)


for path in [
    DATASET_ROOT,
    THERMAL_FILE,
    SWIN_FILE,
    GNN_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing required path:\n{path}"
        )

    print("✓", path)


# ============================================================
# 6. COLLECT DATASET
# ============================================================

def collect_images(split):

    records = []

    split_dir = (
        DATASET_ROOT
        / split
    )

    mapping = {
        "Control": 0,
        "DM": 1,
    }

    for class_name, label in mapping.items():

        class_dir = (
            split_dir
            / class_name
        )

        if not class_dir.exists():

            raise FileNotFoundError(
                class_dir
            )

        for image_path in sorted(
            class_dir.iterdir()
        ):

            if image_path.suffix.lower() not in [
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
            ]:

                continue

            records.append(
                {
                    "image_path": str(
                        image_path
                    ),
                    "filename": image_path.name,
                    "class": class_name,
                    "label": label,
                }
            )

    return pd.DataFrame(records)


print("\n" + "=" * 75)
print("DATASET CHECK")
print("=" * 75)


train_df = collect_images(
    "train"
)

val_df = collect_images(
    "val"
)


print(
    "\nTraining images:",
    len(train_df)
)

print(
    "Validation images:",
    len(val_df)
)


if len(train_df) != 1444:

    raise RuntimeError(
        f"Expected 1444 training images, "
        f"found {len(train_df)}"
    )


if len(val_df) != 422:

    raise RuntimeError(
        f"Expected 422 validation images, "
        f"found {len(val_df)}"
    )


print(
    "✓ Training = 1444"
)

print(
    "✓ Validation = 422"
)


if train_df["image_path"].duplicated().sum() != 0:

    raise RuntimeError(
        "Duplicate training image paths."
    )


if val_df["image_path"].duplicated().sum() != 0:

    raise RuntimeError(
        "Duplicate validation image paths."
    )


if val_df["filename"].duplicated().sum() != 0:

    raise RuntimeError(
        "Duplicate validation filenames."
    )


print(
    "✓ No duplicate validation samples."
)


# ============================================================
# 7. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# 8. LOAD SWIN
# ============================================================

print("\n" + "=" * 75)
print("LOADING TRAINED SWIN")
print("=" * 75)


swin_model = swin_t(
    weights=None
)


swin_model.head = nn.Linear(
    swin_model.head.in_features,
    NUM_CLASSES
)


swin_checkpoint = torch.load(
    SWIN_FILE,
    map_location=DEVICE
)


print(
    "\nCheckpoint keys:"
)

for key in swin_checkpoint.keys():

    print(
        " ",
        key
    )


swin_state = (
    swin_checkpoint[
        "model_state_dict"
    ]
)


clean_swin_state = {}


for key, value in swin_state.items():

    if key.startswith(
        "module."
    ):

        key = key[7:]

    clean_swin_state[
        key
    ] = value


missing, unexpected = (
    swin_model.load_state_dict(
        clean_swin_state,
        strict=True
    )
)


print(
    "\nMissing keys:",
    len(missing)
)

print(
    "Unexpected keys:",
    len(unexpected)
)


if len(missing) != 0:

    raise RuntimeError(
        "Swin checkpoint failed strict loading."
    )


if len(unexpected) != 0:

    raise RuntimeError(
        "Swin checkpoint contains unexpected keys."
    )


print(
    "✓ Swin checkpoint loaded STRICTLY."
)


swin_model = swin_model.to(
    DEVICE
)

swin_model.eval()


# ============================================================
# 9. SWIN EMBEDDING MODEL
# ============================================================

class SwinEmbeddingExtractor(
    nn.Module
):

    def __init__(
        self,
        model
    ):

        super().__init__()

        self.model = model


    def forward(
        self,
        x
    ):

        x = self.model.features(
            x
        )

        x = self.model.norm(
            x
        )

        x = self.model.permute(
            x
        )

        x = self.model.avgpool(
            x
        )

        x = torch.flatten(
            x,
            1
        )

        return x


embedding_model = (
    SwinEmbeddingExtractor(
        swin_model
    )
    .to(DEVICE)
)


embedding_model.eval()


# ============================================================
# 10. EXTRACT SWIN EMBEDDINGS
# ============================================================

def extract_embeddings(
    dataframe
):

    all_embeddings = []

    all_labels = []

    all_filenames = []


    total = len(dataframe)


    with torch.no_grad():

        for start in range(
            0,
            total,
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                total
            )


            batch = dataframe.iloc[
                start:end
            ]


            images = []


            for _, row in batch.iterrows():

                with Image.open(
                    row["image_path"]
                ) as image:

                    image = image.convert(
                        "RGB"
                    )

                    tensor = transform(
                        image
                    )

                images.append(
                    tensor
                )


            batch_tensor = torch.stack(
                images
            ).to(
                DEVICE
            )


            embeddings = (
                embedding_model(
                    batch_tensor
                )
            )


            all_embeddings.append(
                embeddings
                .cpu()
                .numpy()
            )


            all_labels.extend(
                batch["label"].tolist()
            )


            all_filenames.extend(
                batch["filename"].tolist()
            )


            print(
                f"\rProcessed "
                f"{end}/{total}",
                end=""
            )


    print()


    return (
        np.concatenate(
            all_embeddings,
            axis=0
        ),
        np.asarray(
            all_labels,
            dtype=np.int64
        ),
        all_filenames
    )


print(
    "\nExtracting training embeddings..."
)


train_embeddings, train_labels, train_filenames = (
    extract_embeddings(
        train_df
    )
)


print(
    "\nExtracting validation embeddings..."
)


val_embeddings, val_labels, val_filenames = (
    extract_embeddings(
        val_df
    )
)


print(
    "\nTrain embeddings:",
    train_embeddings.shape
)

print(
    "Validation embeddings:",
    val_embeddings.shape
)


if train_embeddings.shape != (
    1444,
    768
):

    raise RuntimeError(
        "Incorrect train embedding shape."
    )


if val_embeddings.shape != (
    422,
    768
):

    raise RuntimeError(
        "Incorrect validation embedding shape."
    )


print(
    "✓ Swin embeddings verified."
)


# ============================================================
# 11. LOAD THERMAL FEATURES
# ============================================================

print("\n" + "=" * 75)
print("LOADING THERMAL FEATURES")
print("=" * 75)


features_df = pd.read_csv(
    THERMAL_FILE
)


print(
    "\nTotal feature records:",
    len(features_df)
)


thermal_df = features_df[
    (
        features_df[
            "noise_type"
        ]
        .astype(str)
        .str.lower()
        == "gaussian"
    )
    &
    (
        features_df[
            "denoising_method"
        ]
        .astype(str)
        .str.lower()
        == "median"
    )
].copy()


print(
    "Gaussian + Median records:",
    len(thermal_df)
)


if len(thermal_df) != 1866:

    raise RuntimeError(
        "Expected 1866 Gaussian + Median records."
    )


print(
    "✓ Gaussian + Median verified."
)


# ============================================================
# 12. NORMALIZE FILENAMES
# ============================================================

def normalize_filename(
    filename
):

    stem = Path(
        str(filename)
    ).stem


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

            if stem.endswith(
                suffix
            ):

                stem = stem[
                    : -len(suffix)
                ]

                changed = True


    return stem


thermal_df[
    "match_key"
] = thermal_df[
    "filename"
].apply(
    normalize_filename
)


train_df[
    "match_key"
] = train_df[
    "filename"
].apply(
    normalize_filename
)


val_df[
    "match_key"
] = val_df[
    "filename"
].apply(
    normalize_filename
)


# ============================================================
# 13. THERMAL DUPLICATE CHECK
# ============================================================

duplicate_keys = (
    thermal_df[
        "match_key"
    ]
    .duplicated()
    .sum()
)


print(
    "\nThermal duplicate keys:",
    duplicate_keys
)


if duplicate_keys != 0:

    raise RuntimeError(
        "Duplicate thermal match keys."
    )


thermal_lookup = (
    thermal_df
    .set_index(
        "match_key"
    )
)


# ============================================================
# 14. MATCH THERMAL FEATURES
# ============================================================

def get_features(
    dataframe
):

    output = []

    labels = []

    missing = []

    mismatch = []


    for _, row in dataframe.iterrows():

        key = row[
            "match_key"
        ]


        if key not in thermal_lookup.index:

            missing.append(
                row["filename"]
            )

            continue


        feature_row = (
            thermal_lookup.loc[key]
        )


        values = (
            feature_row[
                THERMAL_FEATURES
            ]
            .astype(float)
            .values
        )


        output.append(
            values
        )


        thermal_label = int(
            feature_row[
                "label"
            ]
        )


        image_label = int(
            row[
                "label"
            ]
        )


        labels.append(
            thermal_label
        )


        if thermal_label != image_label:

            mismatch.append(
                row["filename"]
            )


    return (
        np.asarray(
            output,
            dtype=np.float32
        ),
        np.asarray(
            labels,
            dtype=np.int64
        ),
        missing,
        mismatch
    )


train_thermal, train_thermal_labels, train_missing, train_mismatch = (
    get_features(
        train_df
    )
)


val_thermal, val_thermal_labels, val_missing, val_mismatch = (
    get_features(
        val_df
    )
)


print(
    "\nTrain thermal shape:",
    train_thermal.shape
)

print(
    "Validation thermal shape:",
    val_thermal.shape
)


print(
    "\nTrain missing:",
    len(train_missing)
)

print(
    "Validation missing:",
    len(val_missing)
)

print(
    "Train label mismatch:",
    len(train_mismatch)
)

print(
    "Validation label mismatch:",
    len(val_mismatch)
)


if len(train_missing) != 0:

    raise RuntimeError(
        "Training thermal feature matching failed."
    )


if len(val_missing) != 0:

    raise RuntimeError(
        "Validation thermal feature matching failed."
    )


if len(train_mismatch) != 0:

    raise RuntimeError(
        "Training thermal labels mismatch."
    )


if len(val_mismatch) != 0:

    raise RuntimeError(
        "Validation thermal labels mismatch."
    )


if not np.array_equal(
    train_labels,
    train_thermal_labels
):

    raise RuntimeError(
        "Training image and thermal labels differ."
    )


if not np.array_equal(
    val_labels,
    val_thermal_labels
):

    raise RuntimeError(
        "Validation image and thermal labels differ."
    )


print(
    "\n✓ All 1444 training samples matched."
)

print(
    "✓ All 422 validation samples matched."
)

print(
    "✓ Labels match."
)


# ============================================================
# 15. FINITE CHECK
# ============================================================

for name, array in [
    (
        "train embeddings",
        train_embeddings
    ),
    (
        "validation embeddings",
        val_embeddings
    ),
    (
        "train thermal",
        train_thermal
    ),
    (
        "validation thermal",
        val_thermal
    ),
]:

    if not np.isfinite(
        array
    ).all():

        raise RuntimeError(
            f"NaN/Inf found in {name}."
        )


print(
    "✓ No NaN/Inf values."
)


# ============================================================
# 16. TRAIN-ONLY SCALING
# ============================================================

print("\n" + "=" * 75)
print("TRAIN-ONLY STANDARDIZATION")
print("=" * 75)


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


print(
    "✓ Swin scaler fitted using training data only."
)

print(
    "✓ Thermal scaler fitted using training data only."
)


# ============================================================
# 17. COMBINE 768 + 7
# ============================================================

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
    "\nTrain feature shape:",
    train_features.shape
)

print(
    "Validation feature shape:",
    val_features.shape
)


if train_features.shape != (
    1444,
    775
):

    raise RuntimeError(
        "Incorrect training feature dimension."
    )


if val_features.shape != (
    422,
    775
):

    raise RuntimeError(
        "Incorrect validation feature dimension."
    )


print(
    "✓ 775-D representation verified."
)


# ============================================================
# 18. BUILD KNN GRAPH
# ============================================================

print("\n" + "=" * 75)
print("BUILDING K=5 KNN GRAPHS")
print("=" * 75)


def build_knn_graph(
    features,
    k
):

    normalized = F.normalize(
        torch.tensor(
            features,
            dtype=torch.float32
        ),
        p=2,
        dim=1
    ).numpy()


    knn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine"
    )


    knn.fit(
        normalized
    )


    distances, indices = (
        knn.kneighbors(
            normalized
        )
    )


    src = []

    dst = []

    weights = []


    for i in range(
        len(features)
    ):

        for position in range(
            1,
            k + 1
        ):

            j = indices[
                i,
                position
            ]


            distance = distances[
                i,
                position
            ]


            weight = max(
                1.0 - float(distance),
                0.0
            )


            src.append(i)

            dst.append(j)

            weights.append(
                weight
            )


    edge_index = torch.tensor(
        [
            src,
            dst
        ],
        dtype=torch.long
    )


    edge_weight = torch.tensor(
        weights,
        dtype=torch.float32
    )


    return (
        edge_index,
        edge_weight
    )


train_edge_index, train_edge_weight = (
    build_knn_graph(
        train_features,
        K
    )
)


val_edge_index, val_edge_weight = (
    build_knn_graph(
        val_features,
        K
    )
)


print(
    "\nK:",
    K
)

print(
    "Train edges:",
    train_edge_index.shape[1]
)

print(
    "Validation edges:",
    val_edge_index.shape[1]
)


# ============================================================
# 19. EXACT GRAPH CONVOLUTION
# ============================================================

class GraphConv(
    nn.Module
):

    def __init__(
        self,
        in_features,
        out_features
    ):

        super().__init__()


        self.linear = nn.Linear(
            in_features,
            out_features
        )


        self.norm = nn.LayerNorm(
            out_features
        )


    def forward(
        self,
        x,
        edge_index,
        edge_weight
    ):

        src = edge_index[0]

        dst = edge_index[1]


        messages = (
            x[src]
            * edge_weight.unsqueeze(
                1
            )
        )


        aggregated = (
            torch.zeros_like(x)
        )


        aggregated.index_add_(
            0,
            dst,
            messages
        )


        degree = torch.zeros(
            x.size(0),
            device=x.device
        )


        degree.index_add_(
            0,
            dst,
            edge_weight
        )


        degree = degree.clamp(
            min=1e-6
        )


        aggregated = (
            aggregated
            / degree.unsqueeze(1)
        )


        output = self.linear(
            aggregated
        )


        output = self.norm(
            output
        )


        return output


# ============================================================
# 20. EXACT IMPROVED GNN ARCHITECTURE
# ============================================================

class ImprovedSwinGNN(
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


        # Actual checkpoint:
        #
        # input_projection.0
        # input_projection.1
        #
        self.input_projection = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            )
        )


        # Actual checkpoint:
        #
        # gcn1.linear
        # gcn1.norm
        #
        self.gcn1 = GraphConv(
            hidden_dim,
            hidden_dim
        )


        self.gcn2 = GraphConv(
            hidden_dim,
            hidden_dim
        )


        self.gcn3 = GraphConv(
            hidden_dim,
            hidden_dim
        )


        # Actual checkpoint classifier:
        #
        # classifier.0
        # classifier.1
        # classifier.4
        #
        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )


    def forward(
        self,
        x,
        edge_index,
        edge_weight
    ):

        x = self.input_projection(
            x
        )


        x = F.gelu(
            x
        )


        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )


        h1 = self.gcn1(
            x,
            edge_index,
            edge_weight
        )


        h1 = F.gelu(
            h1
        )


        h1 = F.dropout(
            h1,
            p=DROPOUT,
            training=self.training
        )


        h2 = self.gcn2(
            h1,
            edge_index,
            edge_weight
        )


        h2 = F.gelu(
            h2
        )


        h2 = F.dropout(
            h2,
            p=DROPOUT,
            training=self.training
        )


        h2 = h2 + h1


        h3 = self.gcn3(
            h2,
            edge_index,
            edge_weight
        )


        h3 = F.gelu(
            h3
        )


        h3 = F.dropout(
            h3,
            p=DROPOUT,
            training=self.training
        )


        h3 = h3 + h2


        logits = self.classifier(
            h3
        )


        return logits


# ============================================================
# 21. LOAD GNN CHECKPOINT
# ============================================================

print("\n" + "=" * 75)
print("LOADING SAVED GNN")
print("=" * 75)


gnn_checkpoint = torch.load(
    GNN_FILE,
    map_location=DEVICE
)


print(
    "\nCheckpoint keys:"
)


for key in gnn_checkpoint.keys():

    print(
        " ",
        key
    )


print(
    "\nSaved best K:",
    gnn_checkpoint.get(
        "best_k"
    )
)


print(
    "Saved best epoch:",
    gnn_checkpoint.get(
        "best_epoch"
    )
)


print(
    "Saved accuracy:",
    gnn_checkpoint.get(
        "accuracy"
    )
)


print(
    "Saved precision:",
    gnn_checkpoint.get(
        "precision"
    )
)


print(
    "Saved recall:",
    gnn_checkpoint.get(
        "recall"
    )
)


print(
    "Saved F1:",
    gnn_checkpoint.get(
        "f1"
    )
)


print(
    "Saved ROC-AUC:",
    gnn_checkpoint.get(
        "roc_auc"
    )
)


# ============================================================
# 22. CREATE EXACT GNN
# ============================================================

gnn_model = ImprovedSwinGNN(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(
    DEVICE
)


gnn_state = (
    gnn_checkpoint[
        "model_state_dict"
    ]
)


clean_gnn_state = {}


for key, value in gnn_state.items():

    if key.startswith(
        "module."
    ):

        key = key[7:]


    clean_gnn_state[
        key
    ] = value


print(
    "\nLoading GNN weights STRICTLY..."
)


try:

    gnn_model.load_state_dict(
        clean_gnn_state,
        strict=True
    )

except RuntimeError as error:

    print(
        "\n✗ GNN architecture still does not "
        "match checkpoint."
    )

    print(
        "\nDetailed error:"
    )

    print(error)

    raise


print(
    "\n✓ GNN checkpoint loaded STRICTLY."
)


gnn_model.eval()


# ============================================================
# 23. VALIDATION INFERENCE
# ============================================================

print("\n" + "=" * 75)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 75)


x_val = torch.tensor(
    val_features,
    dtype=torch.float32,
    device=DEVICE
)


edge_index = (
    val_edge_index
    .to(DEVICE)
)


edge_weight = (
    val_edge_weight
    .to(DEVICE)
)


with torch.no_grad():

    logits = gnn_model(
        x_val,
        edge_index,
        edge_weight
    )


    probabilities = torch.softmax(
        logits,
        dim=1
    )


    predictions = torch.argmax(
        probabilities,
        dim=1
    )


predictions = (
    predictions
    .cpu()
    .numpy()
    .astype(int)
)


probability_dm = (
    probabilities[:, 1]
    .cpu()
    .numpy()
)


true_labels = (
    val_labels
    .astype(int)
)


print(
    "\nTrue labels:",
    len(true_labels)
)

print(
    "Predictions:",
    len(predictions)
)

print(
    "DM probabilities:",
    len(probability_dm)
)


if len(true_labels) != 422:

    raise RuntimeError(
        "Wrong number of validation labels."
    )


if len(predictions) != 422:

    raise RuntimeError(
        "Wrong number of predictions."
    )


print(
    "✓ Exactly 422 validation predictions."
)


# ============================================================
# 24. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "filename": val_filenames,
        "true_label": true_labels,
        "predicted_label": predictions,
        "probability_DM": probability_dm
    }
)


prediction_file = (
    RESULTS_DIR
    / "verified_swin_gnn_predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False
)


print(
    "\n✓ Saved:"
)

print(
    prediction_file
)


# ============================================================
# 25. RECOMPUTE METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)


precision = precision_score(
    true_labels,
    predictions,
    zero_division=0
)


recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)


f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0
)


roc_auc = roc_auc_score(
    true_labels,
    probability_dm
)


cm = confusion_matrix(
    true_labels,
    predictions
)


# ============================================================
# 26. PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("INDEPENDENTLY RECOMPUTED RESULTS")
print("=" * 75)


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
    f"ROC-AUC  : {roc_auc:.6f}"
    f"  ({roc_auc * 100:.2f}%)"
)


print(
    "\nConfusion Matrix:"
)

print(cm)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        true_labels,
        predictions,
        target_names=[
            "Control",
            "DM"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 27. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [
        {
            "model":
                "Improved Swin + GNN",

            "k":
                K,

            "validation_samples":
                len(true_labels),

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "roc_auc":
                roc_auc,
        }
    ]
)


metrics_file = (
    RESULTS_DIR
    / "verified_swin_gnn_metrics.csv"
)


metrics_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# 28. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Control",
        "Actual_DM"
    ],
    columns=[
        "Predicted_Control",
        "Predicted_DM"
    ]
)


cm_file = (
    RESULTS_DIR
    / "verified_swin_gnn_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_file
)


# ============================================================
# 29. COMPARE TO SAVED CHECKPOINT
# ============================================================

print("\n" + "=" * 75)
print("COMPARING VERIFIED RESULTS WITH CHECKPOINT")
print("=" * 75)


verified = {
    "accuracy":
        accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "f1":
        f1,

    "roc_auc":
        roc_auc,
}


saved = {
    "accuracy":
        float(
            gnn_checkpoint[
                "accuracy"
            ]
        ),

    "precision":
        float(
            gnn_checkpoint[
                "precision"
            ]
        ),

    "recall":
        float(
            gnn_checkpoint[
                "recall"
            ]
        ),

    "f1":
        float(
            gnn_checkpoint[
                "f1"
            ]
        ),

    "roc_auc":
        float(
            gnn_checkpoint[
                "roc_auc"
            ]
        ),
}


all_match = True


for key in verified:

    difference = abs(
        verified[key]
        -
        saved[key]
    )


    match = (
        difference < 0.0001
    )


    if not match:

        all_match = False


    status = (
        "MATCH"
        if match
        else "DIFFER"
    )


    print(
        f"{key:10s} | "
        f"Verified={verified[key]:.6f} | "
        f"Checkpoint={saved[key]:.6f} | "
        f"Difference={difference:.6f} | "
        f"{status}"
    )


# ============================================================
# 30. BEST K CHECK
# ============================================================

saved_k = int(
    gnn_checkpoint[
        "best_k"
    ]
)


print(
    "\nSaved best K:",
    saved_k
)


print(
    "Verification K:",
    K
)


if saved_k == K:

    print(
        "✓ K=5 matches."
    )

else:

    print(
        "✗ K mismatch."
    )

    all_match = False


# ============================================================
# 31. FINAL VERDICT
# ============================================================

print("\n" + "=" * 75)
print("FINAL VERIFICATION VERDICT")
print("=" * 75)


if all_match:

    print(
        "\n✓ VERIFIED"
    )

    print(
        "\nThe independently generated "
        "422 validation predictions reproduce "
        "the metrics stored in the trained "
        "improved Swin + GNN checkpoint."
    )


    print(
        "\nFINAL VERIFIED RESULT:"
    )


    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )


    print(
        f"Precision: {precision * 100:.2f}%"
    )


    print(
        f"Recall   : {recall * 100:.2f}%"
    )


    print(
        f"F1 Score : {f1 * 100:.2f}%"
    )


    print(
        f"ROC-AUC  : {roc_auc * 100:.2f}%"
    )


else:

    print(
        "\n⚠ NOT VERIFIED"
    )

    print(
        "\nThe independently recomputed "
        "metrics do not match the checkpoint."
    )


# ============================================================
# 32. OUTPUT FILES
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT FILES")
print("=" * 75)


print(
    "\nPredictions:"
)

print(
    prediction_file
)


print(
    "\nMetrics:"
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


print(
    "\nVerification completed."
)


# ============================================================
# END
# ============================================================