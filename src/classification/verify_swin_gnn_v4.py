# ============================================================
# FINAL INDEPENDENT VERIFICATION
# IMPROVED SWIN TRANSFORMER + GNN
# ============================================================

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
# 2. CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224

NUM_CLASSES = 2

SWIN_DIM = 768

THERMAL_DIM = 7

INPUT_DIM = 775

HIDDEN_DIM = 256

CLASSIFIER_DIM = 128

DROPOUT = 0.4

K = 5

BATCH_SIZE = 16


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
print("FINAL INDEPENDENT SWIN + GNN VERIFICATION V4")
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
# 5. CHECK FILES
# ============================================================

print("\n" + "=" * 75)
print("CHECKING REQUIRED FILES")
print("=" * 75)


for path in [
    DATASET_ROOT,
    THERMAL_FILE,
    SWIN_FILE,
    GNN_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing:\n{path}"
        )

    print(
        "✓",
        path
    )


# ============================================================
# 6. DATASET
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
                    "image_path":
                        str(image_path),

                    "filename":
                        image_path.name,

                    "class":
                        class_name,

                    "label":
                        label,
                }
            )

    return pd.DataFrame(records)


print("\n" + "=" * 75)
print("DATASET VERIFICATION")
print("=" * 75)


train_df = collect_images(
    "train"
)

val_df = collect_images(
    "val"
)


print(
    "\nTrain:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)


if len(train_df) != 1444:

    raise RuntimeError(
        "Training count is not 1444."
    )


if len(val_df) != 422:

    raise RuntimeError(
        "Validation count is not 422."
    )


print(
    "✓ 1444 training images"
)

print(
    "✓ 422 validation images"
)


# ============================================================
# 7. DUPLICATES
# ============================================================

if train_df[
    "image_path"
].duplicated().sum() != 0:

    raise RuntimeError(
        "Duplicate training paths."
    )


if val_df[
    "image_path"
].duplicated().sum() != 0:

    raise RuntimeError(
        "Duplicate validation paths."
    )


if val_df[
    "filename"
].duplicated().sum() != 0:

    raise RuntimeError(
        "Duplicate validation filenames."
    )


print(
    "✓ No duplicate validation samples."
)


# ============================================================
# 8. IMAGE TRANSFORM
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
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ]
)


# ============================================================
# 9. SWIN MODEL
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


if missing or unexpected:

    raise RuntimeError(
        "Swin checkpoint did not load exactly."
    )


print(
    "✓ Swin checkpoint loaded STRICTLY."
)


swin_model = swin_model.to(
    DEVICE
)

swin_model.eval()


# ============================================================
# 10. SWIN EMBEDDING EXTRACTOR
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
# 11. EMBEDDINGS
# ============================================================

def extract_embeddings(
    dataframe
):

    embeddings = []

    labels = []

    filenames = []


    with torch.no_grad():

        for start in range(
            0,
            len(dataframe),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(dataframe)
            )


            batch = dataframe.iloc[
                start:end
            ]


            tensors = []


            for _, row in batch.iterrows():

                with Image.open(
                    row[
                        "image_path"
                    ]
                ) as image:

                    image = image.convert(
                        "RGB"
                    )

                    tensor = transform(
                        image
                    )


                tensors.append(
                    tensor
                )


            x = torch.stack(
                tensors
            ).to(
                DEVICE
            )


            output = (
                embedding_model(x)
            )


            embeddings.append(
                output
                .cpu()
                .numpy()
            )


            labels.extend(
                batch[
                    "label"
                ].tolist()
            )


            filenames.extend(
                batch[
                    "filename"
                ].tolist()
            )


            print(
                f"\rProcessed "
                f"{end}/{len(dataframe)}",
                end=""
            )


    print()


    return (
        np.concatenate(
            embeddings,
            axis=0
        ),

        np.asarray(
            labels,
            dtype=np.int64
        ),

        filenames
    )


print(
    "\nExtracting train embeddings..."
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
    "\nTrain embedding shape:",
    train_embeddings.shape
)

print(
    "Validation embedding shape:",
    val_embeddings.shape
)


if train_embeddings.shape != (
    1444,
    768
):

    raise RuntimeError(
        "Invalid train Swin embedding shape."
    )


if val_embeddings.shape != (
    422,
    768
):

    raise RuntimeError(
        "Invalid validation Swin embedding shape."
    )


print(
    "✓ 768-D Swin embeddings verified."
)


# ============================================================
# 12. THERMAL FEATURES
# ============================================================

print("\n" + "=" * 75)
print("THERMAL FEATURES")
print("=" * 75)


features_df = pd.read_csv(
    THERMAL_FILE
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
    "\nTotal records:",
    len(features_df)
)

print(
    "Gaussian + Median:",
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
# 13. FILENAME MATCHING
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
                    :-len(suffix)
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


if thermal_df[
    "match_key"
].duplicated().sum() != 0:

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
# 14. MATCH FEATURES
# ============================================================

def match_features(
    dataframe
):

    output = []

    labels = []

    missing = []

    mismatches = []


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


        if (
            thermal_label
            !=
            image_label
        ):

            mismatches.append(
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

        mismatches
    )


train_thermal, train_thermal_labels, train_missing, train_mismatch = (
    match_features(
        train_df
    )
)


val_thermal, val_thermal_labels, val_missing, val_mismatch = (
    match_features(
        val_df
    )
)


print(
    "\nTrain thermal:",
    train_thermal.shape
)

print(
    "Validation thermal:",
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


if train_missing:

    raise RuntimeError(
        "Missing train thermal features."
    )


if val_missing:

    raise RuntimeError(
        "Missing validation thermal features."
    )


if train_mismatch:

    raise RuntimeError(
        "Training label mismatch."
    )


if val_mismatch:

    raise RuntimeError(
        "Validation label mismatch."
    )


if not np.array_equal(
    train_labels,
    train_thermal_labels
):

    raise RuntimeError(
        "Training image/thermal labels differ."
    )


if not np.array_equal(
    val_labels,
    val_thermal_labels
):

    raise RuntimeError(
        "Validation image/thermal labels differ."
    )


print(
    "\n✓ All training samples matched."
)

print(
    "✓ All validation samples matched."
)

print(
    "✓ Labels match."
)


# ============================================================
# 15. STANDARDIZATION
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
    "✓ Train-only Swin scaling."
)

print(
    "✓ Train-only thermal scaling."
)


# ============================================================
# 16. 775-D FEATURES
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
    "\nTrain:",
    train_features.shape
)

print(
    "Validation:",
    val_features.shape
)


if train_features.shape != (
    1444,
    775
):

    raise RuntimeError(
        "Invalid train feature dimension."
    )


if val_features.shape != (
    422,
    775
):

    raise RuntimeError(
        "Invalid validation feature dimension."
    )


print(
    "✓ 775-D representation verified."
)


# ============================================================
# 17. KNN GRAPH
# ============================================================

print("\n" + "=" * 75)
print("K=5 KNN GRAPH")
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


    sources = []

    destinations = []

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


            sources.append(i)

            destinations.append(j)

            weights.append(
                weight
            )


    edge_index = torch.tensor(
        [
            sources,
            destinations
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
    "K:",
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
# 18. EXACT GCN
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
            *
            edge_weight.unsqueeze(
                1
            )
        )


        aggregated = torch.zeros_like(
            x
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
            /
            degree.unsqueeze(1)
        )


        output = self.linear(
            aggregated
        )


        output = self.norm(
            output
        )


        return output


# ============================================================
# 19. EXACT SAVED GNN ARCHITECTURE
# ============================================================

class ImprovedSwinGNN(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        # Checkpoint:
        # input_projection.0
        # input_projection.1

        self.input_projection = nn.Sequential(

            nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM
            ),

            nn.LayerNorm(
                HIDDEN_DIM
            )
        )


        # Checkpoint:
        # gcn1.linear
        # gcn1.norm

        self.gcn1 = GraphConv(
            HIDDEN_DIM,
            HIDDEN_DIM
        )


        self.gcn2 = GraphConv(
            HIDDEN_DIM,
            HIDDEN_DIM
        )


        self.gcn3 = GraphConv(
            HIDDEN_DIM,
            HIDDEN_DIM
        )


        # Checkpoint:
        # classifier.0 = LayerNorm(256)
        # classifier.1 = Linear(256,128)
        # classifier.4 = Linear(128,2)

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                HIDDEN_DIM
            ),

            nn.Linear(
                HIDDEN_DIM,
                CLASSIFIER_DIM
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                CLASSIFIER_DIM,
                NUM_CLASSES
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
# 20. LOAD GNN CHECKPOINT
# ============================================================

print("\n" + "=" * 75)
print("LOADING GNN CHECKPOINT")
print("=" * 75)


gnn_checkpoint = torch.load(
    GNN_FILE,
    map_location=DEVICE
)


print(
    "\nSaved best K:",
    gnn_checkpoint[
        "best_k"
    ]
)


print(
    "Saved best epoch:",
    gnn_checkpoint[
        "best_epoch"
    ]
)


print(
    "Saved accuracy:",
    gnn_checkpoint[
        "accuracy"
    ]
)


print(
    "Saved precision:",
    gnn_checkpoint[
        "precision"
    ]
)


print(
    "Saved recall:",
    gnn_checkpoint[
        "recall"
    ]
)


print(
    "Saved F1:",
    gnn_checkpoint[
        "f1"
    ]
)


print(
    "Saved ROC-AUC:",
    gnn_checkpoint[
        "roc_auc"
    ]
)


if int(
    gnn_checkpoint[
        "best_k"
    ]
) != K:

    raise RuntimeError(
        "Checkpoint best K is not 5."
    )


gnn_model = ImprovedSwinGNN().to(
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


# ============================================================
# 21. STRICT ARCHITECTURE CHECK
# ============================================================

print(
    "\nLoading GNN weights STRICTLY..."
)


gnn_model.load_state_dict(
    clean_gnn_state,
    strict=True
)


print(
    "✓ GNN checkpoint loaded STRICTLY."
)


gnn_model.eval()


# ============================================================
# 22. INFERENCE
# ============================================================

print("\n" + "=" * 75)
print("INDEPENDENT VALIDATION INFERENCE")
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
    probabilities[
        :,
        1
    ]
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
    "Probabilities:",
    len(probability_dm)
)


if not (
    len(true_labels)
    ==
    len(predictions)
    ==
    len(probability_dm)
    ==
    422
):

    raise RuntimeError(
        "Prediction count is not 422."
    )


print(
    "✓ Exactly 422 predictions."
)


# ============================================================
# 23. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "filename":
            val_filenames,

        "true_label":
            true_labels,

        "predicted_label":
            predictions,

        "probability_DM":
            probability_dm
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
    "\n✓ Predictions saved:"
)

print(
    prediction_file
)


# ============================================================
# 24. RECOMPUTE METRICS
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
# 25. RESULTS
# ============================================================

print("\n" + "=" * 75)
print("INDEPENDENTLY RECOMPUTED RESULTS")
print("=" * 75)


print(
    f"\nAccuracy : {accuracy:.10f}"
    f"  ({accuracy * 100:.4f}%)"
)


print(
    f"Precision: {precision:.10f}"
    f"  ({precision * 100:.4f}%)"
)


print(
    f"Recall   : {recall:.10f}"
    f"  ({recall * 100:.4f}%)"
)


print(
    f"F1 Score : {f1:.10f}"
    f"  ({f1 * 100:.4f}%)"
)


print(
    f"ROC-AUC  : {roc_auc:.10f}"
    f"  ({roc_auc * 100:.4f}%)"
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
# 26. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [
        {
            "model":
                "Improved Swin + GNN",

            "k":
                K,

            "validation_samples":
                422,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "roc_auc":
                roc_auc
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
# 27. SAVE CONFUSION MATRIX
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
# 28. COMPARE WITH CHECKPOINT
# ============================================================

print("\n" + "=" * 75)
print("CHECKPOINT VS INDEPENDENT VERIFICATION")
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
        roc_auc
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
        )
}


all_match = True


for metric in verified:

    difference = abs(
        verified[metric]
        -
        saved[metric]
    )


    match = (
        difference < 1e-4
    )


    if not match:

        all_match = False


    status = (
        "MATCH"
        if match
        else "DIFFER"
    )


    print(
        f"{metric:10s} | "
        f"Verified={verified[metric]:.10f} | "
        f"Saved={saved[metric]:.10f} | "
        f"Difference={difference:.10f} | "
        f"{status}"
    )


# ============================================================
# 29. CONFUSION MATRIX CONSISTENCY
# ============================================================

saved_cm = np.array(
    [
        [167, 3],
        [18, 234]
    ]
)


print(
    "\nExpected confusion matrix from "
    "reported 95.02% result:"
)

print(
    saved_cm
)


print(
    "\nIndependently generated confusion matrix:"
)

print(
    cm
)


if np.array_equal(
    cm,
    saved_cm
):

    print(
        "✓ Confusion matrix MATCH."
    )

else:

    print(
        "⚠ Confusion matrix DIFFER."
    )

    all_match = False


# ============================================================
# 30. FINAL VERDICT
# ============================================================

print("\n" + "=" * 75)
print("FINAL VERIFICATION VERDICT")
print("=" * 75)


if all_match:

    print(
        "\n✓ VERIFIED"
    )

    print(
        "\nThe saved improved Swin + GNN "
        "checkpoint reproduces its reported "
        "validation performance."
    )


    print(
        "\nFINAL VERIFIED VALIDATION METRICS:"
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
        "results do not match the saved "
        "checkpoint results."
    )


# ============================================================
# 31. OUTPUTS
# ============================================================

print("\n" + "=" * 75)
print("VERIFICATION OUTPUT FILES")
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
    "\nVerification finished."
)

# ============================================================
# END
# ============================================================