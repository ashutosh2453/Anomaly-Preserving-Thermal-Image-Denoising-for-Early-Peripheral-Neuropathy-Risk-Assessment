import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)


# ============================================================
# STEP 9
# SWIN TRANSFORMER + GRAPH NEURAL NETWORK
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


CLASSIFICATION_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "classification"
)


THERMAL_FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "features",
    "thermal_features.csv"
)


SWIN_MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "classification",
    "best_swin_transformer_thermal.pth"
)


MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "classification"
)


METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)


PLOTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "plots"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    METRICS_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0

NUM_CLASSES = 2

RANDOM_SEED = 42

K_NEIGHBORS = 5

GNN_HIDDEN_DIM = 128

GNN_DROPOUT = 0.30

GNN_EPOCHS = 50

GNN_LEARNING_RATE = 0.001

GNN_WEIGHT_DECAY = 0.0001


# ============================================================
# 3. SELECTED THERMAL FEATURES
# ============================================================

THERMAL_FEATURES = [

    "mean_intensity",

    "std_intensity",

    "percentile_90",

    "hotspot_area_ratio",

    "hotspot_mean_intensity",

    "absolute_asymmetry",

    "relative_asymmetry"

]


# ============================================================
# 4. RANDOM SEED
# ============================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# 5. CUDA
# ============================================================

if not torch.cuda.is_available():

    print("=" * 70)
    print("ERROR: CUDA GPU NOT AVAILABLE")
    print("=" * 70)

    raise SystemExit(1)


DEVICE = torch.device(
    "cuda"
)


torch.backends.cudnn.benchmark = True


print("=" * 70)
print("STEP 9 - SWIN TRANSFORMER + GNN INTEGRATION")
print("=" * 70)


print("\nPyTorch:")
print(torch.__version__)


print("\nCUDA:")
print(torch.version.cuda)


print("\nGPU:")
print(torch.cuda.get_device_name(0))


print("\nDevice:")
print(DEVICE)


# ============================================================
# 6. CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(
    THERMAL_FEATURE_FILE
):

    raise FileNotFoundError(
        f"Thermal feature file not found:\n"
        f"{THERMAL_FEATURE_FILE}"
    )


if not os.path.exists(
    SWIN_MODEL_FILE
):

    raise FileNotFoundError(
        f"Swin model not found:\n"
        f"{SWIN_MODEL_FILE}"
    )


# ============================================================
# 7. DATA TRANSFORM
# ============================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
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

])


# ============================================================
# 8. LOAD CLASSIFICATION DATASET
# ============================================================

print("\nLoading classification dataset...")


train_dir = os.path.join(
    CLASSIFICATION_ROOT,
    "train"
)


val_dir = os.path.join(
    CLASSIFICATION_ROOT,
    "val"
)


train_dataset = datasets.ImageFolder(

    train_dir,

    transform=image_transform

)


val_dataset = datasets.ImageFolder(

    val_dir,

    transform=image_transform

)


print("\nClasses:")
print(train_dataset.classes)


print("\nClass mapping:")
print(train_dataset.class_to_idx)


print("\nTraining samples:")
print(len(train_dataset))


print("Validation samples:")
print(len(val_dataset))


# ============================================================
# 9. DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True

)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True

)


# ============================================================
# 10. LOAD SWIN TRANSFORMER
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED SWIN TRANSFORMER")
print("=" * 70)


swin_model = models.swin_t(
    weights=None
)


swin_model.head = nn.Linear(
    swin_model.head.in_features,
    NUM_CLASSES
)


checkpoint = torch.load(

    SWIN_MODEL_FILE,

    map_location=DEVICE,

    weights_only=False

)


swin_model.load_state_dict(
    checkpoint["model_state_dict"]
)


swin_model = swin_model.to(
    DEVICE
)


swin_model.eval()


print("\n✓ Best Swin Transformer loaded")


print(
    "Best epoch:",
    checkpoint.get(
        "epoch",
        "unknown"
    )
)


print(
    "Best validation F1:",
    checkpoint.get(
        "val_f1",
        "unknown"
    )
)


print(
    "Best validation AUC:",
    checkpoint.get(
        "val_auc",
        "unknown"
    )
)


# ============================================================
# 11. SWIN EMBEDDING EXTRACTION
# ============================================================

def extract_swin_embedding(
    images
):

    x = swin_model.features(
        images
    )

    x = swin_model.norm(
        x
    )

    x = x.permute(
        0,
        3,
        1,
        2
    )

    x = swin_model.avgpool(
        x
    )

    x = torch.flatten(
        x,
        1
    )

    return x


# ============================================================
# 12. EXTRACT TRAINING EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING SWIN EMBEDDINGS")
print("=" * 70)


def extract_embeddings(
    loader,
    dataset_name
):

    embeddings = []

    labels = []

    filenames = []


    total_batches = len(
        loader
    )


    for batch_number, (
        images,
        batch_labels
    ) in enumerate(
        loader,
        start=1
    ):

        images = images.to(

            DEVICE,

            non_blocking=True

        )


        with torch.no_grad():

            with torch.autocast(

                device_type="cuda",

                dtype=torch.float16

            ):

                batch_embeddings = (
                    extract_swin_embedding(
                        images
                    )
                )


        embeddings.append(

            batch_embeddings
            .float()
            .cpu()
            .numpy()

        )


        labels.extend(

            batch_labels
            .cpu()
            .numpy()
            .tolist()

        )


        start_index = (
            (batch_number - 1)
            * BATCH_SIZE
        )


        end_index = min(

            start_index
            + len(batch_labels),

            len(loader.dataset)

        )


        for index in range(
            start_index,
            end_index
        ):

            filename = os.path.basename(

                loader.dataset.samples[index][0]

            )

            filenames.append(
                filename
            )


        if (
            batch_number == 1
            or batch_number % 20 == 0
            or batch_number == total_batches
        ):

            print(

                f"{dataset_name}: "
                f"batch {batch_number}/"
                f"{total_batches}"

            )


    embeddings = np.concatenate(
        embeddings,
        axis=0
    )


    labels = np.array(
        labels
    )


    print(
        f"\n{dataset_name} embeddings shape:"
    )

    print(
        embeddings.shape
    )


    return (
        embeddings,
        labels,
        filenames
    )


(
    train_embeddings,
    train_labels,
    train_filenames
) = extract_embeddings(

    train_loader,

    "Train"

)


(
    val_embeddings,
    val_labels,
    val_filenames
) = extract_embeddings(

    val_loader,

    "Validation"

)


# ============================================================
# 13. LOAD THERMAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("LOADING SPARK-SELECTED THERMAL FEATURES")
print("=" * 70)


thermal_df = pd.read_csv(
    THERMAL_FEATURE_FILE
)


print("\nTotal thermal feature records:")
print(len(thermal_df))


# ============================================================
# 14. SELECT THE SAME PIPELINE USED IN STEP 7
# ============================================================

thermal_df = thermal_df[

    (
        thermal_df["noise_type"]
        == "gaussian"
    )

    &

    (
        thermal_df["denoising_method"]
        == "median"
    )

].copy()


print(
    "\nRecords after selecting "
    "Gaussian + Median:"
)

print(
    len(thermal_df)
)


# ============================================================
# 15. CREATE FILENAME-BASED LOOKUP
# ============================================================

thermal_df["filename"] = (
    thermal_df["filename"]
    .astype(str)
)


# If duplicate filenames exist across different
# split/class combinations, use split + class + filename.

thermal_lookup = {}


for _, row in thermal_df.iterrows():

    key = (

        str(row["split"]),

        str(row["class"]),

        str(row["filename"])

    )


    thermal_lookup[key] = row


# ============================================================
# 16. MATCH THERMAL FEATURES TO DATASET
# ============================================================

def get_thermal_features(
    dataset,
    split_name
):

    feature_rows = []

    matched_filenames = []

    missing = []


    for image_path, label in dataset.samples:

        filename = os.path.basename(
            image_path
        )


        class_name = dataset.classes[
            label
        ]


        key = (

            split_name,

            class_name,

            filename

        )


        if key not in thermal_lookup:

            missing.append(
                key
            )

            continue


        row = thermal_lookup[key]


        values = [

            float(
                row[feature]
            )

            for feature
            in THERMAL_FEATURES

        ]


        feature_rows.append(
            values
        )


        matched_filenames.append(
            filename
        )


    if len(missing) > 0:

        print(
            f"\nWARNING: "
            f"{len(missing)} thermal feature "
            f"records were not matched."
        )

        print(
            "First missing example:"
        )

        print(
            missing[0]
        )

    else:

        print(
            f"\n✓ All {len(dataset)} "
            f"{split_name} samples matched."
        )


    feature_array = np.array(
        feature_rows,
        dtype=np.float32
    )


    return (
        feature_array,
        matched_filenames
    )


(
    train_thermal,
    train_thermal_names
) = get_thermal_features(

    train_dataset,

    "train"

)


(
    val_thermal,
    val_thermal_names
) = get_thermal_features(

    val_dataset,

    "val"

)


# ============================================================
# 17. VERIFY MATCHING
# ============================================================

if len(train_thermal) != len(
    train_embeddings
):

    raise RuntimeError(

        "Training embedding/thermal "
        "feature count mismatch."

    )


if len(val_thermal) != len(
    val_embeddings
):

    raise RuntimeError(

        "Validation embedding/thermal "
        "feature count mismatch."

    )


print("\nTraining thermal feature shape:")
print(train_thermal.shape)


print("\nValidation thermal feature shape:")
print(val_thermal.shape)


# ============================================================
# 18. STANDARDIZE THERMAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZING THERMAL FEATURES")
print("=" * 70)


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


# ============================================================
# 19. STANDARDIZE SWIN EMBEDDINGS
# ============================================================

print("\nStandardizing Swin embeddings...")


embedding_scaler = StandardScaler()


train_embeddings_scaled = (
    embedding_scaler.fit_transform(
        train_embeddings
    )
)


val_embeddings_scaled = (
    embedding_scaler.transform(
        val_embeddings
    )
)


# ============================================================
# 20. COMBINE FEATURES
# ============================================================

train_node_features = np.concatenate(

    [

        train_embeddings_scaled,

        train_thermal_scaled

    ],

    axis=1

)


val_node_features = np.concatenate(

    [

        val_embeddings_scaled,

        val_thermal_scaled

    ],

    axis=1

)


print("\nTraining node feature shape:")
print(train_node_features.shape)


print("\nValidation node feature shape:")
print(val_node_features.shape)


print(
    "\nExpected feature dimension:"
)

print(
    "768 Swin + 7 thermal = 775"
)


# ============================================================
# 21. BUILD KNN GRAPH
# ============================================================

def build_knn_graph(
    features,
    k
):

    print(
        f"\nBuilding KNN graph "
        f"with k={k}..."
    )


    # Use thermal features to define
    # graph similarity.
    #
    # This makes the graph represent
    # relationships between samples
    # based on their thermal characteristics.

    neighbor_model = NearestNeighbors(

        n_neighbors=k + 1,

        metric="euclidean"

    )


    neighbor_model.fit(
        features
    )


    distances, indices = (
        neighbor_model.kneighbors(
            features
        )
    )


    n = len(
        features
    )


    adjacency = np.zeros(

        (n, n),

        dtype=np.float32

    )


    for i in range(n):

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


            # Gaussian similarity
            weight = np.exp(
                -(distance ** 2)
                / 2.0
            )


            adjacency[i, j] = weight

            adjacency[j, i] = weight


    # Self connections
    adjacency += np.eye(
        n,
        dtype=np.float32
    )


    # Normalize:
    #
    # A_hat = D^(-1/2)
    #         A
    #         D^(-1/2)

    degree = adjacency.sum(
        axis=1
    )


    degree_inv_sqrt = (
        1.0
        / np.sqrt(
            degree + 1e-8
        )
    )


    normalized_adjacency = (

        degree_inv_sqrt[:, None]

        * adjacency

        * degree_inv_sqrt[None, :]

    )


    edge_count = np.count_nonzero(
        adjacency
    )


    print(
        "Nodes:",
        n
    )


    print(
        "Graph connections:",
        edge_count
    )


    return normalized_adjacency


# ============================================================
# 22. CREATE TRAIN AND VALIDATION GRAPHS
# ============================================================

print("\n" + "=" * 70)
print("GRAPH CONSTRUCTION")
print("=" * 70)


train_adjacency = build_knn_graph(

    train_thermal_scaled,

    K_NEIGHBORS

)


val_adjacency = build_knn_graph(

    val_thermal_scaled,

    K_NEIGHBORS

)


# ============================================================
# 23. CONVERT TO GPU TENSORS
# ============================================================

train_x = torch.tensor(

    train_node_features,

    dtype=torch.float32,

    device=DEVICE

)


val_x = torch.tensor(

    val_node_features,

    dtype=torch.float32,

    device=DEVICE

)


train_adj = torch.tensor(

    train_adjacency,

    dtype=torch.float32,

    device=DEVICE

)


val_adj = torch.tensor(

    val_adjacency,

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


# ============================================================
# 24. GCN LAYER
# ============================================================

class GraphConvolution(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        output_dim
    ):

        super().__init__()


        self.linear = nn.Linear(

            input_dim,

            output_dim

        )


    def forward(
        self,
        x,
        adjacency
    ):

        aggregated = torch.matmul(

            adjacency,

            x

        )


        output = self.linear(
            aggregated
        )


        return output


# ============================================================
# 25. GCN MODEL
# ============================================================

class GCNClassifier(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        dropout
    ):

        super().__init__()


        self.gcn1 = GraphConvolution(

            input_dim,

            hidden_dim

        )


        self.gcn2 = GraphConvolution(

            hidden_dim,

            hidden_dim

        )


        self.classifier = nn.Linear(

            hidden_dim,

            num_classes

        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(
        self,
        x,
        adjacency
    ):

        x = self.gcn1(
            x,
            adjacency
        )


        x = torch.relu(
            x
        )


        x = self.dropout(
            x
        )


        x = self.gcn2(
            x,
            adjacency
        )


        x = torch.relu(
            x
        )


        x = self.dropout(
            x
        )


        logits = self.classifier(
            x
        )


        return logits


# ============================================================
# 26. CREATE GNN
# ============================================================

input_dimension = (
    train_node_features.shape[1]
)


gnn_model = GCNClassifier(

    input_dim=input_dimension,

    hidden_dim=GNN_HIDDEN_DIM,

    num_classes=NUM_CLASSES,

    dropout=GNN_DROPOUT

)


gnn_model = gnn_model.to(
    DEVICE
)


print("\n" + "=" * 70)
print("GNN MODEL")
print("=" * 70)


print("\nInput dimension:")
print(input_dimension)


print("\nHidden dimension:")
print(GNN_HIDDEN_DIM)


print("\nOutput classes:")
print(NUM_CLASSES)


print("\nGNN:")
print(gnn_model)


# ============================================================
# 27. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(

    gnn_model.parameters(),

    lr=GNN_LEARNING_RATE,

    weight_decay=GNN_WEIGHT_DECAY

)


# ============================================================
# 28. TRAIN GNN
# ============================================================

print("\n" + "=" * 70)
print("STARTING GNN TRAINING")
print("=" * 70)


best_f1 = -1.0

best_epoch = 0

best_state = None

history = []


training_start = time.time()


for epoch in range(
    1,
    GNN_EPOCHS + 1
):

    epoch_start = time.time()


    gnn_model.train()


    optimizer.zero_grad()


    train_logits = gnn_model(

        train_x,

        train_adj

    )


    train_loss = criterion(

        train_logits,

        train_y

    )


    train_loss.backward()


    optimizer.step()


    # --------------------------------------------------------
    # TRAIN METRICS
    # --------------------------------------------------------

    train_predictions = (

        torch.argmax(

            train_logits,

            dim=1

        )

    )


    train_accuracy = (

        accuracy_score(

            train_labels,

            train_predictions
            .detach()
            .cpu()
            .numpy()

        )

    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    gnn_model.eval()


    with torch.no_grad():

        val_logits = gnn_model(

            val_x,

            val_adj

        )


        val_loss = criterion(

            val_logits,

            val_y

        )


        val_probabilities = torch.softmax(

            val_logits,

            dim=1

        )[:, 1]


        val_predictions = torch.argmax(

            val_logits,

            dim=1

        )


    val_pred_numpy = (

        val_predictions
        .cpu()
        .numpy()

    )


    val_probability_numpy = (

        val_probabilities
        .cpu()
        .numpy()

    )


    val_metrics = {

        "accuracy":
            accuracy_score(
                val_labels,
                val_pred_numpy
            ),

        "precision":
            precision_score(
                val_labels,
                val_pred_numpy,
                zero_division=0
            ),

        "recall":
            recall_score(
                val_labels,
                val_pred_numpy,
                zero_division=0
            ),

        "f1":
            f1_score(
                val_labels,
                val_pred_numpy,
                zero_division=0
            ),

        "auc":
            roc_auc_score(
                val_labels,
                val_probability_numpy
            )

    }


    epoch_time = (
        time.time()
        - epoch_start
    )


    history.append({

        "epoch":
            epoch,

        "train_loss":
            train_loss.item(),

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss.item(),

        "val_accuracy":
            val_metrics["accuracy"],

        "val_precision":
            val_metrics["precision"],

        "val_recall":
            val_metrics["recall"],

        "val_f1":
            val_metrics["f1"],

        "val_auc":
            val_metrics["auc"],

        "epoch_time_seconds":
            epoch_time

    })


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    if (

        epoch == 1

        or epoch % 5 == 0

        or epoch == GNN_EPOCHS

    ):

        print("\n" + "-" * 70)

        print(
            f"Epoch {epoch}/{GNN_EPOCHS}"
        )

        print(
            f"Train Loss     : "
            f"{train_loss.item():.4f}"
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
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Precision  : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Val Recall     : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Val F1         : "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"Val ROC-AUC    : "
            f"{val_metrics['auc']:.4f}"
        )

        print(
            f"Epoch Time     : "
            f"{epoch_time:.2f} sec"
        )


    # --------------------------------------------------------
    # SAVE BEST GNN
    # --------------------------------------------------------

    if val_metrics["f1"] > best_f1:

        best_f1 = val_metrics["f1"]

        best_epoch = epoch


        best_state = {

            key: value.detach().cpu().clone()

            for key, value
            in gnn_model.state_dict().items()

        }


# ============================================================
# 29. RESTORE BEST GNN
# ============================================================

gnn_model.load_state_dict(
    best_state
)


gnn_model = gnn_model.to(
    DEVICE
)


# ============================================================
# 30. FINAL VALIDATION
# ============================================================

gnn_model.eval()


with torch.no_grad():

    final_logits = gnn_model(

        val_x,

        val_adj

    )


    final_probabilities = (

        torch.softmax(

            final_logits,

            dim=1

        )[:, 1]

        .cpu()
        .numpy()

    )


    final_predictions = (

        torch.argmax(

            final_logits,

            dim=1

        )

        .cpu()
        .numpy()

    )


final_accuracy = accuracy_score(

    val_labels,

    final_predictions

)


final_precision = precision_score(

    val_labels,

    final_predictions,

    zero_division=0

)


final_recall = recall_score(

    val_labels,

    final_predictions,

    zero_division=0

)


final_f1 = f1_score(

    val_labels,

    final_predictions,

    zero_division=0

)


final_auc = roc_auc_score(

    val_labels,

    final_probabilities

)


# ============================================================
# 31. SAVE GNN MODEL
# ============================================================

gnn_model_file = os.path.join(

    MODEL_DIR,

    "best_swin_gnn_thermal.pth"

)


torch.save({

    "model_state_dict":
        gnn_model.state_dict(),

    "input_dimension":
        input_dimension,

    "hidden_dimension":
        GNN_HIDDEN_DIM,

    "num_classes":
        NUM_CLASSES,

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

    "thermal_features":
        THERMAL_FEATURES,

    "k_neighbors":
        K_NEIGHBORS

}, gnn_model_file)


# ============================================================
# 32. PRINT FINAL RESULTS
# ============================================================

total_time = (
    time.time()
    - training_start
)


print("\n" + "=" * 70)
print("GNN TRAINING COMPLETED")
print("=" * 70)


print(
    f"\nBest Epoch : {best_epoch}"
)


print(
    f"Accuracy   : {final_accuracy:.4f}"
)


print(
    f"Precision  : {final_precision:.4f}"
)


print(
    f"Recall     : {final_recall:.4f}"
)


print(
    f"F1         : {final_f1:.4f}"
)


print(
    f"ROC-AUC    : {final_auc:.4f}"
)


print(
    f"Total Time : {total_time:.2f} sec"
)


# ============================================================
# 33. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    val_labels,

    final_predictions

)


print("\nConfusion Matrix:")

print(cm)


cm_file = os.path.join(

    METRICS_DIR,

    "swin_gnn_confusion_matrix.csv"

)


pd.DataFrame(

    cm,

    index=[
        "Actual_Control",
        "Actual_DM"
    ],

    columns=[
        "Predicted_Control",
        "Predicted_DM"
    ]

).to_csv(

    cm_file

)


# ============================================================
# 34. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    val_labels,

    final_predictions,

    target_names=[
        "Control",
        "DM"
    ],

    zero_division=0

)


print("\nClassification Report:")
print(report)


report_file = os.path.join(

    METRICS_DIR,

    "swin_gnn_classification_report.txt"

)


with open(

    report_file,

    "w",

    encoding="utf-8"

) as f:

    f.write(report)


# ============================================================
# 35. FINAL METRICS CSV
# ============================================================

metrics_file = os.path.join(

    METRICS_DIR,

    "swin_gnn_metrics.csv"

)


metrics_df = pd.DataFrame([{

    "model":
        "Swin Transformer + GCN",

    "device":
        "NVIDIA GeForce RTX 4060 Laptop GPU",

    "train_nodes":
        len(train_labels),

    "validation_nodes":
        len(val_labels),

    "swin_embedding_dimension":
        768,

    "thermal_feature_dimension":
        len(THERMAL_FEATURES),

    "combined_feature_dimension":
        input_dimension,

    "k_neighbors":
        K_NEIGHBORS,

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
        final_auc

}])


metrics_df.to_csv(

    metrics_file,

    index=False

)


# ============================================================
# 36. TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_file = os.path.join(

    METRICS_DIR,

    "swin_gnn_training_history.csv"

)


history_df.to_csv(

    history_file,

    index=False

)


# ============================================================
# 37. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(cm)


plt.title(
    "Swin Transformer + GCN Confusion Matrix"
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.xticks(

    [0, 1],

    ["Control", "DM"]

)


plt.yticks(

    [0, 1],

    ["Control", "DM"]

)


for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            str(cm[i, j]),

            ha="center",

            va="center"

        )


plt.colorbar()


plt.tight_layout()


confusion_plot = os.path.join(

    PLOTS_DIR,

    "swin_gnn_confusion_matrix.png"

)


plt.savefig(

    confusion_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 38. GNN LOSS CURVE
# ============================================================

plt.figure(
    figsize=(7, 5)
)


plt.plot(

    history_df["epoch"],

    history_df["train_loss"],

    marker="o",

    label="Train Loss"

)


plt.plot(

    history_df["epoch"],

    history_df["val_loss"],

    marker="o",

    label="Validation Loss"

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Swin + GNN Loss Curve"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


loss_plot = os.path.join(

    PLOTS_DIR,

    "swin_gnn_loss_curve.png"

)


plt.savefig(

    loss_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 39. GNN ACCURACY CURVE
# ============================================================

plt.figure(
    figsize=(7, 5)
)


plt.plot(

    history_df["epoch"],

    history_df["train_accuracy"],

    marker="o",

    label="Train Accuracy"

)


plt.plot(

    history_df["epoch"],

    history_df["val_accuracy"],

    marker="o",

    label="Validation Accuracy"

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Swin + GNN Accuracy Curve"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


accuracy_plot = os.path.join(

    PLOTS_DIR,

    "swin_gnn_accuracy_curve.png"

)


plt.savefig(

    accuracy_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 40. ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(

    val_labels,

    final_probabilities

)


plt.figure(
    figsize=(7, 5)
)


plt.plot(

    fpr,

    tpr,

    label=f"AUC = {final_auc:.4f}"

)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Random Classifier"

)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "Swin Transformer + GCN ROC Curve"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


roc_plot = os.path.join(

    PLOTS_DIR,

    "swin_gnn_roc_curve.png"

)


plt.savefig(

    roc_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 41. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 OUTPUT FILES")
print("=" * 70)


print("\nGNN MODEL:")
print(
    gnn_model_file
)


print("\nMETRICS:")
print(
    metrics_file
)


print("\nTRAINING HISTORY:")
print(
    history_file
)


print("\nCONFUSION MATRIX:")
print(
    cm_file
)


print("\nCLASSIFICATION REPORT:")
print(
    report_file
)


print("\nPLOTS:")

print(
    confusion_plot
)

print(
    loss_plot
)

print(
    accuracy_plot
)

print(
    roc_plot
)


print("\n" + "=" * 70)
print("✓ STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 70)