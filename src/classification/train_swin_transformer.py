import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

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
# STEP 8
# SWIN TRANSFORMER CLASSIFICATION - NVIDIA CUDA VERSION
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "classification"
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


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

# RTX 4060 Laptop GPU ~8 GB VRAM
BATCH_SIZE = 16

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

NUM_CLASSES = 2

NUM_WORKERS = 0

RANDOM_SEED = 42

USE_PRETRAINED = True


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 4. CUDA CONFIGURATION
# ============================================================

if not torch.cuda.is_available():

    print("=" * 70)
    print("ERROR: CUDA GPU IS NOT AVAILABLE")
    print("=" * 70)

    print("\nPyTorch version:")
    print(torch.__version__)

    print("\nCUDA available:")
    print(torch.cuda.is_available())

    print("\nPlease verify your CUDA installation.")

    raise SystemExit(1)


DEVICE = torch.device("cuda")


# Enable optimized CUDA behavior
torch.backends.cudnn.benchmark = True


# ============================================================
# 5. GPU INFORMATION
# ============================================================

print("=" * 70)
print("STEP 8 - SWIN TRANSFORMER CLASSIFICATION")
print("=" * 70)

print("\nPyTorch version:")
print(torch.__version__)

print("\nCUDA available:")
print(torch.cuda.is_available())

print("\nPyTorch CUDA version:")
print(torch.version.cuda)

print("\nGPU:")
print(torch.cuda.get_device_name(0))


gpu_properties = torch.cuda.get_device_properties(0)

gpu_memory_gb = (
    gpu_properties.total_memory
    / (1024 ** 3)
)

print("\nGPU VRAM:")
print(f"{gpu_memory_gb:.2f} GB")

print("\nCUDA device:")
print(torch.cuda.current_device())

print("\nTraining device:")
print(DEVICE)


# ============================================================
# 6. DATASET PATHS
# ============================================================

TRAIN_DIR = os.path.join(
    DATA_ROOT,
    "train"
)

VAL_DIR = os.path.join(
    DATA_ROOT,
    "val"
)


if not os.path.exists(TRAIN_DIR):

    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )


if not os.path.exists(VAL_DIR):

    raise FileNotFoundError(
        f"Validation directory not found:\n{VAL_DIR}"
    )


# ============================================================
# 7. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5
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


val_transform = transforms.Compose([

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
# 8. LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)


print("Loading validation dataset...")

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


print("\nClasses:")
print(train_dataset.classes)


print("\nClass mapping:")
print(train_dataset.class_to_idx)


print("\nTraining images:")
print(len(train_dataset))


print("Validation images:")
print(len(val_dataset))


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

train_class_counts = {
    class_name: 0
    for class_name in train_dataset.classes
}

val_class_counts = {
    class_name: 0
    for class_name in val_dataset.classes
}


for _, label in train_dataset.samples:

    class_name = train_dataset.classes[label]

    train_class_counts[class_name] += 1


for _, label in val_dataset.samples:

    class_name = val_dataset.classes[label]

    val_class_counts[class_name] += 1


print("\nTraining class distribution:")

for class_name, count in train_class_counts.items():

    print(
        f"  {class_name}: {count}"
    )


print("\nValidation class distribution:")

for class_name, count in val_class_counts.items():

    print(
        f"  {class_name}: {count}"
    )


# ============================================================
# 10. DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

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


print("\nBatch size:")
print(BATCH_SIZE)

print("\nTraining batches:")
print(len(train_loader))

print("Validation batches:")
print(len(val_loader))


# ============================================================
# 11. LOAD SWIN TRANSFORMER
# ============================================================

print("\n" + "=" * 70)
print("CREATING SWIN TRANSFORMER")
print("=" * 70)


model = None


if USE_PRETRAINED:

    print("\nAttempting to load pretrained Swin-T weights...")

    try:

        weights = models.Swin_T_Weights.DEFAULT

        model = models.swin_t(
            weights=weights
        )

        print("✓ Pretrained Swin-T loaded")

    except Exception as error:

        print("\nPretrained weights could not be loaded.")

        print("Reason:")
        print(error)

        print("\nCreating Swin-T without pretrained weights.")

        model = models.swin_t(
            weights=None
        )

else:

    print("\nCreating Swin-T without pretrained weights.")

    model = models.swin_t(
        weights=None
    )


# ============================================================
# 12. REPLACE CLASSIFICATION HEAD
# ============================================================

input_features = model.head.in_features


model.head = nn.Linear(
    input_features,
    NUM_CLASSES
)


model = model.to(DEVICE)


print("\nModel:")
print("Swin Transformer Tiny")

print("\nOriginal classifier input:")
print(input_features)

print("\nNew classifier:")
print(
    f"{input_features} -> {NUM_CLASSES}"
)


# ============================================================
# 13. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 14. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ============================================================
# 15. LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=NUM_EPOCHS

)


# ============================================================
# 16. MIXED PRECISION
# ============================================================

use_amp = True

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


print("\nMixed Precision:")
print("Enabled")


# ============================================================
# 17. METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_true,
            y_probability
        )

    except Exception:

        auc = 0.0


    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "auc": auc

    }


# ============================================================
# 18. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    all_labels = []

    all_predictions = []


    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()


        total_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


    epoch_loss = (
        total_loss
        / len(train_dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return (
        epoch_loss,
        accuracy
    )


# ============================================================
# 19. VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    all_labels = []

    all_predictions = []

    all_probabilities = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            total_loss += (
                loss.item()
                * images.size(0)
            )


            all_labels.extend(
                labels.cpu().numpy()
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )


    epoch_loss = (
        total_loss
        / len(val_dataset)
    )


    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities
    )


    return (
        epoch_loss,
        metrics,
        all_labels,
        all_predictions,
        all_probabilities
    )


# ============================================================
# 20. TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING GPU TRAINING")
print("=" * 70)


history = []

best_f1 = -1.0

best_auc = -1.0

best_epoch = 0

best_results = None


training_start = time.time()


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    epoch_start = time.time()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch()
    )


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    (
        val_loss,
        val_metrics,
        y_true,
        y_pred,
        y_probability
    ) = validate()


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step()


    epoch_time = (
        time.time()
        - epoch_start
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    history.append({

        "epoch": epoch,

        "learning_rate":
            optimizer.param_groups[0]["lr"],

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

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
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss     : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy : {train_accuracy:.4f}"
    )

    print(
        f"Val Loss       : {val_loss:.4f}"
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
        f"Learning Rate  : "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )

    print(
        f"Epoch Time     : "
        f"{epoch_time:.2f} sec"
    )


    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )


    print(
        f"GPU Memory     : "
        f"{allocated:.2f} GB allocated / "
        f"{reserved:.2f} GB reserved"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_metrics["f1"] > best_f1:

        best_f1 = val_metrics["f1"]

        best_auc = val_metrics["auc"]

        best_epoch = epoch

        best_results = {

            "y_true": y_true,

            "y_pred": y_pred,

            "y_probability":
                y_probability

        }


        best_model_path = os.path.join(

            MODEL_DIR,

            "best_swin_transformer_thermal.pth"

        )


        torch.save({

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

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

            "class_to_idx":
                train_dataset.class_to_idx

        }, best_model_path)


        print(
            "\n✓ BEST MODEL SAVED"
        )

        print(
            best_model_path
        )


# ============================================================
# 21. TRAINING COMPLETE
# ============================================================

total_training_time = (
    time.time()
    - training_start
)


print("\n" + "=" * 70)
print("GPU TRAINING COMPLETED")
print("=" * 70)


print(
    f"\nBest Epoch : {best_epoch}"
)

print(
    f"Best F1    : {best_f1:.4f}"
)

print(
    f"Best AUC   : {best_auc:.4f}"
)

print(
    f"Total Time : "
    f"{total_training_time:.2f} seconds"
)


# ============================================================
# 22. TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_file = os.path.join(

    METRICS_DIR,

    "swin_transformer_training_history.csv"

)


history_df.to_csv(
    history_file,
    index=False
)


print(
    "\nTraining history saved:"
)

print(history_file)


# ============================================================
# 23. BEST MODEL METRICS
# ============================================================

best_y_true = best_results["y_true"]

best_y_pred = best_results["y_pred"]

best_y_probability = (
    best_results["y_probability"]
)


best_metrics = calculate_metrics(

    best_y_true,

    best_y_pred,

    best_y_probability

)


metrics_file = os.path.join(

    METRICS_DIR,

    "swin_transformer_metrics.csv"

)


metrics_df = pd.DataFrame([{

    "model":
        "Swin Transformer Tiny",

    "device":
        "NVIDIA GeForce RTX 4060 Laptop GPU",

    "epochs":
        NUM_EPOCHS,

    "best_epoch":
        best_epoch,

    "train_images":
        len(train_dataset),

    "validation_images":
        len(val_dataset),

    "batch_size":
        BATCH_SIZE,

    "accuracy":
        best_metrics["accuracy"],

    "precision":
        best_metrics["precision"],

    "recall":
        best_metrics["recall"],

    "f1":
        best_metrics["f1"],

    "roc_auc":
        best_metrics["auc"]

}])


metrics_df.to_csv(

    metrics_file,

    index=False

)


print(
    "\nFinal metrics saved:"
)

print(metrics_file)


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    best_y_true,

    best_y_pred

)


print("\nConfusion Matrix:")

print(cm)


cm_file = os.path.join(

    METRICS_DIR,

    "swin_transformer_confusion_matrix.csv"

)


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


cm_df.to_csv(
    cm_file
)


# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    best_y_true,

    best_y_pred,

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

    "swin_transformer_classification_report.txt"

)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# 26. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(cm)


plt.title(
    "Swin Transformer Confusion Matrix"
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

    "swin_transformer_confusion_matrix.png"

)


plt.savefig(

    confusion_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 27. LOSS CURVE
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
    "Swin Transformer Loss Curve"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


loss_plot = os.path.join(

    PLOTS_DIR,

    "swin_transformer_loss_curve.png"

)


plt.savefig(

    loss_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 28. ACCURACY CURVE
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
    "Swin Transformer Accuracy Curve"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


accuracy_plot = os.path.join(

    PLOTS_DIR,

    "swin_transformer_accuracy_curve.png"

)


plt.savefig(

    accuracy_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 29. F1 CURVE
# ============================================================

plt.figure(
    figsize=(7, 5)
)


plt.plot(

    history_df["epoch"],

    history_df["val_f1"],

    marker="o",

    label="Validation F1"

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "F1 Score"
)


plt.title(
    "Swin Transformer Validation F1"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


f1_plot = os.path.join(

    PLOTS_DIR,

    "swin_transformer_f1_curve.png"

)


plt.savefig(

    f1_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 30. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(

    best_y_true,

    best_y_probability

)


plt.figure(
    figsize=(7, 5)
)


plt.plot(

    fpr,

    tpr,

    label=f"AUC = {best_metrics['auc']:.4f}"

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
    "Swin Transformer ROC Curve"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


roc_plot = os.path.join(

    PLOTS_DIR,

    "swin_transformer_roc_curve.png"

)


plt.savefig(

    roc_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 31. FINAL GPU MEMORY
# ============================================================

final_allocated = (

    torch.cuda.memory_allocated()

    / (1024 ** 3)

)


final_reserved = (

    torch.cuda.memory_reserved()

    / (1024 ** 3)

)


# ============================================================
# 32. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 OUTPUT FILES")
print("=" * 70)


print("\nBEST MODEL:")
print(
    best_model_path
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
    f1_plot
)

print(
    roc_plot
)


print("\nFINAL GPU MEMORY:")

print(
    f"Allocated: {final_allocated:.2f} GB"
)

print(
    f"Reserved : {final_reserved:.2f} GB"
)


print("\n" + "=" * 70)
print("✓ STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 70)