import os
import time
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# STEP 11
# DENOISING -> CLASSIFICATION EXPERIMENT
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


STANDARDIZED_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "standardized"
)


DENOISED_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "denoised_fast"
)

NOISY_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "noisy"
)


DENOSING_RESULTS = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics",
    "denoising_fast_results.csv"
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

EPOCHS = 5

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

RANDOM_SEED = 42


# ============================================================
# 3. EXPERIMENT CONDITIONS
# ============================================================

CONDITIONS = [

    "clean",

    "gaussian_noisy",

    "gaussian_median",

    "gaussian_gaussian",

    "gaussian_bilateral"

]


CONDITION_LABELS = {

    "clean":
        "Clean",

    "gaussian_noisy":
        "Gaussian Noise",

    "gaussian_median":
        "Gaussian + Median",

    "gaussian_gaussian":
        "Gaussian + Gaussian",

    "gaussian_bilateral":
        "Gaussian + Bilateral"

}


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

print("=" * 70)
print("STEP 11 - DENOISING -> CLASSIFICATION EXPERIMENT")
print("=" * 70)


print("\nPyTorch:")
print(torch.__version__)


print("\nCUDA available:")
print(torch.cuda.is_available())


if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is required for Step 11."
    )


DEVICE = torch.device(
    "cuda"
)


print("\nCUDA version:")
print(torch.version.cuda)


print("\nGPU:")
print(torch.cuda.get_device_name(0))


print("\nTraining device:")
print(DEVICE)


# ============================================================
# 6. CHECK DATA
# ============================================================

if not os.path.exists(
    STANDARDIZED_ROOT
):

    raise FileNotFoundError(
        "Standardized image directory not found:\n"
        + STANDARDIZED_ROOT
    )


if not os.path.exists(
    DENOISED_ROOT
):

    raise FileNotFoundError(
        "Denoised image directory not found:\n"
        + DENOISED_ROOT
    )


if not os.path.exists(
    DENOSING_RESULTS
):

    raise FileNotFoundError(
        "Denoising results CSV not found:\n"
        + DENOSING_RESULTS
    )


# ============================================================
# 7. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
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
# 8. BUILD IMAGE LIST
# ============================================================

def get_standardized_samples(
    split
):

    samples = []


    split_root = os.path.join(

        STANDARDIZED_ROOT,

        split

    )


    for class_name in [
        "Control",
        "DM"
    ]:

        class_root = os.path.join(

            split_root,

            class_name

        )


        if not os.path.exists(
            class_root
        ):

            raise FileNotFoundError(
                class_root
            )


        label = (
            0
            if class_name == "Control"
            else 1
        )


        for filename in sorted(
            os.listdir(class_root)
        ):

            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):

                samples.append({

                    "filename":
                        filename,

                    "class":
                        class_name,

                    "label":
                        label,

                    "standardized_path":
                        os.path.join(
                            class_root,
                            filename
                        )

                })


    return samples


# ============================================================
# 9. BUILD CONDITION PATH
# ============================================================

def get_condition_path(
    sample,
    condition
):

    standardized_path = sample["standardized_path"]
    normalized = os.path.normpath(standardized_path)
    parts = normalized.split(os.sep)

    split = None
    for part in parts:
        if part.lower() in ["train", "val"]:
            split = part.lower()
            break

    if split is None:
        raise RuntimeError(f"Could not determine split from path: {standardized_path}")

    class_name = sample["class"]
    filename = sample["filename"]
    stem, ext = os.path.splitext(filename)

    if condition == "clean":
        return standardized_path

    if condition == "gaussian_noisy":
        noisy_filename = f"{stem}_gaussian{ext}"
        return os.path.join(
            NOISY_ROOT,
            split,
            class_name,
            noisy_filename
        )

    if condition == "gaussian_median":
        denoised_filename = f"{stem}_gaussian_median{ext}"
        return os.path.join(
            DENOISED_ROOT,
            split,
            class_name,
            "gaussian",
            "median",
            denoised_filename
        )

    if condition == "gaussian_gaussian":
        denoised_filename = f"{stem}_gaussian_gaussian{ext}"
        return os.path.join(
            DENOISED_ROOT,
            split,
            class_name,
            "gaussian",
            "gaussian",
            denoised_filename
        )

    if condition == "gaussian_bilateral":
        denoised_filename = f"{stem}_gaussian_bilateral{ext}"
        return os.path.join(
            DENOISED_ROOT,
            split,
            class_name,
            "gaussian",
            "bilateral",
            denoised_filename
        )

    raise ValueError(f"Unknown condition: {condition}")


# ============================================================
# 10. DATASET CLASS
# ============================================================

class ThermalConditionDataset(
    Dataset
):

    def __init__(
        self,
        samples,
        condition,
        transform
    ):

        self.samples = samples

        self.condition = condition

        self.transform = transform


        self.valid_samples = []


        for sample in samples:

            path = get_condition_path(

                sample,

                condition

            )


            if os.path.exists(path):

                self.valid_samples.append({

                    "path":
                        path,

                    "label":
                        sample["label"],

                    "filename":
                        sample["filename"],

                    "class":
                        sample["class"]

                })


        if len(self.valid_samples) == 0:
            example = get_condition_path(samples[0], condition) if len(samples) > 0 else "No sample available"
            raise RuntimeError(
                f"No images found for condition: {condition}\n"
                f"Example expected path:\n{example}"
            )


    def __len__(self):

        return len(
            self.valid_samples
        )


    def __getitem__(
        self,
        index
    ):

        item = self.valid_samples[
            index
        ]


        image = Image.open(
            item["path"]
        ).convert("RGB")


        if self.transform:

            image = self.transform(
                image
            )


        label = torch.tensor(

            item["label"],

            dtype=torch.long

        )


        return (

            image,

            label

        )


# ============================================================
# 11. CREATE SWIN MODEL
# ============================================================

def create_swin():

    model = models.swin_t(
        weights=None
    )


    model.head = nn.Linear(

        model.head.in_features,

        NUM_CLASSES

    )


    return model.to(
        DEVICE
    )


# ============================================================
# 12. TRAIN ONE CONDITION
# ============================================================

def train_condition(
    condition,
    train_samples,
    val_samples
):

    print("\n")
    print("=" * 70)
    print(
        "CONDITION:",
        CONDITION_LABELS[condition]
    )
    print("=" * 70)


    train_dataset = (
        ThermalConditionDataset(

            train_samples,

            condition,

            train_transform

        )
    )


    val_dataset = (
        ThermalConditionDataset(

            val_samples,

            condition,

            val_transform

        )
    )


    print(
        "\nTraining images:",
        len(train_dataset)
    )


    print(
        "Validation images:",
        len(val_dataset)
    )


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


    model = create_swin()


    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS

    )


    scaler = torch.amp.GradScaler(
        "cuda"
    )


    best_f1 = -1.0

    best_auc = 0.0

    best_epoch = 0

    best_state = None


    history = []


    start_time = time.time()


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()


        train_loss_total = 0.0

        train_predictions = []

        train_targets = []


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

                outputs = model(
                    images
                )


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


            train_loss_total += (

                loss.item()
                * images.size(0)

            )


            predictions = torch.argmax(

                outputs,

                dim=1

            )


            train_predictions.extend(

                predictions
                .detach()
                .cpu()
                .numpy()

            )


            train_targets.extend(

                labels
                .detach()
                .cpu()
                .numpy()

            )


        scheduler.step()


        train_loss = (

            train_loss_total
            / len(train_dataset)

        )


        train_accuracy = accuracy_score(

            train_targets,

            train_predictions

        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()


        val_loss_total = 0.0

        val_predictions = []

        val_targets = []

        val_probabilities = []


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

                    outputs = model(
                        images
                    )


                    loss = criterion(

                        outputs,

                        labels

                    )


                probabilities = torch.softmax(

                    outputs,

                    dim=1

                )[:, 1]


                predictions = torch.argmax(

                    outputs,

                    dim=1

                )


                val_loss_total += (

                    loss.item()
                    * images.size(0)

                )


                val_predictions.extend(

                    predictions
                    .cpu()
                    .numpy()

                )


                val_targets.extend(

                    labels
                    .cpu()
                    .numpy()

                )


                val_probabilities.extend(

                    probabilities
                    .float()
                    .cpu()
                    .numpy()

                )


        val_loss = (

            val_loss_total
            / len(val_dataset)

        )


        val_accuracy = accuracy_score(

            val_targets,

            val_predictions

        )


        val_precision = precision_score(

            val_targets,

            val_predictions,

            zero_division=0

        )


        val_recall = recall_score(

            val_targets,

            val_predictions,

            zero_division=0

        )


        val_f1 = f1_score(

            val_targets,

            val_predictions,

            zero_division=0

        )


        val_auc = roc_auc_score(

            val_targets,

            val_probabilities

        )


        epoch_time = (

            time.time()
            - epoch_start

        )


        history.append({

            "condition":
                condition,

            "condition_label":
                CONDITION_LABELS[
                    condition
                ],

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,

            "val_precision":
                val_precision,

            "val_recall":
                val_recall,

            "val_f1":
                val_f1,

            "val_auc":
                val_auc,

            "learning_rate":
                scheduler.get_last_lr()[0],

            "epoch_time_seconds":
                epoch_time

        })


        print("\n" + "-" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss     : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss       : "
            f"{val_loss:.4f}"
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
            f"Epoch Time     : "
            f"{epoch_time:.2f} sec"
        )


        # ----------------------------------------------------
        # SAVE BEST MODEL IN MEMORY
        # ----------------------------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_auc = val_auc

            best_epoch = epoch

            best_state = copy.deepcopy(

                model.state_dict()

            )


    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_state
    )


    model.eval()


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    final_predictions = []

    final_targets = []

    final_probabilities = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(

                DEVICE,

                non_blocking=True

            )


            with torch.autocast(

                device_type="cuda",

                dtype=torch.float16

            ):

                outputs = model(
                    images
                )


            probabilities = torch.softmax(

                outputs,

                dim=1

            )[:, 1]


            predictions = torch.argmax(

                outputs,

                dim=1

            )


            final_predictions.extend(

                predictions
                .cpu()
                .numpy()

            )


            final_targets.extend(

                labels
                .numpy()

            )


            final_probabilities.extend(

                probabilities
                .float()
                .cpu()
                .numpy()

            )


    final_accuracy = accuracy_score(

        final_targets,

        final_predictions

    )


    final_precision = precision_score(

        final_targets,

        final_predictions,

        zero_division=0

    )


    final_recall = recall_score(

        final_targets,

        final_predictions,

        zero_division=0

    )


    final_f1 = f1_score(

        final_targets,

        final_predictions,

        zero_division=0

    )


    final_auc = roc_auc_score(

        final_targets,

        final_probabilities

    )


    cm = confusion_matrix(

        final_targets,

        final_predictions

    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    safe_condition = condition.replace(
        "_",
        "-"
    )


    model_file = os.path.join(

        MODEL_DIR,

        f"step11_swin_{safe_condition}.pth"

    )


    torch.save({

        "model_state_dict":
            model.state_dict(),

        "condition":
            condition,

        "condition_label":
            CONDITION_LABELS[
                condition
            ],

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

    }, model_file)


    # ========================================================
    # PRINT FINAL CONDITION RESULT
    # ========================================================

    total_time = (

        time.time()
        - start_time

    )


    print("\n" + "=" * 70)

    print(
        "BEST RESULT:",
        CONDITION_LABELS[
            condition
        ]
    )

    print("=" * 70)


    print(
        "Best Epoch :",
        best_epoch
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


    print("\nConfusion Matrix:")

    print(cm)


    return {

        "condition":
            condition,

        "condition_label":
            CONDITION_LABELS[
                condition
            ],

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

        "train_images":
            len(train_dataset),

        "validation_images":
            len(val_dataset),

        "model_file":
            model_file,

        "confusion_matrix":
            cm,

        "targets":
            np.array(
                final_targets
            ),

        "predictions":
            np.array(
                final_predictions
            ),

        "probabilities":
            np.array(
                final_probabilities
            ),

        "history":
            history

    }


# ============================================================
# 13. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("BUILDING CONTROLLED DATASETS")
print("=" * 70)


train_samples = get_standardized_samples(
    "train"
)


val_samples = get_standardized_samples(
    "val"
)


print(
    "\nOriginal training images:",
    len(train_samples)
)


print(
    "Original validation images:",
    len(val_samples)
)


# ============================================================
# 14. VERIFY EXPECTED COUNTS
# ============================================================

if len(train_samples) != 1444:

    print(
        "\nWARNING:"
        " Expected 1444 training images."
    )


if len(val_samples) != 422:

    print(
        "\nWARNING:"
        " Expected 422 validation images."
    )


# ============================================================
# 15. RUN ALL CONDITIONS
# ============================================================

all_results = []

all_histories = []


for condition in CONDITIONS:

    result = train_condition(

        condition,

        train_samples,

        val_samples

    )


    all_results.append({

        "condition":
            result["condition"],

        "condition_label":
            result["condition_label"],

        "best_epoch":
            result["best_epoch"],

        "train_images":
            result["train_images"],

        "validation_images":
            result["validation_images"],

        "accuracy":
            result["accuracy"],

        "precision":
            result["precision"],

        "recall":
            result["recall"],

        "f1":
            result["f1"],

        "roc_auc":
            result["roc_auc"]

    })


    all_histories.extend(
        result["history"]
    )


# ============================================================
# 16. SAVE COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    all_results
)


comparison_df = (
    comparison_df
    .sort_values(
        "f1",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


comparison_df = (
    comparison_df.round(4)
)


comparison_file = os.path.join(

    METRICS_DIR,

    "step11_denoising_classification_comparison.csv"

)


comparison_df.to_csv(

    comparison_file,

    index=False

)


# ============================================================
# 17. SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    all_histories
)


history_file = os.path.join(

    METRICS_DIR,

    "step11_training_history.csv"

)


history_df.to_csv(

    history_file,

    index=False

)


# ============================================================
# 18. PRINT FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 FINAL CLASSIFICATION COMPARISON")
print("=" * 70)


print(

    comparison_df[

        [

            "condition_label",

            "accuracy",

            "precision",

            "recall",

            "f1",

            "roc_auc"

        ]

    ].to_string(

        index=False

    )

)


# ============================================================
# 19. BEST CONDITION
# ============================================================

best_result = (
    comparison_df
    .iloc[0]
)


print("\n" + "=" * 70)
print("BEST CONDITION")
print("=" * 70)


print(
    best_result["condition_label"]
)


print(
    f"Accuracy : "
    f"{best_result['accuracy']:.4f}"
)


print(
    f"Precision: "
    f"{best_result['precision']:.4f}"
)


print(
    f"Recall   : "
    f"{best_result['recall']:.4f}"
)


print(
    f"F1       : "
    f"{best_result['f1']:.4f}"
)


print(
    f"ROC-AUC  : "
    f"{best_result['roc_auc']:.4f}"
)


# ============================================================
# 20. CLASSIFICATION COMPARISON PLOT
# ============================================================

metrics = [

    "accuracy",

    "precision",

    "recall",

    "f1",

    "roc_auc"

]


plot_df = comparison_df.copy()


x = np.arange(
    len(plot_df)
)


width = 0.15


plt.figure(
    figsize=(12, 7)
)


for i, metric in enumerate(
    metrics
):

    plt.bar(

        x
        + (
            i - 2
        )
        * width,

        plot_df[metric],

        width,

        label=metric.upper()

    )


plt.xticks(

    x,

    plot_df["condition_label"],

    rotation=20,

    ha="right"

)


plt.ylim(
    0,
    1
)


plt.ylabel(
    "Score"
)


plt.title(
    "Denoising-to-Classification Performance"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


comparison_plot = os.path.join(

    PLOTS_DIR,

    "step11_denoising_classification_comparison.png"

)


plt.savefig(

    comparison_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 21. F1 COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.bar(

    plot_df["condition_label"],

    plot_df["f1"]

)


plt.ylabel(
    "F1 Score"
)


plt.xlabel(
    "Image Condition"
)


plt.title(
    "F1 Score Comparison Across Image Conditions"
)


plt.ylim(
    0,
    1
)


plt.xticks(
    rotation=20,
    ha="right"
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


f1_plot = os.path.join(

    PLOTS_DIR,

    "step11_f1_comparison.png"

)


plt.savefig(

    f1_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 22. ROC CURVES
# ============================================================

plt.figure(
    figsize=(8, 7)
)


for condition in CONDITIONS:

    condition_history = None

    # Find the corresponding result
    # by rerunning the stored model output
    # is unnecessary here, so ROC curves are
    # reconstructed only for the final result
    # if available in memory.


# We intentionally save the metric comparison
# separately. Full ROC curves can be generated
# from the saved models in a later explainability
# step.


plt.close()


# ============================================================
# 23. STEP 11 SUMMARY
# ============================================================

summary = pd.DataFrame([

    {

        "experiment":
            "Denoising -> Classification",

        "conditions_tested":
            len(CONDITIONS),

        "training_images_per_condition":
            1444,

        "validation_images_per_condition":
            422,

        "best_condition":
            best_result[
                "condition_label"
            ],

        "best_accuracy":
            best_result[
                "accuracy"
            ],

        "best_precision":
            best_result[
                "precision"
            ],

        "best_recall":
            best_result[
                "recall"
            ],

        "best_f1":
            best_result[
                "f1"
            ],

        "best_roc_auc":
            best_result[
                "roc_auc"
            ]

    }

])


summary_file = os.path.join(

    METRICS_DIR,

    "step11_summary.csv"

)


summary.to_csv(

    summary_file,

    index=False

)


# ============================================================
# 24. OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 OUTPUT FILES")
print("=" * 70)


print("\nComparison:")
print(
    comparison_file
)


print("\nTraining history:")
print(
    history_file
)


print("\nSummary:")
print(
    summary_file
)


print("\nPlots:")
print(
    comparison_plot
)

print(
    f1_plot
)


print("\nModels saved under:")
print(
    MODEL_DIR
)


print("\n" + "=" * 70)
print("✓ STEP 11 COMPLETED SUCCESSFULLY")
print("=" * 70)