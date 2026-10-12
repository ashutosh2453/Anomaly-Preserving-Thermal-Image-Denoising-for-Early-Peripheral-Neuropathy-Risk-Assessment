import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 15 - FINAL RESULTS & VISUALIZATION
# CORRECTED FINAL VERSION
# ============================================================

print("=" * 70)
print("STEP 15 - FINAL RESULTS & VISUALIZATION")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

METRICS_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "final_plots"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# 2. LOAD FILES
# ============================================================

metadata_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "thermal_image_metadata.csv"
)

denoising_path = os.path.join(
    METRICS_ROOT,
    "overall_denoising_comparison.csv"
)

classification_path = os.path.join(
    METRICS_ROOT,
    "classification_model_comparison.csv"
)

step11_path = os.path.join(
    METRICS_ROOT,
    "step11_denoising_classification_comparison.csv"
)


metadata = pd.read_csv(
    metadata_path
)

denoising = pd.read_csv(
    denoising_path
)

classification = pd.read_csv(
    classification_path
)

step11 = pd.read_csv(
    step11_path
)


print("\nFiles loaded successfully.")

print(
    f"Dataset records      : {len(metadata)}"
)

print(
    f"Denoising records    : {len(denoising)}"
)

print(
    f"Classification rows  : {len(classification)}"
)

print(
    f"Step 11 rows         : {len(step11)}"
)


# ============================================================
# 3. DISPLAY COLUMNS
# ============================================================

print("\nDenoising columns:")

for column in denoising.columns:
    print(f"  - {column}")

print("\nClassification columns:")

for column in classification.columns:
    print(f"  - {column}")

print("\nStep 11 columns:")

for column in step11.columns:
    print(f"  - {column}")


# ============================================================
# 4. HELPER
# ============================================================

def save_plot(filename):

    output_path = os.path.join(
        OUTPUT_ROOT,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 5. DATASET CLASS DISTRIBUTION
# ============================================================

print(
    "\nCreating dataset class distribution..."
)

class_counts = (
    metadata["class"]
    .value_counts()
)

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    class_counts.index.astype(str),
    class_counts.values
)

plt.title(
    "Thermal Image Dataset - Class Distribution"
)

plt.xlabel(
    "Class"
)

plt.ylabel(
    "Number of Images"
)

for i, value in enumerate(
    class_counts.values
):

    plt.text(
        i,
        value,
        str(value),
        ha="center",
        va="bottom"
    )

save_plot(
    "01_dataset_class_distribution.png"
)


# ============================================================
# 6. TRAIN / VALIDATION DISTRIBUTION
# ============================================================

print(
    "\nCreating train/validation distribution..."
)

split_counts = (
    metadata["split"]
    .value_counts()
)

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    split_counts.index.astype(str),
    split_counts.values
)

plt.title(
    "Training and Validation Dataset Distribution"
)

plt.xlabel(
    "Dataset Split"
)

plt.ylabel(
    "Number of Images"
)

for i, value in enumerate(
    split_counts.values
):

    plt.text(
        i,
        value,
        str(value),
        ha="center",
        va="bottom"
    )

save_plot(
    "02_dataset_split_distribution.png"
)


# ============================================================
# 7. OVERALL DENOISING PSNR
# ============================================================

print(
    "\nCreating overall denoising PSNR comparison..."
)

methods = (
    denoising["denoising_method"]
    .astype(str)
)

psnr_values = (
    denoising["psnr"]
    .astype(float)
)

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    methods,
    psnr_values
)

plt.title(
    "Overall Denoising Performance - PSNR"
)

plt.xlabel(
    "Denoising Method"
)

plt.ylabel(
    "Average PSNR (dB)"
)

for i, value in enumerate(
    psnr_values
):

    plt.text(
        i,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

save_plot(
    "03_denoising_psnr.png"
)


# ============================================================
# 8. OVERALL DENOISING SSIM
# ============================================================

print(
    "\nCreating overall denoising SSIM comparison..."
)

ssim_values = (
    denoising["ssim"]
    .astype(float)
)

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    methods,
    ssim_values
)

plt.title(
    "Overall Denoising Performance - SSIM"
)

plt.xlabel(
    "Denoising Method"
)

plt.ylabel(
    "Average SSIM"
)

for i, value in enumerate(
    ssim_values
):

    plt.text(
        i,
        value,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

save_plot(
    "04_denoising_ssim.png"
)


# ============================================================
# 9. ANOMALY PRESERVATION
# ============================================================

print(
    "\nCreating anomaly preservation comparison..."
)

anomaly_values = (
    denoising[
        "anomaly_preservation_score"
    ]
    .astype(float)
)

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    methods,
    anomaly_values
)

plt.title(
    "Overall Anomaly Preservation Comparison"
)

plt.xlabel(
    "Denoising Method"
)

plt.ylabel(
    "Anomaly Preservation Score"
)

for i, value in enumerate(
    anomaly_values
):

    plt.text(
        i,
        value,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

save_plot(
    "05_anomaly_preservation.png"
)


# ============================================================
# 10. CLASSIFICATION MODEL COMPARISON
# ============================================================

print(
    "\nCreating classification model comparison..."
)

classification_metrics = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC")
]

model_names = (
    classification["model"]
    .astype(str)
    .tolist()
)

x = np.arange(
    len(model_names)
)

width = 0.15

plt.figure(
    figsize=(12, 6)
)

for i, (
    column,
    label
) in enumerate(
    classification_metrics
):

    values = (
        classification[column]
        .astype(float)
        .values
    )

    plt.bar(
        x + (
            i - 2
        ) * width,
        values,
        width,
        label=label
    )

plt.xticks(
    x,
    model_names
)

plt.ylim(
    0,
    1.05
)

plt.title(
    "Classification Model Performance"
)

plt.ylabel(
    "Score"
)

plt.legend()

save_plot(
    "06_classification_models.png"
)


# ============================================================
# 11. STEP 11 ACCURACY
# ============================================================

print(
    "\nCreating Step 11 accuracy comparison..."
)

conditions = (
    step11["condition_label"]
    .astype(str)
)

accuracy_values = (
    step11["accuracy"]
    .astype(float)
)

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    conditions,
    accuracy_values
)

plt.title(
    "Step 11 - Accuracy by Denoising Condition"
)

plt.xlabel(
    "Condition"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=20,
    ha="right"
)

for i, value in enumerate(
    accuracy_values
):

    plt.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

save_plot(
    "07_step11_accuracy.png"
)


# ============================================================
# 12. STEP 11 F1
# ============================================================

print(
    "\nCreating Step 11 F1 comparison..."
)

f1_values = (
    step11["f1"]
    .astype(float)
)

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    conditions,
    f1_values
)

plt.title(
    "Step 11 - F1 Score by Denoising Condition"
)

plt.xlabel(
    "Condition"
)

plt.ylabel(
    "F1 Score"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=20,
    ha="right"
)

for i, value in enumerate(
    f1_values
):

    plt.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

save_plot(
    "08_step11_f1.png"
)


# ============================================================
# 13. STEP 11 ROC-AUC
# ============================================================

print(
    "\nCreating Step 11 ROC-AUC comparison..."
)

auc_values = (
    step11["roc_auc"]
    .astype(float)
)

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    conditions,
    auc_values
)

plt.title(
    "Step 11 - ROC-AUC by Denoising Condition"
)

plt.xlabel(
    "Condition"
)

plt.ylabel(
    "ROC-AUC"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=20,
    ha="right"
)

for i, value in enumerate(
    auc_values
):

    plt.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

save_plot(
    "09_step11_roc_auc.png"
)


# ============================================================
# 14. BEST STEP 11 CONFUSION MATRIX
# ============================================================

print(
    "\nCreating best-condition confusion matrix..."
)

best_row = step11.loc[
    step11["f1"].idxmax()
]

best_condition = (
    str(best_row["condition_label"])
)

print(
    f"Best Step 11 condition: "
    f"{best_condition}"
)


# Confirmed from Step 11:
# Gaussian + Bilateral
#
# [[145, 25],
#  [ 46,206]]

cm = np.array([
    [145, 25],
    [46, 206]
])


plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm
)

plt.title(
    f"Confusion Matrix - {best_condition}"
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

save_plot(
    "10_best_confusion_matrix.png"
)


# ============================================================
# 15. FINAL RESULTS TABLE
# ============================================================

print(
    "\nCreating final results table..."
)

final_rows = []


# Classification models
for _, row in classification.iterrows():

    final_rows.append({

        "experiment":
            "Classification Model",

        "condition_or_model":
            row["model"],

        "accuracy":
            row["accuracy"],

        "precision":
            row["precision"],

        "recall":
            row["recall"],

        "f1":
            row["f1"],

        "roc_auc":
            row["roc_auc"]

    })


# Step 11
for _, row in step11.iterrows():

    final_rows.append({

        "experiment":
            "Denoising -> Classification",

        "condition_or_model":
            row["condition_label"],

        "accuracy":
            row["accuracy"],

        "precision":
            row["precision"],

        "recall":
            row["recall"],

        "f1":
            row["f1"],

        "roc_auc":
            row["roc_auc"]

    })


final_results = pd.DataFrame(
    final_rows
)

final_results_path = os.path.join(
    OUTPUT_ROOT,
    "final_results_table.csv"
)

final_results.to_csv(
    final_results_path,
    index=False
)

print(
    f"Saved: {final_results_path}"
)


# ============================================================
# 16. DENOISING SUMMARY TABLE
# ============================================================

denoising_summary = denoising[
    [
        "denoising_method",
        "psnr",
        "ssim",
        "anomaly_recall",
        "anomaly_iou",
        "intensity_preservation",
        "area_preservation",
        "anomaly_preservation_score"
    ]
].copy()

denoising_summary_path = os.path.join(
    OUTPUT_ROOT,
    "final_denoising_results.csv"
)

denoising_summary.to_csv(
    denoising_summary_path,
    index=False
)

print(
    f"Saved: {denoising_summary_path}"
)


# ============================================================
# 17. STEP 11 FINAL TABLE
# ============================================================

step11_final_path = os.path.join(
    OUTPUT_ROOT,
    "final_step11_results.csv"
)

step11.to_csv(
    step11_final_path,
    index=False
)

print(
    f"Saved: {step11_final_path}"
)


# ============================================================
# 18. OUTPUT LIST
# ============================================================

print("\n" + "=" * 70)
print("STEP 15 OUTPUT")
print("=" * 70)

for filename in sorted(
    os.listdir(OUTPUT_ROOT)
):

    print(
        os.path.join(
            OUTPUT_ROOT,
            filename
        )
    )


print("\n" + "=" * 70)
print("✓ STEP 15 COMPLETED")
print("=" * 70)