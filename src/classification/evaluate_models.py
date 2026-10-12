import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 10
# COMPLETE MODEL AND DENOISING EVALUATION
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
    METRICS_DIR,
    exist_ok=True
)


os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


print("=" * 70)
print("STEP 10 - COMPLETE MODEL EVALUATION")
print("=" * 70)


# ============================================================
# 2. INPUT FILES
# ============================================================

DENOSING_FILE = os.path.join(
    METRICS_DIR,
    "denoising_fast_results.csv"
)


SWIN_FILE = os.path.join(
    METRICS_DIR,
    "swin_transformer_metrics.csv"
)


GNN_FILE = os.path.join(
    METRICS_DIR,
    "swin_gnn_metrics.csv"
)


# ============================================================
# 3. CHECK FILES
# ============================================================

required_files = [

    DENOSING_FILE,

    SWIN_FILE,

    GNN_FILE

]


print("\nChecking result files...")


for file_path in required_files:

    print(
        os.path.basename(file_path),
        ":",
        "FOUND" if os.path.exists(file_path)
        else "NOT FOUND"
    )


missing_files = [

    file_path

    for file_path in required_files

    if not os.path.exists(file_path)

]


if missing_files:

    print("\nERROR: Required result files are missing.")

    for file_path in missing_files:

        print(file_path)

    raise SystemExit(1)


# ============================================================
# 4. LOAD RESULTS
# ============================================================

print("\nLoading denoising results...")

denoising_df = pd.read_csv(
    DENOSING_FILE
)


print(
    "Denoising experiments:",
    len(denoising_df)
)


print("\nLoading Swin results...")

swin_df = pd.read_csv(
    SWIN_FILE
)


print(
    "Swin result rows:",
    len(swin_df)
)


print("\nLoading GNN results...")

gnn_df = pd.read_csv(
    GNN_FILE
)


print(
    "GNN result rows:",
    len(gnn_df)
)


# ============================================================
# 5. DISPLAY COLUMN INFORMATION
# ============================================================

print("\nDenoising columns:")

print(
    list(denoising_df.columns)
)


print("\nSwin columns:")

print(
    list(swin_df.columns)
)


print("\nGNN columns:")

print(
    list(gnn_df.columns)
)


# ============================================================
# 6. DENOISING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DENOISING EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# Detect the important denoising columns
# ------------------------------------------------------------

print("\nCalculating average denoising performance...")


denoising_grouped = (

    denoising_df

    .groupby(
        [
            "noise_type",
            "denoising_method"
        ]
    )

    .agg({

        "psnr": "mean",

        "ssim": "mean",

        "anomaly_recall": "mean",

        "anomaly_iou": "mean",

        "intensity_preservation": "mean",

        "area_preservation": "mean",

        "anomaly_preservation_score": "mean"

    })

    .reset_index()

)


# Round values

denoising_grouped = (
    denoising_grouped.round(4)
)


print("\nDenoising performance:")

print(
    denoising_grouped.to_string(
        index=False
    )
)


# ============================================================
# 7. BEST DENOISER BY PSNR
# ============================================================

best_psnr = (

    denoising_grouped

    .sort_values(
        "psnr",
        ascending=False
    )

    .groupby(
        "noise_type"
    )

    .first()

    .reset_index()

)


best_psnr_file = os.path.join(

    METRICS_DIR,

    "best_denoiser_by_psnr.csv"

)


best_psnr.to_csv(

    best_psnr_file,

    index=False

)


print("\nBest denoiser by PSNR:")

print(
    best_psnr.to_string(
        index=False
    )
)


# ============================================================
# 8. BEST DENOISER BY ANOMALY PRESERVATION
# ============================================================

best_anomaly = (

    denoising_grouped

    .sort_values(
        "anomaly_preservation_score",
        ascending=False
    )

    .groupby(
        "noise_type"
    )

    .first()

    .reset_index()

)


best_anomaly_file = os.path.join(

    METRICS_DIR,

    "best_denoiser_by_anomaly_preservation.csv"

)


best_anomaly.to_csv(

    best_anomaly_file,

    index=False

)


print("\nBest denoiser by anomaly preservation:")

print(
    best_anomaly.to_string(
        index=False
    )
)


# ============================================================
# 9. OVERALL DENOISING RANKING
# ============================================================

overall_denoising = (

    denoising_grouped

    .groupby(
        "denoising_method"
    )

    .agg({

        "psnr": "mean",

        "ssim": "mean",

        "anomaly_recall": "mean",

        "anomaly_iou": "mean",

        "intensity_preservation": "mean",

        "area_preservation": "mean",

        "anomaly_preservation_score": "mean"

    })

    .reset_index()

)


overall_denoising = (

    overall_denoising

    .sort_values(

        "anomaly_preservation_score",

        ascending=False

    )

    .round(4)

)


overall_denoising_file = os.path.join(

    METRICS_DIR,

    "overall_denoising_comparison.csv"

)


overall_denoising.to_csv(

    overall_denoising_file,

    index=False

)


print("\nOverall denoising ranking:")

print(
    overall_denoising.to_string(
        index=False
    )
)


# ============================================================
# 10. CLASSIFICATION COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# Swin metrics
# ------------------------------------------------------------

swin_row = swin_df.iloc[0]


# ------------------------------------------------------------
# GNN metrics
# ------------------------------------------------------------

gnn_row = gnn_df.iloc[0]


classification_comparison = pd.DataFrame([

    {

        "model":
            "Swin Transformer",

        "accuracy":
            float(swin_row["accuracy"]),

        "precision":
            float(swin_row["precision"]),

        "recall":
            float(swin_row["recall"]),

        "f1":
            float(swin_row["f1"]),

        "roc_auc":
            float(swin_row["roc_auc"]),

        "validation_samples":
            int(swin_row["validation_images"])

    },

    {

        "model":
            "Swin Transformer + GCN",

        "accuracy":
            float(gnn_row["accuracy"]),

        "precision":
            float(gnn_row["precision"]),

        "recall":
            float(gnn_row["recall"]),

        "f1":
            float(gnn_row["f1"]),

        "roc_auc":
            float(gnn_row["roc_auc"]),

        "validation_samples":
            int(gnn_row["validation_nodes"])

    }

])


classification_comparison = (
    classification_comparison.round(4)
)


print("\nClassification comparison:")

print(
    classification_comparison.to_string(
        index=False
    )
)


classification_file = os.path.join(

    METRICS_DIR,

    "classification_model_comparison.csv"

)


classification_comparison.to_csv(

    classification_file,

    index=False

)


# ============================================================
# 11. CALCULATE MODEL DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE DIFFERENCE")
print("=" * 70)


swin_values = (
    classification_comparison.iloc[0]
)


gnn_values = (
    classification_comparison.iloc[1]
)


difference_df = pd.DataFrame([

    {

        "metric":
            "Accuracy",

        "swin":
            swin_values["accuracy"],

        "swin_gnn":
            gnn_values["accuracy"],

        "difference_swin_gnn":
            gnn_values["accuracy"]
            - swin_values["accuracy"]

    },

    {

        "metric":
            "Precision",

        "swin":
            swin_values["precision"],

        "swin_gnn":
            gnn_values["precision"],

        "difference_swin_gnn":
            gnn_values["precision"]
            - swin_values["precision"]

    },

    {

        "metric":
            "Recall",

        "swin":
            swin_values["recall"],

        "swin_gnn":
            gnn_values["recall"],

        "difference_swin_gnn":
            gnn_values["recall"]
            - swin_values["recall"]

    },

    {

        "metric":
            "F1",

        "swin":
            swin_values["f1"],

        "swin_gnn":
            gnn_values["f1"],

        "difference_swin_gnn":
            gnn_values["f1"]
            - swin_values["f1"]

    },

    {

        "metric":
            "ROC-AUC",

        "swin":
            swin_values["roc_auc"],

        "swin_gnn":
            gnn_values["roc_auc"],

        "difference_swin_gnn":
            gnn_values["roc_auc"]
            - swin_values["roc_auc"]

    }

])


difference_df = (
    difference_df.round(4)
)


difference_file = os.path.join(

    METRICS_DIR,

    "swin_vs_gnn_difference.csv"

)


difference_df.to_csv(

    difference_file,

    index=False

)


print(
    difference_df.to_string(
        index=False
    )
)


# ============================================================
# 12. BEST CLASSIFICATION MODEL
# ============================================================

best_model = (

    classification_comparison

    .sort_values(
        "f1",
        ascending=False
    )

    .iloc[0]

)


print("\nBest classification model by F1:")

print(
    best_model["model"]
)


print(
    "F1:",
    f"{best_model['f1']:.4f}"
)


# ============================================================
# 13. CLASSIFICATION METRICS BAR CHART
# ============================================================

print("\nCreating classification comparison chart...")


metrics = [

    "accuracy",

    "precision",

    "recall",

    "f1",

    "roc_auc"

]


x = np.arange(
    len(metrics)
)


width = 0.35


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    x - width / 2,

    classification_comparison.iloc[0][metrics],

    width,

    label="Swin Transformer"

)


plt.bar(

    x + width / 2,

    classification_comparison.iloc[1][metrics],

    width,

    label="Swin + GCN"

)


plt.xticks(

    x,

    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]

)


plt.ylim(
    0,
    1
)


plt.ylabel(
    "Score"
)


plt.title(
    "Classification Model Comparison"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


classification_plot = os.path.join(

    PLOTS_DIR,

    "classification_model_comparison.png"

)


plt.savefig(

    classification_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 14. DENOISING PSNR CHART
# ============================================================

print(
    "Creating PSNR comparison chart..."
)


pivot_psnr = denoising_grouped.pivot(

    index="noise_type",

    columns="denoising_method",

    values="psnr"

)


plt.figure(
    figsize=(10, 6)
)


pivot_psnr.plot(
    kind="bar"
)


plt.ylabel(
    "PSNR (dB)"
)


plt.xlabel(
    "Noise Type"
)


plt.title(
    "Denoising PSNR Comparison"
)


plt.xticks(
    rotation=0
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


psnr_plot = os.path.join(

    PLOTS_DIR,

    "denoising_psnr_comparison.png"

)


plt.savefig(

    psnr_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 15. DENOISING ANOMALY PRESERVATION CHART
# ============================================================

print(
    "Creating anomaly preservation chart..."
)


pivot_anomaly = denoising_grouped.pivot(

    index="noise_type",

    columns="denoising_method",

    values="anomaly_preservation_score"

)


plt.figure(
    figsize=(10, 6)
)


pivot_anomaly.plot(
    kind="bar"
)


plt.ylabel(
    "Anomaly Preservation Score"
)


plt.xlabel(
    "Noise Type"
)


plt.title(
    "Anomaly Preservation Comparison"
)


plt.xticks(
    rotation=0
)


plt.ylim(
    0,
    1
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


anomaly_plot = os.path.join(

    PLOTS_DIR,

    "denoising_anomaly_preservation_comparison.png"

)


plt.savefig(

    anomaly_plot,

    dpi=200,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 16. SUMMARY TABLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT EVALUATION SUMMARY")
print("=" * 70)


summary_df = pd.DataFrame([

    {

        "category":
            "Denoising",

        "best_method":
            "Median for Gaussian/Salt-Pepper; "
            "Bilateral for Speckle",

        "evaluation":
            "PSNR, SSIM, anomaly preservation"

    },

    {

        "category":
            "Classification",

        "best_method":
            "Swin Transformer",

        "evaluation":
            "Accuracy, Precision, Recall, F1, ROC-AUC"

    }

])


print(
    summary_df.to_string(
        index=False
    )
)


summary_file = os.path.join(

    METRICS_DIR,

    "step10_evaluation_summary.csv"

)


summary_df.to_csv(

    summary_file,

    index=False

)


# ============================================================
# 17. FINAL OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 OUTPUT FILES")
print("=" * 70)


print("\nDenoising comparison:")
print(
    overall_denoising_file
)


print("\nBest denoiser by PSNR:")
print(
    best_psnr_file
)


print("\nBest denoiser by anomaly preservation:")
print(
    best_anomaly_file
)


print("\nClassification comparison:")
print(
    classification_file
)


print("\nSwin vs GNN difference:")
print(
    difference_file
)


print("\nProject summary:")
print(
    summary_file
)


print("\nPlots:")

print(
    classification_plot
)

print(
    psnr_plot
)

print(
    anomaly_plot
)


print("\n" + "=" * 70)
print("✓ STEP 10 COMPLETED SUCCESSFULLY")
print("=" * 70)