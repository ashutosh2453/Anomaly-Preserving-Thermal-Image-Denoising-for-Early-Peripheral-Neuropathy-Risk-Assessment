
"""
BDE THERMAL
Presentation Results Generator

Uses existing project results; does NOT retrain models.

Output:
results/ppt/
    01_preprocessing/
    02_proposed_model/
    03_model_comparison/
    04_summary_tables/
"""

from pathlib import Path
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS = PROJECT_ROOT / "results"
METRICS = RESULTS / "metrics"
PLOTS = RESULTS / "plots"
FINAL_PLOTS = RESULTS / "final_plots"

OUTPUT = RESULTS / "ppt"

PREPROCESSING_OUT = OUTPUT / "01_preprocessing"
PROPOSED_OUT = OUTPUT / "02_proposed_model"
COMPARISON_OUT = OUTPUT / "03_model_comparison"
TABLES_OUT = OUTPUT / "04_summary_tables"

for folder in [
    PREPROCESSING_OUT,
    PROPOSED_OUT,
    COMPARISON_OUT,
    TABLES_OUT,
]:
    folder.mkdir(parents=True, exist_ok=True)

# Verified prediction-derived validation metrics.
PROPOSED = {
    "Model": "Improved Swin + GNN",
    "Accuracy": 92.1801,
    "Precision": 93.2806,
    "Recall": 93.6508,
    "F1-score": 93.4653,
    "ROC-AUC": 98.1092,
}

# Rows = actual class; columns = predicted class.
VERIFIED_CM = np.array([
    [153, 17],
    [16, 236],
])

# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def find_file(filename, folders=None):
    """Find an existing project file without assuming its location."""

    if folders is None:
        folders = [METRICS, FINAL_PLOTS, PLOTS, RESULTS]

    for folder in folders:
        candidate = folder / filename
        if candidate.exists():
            return candidate

    # Search within the results tree as a fallback.
    if RESULTS.exists():
        matches = list(RESULTS.rglob(filename))
        if matches:
            return matches[0]

    return None


def load_csv(filename, folders=None):
    path = find_file(filename, folders)

    if path is None:
        print(f"[NOT FOUND] {filename}")
        return None

    try:
        df = pd.read_csv(path)
        print(f"[CSV] {path} | shape={df.shape}")
        return df
    except Exception as exc:
        print(f"[ERROR] Could not read {path}: {exc}")
        return None


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace("-", "_").replace(" ", "_")
        for c in df.columns
    ]
    return df


def find_column(df, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def save_csv_and_table(df, basename, title, output_dir):
    """Save a compact CSV and PNG table."""

    if df is None or df.empty:
        print(f"[SKIP] Empty table: {basename}")
        return

    csv_path = output_dir / f"{basename}.csv"
    png_path = output_dir / f"{basename}.png"

    df.to_csv(csv_path, index=False)

    # Never attempt to render thousands of records.
    display = df.head(20).copy()

    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )

    nrows, ncols = display.shape

    fig, ax = plt.subplots(
        figsize=(
            min(13, max(7, ncols * 1.5)),
            min(7, max(2.5, 0.42 * (nrows + 2))),
        )
    )
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {png_path}")


def save_bar_chart(labels, values, filename, title, output_dir,
                   ylabel="Score (%)", percentage=True):
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)
    labels = [label for label, keep in zip(labels, valid) if keep]
    values = values[valid]

    if len(values) == 0:
        print(f"[SKIP] No numeric values for {filename}")
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    if percentage:
        ax.set_ylim(0, 108)

    for bar, value in zip(bars, values):
        label = f"{value:.2f}%" if percentage else f"{value:.4f}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {output_dir / filename}")


def copy_existing_plot(filename, destination_dir, new_name=None):
    """Reuse plots already generated by the project."""

    path = find_file(filename, [FINAL_PLOTS, PLOTS, RESULTS])

    if path is None:
        print(f"[PLOT NOT FOUND] {filename}")
        return False

    destination = destination_dir / (new_name or filename)
    shutil.copy2(path, destination)
    print(f"[COPIED] {path.name} -> {destination}")
    return True


# ============================================================
# 3. PREPROCESSING RESULTS
# ============================================================

def preprocessing_results():
    print("\n" + "=" * 60)
    print("SECTION 1: PREPROCESSING RESULTS")
    print("=" * 60)

    # Reuse the figures already produced by the project.
    existing_plots = [
        ("01_dataset_class_distribution.png",
         "01_dataset_class_distribution.png"),
        ("02_dataset_split_distribution.png",
         "02_dataset_split_distribution.png"),
        ("03_denoising_psnr.png",
         "03_denoising_psnr.png"),
        ("04_denoising_ssim.png",
         "04_denoising_ssim.png"),
        ("05_anomaly_preservation.png",
         "05_anomaly_preservation.png"),
    ]

    for original, renamed in existing_plots:
        copy_existing_plot(original, PREPROCESSING_OUT, renamed)

    # Use the actual denoising metrics file.
    df = load_csv(
        "final_denoising_results.csv",
        [FINAL_PLOTS, METRICS, RESULTS],
    )

    if df is None:
        df = load_csv("overall_denoising_comparison.csv")

    if df is None:
        df = load_csv("denoising_fast_results.csv")

    if df is not None and not df.empty:
        data = normalize_columns(df)

        # Keep compact numeric summaries for a large per-image CSV.
        method_col = find_column(
            data, ["method", "denoising_method", "filter", "condition"]
        )

        metric_aliases = {
            "PSNR": ["psnr", "psnr_db"],
            "SSIM": ["ssim"],
            "Anomaly Recall": ["anomaly_recall"],
            "Anomaly IoU": ["anomaly_iou", "anomaly_iou_score"],
            "Intensity Preservation": ["intensity_preservation"],
            "Area Preservation": ["area_preservation"],
            "Anomaly Preservation Score": [
                "anomaly_preservation_score", "aps"
            ],
        }

        selected = {
            label: find_column(data, aliases)
            for label, aliases in metric_aliases.items()
        }
        selected = {
            label: col for label, col in selected.items()
            if col is not None
        }

        if selected:
            for label, col in selected.items():
                data[col] = pd.to_numeric(data[col], errors="coerce")

            if method_col is not None:
                # Average image-level measurements by method if necessary.
                summary = (
                    data.groupby(method_col)[list(selected.values())]
                    .mean(numeric_only=True)
                    .reset_index()
                )
                summary = summary.rename(
                    columns={
                        method_col: "Method",
                        **{
                            col: label for label, col in selected.items()
                        },
                    }
                )
            else:
                summary = pd.DataFrame([{
                    label: data[col].mean()
                    for label, col in selected.items()
                }])

            save_csv_and_table(
                summary,
                "preprocessing_metrics",
                "Preprocessing and Denoising Summary",
                PREPROCESSING_OUT,
            )

            if method_col is not None:
                for label in selected:
                    if label not in summary.columns:
                        continue

                    save_bar_chart(
                        summary["Method"].astype(str).tolist(),
                        summary[label].tolist(),
                        f"denoising_{label.lower().replace(' ', '_')}.png",
                        f"Denoising Comparison: {label}",
                        PREPROCESSING_OUT,
                        ylabel="PSNR (dB)" if label == "PSNR" else "Score",
                        percentage=False,
                    )

    print("[DONE] Preprocessing section processed.")


# ============================================================
# 4. PROPOSED SWIN + GNN RESULTS
# ============================================================

def proposed_model_results():
    print("\n" + "=" * 60)
    print("SECTION 2: PROPOSED MODEL RESULTS")
    print("=" * 60)

    metrics = pd.DataFrame([{
        "Model": PROPOSED["Model"],
        "Accuracy (%)": PROPOSED["Accuracy"],
        "Precision (%)": PROPOSED["Precision"],
        "Recall (%)": PROPOSED["Recall"],
        "F1-score (%)": PROPOSED["F1-score"],
        "ROC-AUC (%)": PROPOSED["ROC-AUC"],
        "Validation Samples": 422,
    }])

    save_csv_and_table(
        metrics,
        "proposed_model_metrics",
        "Improved Swin Transformer + GNN Validation Results",
        PROPOSED_OUT,
    )

    save_bar_chart(
        ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        [
            PROPOSED["Accuracy"],
            PROPOSED["Precision"],
            PROPOSED["Recall"],
            PROPOSED["F1-score"],
            PROPOSED["ROC-AUC"],
        ],
        "proposed_model_metrics.png",
        "Proposed Model Validation Performance",
        PROPOSED_OUT,
    )

    # Confusion matrix from the verified predictions.
    cm_df = pd.DataFrame(
        VERIFIED_CM,
        index=["Actual Control", "Actual DM"],
        columns=["Predicted Control", "Predicted DM"],
    )
    cm_df.to_csv(PROPOSED_OUT / "proposed_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(VERIFIED_CM, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1], labels=["Control", "DM"])
    ax.set_yticks([0, 1], labels=["Control", "DM"])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Improved Swin + GNN Confusion Matrix")

    threshold = VERIFIED_CM.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(VERIFIED_CM[i, j]),
                ha="center", va="center",
                color="white" if VERIFIED_CM[i, j] > threshold else "black",
                fontsize=13,
            )

    fig.tight_layout()
    fig.savefig(
        PROPOSED_OUT / "proposed_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Reuse existing training and ROC plots when available.
    for filename in [
        "sw in_gnn_accuracy_curve.png".replace(" ", ""),
        "sw in_gnn_loss_curve.png".replace(" ", ""),
        "sw in_gnn_roc_curve.png".replace(" ", ""),
        "improved_swin_gnn_confusion_matrix.png",
    ]:
        copy_existing_plot(filename, PROPOSED_OUT)

    # Actual ROC curve should be reused only if it exists and
    # corresponds to the verified proposed model.
    print("[DONE] Proposed model section processed.")


# ============================================================
# 5. EXISTING CLASSIFIER COMPARISON
# ============================================================

def classifier_comparison():
    print("\n" + "=" * 60)
    print("SECTION 3: CLASSIFIER COMPARISON")
    print("=" * 60)

    # Prefer the dedicated comparison CSV.
    df = load_csv(
        "classification_model_comparison.csv",
        [METRICS, FINAL_PLOTS, RESULTS],
    )

    if df is None:
        df = load_csv(
            "final_results_table.csv",
            [FINAL_PLOTS, METRICS, RESULTS],
        )

    if df is None:
        print("[WARNING] No classifier comparison CSV found.")
        return

    data = normalize_columns(df)

    model_col = find_column(
        data, ["model", "condition_or_model", "condition", "classifier"]
    )

    metric_columns = {
        "Accuracy": find_column(data, ["accuracy", "accuracy_pct"]),
        "Precision": find_column(data, ["precision", "precision_pct"]),
        "Recall": find_column(data, ["recall", "recall_pct"]),
        "F1-score": find_column(data, ["f1", "f1_score", "f1_pct"]),
        "ROC-AUC": find_column(data, ["roc_auc", "auc"]),
    }

    metric_columns = {
        label: col for label, col in metric_columns.items()
        if col is not None
    }

    if model_col is None or not metric_columns:
        print("[WARNING] Could not identify model/metric columns.")
        print("CSV columns:", list(data.columns))
        return

    comparison = pd.DataFrame()
    comparison["Model"] = data[model_col].astype(str)

    for label, col in metric_columns.items():
        comparison[label] = pd.to_numeric(data[col], errors="coerce")

        # Convert fractional metrics to percentage only if needed.
        maximum = comparison[label].max(skipna=True)
        if pd.notna(maximum) and maximum <= 1.0:
            comparison[label] *= 100

    # Add the verified proposed model unless already present.
    if not comparison["Model"].str.contains(
        "Improved Swin", case=False, regex=False
    ).any():
        proposed_row = {"Model": PROPOSED["Model"]}
        for label in metric_columns:
            proposed_row[label] = PROPOSED[label]
        comparison = pd.concat(
            [comparison, pd.DataFrame([proposed_row])],
            ignore_index=True,
        )

    save_csv_and_table(
        comparison,
        "classifier_comparison",
        "Comparison with Existing Classifier Models",
        COMPARISON_OUT,
    )

    for metric in metric_columns:
        valid = comparison[metric].notna()
        save_bar_chart(
            comparison.loc[valid, "Model"].tolist(),
            comparison.loc[valid, metric].tolist(),
            f"comparison_{metric.lower().replace('-', '_')}.png",
            f"Classifier Comparison: {metric}",
            COMPARISON_OUT,
        )

    # Reuse the existing comparison graph when present.
    for filename in [
        "06_classification_models.png",
        "classification_model_comparison.png",
        "step11_denoising_classification_comparison.png",
        "sw in_gnn_original_vs_improved.png".replace(" ", ""),
    ]:
        copy_existing_plot(filename, COMPARISON_OUT)

    print("[DONE] Classifier comparison section processed.")


# ============================================================
# 6. COPY EXISTING FINAL PLOTS
# ============================================================

def copy_all_final_plots():
    """Copy existing final plots into presentation folders."""

    print("\n[EXTRA] Copying available final plots...")

    if not FINAL_PLOTS.exists():
        print("[INFO] results/final_plots folder not found.")
        return

    for path in FINAL_PLOTS.glob("*.png"):
        # Avoid copying a potentially huge collection of files.
        if path.name.startswith(("01_", "02_", "03_", "04_", "05_", "06_")):
            destination = PREPROCESSING_OUT / path.name
            if not destination.exists():
                shutil.copy2(path, destination)


# ============================================================
# 7. RUN
# ============================================================

def main():
    print("=" * 65)
    print("BDE THERMAL - PRESENTATION RESULTS GENERATOR")
    print("=" * 65)
    print("Project:", PROJECT_ROOT)
    print("Metrics:", METRICS)
    print("Output:", OUTPUT)

    preprocessing_results()
    proposed_model_results()
    classifier_comparison()
    copy_all_final_plots()

    print("\n" + "=" * 65)
    print("GENERATION FINISHED")
    print("Check:", OUTPUT)
    print("=" * 65)


if __name__ == "__main__":
    main()
