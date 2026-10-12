import os
import csv
import pandas as pd


# ============================================================
# STEP 14 - POWER BI DATA PREPARATION
# ============================================================

print("=" * 70)
print("STEP 14 - POWER BI DATA PREPARATION")
print("=" * 70)


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

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data"
)

RESULTS_ROOT = os.path.join(
    PROJECT_ROOT,
    "results"
)

METRICS_ROOT = os.path.join(
    RESULTS_ROOT,
    "metrics"
)

FEATURES_ROOT = os.path.join(
    DATA_ROOT,
    "features"
)

POWERBI_ROOT = os.path.join(
    RESULTS_ROOT,
    "powerbi"
)

os.makedirs(
    POWERBI_ROOT,
    exist_ok=True
)


# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def load_csv(path, name):

    print(
        f"\nLoading {name}:"
    )

    print(path)

    if not os.path.exists(path):

        print(
            f"WARNING: File not found: {path}"
        )

        return None

    df = pd.read_csv(path)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


def save_csv(df, filename):

    output_path = os.path.join(
        POWERBI_ROOT,
        filename
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Saved: {output_path}"
    )

    return output_path


# ============================================================
# 3. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET SUMMARY")
print("=" * 70)


metadata_path = os.path.join(
    DATA_ROOT,
    "processed",
    "thermal_image_metadata.csv"
)

metadata = load_csv(
    metadata_path,
    "thermal image metadata"
)


dataset_rows = []


if metadata is not None:

    total_images = len(metadata)

    dataset_rows.append({
        "category": "Dataset",
        "metric": "Total Images",
        "value": total_images
    })

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    if "class" in metadata.columns:

        class_counts = (
            metadata["class"]
            .value_counts()
        )

        for class_name, count in class_counts.items():

            dataset_rows.append({
                "category": "Class",
                "metric": str(class_name),
                "value": int(count)
            })

    # --------------------------------------------------------
    # Split distribution
    # --------------------------------------------------------

    if "split" in metadata.columns:

        split_counts = (
            metadata["split"]
            .value_counts()
        )

        for split_name, count in split_counts.items():

            dataset_rows.append({
                "category": "Split",
                "metric": str(split_name),
                "value": int(count)
            })

    # --------------------------------------------------------
    # Split + class
    # --------------------------------------------------------

    if (
        "split" in metadata.columns
        and "class" in metadata.columns
    ):

        split_class = (
            metadata
            .groupby(
                ["split", "class"]
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        for _, row in split_class.iterrows():

            dataset_rows.append({
                "category": "Split-Class",
                "metric": (
                    f"{row['split']}-"
                    f"{row['class']}"
                ),
                "value": int(row["count"])
            })


dataset_summary = pd.DataFrame(
    dataset_rows
)

save_csv(
    dataset_summary,
    "dataset_summary.csv"
)


# ============================================================
# 4. DENOISING RESULTS
# ============================================================

print("\n" + "=" * 70)
print("2. DENOISING RESULTS")
print("=" * 70)


denoising_path = os.path.join(
    METRICS_ROOT,
    "overall_denoising_comparison.csv"
)

denoising = load_csv(
    denoising_path,
    "overall denoising comparison"
)


if denoising is not None:

    # Add a project category column
    denoising.insert(
        0,
        "analysis_type",
        "Overall Denoising"
    )

    save_csv(
        denoising,
        "denoising_results.csv"
    )

else:

    print(
        "Denoising results could not be prepared."
    )


# ============================================================
# 5. CLASSIFICATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("3. CLASSIFICATION RESULTS")
print("=" * 70)


classification_path = os.path.join(
    METRICS_ROOT,
    "classification_model_comparison.csv"
)

classification = load_csv(
    classification_path,
    "classification model comparison"
)


if classification is not None:

    classification.insert(
        0,
        "analysis_type",
        "Classification Model"
    )

    save_csv(
        classification,
        "classification_results.csv"
    )

else:

    print(
        "Classification results could not be prepared."
    )


# ============================================================
# 6. STEP 11 RESULTS
# ============================================================

print("\n" + "=" * 70)
print("4. STEP 11 RESULTS")
print("=" * 70)


step11_path = os.path.join(
    METRICS_ROOT,
    "step11_denoising_classification_comparison.csv"
)

step11 = load_csv(
    step11_path,
    "Step 11 denoising-classification comparison"
)


if step11 is not None:

    step11.insert(
        0,
        "analysis_type",
        "Denoising to Classification"
    )

    save_csv(
        step11,
        "step11_results.csv"
    )

else:

    print(
        "Step 11 results could not be prepared."
    )


# ============================================================
# 7. THERMAL FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("5. THERMAL FEATURE SUMMARY")
print("=" * 70)


thermal_features_path = os.path.join(
    FEATURES_ROOT,
    "thermal_features.csv"
)

thermal_features = load_csv(
    thermal_features_path,
    "thermal features"
)


if thermal_features is not None:

    print(
        "\nCreating feature summary..."
    )

    numeric_features = [
        "mean_intensity",
        "min_intensity",
        "max_intensity",
        "std_intensity",
        "percentile_90",
        "hotspot_pixels",
        "hotspot_area_ratio",
        "hotspot_mean_intensity",
        "left_mean_intensity",
        "right_mean_intensity",
        "absolute_asymmetry",
        "relative_asymmetry"
    ]

    available_features = [
        column
        for column in numeric_features
        if column in thermal_features.columns
    ]

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    summary_records = []

    for feature in available_features:

        values = pd.to_numeric(
            thermal_features[feature],
            errors="coerce"
        )

        summary_records.append({
            "group": "Overall",
            "class": "All",
            "noise_type": "All",
            "denoising_method": "All",
            "feature": feature,
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "median": float(values.median()),
            "max": float(values.max())
        })

    # --------------------------------------------------------
    # Class-wise statistics
    # --------------------------------------------------------

    if "class" in thermal_features.columns:

        for class_name, group in (
            thermal_features
            .groupby("class")
        ):

            for feature in available_features:

                values = pd.to_numeric(
                    group[feature],
                    errors="coerce"
                )

                summary_records.append({
                    "group": "Class",
                    "class": class_name,
                    "noise_type": "All",
                    "denoising_method": "All",
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "median": float(values.median()),
                    "max": float(values.max())
                })

    # --------------------------------------------------------
    # Noise type statistics
    # --------------------------------------------------------

    if "noise_type" in thermal_features.columns:

        for noise_name, group in (
            thermal_features
            .groupby("noise_type")
        ):

            for feature in available_features:

                values = pd.to_numeric(
                    group[feature],
                    errors="coerce"
                )

                summary_records.append({
                    "group": "Noise Type",
                    "class": "All",
                    "noise_type": noise_name,
                    "denoising_method": "All",
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "median": float(values.median()),
                    "max": float(values.max())
                })

    # --------------------------------------------------------
    # Denoising method statistics
    # --------------------------------------------------------

    if "denoising_method" in thermal_features.columns:

        for method_name, group in (
            thermal_features
            .groupby("denoising_method")
        ):

            for feature in available_features:

                values = pd.to_numeric(
                    group[feature],
                    errors="coerce"
                )

                summary_records.append({
                    "group": "Denoising Method",
                    "class": "All",
                    "noise_type": "All",
                    "denoising_method": method_name,
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "median": float(values.median()),
                    "max": float(values.max())
                })

    thermal_feature_summary = pd.DataFrame(
        summary_records
    )

    save_csv(
        thermal_feature_summary,
        "thermal_feature_summary.csv"
    )

else:

    print(
        "Thermal feature file could not be prepared."
    )


# ============================================================
# 8. STEP 13 LEAKAGE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("6. DATA INTEGRITY SUMMARY")
print("=" * 70)


leakage_path = os.path.join(
    METRICS_ROOT,
    "group_leakage_summary.csv"
)

leakage = load_csv(
    leakage_path,
    "group leakage summary"
)


if leakage is not None:

    save_csv(
        leakage,
        "group_leakage_summary.csv"
    )


# ============================================================
# 9. POWER BI FILE INDEX
# ============================================================

print("\n" + "=" * 70)
print("POWER BI FILE INDEX")
print("=" * 70)


files = sorted(
    os.listdir(POWERBI_ROOT)
)

index_rows = []

for filename in files:

    if filename.lower().endswith(
        ".csv"
    ):

        path = os.path.join(
            POWERBI_ROOT,
            filename
        )

        try:

            df = pd.read_csv(path)

            index_rows.append({
                "file_name": filename,
                "rows": len(df),
                "columns": len(df.columns)
            })

        except Exception:

            index_rows.append({
                "file_name": filename,
                "rows": "Error",
                "columns": "Error"
            })


file_index = pd.DataFrame(
    index_rows
)

save_csv(
    file_index,
    "powerbi_file_index.csv"
)


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 14 OUTPUT")
print("=" * 70)

for filename in sorted(
    os.listdir(POWERBI_ROOT)
):

    print(
        os.path.join(
            POWERBI_ROOT,
            filename
        )
    )


print("\n" + "=" * 70)
print("✓ STEP 14 COMPLETED")
print("=" * 70)