import os
import sys
import csv

# ============================================================
# FIX WINDOWS PYTHON PATH FOR SPARK
# ============================================================

PYTHON_EXE = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXE


from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    min,
    max,
    stddev
)


# ============================================================
# THERMAL IMAGE DIABETIC FOOT
# STEP 6 - APACHE SPARK FEATURE ANALYSIS & SELECTION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "thermal_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "features"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "spark_feature_summary.csv"
)

SELECTED_FILE = os.path.join(
    OUTPUT_DIR,
    "selected_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("THERMAL IMAGE DIABETIC FOOT")
print("STEP 6 - APACHE SPARK FEATURE ANALYSIS & SELECTION")
print("=" * 70)

print("\nPython executable:")
print(PYTHON_EXE)

print("\nInput:")
print(INPUT_FILE)

print("\nSummary output:")
print(SUMMARY_FILE)

print("\nSelected features output:")
print(SELECTED_FILE)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):

    print("\nERROR: thermal_features.csv not found.")

    raise SystemExit(1)


# ============================================================
# START SPARK
# ============================================================

print("\nStarting Apache Spark...")

spark = (
    SparkSession.builder
    .appName("ThermalFeatureAnalysis")
    .master("local[*]")
    .config(
        "spark.sql.shuffle.partitions",
        "4"
    )
    .config(
        "spark.pyspark.python",
        PYTHON_EXE
    )
    .config(
        "spark.pyspark.driver.python",
        PYTHON_EXE
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("✓ Apache Spark started")

print(
    "Spark version:",
    spark.version
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading thermal feature dataset...")

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(INPUT_FILE)
)

print("✓ Dataset loaded")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

total_rows = df.count()

print(
    "\nTotal feature records:",
    total_rows
)

print(
    "Total columns:",
    len(df.columns)
)


print("\nSchema:")

df.printSchema()


# ============================================================
# CONVERT NUMERICAL FEATURES
# ============================================================

feature_columns = [

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

for feature in feature_columns:

    df = df.withColumn(
        feature,
        col(feature).cast("double")
    )

df = df.withColumn(
    "label",
    col("label").cast("double")
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_distribution = (
    df.groupBy("class")
    .agg(
        count("*").alias("image_count"),

        avg("mean_intensity").alias(
            "avg_mean_intensity"
        ),

        avg("hotspot_area_ratio").alias(
            "avg_hotspot_area_ratio"
        ),

        avg("relative_asymmetry").alias(
            "avg_relative_asymmetry"
        )
    )
    .orderBy("class")
)

class_distribution.show(
    truncate=False
)


# ============================================================
# TRAIN / VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION DISTRIBUTION")
print("=" * 70)

split_distribution = (
    df.groupBy("split")
    .agg(
        count("*").alias("record_count")
    )
    .orderBy("split")
)

split_distribution.show()


# ============================================================
# NOISE TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("NOISE TYPE ANALYSIS")
print("=" * 70)

noise_analysis = (
    df.groupBy("noise_type")
    .agg(

        count("*").alias(
            "record_count"
        ),

        avg("mean_intensity").alias(
            "avg_mean_intensity"
        ),

        avg("std_intensity").alias(
            "avg_std_intensity"
        ),

        avg("hotspot_area_ratio").alias(
            "avg_hotspot_area_ratio"
        ),

        avg("relative_asymmetry").alias(
            "avg_relative_asymmetry"
        )
    )
    .orderBy("noise_type")
)

noise_analysis.show(
    truncate=False
)


# ============================================================
# DENOISING METHOD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DENOISING METHOD ANALYSIS")
print("=" * 70)

method_analysis = (
    df.groupBy("denoising_method")
    .agg(

        count("*").alias(
            "record_count"
        ),

        avg("mean_intensity").alias(
            "avg_mean_intensity"
        ),

        avg("std_intensity").alias(
            "avg_std_intensity"
        ),

        avg("hotspot_area_ratio").alias(
            "avg_hotspot_area_ratio"
        ),

        avg("hotspot_mean_intensity").alias(
            "avg_hotspot_mean_intensity"
        ),

        avg("relative_asymmetry").alias(
            "avg_relative_asymmetry"
        )
    )
    .orderBy("denoising_method")
)

method_analysis.show(
    truncate=False
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)


# Use Spark aggregation directly.
# Results are collected to the driver as rows.
# This avoids creating a Spark DataFrame from Python objects.

feature_summary_rows = []

for feature in feature_columns:

    result = (
        df.select(
            avg(col(feature)).alias("mean"),
            min(col(feature)).alias("minimum"),
            max(col(feature)).alias("maximum"),
            stddev(col(feature)).alias("stddev")
        )
        .collect()[0]
    )

    feature_summary_rows.append(
        [
            feature,
            result["mean"],
            result["minimum"],
            result["maximum"],
            result["stddev"]
        ]
    )


print("\nGlobal feature statistics:")

for row in feature_summary_rows:

    print(
        f"{row[0]:30s} "
        f"mean={row[1]:.4f} "
        f"min={row[2]:.4f} "
        f"max={row[3]:.4f} "
        f"std={row[4]:.4f}"
    )


# ============================================================
# FEATURE CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE CORRELATION WITH CLASS LABEL")
print("=" * 70)


correlation_rows = []

for feature in feature_columns:

    correlation = df.stat.corr(
        feature,
        "label"
    )

    if correlation is None:

        correlation = 0.0

    correlation_rows.append(
        [
            feature,
            float(correlation),
            abs(float(correlation))
        ]
    )


# Sort by absolute correlation

correlation_rows.sort(
    key=lambda x: x[2],
    reverse=True
)


print(
    "\nFeatures ranked by absolute correlation:"
)

for row in correlation_rows:

    print(
        f"{row[0]:30s} "
        f"correlation={row[1]:.6f} "
        f"absolute={row[2]:.6f}"
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)


selected_features = [

    "mean_intensity",

    "std_intensity",

    "percentile_90",

    "hotspot_area_ratio",

    "hotspot_mean_intensity",

    "absolute_asymmetry",

    "relative_asymmetry"

]


print("\nSelected thermal features:")

for feature in selected_features:

    print(
        " -",
        feature
    )


# ============================================================
# CREATE SELECTED DATASET
# ============================================================

selected_columns = [

    "image_path",

    "filename",

    "split",

    "class",

    "label",

    "noise_type",

    "denoising_method"

] + selected_features


selected_df = df.select(
    *selected_columns
)


selected_count = selected_df.count()

print(
    "\nSelected dataset records:",
    selected_count
)


# ============================================================
# WRITE SELECTED CSV
# ============================================================

print("\nSaving selected features...")

selected_rows = selected_df.collect()

with open(
    SELECTED_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        selected_columns
    )

    for row in selected_rows:

        writer.writerow(
            row
        )


print(
    "✓ Selected features saved:"
)

print(
    SELECTED_FILE
)


# ============================================================
# WRITE FEATURE SUMMARY CSV
# ============================================================

print("\nSaving Spark feature summary...")

correlation_lookup = {
    row[0]: row
    for row in correlation_rows
}


with open(
    SUMMARY_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "feature",
            "mean",
            "minimum",
            "maximum",
            "stddev",
            "correlation_with_label",
            "absolute_correlation"
        ]
    )


    for row in feature_summary_rows:

        feature = row[0]

        correlation = (
            correlation_lookup[feature]
        )

        writer.writerow(
            [
                feature,
                row[1],
                row[2],
                row[3],
                row[4],
                correlation[1],
                correlation[2]
            ]
        )


print(
    "✓ Spark feature summary saved:"
)

print(
    SUMMARY_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 SUMMARY")
print("=" * 70)

print(
    "\nTotal records:",
    total_rows
)

print(
    "Selected dataset records:",
    selected_count
)

print(
    "Total original columns:",
    len(df.columns)
)

print(
    "Selected feature count:",
    len(selected_features)
)

print(
    "Apache Spark version:",
    spark.version
)


print("\nSelected features:")

for feature in selected_features:

    print(
        " -",
        feature
    )


print("\nOutput files:")

print(
    " -",
    SUMMARY_FILE
)

print(
    " -",
    SELECTED_FILE
)


print("\n" + "=" * 70)

print(
    "✓ STEP 6 COMPLETED SUCCESSFULLY"
)

print("=" * 70)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()