# ============================================================
# STEP 1: APACHE SPARK DATASET INSPECTION
# Thermal Image Diabetic Foot Project
# ============================================================

import sys
from pathlib import Path

from PIL import Image
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType
)
from pyspark.sql.functions import col


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "archive" / "ThermoDataBase"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("THERMAL IMAGE DIABETIC FOOT")
print("APACHE SPARK - DATASET INSPECTION")
print("=" * 70)

print("\nProject Root:")
print(PROJECT_ROOT)

print("\nDataset Root:")
print(DATASET_ROOT)


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not DATASET_ROOT.exists():

    print("\nERROR: Dataset folder does not exist.")
    print(DATASET_ROOT)

    sys.exit(1)

print("\n✓ Dataset directory found")


# ============================================================
# 3. CHECK DATASET STRUCTURE
# ============================================================

train_dir = DATASET_ROOT / "train"
val_dir = DATASET_ROOT / "val"

print("\nChecking dataset structure...")

if train_dir.exists():
    print("✓ train/")
else:
    print("✗ train/ missing")

if val_dir.exists():
    print("✓ val/")
else:
    print("✗ val/ missing")


classes = [
    "Control Group",
    "DM Group"
]

for split_dir in [train_dir, val_dir]:

    if not split_dir.exists():
        continue

    print(f"\nChecking {split_dir.name}/")

    for class_name in classes:

        class_dir = split_dir / class_name

        if class_dir.exists():
            print(f"✓ {class_name}")
        else:
            print(f"✗ {class_name} missing")


# ============================================================
# 4. FIND IMAGE FILES
# ============================================================

print("\n" + "=" * 70)
print("SCANNING IMAGE FILES")
print("=" * 70)

supported_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

records = []

invalid_files = []

for split in ["train", "val"]:

    split_path = DATASET_ROOT / split

    if not split_path.exists():
        continue

    for class_name in classes:

        class_path = split_path / class_name

        if not class_path.exists():
            continue

        for file_path in class_path.rglob("*"):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in supported_extensions:
                continue

            # ------------------------------------------------
            # Determine class label
            # ------------------------------------------------

            if class_name == "DM Group":
                label = "DM"
                label_id = 1

            elif class_name == "Control Group":
                label = "Control"
                label_id = 0

            else:
                label = "Unknown"
                label_id = -1

            # ------------------------------------------------
            # Read image properties
            # ------------------------------------------------

            try:

                with Image.open(file_path) as img:

                    width, height = img.size

                    # Number of channels
                    if img.mode == "L":
                        channels = 1

                    elif img.mode == "RGB":
                        channels = 3

                    elif img.mode == "RGBA":
                        channels = 4

                    else:
                        channels = len(img.getbands())

                    image_format = img.format

                    records.append(
                        (
                            str(file_path),
                            file_path.name,
                            split,
                            label,
                            label_id,
                            width,
                            height,
                            channels,
                            image_format,
                            file_path.stat().st_size
                        )
                    )

            except Exception as e:

                invalid_files.append(
                    (
                        str(file_path),
                        str(e)
                    )
                )


print("\nImage scanning completed.")

print("Valid images found:", len(records))
print("Invalid/corrupt images:", len(invalid_files))


# ============================================================
# 5. CREATE SPARK SESSION
# ============================================================

print("\n" + "=" * 70)
print("STARTING APACHE SPARK")
print("=" * 70)

# ============================================================
# 5. CONFIGURE PYTHON FOR SPARK WORKERS
# ============================================================

import os

# Use the exact Python interpreter running this script.
# This prevents Windows from redirecting Spark workers
# to the Microsoft Store Python alias.

python_executable = sys.executable

os.environ["PYSPARK_PYTHON"] = python_executable
os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

print("\nPython executable being used:")
print(python_executable)


# ============================================================
# 6. CREATE SPARK SESSION
# ============================================================

print("\n" + "=" * 70)
print("STARTING APACHE SPARK")
print("=" * 70)

spark = (
    SparkSession.builder
    .appName("ThermalDiabeticFootDatasetInspection")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.pyspark.python", python_executable)
    .config("spark.pyspark.driver.python", python_executable)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n✓ Spark started")
print("Spark Version:", spark.version)

spark.sparkContext.setLogLevel("WARN")

print("\n✓ Spark started")
print("Spark Version:", spark.version)


# ============================================================
# 6. DEFINE SPARK SCHEMA
# ============================================================

schema = StructType([

    StructField(
        "image_path",
        StringType(),
        False
    ),

    StructField(
        "filename",
        StringType(),
        False
    ),

    StructField(
        "split",
        StringType(),
        False
    ),

    StructField(
        "class",
        StringType(),
        False
    ),

    StructField(
        "label",
        IntegerType(),
        False
    ),

    StructField(
        "width",
        IntegerType(),
        False
    ),

    StructField(
        "height",
        IntegerType(),
        False
    ),

    StructField(
        "channels",
        IntegerType(),
        False
    ),

    StructField(
        "image_format",
        StringType(),
        False
    ),

    StructField(
        "file_size_bytes",
        IntegerType(),
        False
    )
])


# ============================================================
# 7. CREATE SPARK DATAFRAME
# ============================================================

print("\n" + "=" * 70)
print("CREATING SPARK DATAFRAME")
print("=" * 70)

metadata_df = spark.createDataFrame(
    records,
    schema=schema
)

print("\n✓ Spark DataFrame created")


# ============================================================
# 8. DISPLAY SCHEMA
# ============================================================

print("\n" + "-" * 70)
print("SPARK DATAFRAME SCHEMA")
print("-" * 70)

metadata_df.printSchema()


# ============================================================
# 9. TOTAL IMAGE COUNT
# ============================================================

print("\n" + "-" * 70)
print("TOTAL DATASET COUNT")
print("-" * 70)

total_images = metadata_df.count()

print("Total images:", total_images)

expected_total = 1866

if total_images == expected_total:

    print(
        f"✓ Dataset matches expected total of {expected_total}"
    )

else:

    print(
        f"⚠ Expected {expected_total}, "
        f"but found {total_images}"
    )


# ============================================================
# 10. CLASS DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("CLASS DISTRIBUTION")
print("-" * 70)

class_distribution = (
    metadata_df
    .groupBy("class", "label")
    .count()
    .orderBy("label")
)

class_distribution.show(
    truncate=False
)


# ============================================================
# 11. TRAIN / VALIDATION DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("TRAIN / VALIDATION DISTRIBUTION")
print("-" * 70)

split_distribution = (
    metadata_df
    .groupBy("split")
    .count()
    .orderBy("split")
)

split_distribution.show(
    truncate=False
)


# ============================================================
# 12. CLASS + SPLIT DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("CLASS + SPLIT DISTRIBUTION")
print("-" * 70)

class_split_distribution = (
    metadata_df
    .groupBy("split", "class")
    .count()
    .orderBy("split", "class")
)

class_split_distribution.show(
    truncate=False
)


# ============================================================
# 13. IMAGE DIMENSION DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("IMAGE DIMENSIONS")
print("-" * 70)

dimension_distribution = (
    metadata_df
    .groupBy(
        "width",
        "height",
        "channels"
    )
    .count()
    .orderBy(
        col("count").desc()
    )
)

dimension_distribution.show(
    20,
    truncate=False
)


# ============================================================
# 14. IMAGE FORMAT DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("IMAGE FORMAT DISTRIBUTION")
print("-" * 70)

format_distribution = (
    metadata_df
    .groupBy("image_format")
    .count()
    .orderBy("image_format")
)

format_distribution.show(
    truncate=False
)


# ============================================================
# 15. DUPLICATE PATH CHECK
# ============================================================

print("\n" + "-" * 70)
print("DUPLICATE PATH CHECK")
print("-" * 70)

unique_paths = (
    metadata_df
    .select("image_path")
    .distinct()
    .count()
)

duplicate_paths = total_images - unique_paths

print("Total records:", total_images)
print("Unique paths:", unique_paths)
print("Duplicate paths:", duplicate_paths)

if duplicate_paths == 0:
    print("✓ No duplicate paths found")
else:
    print("⚠ Duplicate paths detected")


# ============================================================
# 16. NULL VALUE CHECK
# ============================================================

print("\n" + "-" * 70)
print("NULL VALUE CHECK")
print("-" * 70)

columns_to_check = [
    "image_path",
    "filename",
    "split",
    "class",
    "label",
    "width",
    "height",
    "channels",
    "image_format"
]

for column_name in columns_to_check:

    null_count = (
        metadata_df
        .filter(
            col(column_name).isNull()
        )
        .count()
    )

    print(
        f"{column_name}: {null_count} null values"
    )


# ============================================================
# 17. INVALID FILE REPORT
# ============================================================

print("\n" + "-" * 70)
print("INVALID IMAGE FILE CHECK")
print("-" * 70)

if len(invalid_files) == 0:

    print("✓ No invalid/corrupt images detected")

else:

    print(
        f"⚠ {len(invalid_files)} invalid images detected"
    )

    print("\nFirst 10 invalid files:")

    for path, error in invalid_files[:10]:

        print("\nFile:", path)
        print("Error:", error)


# ============================================================
# 18. SAVE METADATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING METADATA")
print("=" * 70)

metadata_csv = OUTPUT_DIR / "thermal_image_metadata.csv"

# Collect the already-created Spark DataFrame to Pandas
# for local CSV storage.
metadata_pandas = metadata_df.toPandas()

metadata_pandas.to_csv(
    metadata_csv,
    index=False
)

print("\n✓ Metadata CSV saved successfully:")
print(metadata_csv)


# ============================================================
# 19. DISPLAY SAMPLE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE SPARK DATA")
print("=" * 70)

metadata_df.show(
    10,
    truncate=False
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 COMPLETED")
print("=" * 70)

print("\nDataset:")
print("ThermoDataBase")

print("\nTotal images:")
print(total_images)

print("\nExpected images:")
print(expected_total)

print("\nInvalid images:")
print(len(invalid_files))

print("\nUnique image paths:")
print(unique_paths)

print("\nDuplicate paths:")
print(duplicate_paths)

print("\nMetadata:")
print(metadata_csv)

print("\n✓ Apache Spark dataset inspection completed.")


# ============================================================
# 21. STOP SPARK
# ============================================================

spark.stop()

print("\n✓ Spark session stopped.")
print("DONE.")
