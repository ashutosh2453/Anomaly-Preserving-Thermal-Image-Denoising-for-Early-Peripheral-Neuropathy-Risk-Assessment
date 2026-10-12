import os
import csv
import shutil
from collections import Counter

# ============================================================
# STEP 7
# CLASSIFICATION DATASET PREPARATION
# ============================================================

# Project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Input directory containing denoised images
DENOISED_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "denoised_fast"
)

# Output classification dataset
OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "classification"
)

# Metadata files
FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "features",
    "thermal_features.csv"
)

SELECTED_FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "features",
    "selected_features.csv"
)

OUTPUT_METADATA = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "classification_metadata.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Primary denoising method selected for classification
PRIMARY_DENOISING_METHOD = "median"

# We use the three noise conditions, but select the
# median-denoised image for the final dataset.
#
# To avoid duplicate samples, we select one fixed
# condition for the primary classification dataset.
PRIMARY_NOISE_TYPE = "gaussian"

# Dataset classes
CLASSES = {
    "Control": 0,
    "DM": 1
}

SPLITS = ["train", "val"]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in SPLITS:
    for class_name in CLASSES:
        output_dir = os.path.join(
            OUTPUT_ROOT,
            split,
            class_name
        )

        os.makedirs(output_dir, exist_ok=True)


# ============================================================
# LOAD SELECTED FEATURES
# ============================================================

selected_features = []

if os.path.exists(SELECTED_FEATURE_FILE):

    with open(
        SELECTED_FEATURE_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            if "feature" in row:
                selected_features.append(row["feature"])

print("=" * 70)
print("STEP 7 - CLASSIFICATION DATASET PREPARATION")
print("=" * 70)

print("\nPrimary denoising method:")
print(PRIMARY_DENOISING_METHOD)

print("\nPrimary noise condition:")
print(PRIMARY_NOISE_TYPE)

print("\nSelected thermal features:")
for feature in selected_features:
    print(" -", feature)


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not os.path.exists(DENOISED_ROOT):

    print("\nERROR: Denoised dataset not found:")
    print(DENOISED_ROOT)
    raise SystemExit(1)


# ============================================================
# PROCESS DATASET
# ============================================================

records = []

total_found = 0
total_copied = 0
total_failed = 0

split_counts = Counter()
class_counts = Counter()


for split in SPLITS:

    for class_name, label in CLASSES.items():

        source_dir = os.path.join(
            DENOISED_ROOT,
            split,
            class_name,
            PRIMARY_NOISE_TYPE,
            PRIMARY_DENOISING_METHOD
        )

        destination_dir = os.path.join(
            OUTPUT_ROOT,
            split,
            class_name
        )

        print("\nProcessing:")
        print(" Split :", split)
        print(" Class :", class_name)
        print(" Source:", source_dir)

        if not os.path.exists(source_dir):

            print(" WARNING: Source directory not found.")
            continue

        files = sorted(os.listdir(source_dir))

        for filename in files:

            if not filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            ):
                continue

            total_found += 1

            source_path = os.path.join(
                source_dir,
                filename
            )

            destination_path = os.path.join(
                destination_dir,
                filename
            )

            try:

                shutil.copy2(
                    source_path,
                    destination_path
                )

                total_copied += 1

                split_counts[split] += 1
                class_counts[class_name] += 1

                records.append({

                    "image_path": destination_path,
                    "original_denoised_path": source_path,
                    "filename": filename,
                    "split": split,
                    "class": class_name,
                    "label": label,
                    "noise_type": PRIMARY_NOISE_TYPE,
                    "denoising_method": PRIMARY_DENOISING_METHOD

                })

            except Exception as e:

                total_failed += 1

                print(
                    " Failed:",
                    filename,
                    "| Error:",
                    e
                )


# ============================================================
# SAVE METADATA
# ============================================================

fieldnames = [
    "image_path",
    "original_denoised_path",
    "filename",
    "split",
    "class",
    "label",
    "noise_type",
    "denoising_method"
]

with open(
    OUTPUT_METADATA,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(records)


# ============================================================
# DATASET VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION DATASET SUMMARY")
print("=" * 70)

print("\nTotal images found :", total_found)
print("Total images copied:", total_copied)
print("Total failed       :", total_failed)

print("\nSplit distribution:")

for split in SPLITS:
    print(
        f" {split:5s}:",
        split_counts[split]
    )

print("\nClass distribution:")

for class_name in CLASSES:
    print(
        f" {class_name:8s}:",
        class_counts[class_name]
    )


# ============================================================
# EXPECTED DATASET CHECK
# ============================================================

expected_total = 1866

print("\nExpected total original images:", expected_total)
print("Actual classification images   :", total_copied)

if total_copied == expected_total:

    print("\n✓ Dataset size check PASSED")

else:

    print("\n⚠ Dataset size differs from expected 1866")


# ============================================================
# CLASS BALANCE
# ============================================================

control_count = class_counts["Control"]
dm_count = class_counts["DM"]

if control_count > 0 and dm_count > 0:

    difference = abs(
        control_count - dm_count
    )

    ratio = min(
        control_count,
        dm_count
    ) / max(
        control_count,
        dm_count
    )

    print("\nClass balance:")

    print(" Control:", control_count)
    print(" DM     :", dm_count)

    print(
        " Balance ratio:",
        round(ratio, 4)
    )


# ============================================================
# TRAIN / VALIDATION CHECK
# ============================================================

print("\nTrain / Validation distribution:")

for split in SPLITS:

    print(
        f"\n{split.upper()}"
    )

    for class_name in CLASSES:

        count = 0

        for record in records:

            if (
                record["split"] == split
                and record["class"] == class_name
            ):
                count += 1

        print(
            f" {class_name}: {count}"
        )


# ============================================================
# DUPLICATE FILENAME CHECK
# ============================================================

filenames = [
    record["filename"]
    for record in records
]

duplicate_filenames = (
    len(filenames)
    - len(set(filenames))
)

print("\nDuplicate filename count:",
      duplicate_filenames)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\nOutput dataset:")
print(OUTPUT_ROOT)

print("\nMetadata:")
print(OUTPUT_METADATA)

print("\n" + "=" * 70)
print("✓ STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 70)