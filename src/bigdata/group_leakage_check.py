import os
import re
import csv
from collections import defaultdict


# ============================================================
# STEP 13 - GROUP / DATA LEAKAGE CHECK
# ============================================================

print("=" * 70)
print("STEP 13 - TRAIN / VALIDATION GROUP LEAKAGE CHECK")
print("=" * 70)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "standardized"
)

RESULTS_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

os.makedirs(
    RESULTS_ROOT,
    exist_ok=True
)


# ============================================================
# 2. IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# 3. GROUP ID EXTRACTION
# ============================================================

def get_group_id(filename):
    """
    Create an image-family/group identifier.

    Examples:

        CG009_M_R.png
        CG009_M_R-rotated1.png
        CG009_M_R-rotated2.png
        CG009_M_R-sharpened.png

    are grouped as:

        CG009_M_R

    This is a GROUP/IMAGE-FAMILY proxy.
    It is NOT automatically a patient ID.
    """

    stem = os.path.splitext(
        filename
    )[0]

    group = stem

    # Remove repeated augmentation suffixes.
    # Example:
    # image-rotated1-rotated2
    # becomes:
    # image
    while True:

        new_group = re.sub(
            r"-(?:rotated\d+|sharpened)$",
            "",
            group,
            flags=re.IGNORECASE
        )

        if new_group == group:
            break

        group = new_group

    return group


# ============================================================
# 4. COLLECT DATA
# ============================================================

records = []

for split in ["train", "val"]:

    for class_name in ["Control", "DM"]:

        folder = os.path.join(
            DATASET_ROOT,
            split,
            class_name
        )

        if not os.path.isdir(folder):

            print(
                f"WARNING: Folder not found:\n{folder}"
            )

            continue

        for filename in sorted(
            os.listdir(folder)
        ):

            if not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue

            group_id = get_group_id(
                filename
            )

            records.append({
                "filename": filename,
                "split": split,
                "class": class_name,
                "group_id": group_id
            })


# ============================================================
# 5. BASIC COUNTS
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

total_images = len(records)

train_records = [
    r for r in records
    if r["split"] == "train"
]

val_records = [
    r for r in records
    if r["split"] == "val"
]

print(
    f"Total images       : {total_images}"
)

print(
    f"Training images    : {len(train_records)}"
)

print(
    f"Validation images  : {len(val_records)}"
)


# ============================================================
# 6. GROUPS BY SPLIT
# ============================================================

train_groups = set(
    r["group_id"]
    for r in train_records
)

val_groups = set(
    r["group_id"]
    for r in val_records
)

overlap_groups = (
    train_groups &
    val_groups
)

all_groups = (
    train_groups |
    val_groups
)


# ============================================================
# 7. GROUP COUNTS
# ============================================================

print("\n" + "=" * 70)
print("GROUP ANALYSIS")
print("=" * 70)

print(
    f"Unique groups overall : {len(all_groups)}"
)

print(
    f"Training groups       : {len(train_groups)}"
)

print(
    f"Validation groups     : {len(val_groups)}"
)

print(
    f"Overlapping groups    : {len(overlap_groups)}"
)


# ============================================================
# 8. DISPLAY OVERLAPPING GROUPS
# ============================================================

if overlap_groups:

    print("\n" + "=" * 70)
    print("WARNING - GROUP OVERLAP FOUND")
    print("=" * 70)

    print(
        "\nThe following groups occur in BOTH "
        "training and validation:\n"
    )

    for group_id in sorted(
        overlap_groups
    ):

        print(
            f"\nGroup: {group_id}"
        )

        train_files = [
            r["filename"]
            for r in train_records
            if r["group_id"] == group_id
        ]

        val_files = [
            r["filename"]
            for r in val_records
            if r["group_id"] == group_id
        ]

        print(
            "  Train:"
        )

        for filename in train_files:
            print(
                f"    {filename}"
            )

        print(
            "  Validation:"
        )

        for filename in val_files:
            print(
                f"    {filename}"
            )

else:

    print("\n" + "=" * 70)
    print("✓ NO GROUP OVERLAP FOUND")
    print("=" * 70)

    print(
        "\nNo image-family group was found in both "
        "training and validation."
    )


# ============================================================
# 9. GROUP SIZE ANALYSIS
# ============================================================

group_to_files = defaultdict(list)

for record in records:

    group_to_files[
        record["group_id"]
    ].append(record)


group_sizes = [
    len(files)
    for files in group_to_files.values()
]


if group_sizes:

    print("\n" + "=" * 70)
    print("GROUP SIZE ANALYSIS")
    print("=" * 70)

    print(
        f"Minimum images/group : "
        f"{min(group_sizes)}"
    )

    print(
        f"Maximum images/group : "
        f"{max(group_sizes)}"
    )

    print(
        f"Average images/group : "
        f"{sum(group_sizes) / len(group_sizes):.2f}"
    )


# ============================================================
# 10. SAME GROUP WITH DIFFERENT CLASSES
# ============================================================

group_classes = defaultdict(set)

for record in records:

    group_classes[
        record["group_id"]
    ].add(
        record["class"]
    )


mixed_class_groups = {
    group_id
    for group_id, classes
    in group_classes.items()
    if len(classes) > 1
}


print("\n" + "=" * 70)
print("GROUP / CLASS CONSISTENCY")
print("=" * 70)

print(
    f"Groups containing multiple classes: "
    f"{len(mixed_class_groups)}"
)


if mixed_class_groups:

    print(
        "\nWARNING: Some groups appear under "
        "both Control and DM:"
    )

    for group_id in sorted(
        mixed_class_groups
    ):

        print(
            f"  {group_id}: "
            f"{sorted(group_classes[group_id])}"
        )

else:

    print(
        "✓ No group was found with multiple classes."
    )


# ============================================================
# 11. SAVE DETAILED GROUP REPORT
# ============================================================

group_report_path = os.path.join(
    RESULTS_ROOT,
    "group_leakage_report.csv"
)

with open(
    group_report_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "group_id",
        "split",
        "class",
        "filename"
    ])

    for record in sorted(
        records,
        key=lambda x: (
            x["group_id"],
            x["split"],
            x["filename"]
        )
    ):

        writer.writerow([
            record["group_id"],
            record["split"],
            record["class"],
            record["filename"]
        ])


# ============================================================
# 12. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_ROOT,
    "group_leakage_summary.csv"
)

leakage_status = (
    "OVERLAP_FOUND"
    if overlap_groups
    else "NO_OVERLAP_FOUND"
)

summary_rows = [

    ["total_images", total_images],

    ["training_images", len(train_records)],

    ["validation_images", len(val_records)],

    ["unique_groups", len(all_groups)],

    ["training_groups", len(train_groups)],

    ["validation_groups", len(val_groups)],

    ["overlapping_groups", len(overlap_groups)],

    ["mixed_class_groups", len(mixed_class_groups)],

    ["leakage_status", leakage_status]

]


with open(
    summary_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "metric",
        "value"
    ])

    writer.writerows(
        summary_rows
    )


# ============================================================
# 13. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 INTERPRETATION")
print("=" * 70)

if overlap_groups:

    print(
        "\n⚠ GROUP OVERLAP DETECTED."
    )

    print(
        "Related image families occur in both "
        "training and validation."
    )

    print(
        "\nThe current validation results should "
        "therefore be interpreted cautiously."
    )

    print(
        "\nA group-aware train/validation split "
        "may be required before reporting final "
        "generalization performance."
    )

else:

    print(
        "\n✓ NO GROUP OVERLAP DETECTED."
    )

    print(
        "The detected image families are separated "
        "between training and validation."
    )

    print(
        "\nThis supports using the existing train/val "
        "split for the current experiments."
    )


# ============================================================
# 14. OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 OUTPUT FILES")
print("=" * 70)

print(
    "\nDetailed report:"
)

print(
    group_report_path
)

print(
    "\nSummary:"
)

print(
    summary_path
)

print("\n" + "=" * 70)
print("STEP 13 COMPLETED")
print("=" * 70)