import os
import csv
import numpy as np
from PIL import Image


# ============================================================
# THERMAL IMAGE DIABETIC FOOT
# STEP 5 - THERMAL ANOMALY & ASYMMETRY FEATURE EXTRACTION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "denoised_fast"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "features"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "thermal_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

HOTSPOT_PERCENTILE = 90


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(image_path):

    image = Image.open(image_path).convert("L")

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # BASIC INTENSITY FEATURES
    # --------------------------------------------------------

    mean_intensity = float(
        np.mean(image_array)
    )

    min_intensity = float(
        np.min(image_array)
    )

    max_intensity = float(
        np.max(image_array)
    )

    std_intensity = float(
        np.std(image_array)
    )

    percentile_90 = float(
        np.percentile(
            image_array,
            HOTSPOT_PERCENTILE
        )
    )


    # --------------------------------------------------------
    # HOTSPOT FEATURES
    # --------------------------------------------------------

    hotspot_mask = (
        image_array >= percentile_90
    )

    hotspot_pixels = int(
        np.sum(hotspot_mask)
    )

    total_pixels = int(
        image_array.size
    )

    hotspot_area_ratio = (
        hotspot_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    if hotspot_pixels > 0:

        hotspot_mean_intensity = float(
            np.mean(
                image_array[hotspot_mask]
            )
        )

    else:

        hotspot_mean_intensity = 0.0


    # --------------------------------------------------------
    # LEFT-RIGHT IMAGE INTENSITY ASYMMETRY
    # --------------------------------------------------------

    height, width = image_array.shape

    middle = width // 2

    left_half = image_array[:, :middle]

    right_half = image_array[:, middle:]


    left_mean = float(
        np.mean(left_half)
    )

    right_mean = float(
        np.mean(right_half)
    )


    absolute_asymmetry = abs(
        left_mean - right_mean
    )


    average_side_intensity = (
        left_mean + right_mean
    ) / 2.0


    if average_side_intensity > 0:

        relative_asymmetry = (
            absolute_asymmetry
            / average_side_intensity
        )

    else:

        relative_asymmetry = 0.0


    return {
        "mean_intensity": mean_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "std_intensity": std_intensity,
        "percentile_90": percentile_90,
        "hotspot_pixels": hotspot_pixels,
        "hotspot_area_ratio": hotspot_area_ratio,
        "hotspot_mean_intensity": hotspot_mean_intensity,
        "left_mean_intensity": left_mean,
        "right_mean_intensity": right_mean,
        "absolute_asymmetry": absolute_asymmetry,
        "relative_asymmetry": relative_asymmetry
    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print("THERMAL IMAGE DIABETIC FOOT")
print("STEP 5 - THERMAL ANOMALY & ASYMMETRY FEATURE EXTRACTION")
print("=" * 70)

print("\nInput:")
print(INPUT_DIR)

print("\nOutput:")
print(OUTPUT_FILE)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_DIR):

    print("\nERROR: Step 4 output folder not found.")

    raise SystemExit(1)


# ============================================================
# FIND ALL IMAGES
# ============================================================

image_files = []

for root, dirs, files in os.walk(INPUT_DIR):

    for filename in files:

        if filename.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff"
            )
        ):

            image_files.append(
                os.path.join(
                    root,
                    filename
                )
            )


image_files.sort()


print(
    "\nTotal denoised images found:",
    len(image_files)
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

records = []

failed = 0


print("\nExtracting thermal features...\n")


for index, image_path in enumerate(
    image_files,
    start=1
):

    try:

        features = extract_features(
            image_path
        )


        # ----------------------------------------------------
        # GET RELATIVE PATH
        # ----------------------------------------------------

        relative_path = os.path.relpath(
            image_path,
            INPUT_DIR
        )

        parts = relative_path.split(
            os.sep
        )


        # ----------------------------------------------------
        # ACTUAL STEP 4 STRUCTURE
        #
        # split / class / noise_type /
        # denoising_method / image
        # ----------------------------------------------------

        split = (
            parts[0]
            if len(parts) > 0
            else "unknown"
        )

        class_name = (
            parts[1]
            if len(parts) > 1
            else "unknown"
        )

        noise_type = (
            parts[2]
            if len(parts) > 2
            else "unknown"
        )

        denoising_method = (
            parts[3]
            if len(parts) > 3
            else "unknown"
        )


        # ----------------------------------------------------
        # CLASS LABEL
        # ----------------------------------------------------

        if class_name.lower() == "dm":

            label = 1

        elif class_name.lower() == "control":

            label = 0

        else:

            label = -1


        # ----------------------------------------------------
        # CREATE RECORD
        # ----------------------------------------------------

        record = {

            "image_path": image_path,

            "filename": os.path.basename(
                image_path
            ),

            "split": split,

            "class": class_name,

            "label": label,

            "noise_type": noise_type,

            "denoising_method": denoising_method,

            **features

        }


        records.append(
            record
        )


    except Exception as error:

        failed += 1

        print(
            "Failed:",
            image_path,
            "->",
            error
        )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        index % 500 == 0
        or index == len(image_files)
    ):

        print(
            f"Images processed: "
            f"{index}/{len(image_files)}"
        )


# ============================================================
# SAVE CSV
# ============================================================

if len(records) > 0:

    fieldnames = list(
        records[0].keys()
    )


    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            records
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 SUMMARY")
print("=" * 70)

print(
    "\nImages found:",
    len(image_files)
)

print(
    "Features extracted:",
    len(records)
)

print(
    "Failed:",
    failed
)


# ============================================================
# EXPECTED STRUCTURE CHECK
# ============================================================

if records:

    splits = sorted(
        set(
            r["split"]
            for r in records
        )
    )

    classes = sorted(
        set(
            r["class"]
            for r in records
        )
    )

    noise_types = sorted(
        set(
            r["noise_type"]
            for r in records
        )
    )

    methods = sorted(
        set(
            r["denoising_method"]
            for r in records
        )
    )


    print("\nSplits found:")
    print(splits)

    print("\nClasses found:")
    print(classes)

    print("\nNoise types found:")
    print(noise_types)

    print("\nDenoising methods found:")
    print(methods)


    print("\nFeature columns:")

    for column in records[0].keys():

        print(
            " -",
            column
        )


print("\nResults saved to:")

print(
    OUTPUT_FILE
)


print("\n" + "=" * 70)

print(
    "✓ STEP 5 COMPLETED SUCCESSFULLY"
)

print("=" * 70)