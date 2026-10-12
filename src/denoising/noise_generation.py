"""
STEP 3
Thermal Image Diabetic Foot Project

NOISE ANALYSIS AND CONTROLLED NOISE ADDITION

Purpose:
1. Read standardized thermal images
2. Analyze clean image statistics
3. Add controlled synthetic noise
4. Generate noisy thermal images
5. Preserve train/validation split
6. Preserve Control/DM classes
7. Generate noise metadata
8. Calculate quality metrics between clean and noisy images

Noise types:
- Gaussian noise
- Salt-and-pepper noise
- Speckle noise

The original and standardized images are NEVER modified.
"""

from pathlib import Path
import csv
import random
import math

import numpy as np
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "standardized"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "noisy"
)

OUTPUT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "noise_metadata.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

# ------------------------------------------------------------
# Noise configuration
# ------------------------------------------------------------

NOISE_TYPES = [
    "gaussian",
    "salt_pepper",
    "speckle"
]

GAUSSIAN_SIGMA = 0.05

SALT_PEPPER_AMOUNT = 0.02

SPECKLE_SIGMA = 0.05

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# PRINTING
# ============================================================

def print_header(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_output_directories():

    for split in ["train", "val"]:

        for class_name in ["Control", "DM"]:

            directory = (
                OUTPUT_ROOT
                / split
                / class_name
            )

            directory.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

    return image_array


# ============================================================
# IMAGE SAVING
# ============================================================

def save_image(image_array, output_path):

    image_array = np.clip(
        image_array,
        0.0,
        1.0
    )

    image_uint8 = (
        image_array * 255.0
    ).astype(np.uint8)

    image = Image.fromarray(
        image_uint8,
        mode="RGB"
    )

    image.save(
        output_path,
        format="PNG"
    )


# ============================================================
# GAUSSIAN NOISE
# ============================================================

def add_gaussian_noise(
    image,
    sigma=GAUSSIAN_SIGMA
):

    noise = np.random.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape
    ).astype(np.float32)

    noisy_image = image + noise

    return np.clip(
        noisy_image,
        0.0,
        1.0
    )


# ============================================================
# SALT AND PEPPER NOISE
# ============================================================

def add_salt_pepper_noise(
    image,
    amount=SALT_PEPPER_AMOUNT
):

    noisy_image = image.copy()

    total_pixels = (
        image.shape[0]
        * image.shape[1]
    )

    number_of_noisy_pixels = int(
        total_pixels * amount
    )

    # --------------------------------------------------------
    # Salt pixels
    # --------------------------------------------------------

    salt_count = (
        number_of_noisy_pixels // 2
    )

    salt_y = np.random.randint(
        0,
        image.shape[0],
        salt_count
    )

    salt_x = np.random.randint(
        0,
        image.shape[1],
        salt_count
    )

    noisy_image[
        salt_y,
        salt_x,
        :
    ] = 1.0

    # --------------------------------------------------------
    # Pepper pixels
    # --------------------------------------------------------

    pepper_count = (
        number_of_noisy_pixels
        - salt_count
    )

    pepper_y = np.random.randint(
        0,
        image.shape[0],
        pepper_count
    )

    pepper_x = np.random.randint(
        0,
        image.shape[1],
        pepper_count
    )

    noisy_image[
        pepper_y,
        pepper_x,
        :
    ] = 0.0

    return noisy_image


# ============================================================
# SPECKLE NOISE
# ============================================================

def add_speckle_noise(
    image,
    sigma=SPECKLE_SIGMA
):

    noise = np.random.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape
    ).astype(np.float32)

    noisy_image = (
        image
        + image * noise
    )

    return np.clip(
        noisy_image,
        0.0,
        1.0
    )


# ============================================================
# NOISE SELECTOR
# ============================================================

def add_noise(
    image,
    noise_type
):

    if noise_type == "gaussian":

        return add_gaussian_noise(image)

    elif noise_type == "salt_pepper":

        return add_salt_pepper_noise(image)

    elif noise_type == "speckle":

        return add_speckle_noise(image)

    else:

        raise ValueError(
            f"Unknown noise type: {noise_type}"
        )


# ============================================================
# IMAGE STATISTICS
# ============================================================

def calculate_image_statistics(image):

    gray = np.mean(
        image,
        axis=2
    )

    mean_value = float(
        np.mean(gray)
    )

    std_value = float(
        np.std(gray)
    )

    minimum = float(
        np.min(gray)
    )

    maximum = float(
        np.max(gray)
    )

    return (
        mean_value,
        std_value,
        minimum,
        maximum
    )


# ============================================================
# MSE
# ============================================================

def calculate_mse(
    original,
    noisy
):

    mse = np.mean(
        (original - noisy) ** 2
    )

    return float(mse)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    original,
    noisy
):

    mse = calculate_mse(
        original,
        noisy
    )

    if mse == 0:

        return float("inf")

    psnr = (
        10
        * math.log10(
            1.0 / mse
        )
    )

    return float(psnr)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "THERMAL IMAGE DIABETIC FOOT\n"
        "STEP 3 - NOISE ANALYSIS & CONTROLLED NOISE ADDITION"
    )

    print(
        f"\nProject Root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nInput Directory:\n{INPUT_ROOT}"
    )

    print(
        f"\nOutput Directory:\n{OUTPUT_ROOT}"
    )

    print(
        f"\nNoise Types:\n{NOISE_TYPES}"
    )

    # ========================================================
    # CHECK INPUT
    # ========================================================

    print_header(
        "CHECKING STANDARDIZED DATASET"
    )

    if not INPUT_ROOT.exists():

        print(
            "ERROR: Standardized dataset "
            "not found."
        )

        print(
            f"Expected:\n{INPUT_ROOT}"
        )

        return

    print(
        "✓ Standardized dataset found"
    )

    # ========================================================
    # CREATE OUTPUT
    # ========================================================

    print_header(
        "CREATING NOISY DATASET DIRECTORIES"
    )

    create_output_directories()

    print(
        "✓ Output directories created"
    )

    # ========================================================
    # FIND IMAGES
    # ========================================================

    image_files = []

    for split in ["train", "val"]:

        for class_name in [
            "Control",
            "DM"
        ]:

            directory = (
                INPUT_ROOT
                / split
                / class_name
            )

            if not directory.exists():

                print(
                    f"WARNING: Missing directory: "
                    f"{directory}"
                )

                continue

            files = list(
                directory.glob("*.png")
            )

            for file in files:

                image_files.append(
                    (
                        file,
                        split,
                        class_name
                    )
                )

    print(
        f"Standardized images found: "
        f"{len(image_files)}"
    )

    # ========================================================
    # EXPECTED DATASET SIZE
    # ========================================================

    if len(image_files) == 1866:

        print(
            "✓ Dataset contains expected "
            "1866 images"
        )

    else:

        print(
            "⚠ WARNING: Expected 1866 "
            f"images but found {len(image_files)}"
        )

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    print_header(
        "GENERATING CONTROLLED NOISE"
    )

    records = []

    total_generated = 0

    failed_images = 0

    noise_counters = {
        "gaussian": 0,
        "salt_pepper": 0,
        "speckle": 0
    }

    # --------------------------------------------------------
    # Process each clean image
    # --------------------------------------------------------

    for index, (
        source_path,
        split,
        class_name
    ) in enumerate(
        image_files,
        start=1
    ):

        try:

            clean_image = load_image(
                source_path
            )

            # ------------------------------------------------
            # Verify dimensions
            # ------------------------------------------------

            if (
                clean_image.shape[0]
                != IMAGE_SIZE[1]
                or
                clean_image.shape[1]
                != IMAGE_SIZE[0]
            ):

                print(
                    f"[ERROR] Wrong input size: "
                    f"{source_path.name}"
                )

                failed_images += 1

                continue

            # ------------------------------------------------
            # Clean image statistics
            # ------------------------------------------------

            (
                clean_mean,
                clean_std,
                clean_min,
                clean_max
            ) = calculate_image_statistics(
                clean_image
            )

            # ------------------------------------------------
            # Generate every noise type
            # ------------------------------------------------

            for noise_type in NOISE_TYPES:

                noisy_image = add_noise(
                    clean_image,
                    noise_type
                )

                # --------------------------------------------
                # Noisy statistics
                # --------------------------------------------

                (
                    noisy_mean,
                    noisy_std,
                    noisy_min,
                    noisy_max
                ) = calculate_image_statistics(
                    noisy_image
                )

                # --------------------------------------------
                # Quality metrics
                # --------------------------------------------

                mse = calculate_mse(
                    clean_image,
                    noisy_image
                )

                psnr = calculate_psnr(
                    clean_image,
                    noisy_image
                )

                # --------------------------------------------
                # Output path
                # --------------------------------------------

                output_directory = (
                    OUTPUT_ROOT
                    / split
                    / class_name
                )

                output_filename = (
                    f"{source_path.stem}"
                    f"_{noise_type}.png"
                )

                output_path = (
                    output_directory
                    / output_filename
                )

                # --------------------------------------------
                # Save noisy image
                # --------------------------------------------

                save_image(
                    noisy_image,
                    output_path
                )

                # --------------------------------------------
                # Parameters
                # --------------------------------------------

                if noise_type == "gaussian":

                    parameter = (
                        f"sigma={GAUSSIAN_SIGMA}"
                    )

                elif noise_type == "salt_pepper":

                    parameter = (
                        f"amount={SALT_PEPPER_AMOUNT}"
                    )

                else:

                    parameter = (
                        f"sigma={SPECKLE_SIGMA}"
                    )

                # --------------------------------------------
                # Metadata
                # --------------------------------------------

                records.append(
                    {
                        "original_path":
                            str(source_path),

                        "noisy_path":
                            str(output_path),

                        "filename":
                            source_path.name,

                        "noisy_filename":
                            output_filename,

                        "split":
                            split,

                        "class":
                            class_name,

                        "label":
                            0
                            if class_name == "Control"
                            else 1,

                        "width":
                            IMAGE_SIZE[0],

                        "height":
                            IMAGE_SIZE[1],

                        "channels":
                            3,

                        "noise_type":
                            noise_type,

                        "noise_parameter":
                            parameter,

                        "clean_mean":
                            clean_mean,

                        "clean_std":
                            clean_std,

                        "clean_min":
                            clean_min,

                        "clean_max":
                            clean_max,

                        "noisy_mean":
                            noisy_mean,

                        "noisy_std":
                            noisy_std,

                        "noisy_min":
                            noisy_min,

                        "noisy_max":
                            noisy_max,

                        "mse":
                            mse,

                        "psnr":
                            psnr
                    }
                )

                total_generated += 1

                noise_counters[
                    noise_type
                ] += 1

        except Exception as error:

            failed_images += 1

            print(
                f"[ERROR] {source_path.name}"
            )

            print(
                f"        {error}"
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % 100 == 0
            or
            index == len(image_files)
        ):

            print(
                f"Processed clean images: "
                f"{index}/{len(image_files)}"
            )


    # ========================================================
    # SAVE METADATA
    # ========================================================

    print_header(
        "SAVING NOISE METADATA"
    )

    if records:

        fieldnames = list(
            records[0].keys()
        )

        with open(
            OUTPUT_METADATA,
            "w",
            encoding="utf-8",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                records
            )

        print(
            "✓ Noise metadata saved:"
        )

        print(
            OUTPUT_METADATA
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_header(
        "STEP 3 SUMMARY"
    )

    print(
        f"""
Clean images processed:
{len(image_files)}

Failed images:
{failed_images}

Gaussian noisy images:
{noise_counters["gaussian"]}

Salt-and-pepper noisy images:
{noise_counters["salt_pepper"]}

Speckle noisy images:
{noise_counters["speckle"]}

Total noisy images:
{total_generated}

Target dimensions:
224 x 224 x 3

Random seed:
{RANDOM_SEED}

Noise metadata:
{OUTPUT_METADATA}
"""
    )

    # ========================================================
    # EXPECTED TOTAL
    # ========================================================

    expected_noisy_images = (
        len(image_files)
        * len(NOISE_TYPES)
    )

    print_header(
        "VALIDATION"
    )

    print(
        f"Expected noisy images: "
        f"{expected_noisy_images}"
    )

    print(
        f"Generated noisy images: "
        f"{total_generated}"
    )

    if (
        total_generated
        ==
        expected_noisy_images
    ):

        print(
            "✓ Correct number of noisy "
            "images generated"
        )

    else:

        print(
            "⚠ WARNING: Generated image "
            "count does not match expected count"
        )

    # ========================================================
    # STEP COMPLETE
    # ========================================================

    print_header(
        "STEP 3 COMPLETED"
    )

    if (
        failed_images == 0
        and
        total_generated
        == expected_noisy_images
    ):

        print(
            "✓ STEP 3 SUCCESSFULLY COMPLETED"
        )

    else:

        print(
            "⚠ STEP 3 COMPLETED WITH WARNINGS"
        )


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":

    main()