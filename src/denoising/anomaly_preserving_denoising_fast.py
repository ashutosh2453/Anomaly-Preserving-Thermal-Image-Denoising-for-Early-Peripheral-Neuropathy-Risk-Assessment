"""
STEP 4 - FAST ANOMALY-PRESERVING DENOISING

Thermal Image Diabetic Foot Project

This version uses optimized OpenCV filters.

Methods:
    - Median Filter
    - Gaussian Filter
    - Bilateral Filter

Noise:
    - Gaussian
    - Salt-and-Pepper
    - Speckle

Metrics:
    - MSE
    - PSNR
    - SSIM
    - Anomaly Recall
    - Anomaly IoU
    - Intensity Preservation
    - Anomaly Preservation Score

TEST MODE:
    Processes only 100 images first.

After successful testing:
    Change MAX_IMAGES = None
    to process all 1,866 images.
"""

from pathlib import Path
import csv
import math

import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "standardized"
)

NOISY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "noisy"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "denoised_fast"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

OUTPUT_METADATA = (
    RESULTS_ROOT
    / "denoising_fast_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

NOISE_TYPES = [
    "gaussian",
    "salt_pepper",
    "speckle"
]

DENOISING_METHODS = [
    "median",
    "gaussian",
    "bilateral"
]

# ------------------------------------------------------------
# IMPORTANT:
# First run = 100 images
#
# After successful test:
#
# MAX_IMAGES = None
#
# ------------------------------------------------------------

MAX_IMAGES = None

# Thermal anomaly threshold
ANOMALY_PERCENTILE = 90.0


# ============================================================
# PRINT HEADER
# ============================================================

def print_header(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Could not read image: {path}"
        )

    # OpenCV loads BGR.
    # Convert to RGB.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(image, path):

    # RGB → BGR
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(path),
        image_bgr
    )


# ============================================================
# GRAYSCALE THERMAL INTENSITY
# ============================================================

def to_gray(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    return gray.astype(
        np.float32
    ) / 255.0


# ============================================================
# MEDIAN FILTER
# ============================================================

def median_denoise(image):

    return cv2.medianBlur(
        image,
        3
    )


# ============================================================
# GAUSSIAN FILTER
# ============================================================

def gaussian_denoise(image):

    return cv2.GaussianBlur(
        image,
        (5, 5),
        1.0
    )


# ============================================================
# BILATERAL FILTER
# ============================================================

def bilateral_denoise(image):

    return cv2.bilateralFilter(
        image,
        5,
        30,
        30
    )


# ============================================================
# SELECT DENOISING METHOD
# ============================================================

def apply_denoising(
    image,
    method
):

    if method == "median":

        return median_denoise(
            image
        )

    if method == "gaussian":

        return gaussian_denoise(
            image
        )

    if method == "bilateral":

        return bilateral_denoise(
            image
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# ============================================================
# MSE
# ============================================================

def calculate_mse(
    original,
    processed
):

    return float(
        np.mean(
            (
                original
                -
                processed
            ) ** 2
        )
    )


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    original,
    processed
):

    mse = calculate_mse(
        original,
        processed
    )

    if mse == 0:

        return 100.0

    return float(
        10
        *
        math.log10(
            1.0 / mse
        )
    )


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(
    original,
    processed
):

    original = original.astype(
        np.float64
    )

    processed = processed.astype(
        np.float64
    )

    mean_original = np.mean(
        original
    )

    mean_processed = np.mean(
        processed
    )

    variance_original = np.var(
        original
    )

    variance_processed = np.var(
        processed
    )

    covariance = np.mean(
        (
            original
            -
            mean_original
        )
        *
        (
            processed
            -
            mean_processed
        )
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (
        (
            2
            *
            mean_original
            *
            mean_processed
            +
            c1
        )
        *
        (
            2
            *
            covariance
            +
            c2
        )
    )

    denominator = (
        (
            mean_original ** 2
            +
            mean_processed ** 2
            +
            c1
        )
        *
        (
            variance_original
            +
            variance_processed
            +
            c2
        )
    )

    if denominator == 0:

        return 1.0

    return float(
        numerator
        /
        denominator
    )


# ============================================================
# ANOMALY MASK
# ============================================================

def create_anomaly_mask(
    clean_gray
):

    threshold = np.percentile(
        clean_gray,
        ANOMALY_PERCENTILE
    )

    mask = (
        clean_gray
        >=
        threshold
    )

    return (
        mask,
        float(threshold)
    )


# ============================================================
# ANOMALY METRICS
# ============================================================

def calculate_anomaly_metrics(
    clean_gray,
    denoised_gray,
    clean_mask
):

    # --------------------------------------------------------
    # Use the CLEAN image threshold
    # --------------------------------------------------------

    threshold = np.percentile(
        clean_gray,
        ANOMALY_PERCENTILE
    )

    denoised_mask = (
        denoised_gray
        >=
        threshold
    )

    # --------------------------------------------------------
    # Intersection / union
    # --------------------------------------------------------

    intersection = np.logical_and(
        clean_mask,
        denoised_mask
    )

    union = np.logical_or(
        clean_mask,
        denoised_mask
    )

    clean_pixels = np.sum(
        clean_mask
    )

    denoised_pixels = np.sum(
        denoised_mask
    )

    intersection_pixels = np.sum(
        intersection
    )

    union_pixels = np.sum(
        union
    )

    # --------------------------------------------------------
    # Anomaly recall
    # --------------------------------------------------------

    if clean_pixels > 0:

        anomaly_recall = (
            intersection_pixels
            /
            clean_pixels
        )

    else:

        anomaly_recall = 0.0

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if union_pixels > 0:

        anomaly_iou = (
            intersection_pixels
            /
            union_pixels
        )

    else:

        anomaly_iou = 1.0

    # --------------------------------------------------------
    # Intensity preservation
    # --------------------------------------------------------

    if clean_pixels > 0:

        clean_mean = np.mean(
            clean_gray[
                clean_mask
            ]
        )

        denoised_mean = np.mean(
            denoised_gray[
                clean_mask
            ]
        )

        intensity_error = abs(
            clean_mean
            -
            denoised_mean
        )

        intensity_preservation = (
            1.0
            -
            intensity_error
        )

        intensity_preservation = max(
            0.0,
            min(
                1.0,
                intensity_preservation
            )
        )

    else:

        intensity_preservation = 0.0

    # --------------------------------------------------------
    # Area preservation
    # --------------------------------------------------------

    clean_area_ratio = (
        clean_pixels
        /
        clean_gray.size
    )

    denoised_area_ratio = (
        denoised_pixels
        /
        denoised_gray.size
    )

    if clean_area_ratio > 0:

        area_preservation = (
            1.0
            -
            abs(
                clean_area_ratio
                -
                denoised_area_ratio
            )
            /
            clean_area_ratio
        )

        area_preservation = max(
            0.0,
            min(
                1.0,
                area_preservation
            )
        )

    else:

        area_preservation = 0.0

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    anomaly_score = (
        0.40 * anomaly_recall
        +
        0.30 * anomaly_iou
        +
        0.20 * intensity_preservation
        +
        0.10 * area_preservation
    )

    return {

        "anomaly_threshold":
            float(threshold),

        "clean_anomaly_pixels":
            int(clean_pixels),

        "denoised_anomaly_pixels":
            int(denoised_pixels),

        "clean_anomaly_area_ratio":
            float(clean_area_ratio),

        "denoised_anomaly_area_ratio":
            float(denoised_area_ratio),

        "anomaly_recall":
            float(anomaly_recall),

        "anomaly_iou":
            float(anomaly_iou),

        "intensity_preservation":
            float(
                intensity_preservation
            ),

        "area_preservation":
            float(
                area_preservation
            ),

        "anomaly_preservation_score":
            float(
                anomaly_score
            )
    }


# ============================================================
# GET NOISY IMAGE PATH
# ============================================================

def get_noisy_path(
    split,
    class_name,
    filename,
    noise_type
):

    stem = Path(
        filename
    ).stem

    noisy_filename = (
        f"{stem}_{noise_type}.png"
    )

    return (
        NOISY_ROOT
        / split
        / class_name
        / noisy_filename
    )


# ============================================================
# GET OUTPUT PATH
# ============================================================

def get_output_path(
    split,
    class_name,
    filename,
    noise_type,
    method
):

    stem = Path(
        filename
    ).stem

    output_directory = (
        OUTPUT_ROOT
        / split
        / class_name
        / noise_type
        / method
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_filename = (
        f"{stem}_"
        f"{noise_type}_"
        f"{method}.png"
    )

    return (
        output_directory
        / output_filename
    )


# ============================================================
# FIND CLEAN IMAGES
# ============================================================

def find_clean_images():

    image_files = []

    for split in [
        "train",
        "val"
    ]:

        for class_name in [
            "Control",
            "DM"
        ]:

            directory = (
                CLEAN_ROOT
                / split
                / class_name
            )

            if not directory.exists():

                continue

            files = sorted(
                directory.glob(
                    "*.png"
                )
            )

            for file in files:

                image_files.append(
                    (
                        file,
                        split,
                        class_name
                    )
                )

    return image_files


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "THERMAL IMAGE DIABETIC FOOT"
        "\nSTEP 4 - FAST ANOMALY-PRESERVING DENOISING"
    )

    print(
        f"\nClean dataset:\n{CLEAN_ROOT}"
    )

    print(
        f"\nNoisy dataset:\n{NOISY_ROOT}"
    )

    print(
        f"\nOutput:\n{OUTPUT_ROOT}"
    )

    print(
        f"\nResults:\n{OUTPUT_METADATA}"
    )

    print(
        f"\nMaximum images for this run: "
        f"{MAX_IMAGES}"
    )

    # ========================================================
    # CHECK INPUTS
    # ========================================================

    print_header(
        "CHECKING INPUT DATA"
    )

    if not CLEAN_ROOT.exists():

        print(
            "ERROR: Standardized dataset "
            "does not exist."
        )

        return

    if not NOISY_ROOT.exists():

        print(
            "ERROR: Noisy dataset "
            "does not exist."
        )

        return

    print(
        "✓ Standardized dataset found"
    )

    print(
        "✓ Noisy dataset found"
    )

    # ========================================================
    # FIND IMAGES
    # ========================================================

    image_files = find_clean_images()

    print(
        f"Total clean images available: "
        f"{len(image_files)}"
    )

    if MAX_IMAGES is not None:

        image_files = image_files[
            :MAX_IMAGES
        ]

    print(
        f"Images selected for this run: "
        f"{len(image_files)}"
    )

    # ========================================================
    # EXPECTED EXPERIMENTS
    # ========================================================

    expected_experiments = (
        len(image_files)
        *
        len(NOISE_TYPES)
        *
        len(DENOISING_METHODS)
    )

    print(
        f"Expected experiments: "
        f"{expected_experiments}"
    )

    # ========================================================
    # PROCESS
    # ========================================================

    print_header(
        "RUNNING FAST DENOISING EXPERIMENT"
    )

    records = []

    completed = 0

    failed = 0

    for index, (
        clean_path,
        split,
        class_name
    ) in enumerate(
        image_files,
        start=1
    ):

        try:

            clean_image = load_image(
                clean_path
            )

            clean_gray = to_gray(
                clean_image
            )

            clean_mask, _ = (
                create_anomaly_mask(
                    clean_gray
                )
            )

            for noise_type in NOISE_TYPES:

                noisy_path = get_noisy_path(
                    split,
                    class_name,
                    clean_path.name,
                    noise_type
                )

                if not noisy_path.exists():

                    print(
                        f"[WARNING] Missing noisy "
                        f"image: {noisy_path.name}"
                    )

                    failed += 1

                    continue

                noisy_image = load_image(
                    noisy_path
                )

                noisy_gray = to_gray(
                    noisy_image
                )

                # ------------------------------------------------
                # Apply each denoising method
                # ------------------------------------------------

                for method in DENOISING_METHODS:

                    denoised_image = (
                        apply_denoising(
                            noisy_image,
                            method
                        )
                    )

                    denoised_gray = to_gray(
                        denoised_image
                    )

                    # ------------------------------------------------
                    # Image quality metrics
                    # ------------------------------------------------

                    mse = calculate_mse(
                        clean_gray,
                        denoised_gray
                    )

                    psnr = calculate_psnr(
                        clean_gray,
                        denoised_gray
                    )

                    ssim = calculate_ssim(
                        clean_gray,
                        denoised_gray
                    )

                    # ------------------------------------------------
                    # Noisy baseline metrics
                    # ------------------------------------------------

                    noisy_mse = calculate_mse(
                        clean_gray,
                        noisy_gray
                    )

                    noisy_psnr = calculate_psnr(
                        clean_gray,
                        noisy_gray
                    )

                    noisy_ssim = calculate_ssim(
                        clean_gray,
                        noisy_gray
                    )

                    # ------------------------------------------------
                    # Anomaly metrics
                    # ------------------------------------------------

                    anomaly_metrics = (
                        calculate_anomaly_metrics(
                            clean_gray,
                            denoised_gray,
                            clean_mask
                        )
                    )

                    # ------------------------------------------------
                    # Save image
                    # ------------------------------------------------

                    output_path = (
                        get_output_path(
                            split,
                            class_name,
                            clean_path.name,
                            noise_type,
                            method
                        )
                    )

                    save_image(
                        denoised_image,
                        output_path
                    )

                    # ------------------------------------------------
                    # Result record
                    # ------------------------------------------------

                    record = {

                        "original_path":
                            str(clean_path),

                        "noisy_path":
                            str(noisy_path),

                        "denoised_path":
                            str(output_path),

                        "filename":
                            clean_path.name,

                        "split":
                            split,

                        "class":
                            class_name,

                        "label":
                            (
                                0
                                if class_name == "Control"
                                else 1
                            ),

                        "noise_type":
                            noise_type,

                        "denoising_method":
                            method,

                        "width":
                            224,

                        "height":
                            224,

                        "channels":
                            3,

                        "noisy_mse":
                            noisy_mse,

                        "noisy_psnr":
                            noisy_psnr,

                        "noisy_ssim":
                            noisy_ssim,

                        "mse":
                            mse,

                        "psnr":
                            psnr,

                        "ssim":
                            ssim
                    }

                    record.update(
                        anomaly_metrics
                    )

                    records.append(
                        record
                    )

                    completed += 1

        except Exception as error:

            failed += 1

            print(
                f"[ERROR] {clean_path.name}"
            )

            print(
                f"        {error}"
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % 10 == 0
            or
            index == len(image_files)
        ):

            print(
                f"Images processed: "
                f"{index}/{len(image_files)}"
            )

            print(
                f"Experiments completed: "
                f"{completed}/{expected_experiments}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print_header(
        "SAVING RESULTS"
    )

    if records:

        RESULTS_ROOT.mkdir(
            parents=True,
            exist_ok=True
        )

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
            "✓ Results saved:"
        )

        print(
            OUTPUT_METADATA
        )

    # ========================================================
    # PERFORMANCE SUMMARY
    # ========================================================

    print_header(
        "DENOISING PERFORMANCE"
    )

    for noise_type in NOISE_TYPES:

        print(
            f"\n{noise_type.upper()}"
        )

        for method in DENOISING_METHODS:

            subset = [

                record
                for record in records

                if (
                    record["noise_type"]
                    ==
                    noise_type
                    and
                    record[
                        "denoising_method"
                    ]
                    ==
                    method
                )
            ]

            if not subset:

                continue

            avg_psnr = np.mean(
                [
                    r["psnr"]
                    for r in subset
                ]
            )

            avg_ssim = np.mean(
                [
                    r["ssim"]
                    for r in subset
                ]
            )

            avg_anomaly = np.mean(
                [
                    r[
                        "anomaly_preservation_score"
                    ]
                    for r in subset
                ]
            )

            print(
                f"  {method:<10}"
                f" PSNR={avg_psnr:.4f}"
                f" | SSIM={avg_ssim:.4f}"
                f" | Anomaly="
                f"{avg_anomaly:.4f}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_header(
        "STEP 4 TEST SUMMARY"
    )

    print(
        f"""
Images selected:
{len(image_files)}

Noise types:
{len(NOISE_TYPES)}

Denoising methods:
{len(DENOISING_METHODS)}

Expected experiments:
{expected_experiments}

Completed experiments:
{completed}

Failed experiments:
{failed}

Results file:
{OUTPUT_METADATA}
"""
    )

    if (
        completed
        ==
        expected_experiments
        and
        failed == 0
    ):

        print(
            "✓ STEP 4 TEST SUCCESSFULLY COMPLETED"
        )

        print(
            "\nThe optimized pipeline is working."
        )

        print(
            "Next: change MAX_IMAGES = None "
            "for the full dataset."
        )

    else:

        print(
            "⚠ STEP 4 TEST COMPLETED "
            "WITH WARNINGS"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()