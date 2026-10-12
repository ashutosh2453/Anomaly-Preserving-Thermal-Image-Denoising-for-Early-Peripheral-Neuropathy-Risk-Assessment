"""
STEP 2
Thermal Image Diabetic Foot Project
Image Preprocessing and Standardization

Purpose:
1. Read dataset metadata generated in Step 1
2. Validate image files
3. Convert images to RGB
4. Resize all images to 224x224
5. Preserve train/validation split
6. Preserve Control/DM classes
7. Save processed images
8. Generate preprocessing metadata
"""

from pathlib import Path
import sys
import csv
from PIL import Image, ImageOps


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "archive"
    / "ThermoDataBase"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "thermal_image_metadata.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "standardized"
)

OUTPUT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed_metadata.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SIZE = (224, 224)

SUPPORTED_FORMATS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def create_output_directories():
    """
    Create:

    standardized/
        train/
            Control/
            DM/
        val/
            Control/
            DM/
    """

    for split in ["train", "val"]:
        for class_name in ["Control", "DM"]:
            output_dir = OUTPUT_ROOT / split / class_name
            output_dir.mkdir(parents=True, exist_ok=True)


def load_metadata():
    """
    Load metadata generated during Step 1.
    """

    if not METADATA_FILE.exists():
        print("ERROR: Metadata file not found.")
        print(f"Expected location:")
        print(METADATA_FILE)
        sys.exit(1)

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    print(f"Metadata records loaded: {len(rows)}")

    return rows


def validate_image(image_path):
    """
    Check whether an image can be opened
    and verified successfully.
    """

    try:

        with Image.open(image_path) as image:

            image.verify()

        return True, ""

    except Exception as error:

        return False, str(error)


def preprocess_image(source_path, destination_path):
    """
    Preprocess one image.

    Operations:
    - Open image
    - Convert to RGB
    - Resize to 224x224
    - Save as PNG
    """

    try:

        with Image.open(source_path) as image:

            original_width, original_height = image.size

            # ------------------------------------------------
            # Convert to RGB
            # ------------------------------------------------

            image = image.convert("RGB")

            # ------------------------------------------------
            # Resize
            # ------------------------------------------------

            image = image.resize(
                TARGET_SIZE,
                Image.Resampling.LANCZOS
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            image.save(
                destination_path,
                format="PNG"
            )

        return True, original_width, original_height, ""

    except Exception as error:

        return (
            False,
            None,
            None,
            str(error)
        )


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def main():

    print_header(
        "THERMAL IMAGE DIABETIC FOOT\n"
        "STEP 2 - IMAGE PREPROCESSING & STANDARDIZATION"
    )

    print(f"\nProject Root:")
    print(PROJECT_ROOT)

    print(f"\nDataset Root:")
    print(DATASET_ROOT)

    print(f"\nMetadata:")
    print(METADATA_FILE)

    print(f"\nOutput Directory:")
    print(OUTPUT_ROOT)

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    print_header("CHECKING INPUT DATASET")

    if not DATASET_ROOT.exists():

        print("ERROR: Dataset directory not found.")

        sys.exit(1)

    print("✓ Dataset directory found")

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    print_header("CREATING OUTPUT DIRECTORIES")

    create_output_directories()

    print("✓ Output directories created")

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    print_header("LOADING STEP 1 METADATA")

    rows = load_metadata()

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    print_header("STARTING IMAGE PREPROCESSING")

    processed_count = 0
    failed_count = 0
    already_224_count = 0
    resized_count = 0

    preprocessing_records = []

    total = len(rows)

    for index, row in enumerate(rows, start=1):

        source_path = Path(row["image_path"])

        split = row["split"]

        class_name = row["class"]

        label = int(row["label"])

        filename = row["filename"]

        # ----------------------------------------------------
        # Validate source
        # ----------------------------------------------------

        if not source_path.exists():

            failed_count += 1

            print(
                f"[ERROR] Missing file: {source_path}"
            )

            continue

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        valid, error_message = validate_image(
            source_path
        )

        if not valid:

            failed_count += 1

            print(
                f"[ERROR] Invalid image: {filename}"
            )

            continue

        # ----------------------------------------------------
        # Destination
        # ----------------------------------------------------

        destination_dir = (
            OUTPUT_ROOT
            / split
            / class_name
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination_path = (
            destination_dir
            / filename
        )

        # ----------------------------------------------------
        # Process image
        # ----------------------------------------------------

        success, original_width, original_height, error = (
            preprocess_image(
                source_path,
                destination_path
            )
        )

        if not success:

            failed_count += 1

            print(
                f"[ERROR] Failed: {filename}"
            )

            print(
                f"        {error}"
            )

            continue

        # ----------------------------------------------------
        # Count resizing
        # ----------------------------------------------------

        if (
            original_width == TARGET_SIZE[0]
            and
            original_height == TARGET_SIZE[1]
        ):

            already_224_count += 1

        else:

            resized_count += 1

        # ----------------------------------------------------
        # Metadata record
        # ----------------------------------------------------

        preprocessing_records.append(
            {
                "image_path": str(destination_path),
                "original_path": str(source_path),
                "filename": filename,
                "split": split,
                "class": class_name,
                "label": label,
                "original_width": original_width,
                "original_height": original_height,
                "processed_width": TARGET_SIZE[0],
                "processed_height": TARGET_SIZE[1],
                "channels": 3,
                "format": "PNG",
                "preprocessing": "RGB + Resize 224x224"
            }
        )

        processed_count += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % 100 == 0
            or index == total
        ):

            print(
                f"Processed: {index}/{total}"
            )


    # ========================================================
    # SAVE PREPROCESSING METADATA
    # ========================================================

    print_header("SAVING PREPROCESSING METADATA")

    if preprocessing_records:

        fieldnames = list(
            preprocessing_records[0].keys()
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
                preprocessing_records
            )

        print(
            "✓ Preprocessing metadata saved:"
        )

        print(OUTPUT_METADATA)

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print_header("PREPROCESSING VALIDATION")

    print(
        f"Original records:       {total}"
    )

    print(
        f"Successfully processed: {processed_count}"
    )

    print(
        f"Failed images:          {failed_count}"
    )

    print(
        f"Already 224x224:        {already_224_count}"
    )

    print(
        f"Images resized:         {resized_count}"
    )

    # --------------------------------------------------------
    # Verify output count
    # --------------------------------------------------------

    output_count = 0

    for split in ["train", "val"]:

        for class_name in ["Control", "DM"]:

            directory = (
                OUTPUT_ROOT
                / split
                / class_name
            )

            if directory.exists():

                output_count += len(
                    list(
                        directory.glob("*.png")
                    )
                )

    print(
        f"Output images:          {output_count}"
    )

    # --------------------------------------------------------
    # Verify expected size
    # --------------------------------------------------------

    print_header("CHECKING PROCESSED IMAGE DIMENSIONS")

    dimension_errors = 0

    for record in preprocessing_records:

        processed_path = Path(
            record["image_path"]
        )

        try:

            with Image.open(processed_path) as image:

                width, height = image.size

                if (
                    width != TARGET_SIZE[0]
                    or
                    height != TARGET_SIZE[1]
                ):

                    dimension_errors += 1

                    print(
                        f"[ERROR] Wrong size: "
                        f"{processed_path}"
                    )

        except Exception as error:

            dimension_errors += 1

            print(
                f"[ERROR] Cannot open: "
                f"{processed_path}"
            )

    if dimension_errors == 0:

        print(
            "✓ All processed images are "
            "224x224"
        )

    else:

        print(
            f"WARNING: {dimension_errors} "
            f"dimension errors found"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_header("STEP 2 COMPLETED")

    print(
        f"""
Dataset:
ThermoDataBase

Original images:
{total}

Processed images:
{processed_count}

Failed images:
{failed_count}

Images already 224x224:
{already_224_count}

Images resized:
{resized_count}

Output images:
{output_count}

Target dimensions:
224 x 224 x 3

Train/Validation split:
Preserved

Classes:
Control = 0
DM      = 1

Preprocessing metadata:
{OUTPUT_METADATA}
"""
    )

    if (
        processed_count == total
        and
        failed_count == 0
        and
        output_count == total
        and
        dimension_errors == 0
    ):

        print(
            "✓ STEP 2 SUCCESSFULLY COMPLETED"
        )

    else:

        print(
            "⚠ STEP 2 COMPLETED WITH WARNINGS"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()