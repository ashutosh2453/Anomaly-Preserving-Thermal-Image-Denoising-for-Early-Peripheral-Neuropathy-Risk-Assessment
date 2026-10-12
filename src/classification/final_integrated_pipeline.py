import os
import csv
import time
import numpy as np
from PIL import Image

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# STEP 12 - FINAL INTEGRATED PIPELINE
# Thermal Image -> Preprocessing -> Gaussian Noise
# -> Bilateral Denoising -> Swin Transformer -> Prediction
# ============================================================

print("=" * 70)
print("STEP 12 - FINAL INTEGRATED THERMAL PIPELINE")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

STANDARDIZED_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "standardized"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "classification",
    "step11_swin_gaussian-bilateral.pth"
)

RESULTS_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

IMAGES_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "images",
    "final_pipeline_examples"
)

os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(IMAGES_ROOT, exist_ok=True)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nPyTorch:")
print(torch.__version__)

print("\nCUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

print("\nTraining / inference device:")
print(device)


# ============================================================
# 3. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

print("\nModel:")
print(MODEL_PATH)


# ============================================================
# 4. CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = {
    0: "Control",
    1: "DM"
}


# ============================================================
# 5. SWIN TRANSFORMER
# ============================================================

print("\nLoading Swin Transformer...")

model = models.swin_t(weights=None)

# Step 11 uses binary classification
in_features = model.head.in_features

model.head = nn.Linear(
    in_features,
    2
)


# ============================================================
# 6. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

# Handle different possible checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# Remove possible "module." prefix
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key.replace("module.", "", 1)

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model = model.to(device)
model.eval()

print("✓ Swin Transformer checkpoint loaded")


# ============================================================
# 7. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 8. GAUSSIAN NOISE
# ============================================================

GAUSSIAN_SIGMA = 0.05


def add_gaussian_noise(image):
    """
    Add controlled Gaussian noise.

    Input:
        RGB uint8 image

    Output:
        RGB uint8 noisy image
    """

    image_float = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    noise = np.random.normal(
        0,
        GAUSSIAN_SIGMA,
        image_float.shape
    ).astype(np.float32)

    noisy = image_float + noise

    noisy = np.clip(
        noisy,
        0.0,
        1.0
    )

    noisy = (
        noisy * 255.0
    ).astype(np.uint8)

    return noisy


# ============================================================
# 9. BILATERAL DENOISING
# ============================================================

def bilateral_denoise(image):
    """
    Bilateral filtering preserves edges while
    reducing noise.
    """

    denoised = cv2.bilateralFilter(
        image,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    return denoised


# ============================================================
# 10. THERMAL FEATURES
# ============================================================

def extract_thermal_features(image):
    """
    Extract the same basic thermal image features
    used in the project.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    mean_intensity = float(
        np.mean(gray)
    )

    min_intensity = float(
        np.min(gray)
    )

    max_intensity = float(
        np.max(gray)
    )

    std_intensity = float(
        np.std(gray)
    )

    percentile_90 = float(
        np.percentile(gray, 90)
    )

    hotspot_mask = (
        gray >= percentile_90
    )

    hotspot_pixels = int(
        np.sum(hotspot_mask)
    )

    total_pixels = gray.size

    hotspot_area_ratio = (
        hotspot_pixels / total_pixels
    )

    if hotspot_pixels > 0:
        hotspot_mean_intensity = float(
            np.mean(
                gray[hotspot_mask]
            )
        )
    else:
        hotspot_mean_intensity = 0.0

    height, width = gray.shape

    midpoint = width // 2

    left = gray[:, :midpoint]
    right = gray[:, midpoint:]

    left_mean = float(
        np.mean(left)
    )

    right_mean = float(
        np.mean(right)
    )

    absolute_asymmetry = abs(
        left_mean - right_mean
    )

    denominator = (
        (left_mean + right_mean) / 2.0
    )

    if denominator > 0:
        relative_asymmetry = (
            absolute_asymmetry /
            denominator
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
# 11. MODEL PREDICTION
# ============================================================

def predict(image):

    pil_image = Image.fromarray(
        image
    ).convert("RGB")

    tensor = transform(
        pil_image
    ).unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        probability_dm = float(
            probabilities[0, 1].item()
        )

    return prediction, probability_dm


# ============================================================
# 12. DATASET COLLECTION
# ============================================================

records = []

for split in ["train", "val"]:

    for class_name in ["Control", "DM"]:

        folder = os.path.join(
            STANDARDIZED_ROOT,
            split,
            class_name
        )

        if not os.path.exists(folder):
            print(
                f"Warning: folder not found: {folder}"
            )
            continue

        for filename in sorted(
            os.listdir(folder)
        ):

            if not filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                continue

            image_path = os.path.join(
                folder,
                filename
            )

            label = (
                1 if class_name == "DM"
                else 0
            )

            records.append({
                "image_path": image_path,
                "filename": filename,
                "split": split,
                "actual_class": class_name,
                "label": label
            })


print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

print(
    f"Total images: {len(records)}"
)

train_count = sum(
    r["split"] == "train"
    for r in records
)

val_count = sum(
    r["split"] == "val"
    for r in records
)

print(
    f"Training images: {train_count}"
)

print(
    f"Validation images: {val_count}"
)


# ============================================================
# 13. FINAL PIPELINE
# ============================================================

results = []

start_time = time.time()

print("\n" + "=" * 70)
print("RUNNING FINAL PIPELINE")
print("=" * 70)


for index, record in enumerate(
    records,
    start=1
):

    try:

        # ----------------------------------------------------
        # Original standardized image
        # ----------------------------------------------------

        image = Image.open(
            record["image_path"]
        ).convert("RGB")

        image = np.asarray(
            image,
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # Gaussian noise
        # ----------------------------------------------------

        noisy_image = add_gaussian_noise(
            image
        )

        # ----------------------------------------------------
        # Bilateral denoising
        # ----------------------------------------------------

        denoised_image = bilateral_denoise(
            noisy_image
        )

        # ----------------------------------------------------
        # Thermal feature extraction
        # ----------------------------------------------------

        features = extract_thermal_features(
            denoised_image
        )

        # ----------------------------------------------------
        # Swin prediction
        # ----------------------------------------------------

        prediction, probability_dm = predict(
            denoised_image
        )

        predicted_class = CLASS_NAMES[
            prediction
        ]

        # ----------------------------------------------------
        # Correct prediction
        # ----------------------------------------------------

        correct = (
            prediction == record["label"]
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "filename": record["filename"],
            "split": record["split"],
            "actual_class": record["actual_class"],
            "predicted_class": predicted_class,
            "actual_label": record["label"],
            "predicted_label": prediction,
            "probability_dm": probability_dm,
            "correct": int(correct)
        }

        result.update(features)

        results.append(result)

        # ----------------------------------------------------
        # Save first few examples
        # ----------------------------------------------------

        if index <= 10:

            example_name = (
                f"{index:02d}_"
                f"{record['filename']}"
            )

            example_path = os.path.join(
                IMAGES_ROOT,
                example_name
            )

            Image.fromarray(
                denoised_image
            ).save(
                example_path
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if index % 100 == 0:

            elapsed = (
                time.time() -
                start_time
            )

            print(
                f"Processed "
                f"{index}/{len(records)} "
                f"images | "
                f"Time: {elapsed:.1f}s"
            )

    except Exception as e:

        print(
            f"\nERROR: {record['filename']}"
        )

        print(e)


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

predictions_path = os.path.join(
    RESULTS_ROOT,
    "final_predictions.csv"
)

if results:

    fieldnames = list(
        results[0].keys()
    )

    with open(
        predictions_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# 15. FINAL METRICS
# ============================================================

total = len(results)

correct = sum(
    r["correct"]
    for r in results
)

accuracy = (
    correct / total
    if total > 0
    else 0.0
)

control_results = [
    r for r in results
    if r["actual_label"] == 0
]

dm_results = [
    r for r in results
    if r["actual_label"] == 1
]

true_negative = sum(
    r["actual_label"] == 0
    and r["predicted_label"] == 0
    for r in results
)

false_positive = sum(
    r["actual_label"] == 0
    and r["predicted_label"] == 1
    for r in results
)

false_negative = sum(
    r["actual_label"] == 1
    and r["predicted_label"] == 0
    for r in results
)

true_positive = sum(
    r["actual_label"] == 1
    and r["predicted_label"] == 1
    for r in results
)

precision = (
    true_positive /
    (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0.0
)

recall = (
    true_positive /
    (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0.0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0.0
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

confusion_matrix = [
    [true_negative, false_positive],
    [false_negative, true_positive]
]


# ============================================================
# 17. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_ROOT,
    "final_pipeline_summary.csv"
)

summary_rows = [

    {
        "metric": "Total Images",
        "value": total
    },

    {
        "metric": "Correct Predictions",
        "value": correct
    },

    {
        "metric": "Accuracy",
        "value": accuracy
    },

    {
        "metric": "Precision",
        "value": precision
    },

    {
        "metric": "Recall",
        "value": recall
    },

    {
        "metric": "F1",
        "value": f1
    },

    {
        "metric": "True Negative",
        "value": true_negative
    },

    {
        "metric": "False Positive",
        "value": false_positive
    },

    {
        "metric": "False Negative",
        "value": false_negative
    },

    {
        "metric": "True Positive",
        "value": true_positive
    }

]


with open(
    summary_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["metric", "value"]
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# 18. DISPLAY RESULTS
# ============================================================

total_time = (
    time.time() -
    start_time
)

print("\n" + "=" * 70)
print("STEP 12 FINAL PIPELINE RESULTS")
print("=" * 70)

print(
    f"Images processed : {total}"
)

print(
    f"Correct          : {correct}"
)

print(
    f"Accuracy         : {accuracy:.4f}"
)

print(
    f"Precision        : {precision:.4f}"
)

print(
    f"Recall           : {recall:.4f}"
)

print(
    f"F1 Score         : {f1:.4f}"
)

print("\nConfusion Matrix:")

print(
    f"[[{true_negative:3d} {false_positive:3d}]"
)

print(
    f" [{false_negative:3d} {true_positive:3d}]]"
)

print(
    f"\nTotal Time       : {total_time:.2f} sec"
)


# ============================================================
# 19. OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 OUTPUT FILES")
print("=" * 70)

print(
    "\nPredictions:"
)

print(
    predictions_path
)

print(
    "\nSummary:"
)

print(
    summary_path
)

print(
    "\nExample processed images:"
)

print(
    IMAGES_ROOT
)


print("\n" + "=" * 70)
print("✓ STEP 12 COMPLETED SUCCESSFULLY")
print("=" * 70)