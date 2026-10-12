# Dataset

The raw thermal image dataset (`archive/ThermoDataBase/`) and processed image directories (`data/processed/`) are excluded from version control due to size.

## Dataset Structure

```
archive/ThermoDataBase/
├── train/
│   ├── Control Group/   # 720 images
│   └── DM Group/        # 724 images
└── val/
    ├── Control Group/   # 170 images
    └── DM Group/        # 252 images
```

Total: **1,866 images** (224×224 PNG) — 976 DM, 890 Control.

## Reproducing Processed Data

Once you have the raw dataset in place, run the pipeline steps in order:

```bash
# Step 2: Standardize images to 224x224 PNG
python src/preprocessing/preprocess_images.py

# Step 3: Generate noisy variants
python src/denoising/noise_generation.py

# Step 4: Apply denoising filters
python src/denoising/anomaly_preserving_denoising_fast.py

# Step 5: Extract thermal features
python src/features/thermal_features.py
```

## What Is Tracked

- `data/features/spark_feature_summary.csv` — Spark-computed feature statistics
- `data/features/selected_features.csv` — 7 Spark-selected features for GNN
