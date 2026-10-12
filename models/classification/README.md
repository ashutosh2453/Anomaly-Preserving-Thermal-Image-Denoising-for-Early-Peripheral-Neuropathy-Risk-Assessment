# Model Checkpoints

Large Swin Transformer model checkpoints (>100 MB) exceed GitHub's file size limit and are excluded from version control.

## Available in Repository

| File | Size | Description |
|:-----|-----:|:------------|
| `best_swin_gnn_improved.pth` | ~1.7 MB | Best Improved Swin + GNN checkpoint (k=5, epoch 9) |
| `best_swin_gnn_thermal.pth` | ~0.5 MB | Swin + GCN baseline checkpoint |

## Excluded (Too Large)

| File | Size | Description |
|:-----|-----:|:------------|
| `best_swin_transformer_thermal.pth` | ~315 MB | Fine-tuned Swin Transformer backbone |
| `step11_swin_gaussian-bilateral.pth` | ~105 MB | Step 11 — Gaussian + Bilateral condition |
| `step11_swin_gaussian-gaussian.pth` | ~105 MB | Step 11 — Gaussian + Gaussian condition |
| `step11_swin_gaussian-median.pth` | ~105 MB | Step 11 — Gaussian + Median condition |
| `step11_swin_gaussian-noisy.pth` | ~105 MB | Step 11 — Gaussian Noise condition |
| `step11_swin_clean.pth` | ~105 MB | Step 11 — Clean condition |

## Reproducing

Re-train the Swin Transformer backbone:
```bash
python src/classification/train_swin_transformer.py
```

Re-train the Improved Swin + GNN:
```bash
python src/classification/improved_swin_gnn.py
```
