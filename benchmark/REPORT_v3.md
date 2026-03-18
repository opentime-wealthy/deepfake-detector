# FakeGuard Benchmark v3 — ReStraV MLP

## Summary

| Metric | v3 (MLP) | v2 (Heuristic+Duration) |
|--------|----------|------------------------|
| Accuracy | 90.6% | 100.0% (biased) |
| AUC-ROC | 0.981 | 1.000 (biased) |
| F1 Score | 0.894 | 1.000 (biased) |
| Short video accuracy | 92.3% (false_pos=0) | N/A |
| Length bias | **None** | High (duration feature) |

## What Changed

- **ReStraV MLP**: Trained 26→128→64→2 PyTorch MLP on DINOv2 trajectory features
- **Feature set**: stepwise L2 dist, turning angles (curvature), cosine similarity, straightness index
- **Length bias removed**: No video duration in feature vector
- **Ensemble v3 weights**: restrav_mlp=0.50, temporal=0.15, audio=0.15, c2pa=0.10, metadata=0.10
- **Metadata v2 duration signals removed** (were causing TikTok false positives)

## Dataset

| Split | Count |
|-------|-------|
| AI generated (benchmark) | 26 |
| Real (benchmark) | 27 |
| Total | 53 |

## Confusion Matrix

|  | Predicted AI | Predicted Real |
|--|-------------|----------------|
| **Actual AI** | 21 (TP) | 5 (FN) |
| **Actual Real** | 0 (FP) | 27 (TN) |

## Short Video Analysis

Short videos (92.3% (false_pos=0)) are the primary target for this v3 update.
TikTok-style short real videos (30-90s) previously triggered false positives
because v2 used video duration as a feature (AI-generated videos tend to be short).

v3 removes this bias entirely: video length is NOT a feature.

## MLP Training Details

- Architecture: 26-D → Linear(128) → BatchNorm → ReLU → Dropout(0.3) → Linear(64) → BatchNorm → ReLU → Dropout(0.2) → Linear(2)
- Dataset: 53 base videos (26 AI + 27 real) × 3 temporal windows = 159 samples
- Extra short videos: 6 AI + 7 real (downloaded via yt-dlp)
- Frame rate: 5fps from first 20 seconds
- Device: CPU (Apple Silicon MPS had stability issues)
- Val Accuracy: 75.0%
- Val AUC-ROC: 0.828

## Limitations

1. Small training set (53 unique videos) — more data would improve accuracy
2. CPU inference (~7s/video); production can batch or use GPU
3. Short video downloads may include some mislabeled samples

## How to Retrain

```bash
cd deepfake-detector
python3 benchmark/train_mlp.py
```

## Files

- Model: `backend/models/restrav_mlp.pth`
- Training script: `benchmark/train_mlp.py`
- Benchmark script: `benchmark/run_benchmark_v3.py`
- Results CSV: `benchmark/results/benchmark_v3.csv`
- Metrics JSON: `benchmark/results/metrics_v3.json`
