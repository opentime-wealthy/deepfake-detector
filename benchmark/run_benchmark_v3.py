#!/usr/bin/env python3
"""
FakeGuard Benchmark v3 — ReStraV MLP (length-bias free)

Key changes from v2:
  - ReStraV now uses trained MLP classifier (not heuristic)
  - Ensemble weights: restrav_mlp=0.50, temporal=0.15, audio=0.15, c2pa=0.10, metadata=0.10
  - Metadata v2 duration/length signals REMOVED
  - Tests short real videos to verify no false positives
  - Results saved to benchmark/REPORT_v3.md

Pipeline per video:
  1. Frame extraction at 5fps (max 60 frames, first 20s)
  2. ReStraV MLP: DINOv2 26-D trajectory features → MLP classifier
  3. Temporal: optical flow variance
  4. Audio: MFCC variance + spectral flatness
  5. C2PA: content credentials check
  6. Metadata: AI tool keyword detection only (NO duration signal)
  7. Ensemble v3: weighted → final verdict
"""

import os
import sys
import csv
import json
import time
import logging
import gc
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import librosa
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("benchmark_v3")

# ─── Config ───────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).parent
DATASET_DIR = BENCHMARK_DIR / "dataset"
BACKEND_DIR = BENCHMARK_DIR.parent / "backend"
RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = BACKEND_DIR / "models"
MLP_PATH = MODELS_DIR / "restrav_mlp.pth"

SAMPLE_FPS = 5
MAX_FRAMES = 60
MAX_VIDEO_SEC = 20

ENSEMBLE_WEIGHTS = {
    "restrav":   0.50,
    "temporal":  0.15,
    "audio":     0.15,
    "c2pa":      0.10,
    "metadata":  0.10,
}
DECISION_THRESHOLD = 50.0  # neutral threshold (MLP outputs calibrated probabilities)

# ─── DINOv2 + MLP (singletons) ────────────────────────────────────────────────
_HF_MODEL = None
_HF_PROC = None
_MLP = None
_MLP_SCALER = None


def load_models():
    global _HF_MODEL, _HF_PROC, _MLP, _MLP_SCALER
    if _HF_MODEL is None:
        from transformers import AutoModel, AutoImageProcessor
        logger.info("Loading DINOv2 (HF)…")
        _HF_PROC = AutoImageProcessor.from_pretrained("facebook/dinov2-small", use_fast=False)
        _HF_MODEL = AutoModel.from_pretrained("facebook/dinov2-small")
        _HF_MODEL.eval()
        logger.info("DINOv2 loaded OK")

    if _MLP is None and MLP_PATH.exists():
        logger.info(f"Loading MLP from {MLP_PATH}…")
        ck = torch.load(str(MLP_PATH), map_location="cpu", weights_only=False)

        class _RMLP(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 2),
                )
            def forward(self, x): return self.net(x)

        m = _RMLP(ck.get("input_dim", 26))
        m.load_state_dict(ck["model_state"])
        m.eval()
        _MLP = m
        _MLP_SCALER = (
            np.array(ck["scaler_mean"], dtype=np.float32),
            np.array(ck["scaler_scale"], dtype=np.float32),
        )
        logger.info(
            f"MLP loaded (val_acc={ck.get('val_accuracy', 0):.1%}, "
            f"val_auc={ck.get('val_auc', 0):.3f})"
        )


# ─── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path: str):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / SAMPLE_FPS)))
    max_fi = int(fps * MAX_VIDEO_SEC)

    frames, fi = [], 0
    while True:
        ret, f = cap.read()
        if not ret or fi > max_fi:
            break
        if fi % step == 0:
            h, w = f.shape[:2]
            if w > 640:
                f = cv2.resize(f, (640, int(h * 640 / w)))
            frames.append(f)
            if len(frames) >= MAX_FRAMES:
                break
        fi += 1

    cap.release()
    return frames, fps


# ─── ReStraV MLP ──────────────────────────────────────────────────────────────

def compute_features_26d(embeddings: np.ndarray) -> np.ndarray:
    N = len(embeddings)
    if N < 3:
        return np.zeros(26, dtype=np.float32)

    diffs = embeddings[1:] - embeddings[:-1]
    l2 = np.linalg.norm(diffs, axis=1).astype(np.float32)

    angles = []
    for i in range(len(diffs) - 1):
        n1, n2 = l2[i], l2[i + 1]
        if n1 < 1e-8 or n2 < 1e-8:
            angles.append(0.0)
            continue
        c = float(np.dot(diffs[i], diffs[i + 1]) / (n1 * n2))
        angles.append(float(np.arccos(np.clip(c, -1.0, 1.0))))
    angles = np.array(angles, dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms
    cos_sims = np.array([np.dot(normed[i], normed[i+1]) for i in range(N-1)], dtype=np.float32)

    total = float(np.sum(l2)) + 1e-8
    e2e = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
    straight = e2e / total

    def p7(a): lst=a.tolist(); m=np.mean(lst) if lst else 0.0; [lst.append(float(m)) for _ in range(7-len(lst))]; return lst[:7]
    def p6(a): lst=a.tolist(); m=np.mean(lst) if lst else 0.0; [lst.append(float(m)) for _ in range(6-len(lst))]; return lst[:6]
    def s4(a): return [float(np.mean(a)), float(np.min(a)), float(np.max(a)), float(np.var(a))] if len(a) else [0.]*4

    return np.array(p7(l2)+p6(angles)+s4(l2)+s4(angles)+s4(cos_sims)+[straight], dtype=np.float32)


def analyze_restrav(frames: list) -> float:
    if _HF_MODEL is None or len(frames) < 3:
        return 50.0
    try:
        imgs = [Image.fromarray(f[:, :, ::-1].astype(np.uint8)) for f in frames]
        all_embs = []
        for i in range(0, len(imgs), 8):
            inp = _HF_PROC(images=imgs[i:i+8], return_tensors="pt")
            with torch.no_grad():
                out = _HF_MODEL(**inp)
            all_embs.append(out.last_hidden_state[:, 0, :].numpy())
        embs = np.concatenate(all_embs, axis=0).astype(np.float32)

        feat = compute_features_26d(embs)

        if _MLP is not None and _MLP_SCALER is not None:
            mean, scale = _MLP_SCALER
            feat_s = (feat - mean) / (scale + 1e-8)
            x = torch.tensor(feat_s, dtype=torch.float32).unsqueeze(0)
            _MLP.eval()
            with torch.no_grad():
                logits = _MLP(x)
                proba = torch.softmax(logits, dim=1)[0, 1].item()
            return round(proba * 100.0, 2)
        else:
            # Heuristic fallback
            angles_arr = []
            if len(embs) >= 3:
                diffs = embs[1:] - embs[:-1]
                l2 = np.linalg.norm(diffs, axis=1)
                for i in range(len(diffs)-1):
                    n1, n2 = l2[i], l2[i+1]
                    if n1 < 1e-8 or n2 < 1e-8: continue
                    c = float(np.dot(diffs[i], diffs[i+1]) / (n1*n2))
                    angles_arr.append(float(np.arccos(np.clip(c, -1.0, 1.0))))
            mean_t = float(np.mean(angles_arr)) if angles_arr else 0.0
            score = (mean_t - 0.05) / (0.60 - 0.05)
            return round(float(np.clip(score, 0.0, 1.0)) * 100.0, 2)
    except Exception as e:
        logger.warning(f"ReStraV error: {e}")
        return 50.0


# ─── Temporal analysis ────────────────────────────────────────────────────────

def analyze_temporal(frames: list) -> float:
    if len(frames) < 4:
        return 50.0
    try:
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        flows = []
        for i in range(len(gray) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray[i], gray[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flows.append(float(np.mean(mag)))

        mean_flow = float(np.mean(flows))
        var_flow = float(np.var(flows))

        if mean_flow < 0.3:
            return 75.0  # Very static → possibly AI
        if var_flow < 0.01 and mean_flow < 1.0:
            return 65.0  # Uniform motion → possibly AI
        return max(20.0, 50.0 - mean_flow * 2)
    except Exception as e:
        logger.debug(f"Temporal error: {e}")
        return 50.0


# ─── Audio analysis ────────────────────────────────────────────────────────────

def extract_audio(video_path: str):
    try:
        tmp = tempfile.mktemp(suffix=".wav")
        r = subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "22050", "-ac", "1", "-t", "60", tmp, "-y", "-loglevel", "error"],
            capture_output=True, timeout=30
        )
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            y, sr = librosa.load(tmp, sr=22050, duration=60)
            os.unlink(tmp)
            return y, sr
        if os.path.exists(tmp):
            os.unlink(tmp)
    except Exception:
        pass
    return None


def analyze_audio(video_path: str) -> float:
    audio = extract_audio(video_path)
    if audio is None:
        return 40.0  # no audio → slight AI bias (many AI clips are silent)
    y, sr = audio
    if len(y) < sr:
        return 40.0
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfcc, axis=1)))
        spec = librosa.feature.spectral_flatness(y=y)[0]
        flat = float(np.mean(spec))
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr_var = float(np.var(zcr))

        # High MFCC variance → natural speech → real
        # Low variance → TTS/silence → AI
        if mfcc_var > 50:
            return 20.0
        elif mfcc_var > 20:
            return 35.0
        elif mfcc_var > 5:
            return 55.0
        else:
            return 70.0
    except Exception:
        return 40.0


# ─── C2PA check ────────────────────────────────────────────────────────────────

def analyze_c2pa(video_path: str) -> float:
    try:
        result = subprocess.run(
            ["c2patool", str(video_path)],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout + result.stderr
        if "ai_generated" in output.lower() or "generative_ai" in output.lower():
            return 90.0
        if "adobe" in output.lower() or "camera" in output.lower():
            return 15.0
        if "c2pa" in output.lower() or "manifest" in output.lower():
            return 30.0
    except Exception:
        pass
    return 50.0  # No C2PA data → neutral


# ─── Metadata analysis (AI tool keywords only, NO duration) ──────────────────

AI_TOOL_KEYWORDS = [
    "sora", "veo", "pika", "runway", "kling", "genmo", "luma",
    "stable diffusion", "animatediff", "cogvideo", "ai generated",
    "ai-generated", "synthetically generated", "runwayml"
]


def analyze_metadata(video_path: str) -> float:
    """Check only for AI tool signatures. NO video duration signal."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(video_path)],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode != 0:
            return 30.0
        meta = json.loads(r.stdout)
        fmt = meta.get("format", {})
        tags = {k.lower(): v.lower() for k, v in fmt.get("tags", {}).items()}
        text = " ".join(tags.values())
        for kw in AI_TOOL_KEYWORDS:
            if kw in text:
                return 90.0
        return 25.0  # no AI keywords → slightly real-biased
    except Exception:
        return 30.0


# ─── Ensemble ─────────────────────────────────────────────────────────────────

def ensemble(scores: dict) -> float:
    """Weighted ensemble with renormalization for missing analyzers."""
    weights = {k: v for k, v in ENSEMBLE_WEIGHTS.items() if k in scores}
    total_w = sum(weights.values())
    if total_w == 0:
        return 50.0
    return sum(scores[k] * weights[k] / total_w for k in weights)


# ─── Per-video analysis ───────────────────────────────────────────────────────

def analyze_video(video_path: str, label: int) -> dict:
    """
    Analyze one video. Returns result dict.
    label: 0=real, 1=ai_generated
    """
    name = Path(video_path).name
    logger.info(f"\n[{'AI' if label else 'REAL'}] {name}")
    t0 = time.time()

    frames, fps = extract_frames(video_path)
    if not frames:
        logger.warning(f"  No frames extracted")
        return {
            "file": name, "label": label, "final_score": 50.0,
            "verdict": "error", "restrav": 50.0, "error": "no_frames"
        }

    logger.info(f"  {len(frames)} frames @ {SAMPLE_FPS}fps (orig {fps:.1f}fps)")

    scores = {}
    scores["restrav"] = analyze_restrav(frames)
    scores["temporal"] = analyze_temporal(frames)
    scores["audio"] = analyze_audio(video_path)
    scores["c2pa"] = analyze_c2pa(video_path)
    scores["metadata"] = analyze_metadata(video_path)

    final = ensemble(scores)
    predicted = 1 if final >= DECISION_THRESHOLD else 0
    correct = predicted == label

    elapsed = time.time() - t0
    logger.info(
        f"  scores: restrav={scores['restrav']:.1f} temporal={scores['temporal']:.1f} "
        f"audio={scores['audio']:.1f} c2pa={scores['c2pa']:.1f} meta={scores['metadata']:.1f}"
    )
    logger.info(f"  final={final:.1f} → {'AI' if predicted else 'real'} | {'✓' if correct else '✗'} [{elapsed:.1f}s]")

    return {
        "file": name,
        "label": label,
        "final_score": round(final, 2),
        "predicted": predicted,
        "correct": correct,
        "restrav": scores["restrav"],
        "temporal": scores["temporal"],
        "audio": scores["audio"],
        "c2pa": scores["c2pa"],
        "metadata": scores["metadata"],
        "elapsed_sec": round(elapsed, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== FakeGuard Benchmark v3 — ReStraV MLP ===")
    load_models()

    results = []

    # Existing 40-video benchmark dataset
    for split, label in [("ai_generated", 1), ("real", 0)]:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
        for vp in sorted(split_dir.glob("*.mp4")):
            r = analyze_video(str(vp), label)
            r["split"] = split
            results.append(r)

    # Short videos (extra downloads)
    for split, label in [("extra_ai_short", 1), ("extra_real_short", 0)]:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
        for vp in sorted(list(split_dir.glob("*.mp4")) + list(split_dir.glob("*.webm"))):
            r = analyze_video(str(vp), label)
            r["split"] = f"{split}__SHORT"
            results.append(r)

    # ── Metrics ─────────────────────────────────────────────────────────────
    valid = [r for r in results if "predicted" in r]
    y_true = [r["label"] for r in valid]
    y_pred = [r["predicted"] for r in valid]
    y_score = [r["final_score"] for r in valid]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(y_true, y_pred)

    # Short-video focused metrics
    short_results = [r for r in valid if "SHORT" in r.get("split", "")]
    if short_results:
        sh_true = [r["label"] for r in short_results]
        sh_pred = [r["predicted"] for r in short_results]
        sh_acc = accuracy_score(sh_true, sh_pred)
        sh_fp_real = sum(1 for r in short_results if r["label"] == 0 and r["predicted"] == 1)
    else:
        sh_acc = None
        sh_fp_real = None

    logger.info(f"\n{'='*60}")
    logger.info(f"=== RESULTS (n={len(valid)}) ===")
    logger.info(f"Accuracy:  {acc:.3f} ({acc*100:.1f}%)")
    logger.info(f"Precision: {prec:.3f}")
    logger.info(f"Recall:    {rec:.3f}")
    logger.info(f"F1:        {f1:.3f}")
    logger.info(f"AUC-ROC:   {auc:.3f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    if sh_acc is not None:
        logger.info(f"Short video accuracy: {sh_acc:.1%} (false_pos_real={sh_fp_real})")

    # Save CSV
    csv_path = RESULTS_DIR / "benchmark_v3.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "split", "label", "predicted", "correct", "final_score",
            "restrav", "temporal", "audio", "c2pa", "metadata", "elapsed_sec"
        ])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    # Save metrics
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc_roc": float(auc),
        "confusion_matrix": cm.tolist(),
        "n_total": len(valid),
        "n_ai": sum(y_true),
        "n_real": len(y_true) - sum(y_true),
        "short_accuracy": float(sh_acc) if sh_acc is not None else None,
        "short_false_positives_real": sh_fp_real,
        "threshold": DECISION_THRESHOLD,
        "weights": ENSEMBLE_WEIGHTS,
    }

    metrics_path = RESULTS_DIR / "metrics_v3.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nCSV: {csv_path}")
    logger.info(f"Metrics: {metrics_path}")

    return metrics


if __name__ == "__main__":
    metrics = main()

    # Write REPORT_v3.md
    REPORT_PATH = BENCHMARK_DIR / "REPORT_v3.md"
    acc = metrics["accuracy"]
    auc = metrics["auc_roc"]

    sh_acc_str = (
        f"{metrics['short_accuracy']:.1%} (false_pos={metrics['short_false_positives_real']})"
        if metrics.get("short_accuracy") is not None
        else "N/A (no short videos)"
    )

    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    report = f"""# FakeGuard Benchmark v3 — ReStraV MLP

## Summary

| Metric | v3 (MLP) | v2 (Heuristic+Duration) |
|--------|----------|------------------------|
| Accuracy | {acc:.1%} | 100.0% (biased) |
| AUC-ROC | {auc:.3f} | 1.000 (biased) |
| F1 Score | {metrics['f1']:.3f} | 1.000 (biased) |
| Short video accuracy | {sh_acc_str} | N/A |
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
| AI generated (benchmark) | {metrics['n_ai']} |
| Real (benchmark) | {metrics['n_real']} |
| Total | {metrics['n_total']} |

## Confusion Matrix

|  | Predicted AI | Predicted Real |
|--|-------------|----------------|
| **Actual AI** | {tp} (TP) | {fn} (FN) |
| **Actual Real** | {fp} (FP) | {tn} (TN) |

## Short Video Analysis

Short videos ({sh_acc_str}) are the primary target for this v3 update.
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
"""

    REPORT_PATH.write_text(report)
    print(f"\nReport: {REPORT_PATH}")
