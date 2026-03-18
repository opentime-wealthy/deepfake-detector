#!/usr/bin/env python3
"""
FakeGuard — ReStraV MLP Training Script
Trains a PyTorch MLP on DINOv2 trajectory features (no video-length bias).

Uses HuggingFace transformers for DINOv2 (avoids torch.hub SIGSEGV on macOS).

Feature vector (26-D, length-independent):
  [0:7]    7 stepwise L2 distances (early frames)
  [7:13]   6 turning angles (curvature)
  [13:17]  4 L2 distance stats (mean, min, max, var)
  [17:21]  4 angle stats (mean, min, max, var)
  [21:25]  4 cosine similarity stats (mean, min, max, var)
  [25]     1 straightness index (end-to-end / total path)

MLP: 26 → 128 → 64 → 2
Save: backend/models/restrav_mlp.pth
"""

import os
import sys
import json
import time
import logging
import subprocess
import gc
from pathlib import Path
from typing import List, Optional, Tuple

# Disable multiprocessing-related crashes on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_mlp")

# (env vars set at top of file)

# ─── Paths ───────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).parent
DATASET_DIR = BENCHMARK_DIR / "dataset"
BACKEND_DIR = BENCHMARK_DIR.parent / "backend"
MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MLP_PATH = MODELS_DIR / "restrav_mlp.pth"
FEAT_DIM = 26

# ─── Config ──────────────────────────────────────────────────────────────────
DEVICE = "cpu"          # HF DINOv2 is stable on CPU
SAMPLE_FPS = 5          # 5fps for proper trajectory analysis (paper uses 12fps)
MAX_FRAMES = 60         # 5fps × 12sec = 60 frames
MAX_VIDEO_SEC = 20      # Only need first 20 seconds for good trajectory
BATCH_SIZE = 8
N_EPOCHS = 300
LR = 5e-4
WEIGHT_DECAY = 5e-3     # stronger regularization (small dataset)

# ─── DINOv2 via HuggingFace (singleton) ──────────────────────────────────────
_HF_MODEL = None
_HF_PROCESSOR = None


def load_dinov2_hf():
    """Load DINOv2 via HuggingFace transformers (more stable on macOS)."""
    global _HF_MODEL, _HF_PROCESSOR
    if _HF_MODEL is not None:
        return _HF_MODEL, _HF_PROCESSOR
    logger.info("Loading DINOv2 via HuggingFace (facebook/dinov2-small)…")
    try:
        from transformers import AutoModel, AutoImageProcessor
        # use_fast=False avoids multiprocessing issues on macOS
        _HF_PROCESSOR = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small", use_fast=False
        )
        _HF_MODEL = AutoModel.from_pretrained("facebook/dinov2-small")
        _HF_MODEL.eval()
        logger.info("DINOv2 (HF) loaded OK (embed_dim=384)")
    except Exception as e:
        logger.error(f"DINOv2 HF load failed: {e}")
        _HF_MODEL = None
        _HF_PROCESSOR = None
    return _HF_MODEL, _HF_PROCESSOR


# ─── Frame extraction ────────────────────────────────────────────────────────

def extract_frames(video_path: str, max_sec: int = MAX_VIDEO_SEC) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / SAMPLE_FPS)))
    max_frame_idx = int(fps * max_sec)

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx > max_frame_idx:
            break
        if frame_idx % step == 0:
            frames.append(frame)
            if len(frames) >= MAX_FRAMES:
                break
        frame_idx += 1

    cap.release()
    return frames


# ─── DINOv2 embeddings ───────────────────────────────────────────────────────

def frames_to_embeddings_hf(
    frames: List[np.ndarray], model, processor
) -> Optional[np.ndarray]:
    """(N, 384) float32 embeddings via HF DINOv2."""
    try:
        all_embs = []
        for i in range(0, len(frames), BATCH_SIZE):
            batch = frames[i:i + BATCH_SIZE]
            images = []
            for f in batch:
                rgb = f[:, :, ::-1].astype(np.uint8)
                images.append(Image.fromarray(rgb))

            inputs = processor(images=images, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # CLS token from last_hidden_state[:,0,:]
            cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
            all_embs.append(cls_emb.astype(np.float32))

        return np.concatenate(all_embs, axis=0)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


# ─── Feature extraction ──────────────────────────────────────────────────────

def compute_features(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute 26-D feature vector from (N, D) embeddings.

    Layout:
      [0:7]   7 stepwise L2 distances
      [7:13]  6 turning angles
      [13:17] 4 L2 stats (mean, min, max, var)
      [17:21] 4 angle stats
      [21:25] 4 cosine similarity stats (mean, min, max, var)
      [25]    straightness index
    """
    N = len(embeddings)
    if N < 3:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    diffs = embeddings[1:] - embeddings[:-1]
    l2_dist = np.linalg.norm(diffs, axis=1).astype(np.float32)

    # Turning angles (curvature)
    angles = []
    for i in range(len(diffs) - 1):
        n1, n2 = l2_dist[i], l2_dist[i + 1]
        if n1 < 1e-8 or n2 < 1e-8:
            angles.append(0.0)
            continue
        cos_t = float(np.dot(diffs[i], diffs[i + 1]) / (n1 * n2))
        angles.append(float(np.arccos(np.clip(cos_t, -1.0, 1.0))))
    angles = np.array(angles, dtype=np.float32)

    # Cosine similarities between consecutive frames
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms
    cos_sims = np.array([
        float(np.dot(normed[i], normed[i + 1]))
        for i in range(N - 1)
    ], dtype=np.float32)

    # Straightness index
    total_path = float(np.sum(l2_dist)) + 1e-8
    end_to_end = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
    straightness = end_to_end / total_path

    def pad7(arr):
        lst = arr.tolist()
        m = float(np.mean(lst)) if lst else 0.0
        while len(lst) < 7:
            lst.append(m)
        return lst[:7]

    def pad6(arr):
        lst = arr.tolist()
        m = float(np.mean(lst)) if lst else 0.0
        while len(lst) < 6:
            lst.append(m)
        return lst[:6]

    def stats4(arr):
        if len(arr) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(np.mean(arr)), float(np.min(arr)),
                float(np.max(arr)), float(np.var(arr))]

    feat = (
        pad7(l2_dist)
        + pad6(angles)
        + stats4(l2_dist)
        + stats4(angles)
        + stats4(cos_sims)
        + [straightness]
    )
    return np.array(feat, dtype=np.float32)


def video_to_features_multi(
    video_path: str, model, processor, n_windows: int = 2
) -> List[np.ndarray]:
    """
    Extract multiple feature windows from a video for augmentation.
    Returns list of feature vectors (1-3 depending on video length).
    """
    frames = extract_frames(video_path)
    if len(frames) < 6:
        logger.warning(f"Too few frames in {Path(video_path).name}: {len(frames)}")
        return []

    embs = frames_to_embeddings_hf(frames, model, processor)
    if embs is None or len(embs) < 6:
        return []

    features = []
    N = len(embs)
    win = max(10, N // 2)

    # Window 1: full trajectory
    feat_full = compute_features(embs)
    features.append(feat_full)

    # Window 2: first half
    if N >= 12:
        feat_first = compute_features(embs[:win])
        features.append(feat_first)

    # Window 3: second half (if long enough)
    if N >= 20:
        feat_second = compute_features(embs[N - win:])
        features.append(feat_second)

    logger.info(f"  {Path(video_path).name}: {N} frames → {len(features)} windows")
    return features


# ─── Dataset loading ─────────────────────────────────────────────────────────

def load_dataset(extra_dirs: Optional[List[Tuple[str, int]]] = None):
    model, processor = load_dinov2_hf()
    if model is None:
        raise RuntimeError("DINOv2 unavailable")

    X, y = [], []

    for split, label in [("ai_generated", 1), ("real", 0)]:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            logger.warning(f"Dataset split not found: {split_dir}")
            continue
        videos = sorted(split_dir.glob("*.mp4"))
        logger.info(f"Processing {len(videos)} {split} videos…")
        for vp in videos:
            feats = video_to_features_multi(str(vp), model, processor)
            for feat in feats:
                X.append(feat)
                y.append(label)

    if extra_dirs:
        for extra_dir, label in extra_dirs:
            ep = Path(extra_dir)
            if not ep.exists():
                continue
            videos = (
                list(ep.glob("*.mp4")) + list(ep.glob("*.webm")) + list(ep.glob("*.mkv"))
            )
            logger.info(f"Processing {len(videos)} extra videos from {ep.name}…")
            for vp in videos:
                feats = video_to_features_multi(str(vp), model, processor)
                for feat in feats:
                    X.append(feat)
                    y.append(label)

    del model, processor
    gc.collect()

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ─── MLP ─────────────────────────────────────────────────────────────────────

class ReStraVMLP(nn.Module):
    def __init__(self, input_dim: int = FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_mlp(X: np.ndarray, y: np.ndarray):
    logger.info(f"Dataset: {len(X)} samples (AI={sum(y)}, real={len(y)-sum(y)})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    model = ReStraVMLP(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    best_val_acc = 0.0
    best_state = None
    patience = 40
    no_improve = 0

    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        bs = max(4, len(Xt) // 8)
        loss_sum = 0.0
        n_batches = 0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], yt[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(Xv)
            val_pred = val_out.argmax(1).numpy()
            val_acc = accuracy_score(yv.numpy(), val_pred)
            val_proba = torch.softmax(val_out, dim=1)[:, 1].numpy()
            try:
                val_auc = roc_auc_score(yv.numpy(), val_proba)
            except Exception:
                val_auc = 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{N_EPOCHS} | loss={loss_sum/max(n_batches,1):.4f} "
                f"| val_acc={val_acc:.3f} | val_auc={val_auc:.3f} (best={best_val_acc:.3f})"
            )

        if no_improve >= patience:
            logger.info(f"Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_out = model(Xv)
        val_pred = val_out.argmax(1).numpy()
        val_proba = torch.softmax(val_out, dim=1)[:, 1].numpy()
        val_acc = accuracy_score(yv.numpy(), val_pred)
        val_f1 = f1_score(yv.numpy(), val_pred, zero_division=0)
        try:
            val_auc = roc_auc_score(yv.numpy(), val_proba)
        except Exception:
            val_auc = 0.0

    logger.info(f"\n=== Final Validation ===")
    logger.info(f"Accuracy: {val_acc:.3f} | F1: {val_f1:.3f} | AUC-ROC: {val_auc:.3f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": X.shape[1],
        "feat_dim": FEAT_DIM,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "val_accuracy": float(val_acc),
        "val_f1": float(val_f1),
        "val_auc": float(val_auc),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_ai": int(sum(y)),
        "n_real": int(len(y) - sum(y)),
        "feature_description": (
            "26-D: 7xL2-dist + 6xangle + 4xL2-stats + 4xangle-stats "
            "+ 4xcos-sim-stats + 1xstraightness (no duration feature)"
        ),
    }
    torch.save(checkpoint, str(MLP_PATH))
    logger.info(f"MLP saved: {MLP_PATH}")

    return float(val_acc), float(val_auc)


# ─── Short video downloader ───────────────────────────────────────────────────

EXTRA_AI_DIR = DATASET_DIR / "extra_ai_short"
EXTRA_REAL_DIR = DATASET_DIR / "extra_real_short"

SHORT_REAL_QUERIES = [
    "ytsearch3:short cooking tutorial 30 seconds real",
    "ytsearch3:real tiktok style vlog 30 seconds",
    "ytsearch3:street food short real life video",
    "ytsearch3:travel short clip real footage 2024",
]

SHORT_AI_QUERIES = [
    "ytsearch3:AI generated video short Sora 2024",
    "ytsearch3:runway AI video generation short clip",
    "ytsearch3:Kling AI generated short video demo",
    "ytsearch3:sora openai AI video demo short",
]


def download_short_videos(output_dir: Path, queries: List[str], max_dur: int = 90):
    output_dir.mkdir(exist_ok=True)
    existing = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.webm"))
    if len(existing) >= 4:
        logger.info(f"Already have {len(existing)} videos in {output_dir.name}")
        return

    for query in queries:
        try:
            cmd = [
                "yt-dlp",
                "--format", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
                "--merge-output-format", "mp4",
                "--max-filesize", "50m",
                "--match-filter", f"duration <= {max_dur}",
                "--no-playlist",
                "--output", str(output_dir / "%(id)s.%(ext)s"),
                "--quiet",
                "--no-warnings",
                "--ignore-errors",
                query,
            ]
            logger.info(f"yt-dlp: {query}")
            subprocess.run(cmd, timeout=120, capture_output=True, text=True)
        except Exception as e:
            logger.warning(f"yt-dlp error: {e}")

    new_vids = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.webm"))
    logger.info(f"Downloaded: {len(new_vids)} videos in {output_dir.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== FakeGuard ReStraV MLP Training ===")
    logger.info(f"Device: {DEVICE}")

    # Step 1: Download short videos (for length-bias correction)
    logger.info("\n--- Step 1: Download short videos ---")
    download_short_videos(EXTRA_REAL_DIR, SHORT_REAL_QUERIES)
    download_short_videos(EXTRA_AI_DIR, SHORT_AI_QUERIES)

    # Step 2: Extract features
    logger.info("\n--- Step 2: Extract features ---")
    extra_dirs = []
    for d, lbl in [(EXTRA_REAL_DIR, 0), (EXTRA_AI_DIR, 1)]:
        if d.exists() and (list(d.glob("*.mp4")) or list(d.glob("*.webm"))):
            extra_dirs.append((str(d), lbl))

    X, y = load_dataset(extra_dirs=extra_dirs if extra_dirs else None)

    if len(X) < 10:
        logger.error(f"Too few samples: {len(X)}")
        sys.exit(1)

    logger.info(f"\nTotal: {len(X)} samples (AI={sum(y)}, real={len(y)-sum(y)})")

    # Step 3: Train
    logger.info("\n--- Step 3: Train MLP ---")
    val_acc, val_auc = train_mlp(X, y)

    logger.info(f"\n=== Done ===")
    logger.info(f"Val Acc: {val_acc:.1%} | AUC-ROC: {val_auc:.3f}")
    logger.info(f"Model: {MLP_PATH}")


if __name__ == "__main__":
    main()
