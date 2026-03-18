# © 2026 TimeWealthy Limited — FakeGuard
"""
ReStraVAnalyzer: AI-Generated Video Detection via Perceptual Straightening.

Based on "AI-Generated Video Detection via Perceptual Straightening" (NeurIPS 2025).
GitHub: https://github.com/ChristianInterno/ReStraV

Key changes in v3:
  - Uses HuggingFace transformers for DINOv2 (torch.hub causes SIGSEGV on macOS)
  - Primary: MLP classifier on 26-D trajectory features (length-bias free)
  - Fallback: geometry-based heuristic (mean_theta thresholding)
  - Frame extraction at 5fps for proper trajectory analysis (paper: 12fps)
  - NO video duration in feature vector (removes short-video false positives)

Feature vector (26-D, length-independent):
  [0:7]    7 stepwise L2 distances (early frames)
  [7:13]   6 turning angles (curvature)
  [13:17]  4 L2 distance stats (mean, min, max, var)
  [17:21]  4 angle stats (mean, min, max, var)
  [21:25]  4 cosine similarity stats (mean, min, max, var)
  [25]     1 straightness index (end-to-end / total path)

MLP: 26 → 128 → 64 → 2  (loaded from backend/models/restrav_mlp.pth)
"""

import logging
import os
import numpy as np
from pathlib import Path
from typing import List, Optional

from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)

# Disable OMP/parallelism issues on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ─── Feature constants ────────────────────────────────────────────────────────
FEAT_DIM = 26
BATCH_SIZE = 8      # DINOv2 inference batch size

# Heuristic fallback thresholds
_THETA_LOW = 0.05   # rad — very straight (certainly real)
_THETA_HIGH = 0.60  # rad — very curved (certainly AI)

# ─── Singletons ───────────────────────────────────────────────────────────────
_HF_MODEL = None
_HF_PROC = None
_HF_LOADED = False
_MLP_MODEL = None
_MLP_SCALER = None
_MLP_LOADED = False


def _get_models_dir() -> Path:
    """Resolve backend/models directory relative to this file."""
    # backend/app/analyzers → backend/models
    return Path(__file__).parent.parent.parent / "models"


def _load_dinov2():
    """Load DINOv2 ViT-S/14 via HuggingFace transformers (stable on macOS)."""
    global _HF_MODEL, _HF_PROC, _HF_LOADED
    if _HF_LOADED:
        return _HF_MODEL, _HF_PROC
    _HF_LOADED = True
    try:
        from transformers import AutoModel, AutoImageProcessor
        _HF_PROC = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small", use_fast=False
        )
        _HF_MODEL = AutoModel.from_pretrained("facebook/dinov2-small")
        _HF_MODEL.eval()
        logger.info("DINOv2 ViT-S/14 (HF) loaded OK (embed_dim=384)")
    except Exception as e:
        logger.warning(f"DINOv2 load failed: {e}")
        _HF_MODEL = None
        _HF_PROC = None
    return _HF_MODEL, _HF_PROC


def _load_mlp():
    """Load trained MLP + scaler from backend/models/restrav_mlp.pth."""
    global _MLP_MODEL, _MLP_SCALER, _MLP_LOADED
    if _MLP_LOADED:
        return _MLP_MODEL, _MLP_SCALER
    _MLP_LOADED = True

    mlp_path = _get_models_dir() / "restrav_mlp.pth"
    if not mlp_path.exists():
        logger.info(f"MLP not found at {mlp_path}. Using heuristic fallback.")
        return None, None

    try:
        import torch
        import torch.nn as nn

        ck = torch.load(str(mlp_path), map_location="cpu", weights_only=False)
        input_dim = ck.get("input_dim", FEAT_DIM)

        class _MLP(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 2),
                )

            def forward(self, x):
                return self.net(x)

        model = _MLP(input_dim)
        model.load_state_dict(ck["model_state"])
        model.eval()
        _MLP_MODEL = model

        mean = ck.get("scaler_mean")
        scale = ck.get("scaler_scale")
        if mean is not None and scale is not None:
            _MLP_SCALER = (
                np.array(mean, dtype=np.float32),
                np.array(scale, dtype=np.float32),
            )

        val_acc = ck.get("val_accuracy", 0.0)
        val_auc = ck.get("val_auc", 0.0)
        logger.info(
            f"ReStraV MLP loaded (dim={input_dim}, "
            f"val_acc={val_acc:.1%}, val_auc={val_auc:.3f})"
        )

    except Exception as e:
        logger.warning(f"MLP load failed: {e}. Heuristic fallback active.")
        _MLP_MODEL = None
        _MLP_SCALER = None

    return _MLP_MODEL, _MLP_SCALER


# ─── Embedding extraction ─────────────────────────────────────────────────────

def _frames_to_dinov2_embeddings(
    frames: List[np.ndarray],
    model,
    processor,
    batch_size: int = BATCH_SIZE,
) -> Optional[np.ndarray]:
    """(N, 384) float32 CLS embeddings via HuggingFace DINOv2."""
    try:
        from PIL import Image
        import torch

        all_embs = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            imgs = [
                Image.fromarray(f[:, :, ::-1].astype(np.uint8))
                for f in batch
            ]
            inputs = processor(images=imgs, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].numpy()
            all_embs.append(cls.astype(np.float32))

        return np.concatenate(all_embs, axis=0)

    except Exception as e:
        logger.warning(f"DINOv2 embedding failed: {e}")
        return None


# ─── Feature computation ──────────────────────────────────────────────────────

def _compute_features_26d(embeddings: np.ndarray) -> dict:
    """
    Compute 26-D length-independent trajectory features.

    Layout:
      [0:7]   7 stepwise L2 distances
      [7:13]  6 turning angles (curvature)
      [13:17] 4 L2 distance stats
      [17:21] 4 angle stats
      [21:25] 4 cosine similarity stats
      [25]    straightness index
    """
    N = len(embeddings)
    zero = np.zeros(FEAT_DIM, dtype=np.float32)
    if N < 3:
        return {
            "features_26d": zero,
            "mean_theta": 0.0, "max_theta": 0.0, "var_theta": 0.0,
            "mean_dist": 0.0, "mean_cos_sim": 1.0, "straightness": 1.0,
        }

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
    straightness = float(end_to_end / total_path)

    def _pad7(arr):
        lst = arr.tolist()
        m = float(np.mean(lst)) if lst else 0.0
        while len(lst) < 7: lst.append(m)
        return lst[:7]

    def _pad6(arr):
        lst = arr.tolist()
        m = float(np.mean(lst)) if lst else 0.0
        while len(lst) < 6: lst.append(m)
        return lst[:6]

    def _stats4(arr):
        if len(arr) == 0: return [0.0, 0.0, 0.0, 0.0]
        return [float(np.mean(arr)), float(np.min(arr)),
                float(np.max(arr)), float(np.var(arr))]

    features_26d = np.array(
        _pad7(l2_dist) + _pad6(angles) + _stats4(l2_dist)
        + _stats4(angles) + _stats4(cos_sims) + [straightness],
        dtype=np.float32,
    )

    return {
        "features_26d": features_26d,
        "mean_theta": float(np.mean(angles)) if len(angles) > 0 else 0.0,
        "max_theta": float(np.max(angles)) if len(angles) > 0 else 0.0,
        "var_theta": float(np.var(angles)) if len(angles) > 0 else 0.0,
        "mean_dist": float(np.mean(l2_dist)),
        "mean_cos_sim": float(np.mean(cos_sims)),
        "straightness": straightness,
    }


# ─── Score computation ────────────────────────────────────────────────────────

def _mlp_score(features_26d: np.ndarray, mlp_model, mlp_scaler) -> float:
    """Run MLP inference → AI probability [0, 100]. Returns -1 on failure."""
    try:
        import torch
        feat = features_26d.copy()
        if mlp_scaler is not None:
            mean, scale = mlp_scaler
            feat = (feat - mean) / (scale + 1e-8)

        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        mlp_model.eval()
        with torch.no_grad():
            logits = mlp_model(x)
            proba = torch.softmax(logits, dim=1)[0, 1].item()
        return round(proba * 100.0, 2)
    except Exception as e:
        logger.warning(f"MLP inference failed: {e}")
        return -1.0


def _heuristic_score(mean_theta: float, max_theta: float, var_theta: float) -> float:
    """Fallback geometry-based heuristic score [0, 100]."""
    score = (mean_theta - _THETA_LOW) / (_THETA_HIGH - _THETA_LOW)
    score = float(np.clip(score, 0.0, 1.0)) * 100.0
    if max_theta > 1.2:
        score = min(100.0, score + 10.0)
    if var_theta > 0.05:
        score = min(100.0, score + 5.0)
    return round(score, 2)


# ─── Main analyzer class ──────────────────────────────────────────────────────

class ReStraVAnalyzer(BaseAnalyzer):
    """
    Analyzes video frames using perceptual straightening in DINOv2 feature space.

    Primary: MLP classifier on 26-D trajectory features (length-bias free).
    Fallback: geometry-based heuristic (mean_theta thresholding).
    """

    def __init__(self, device: str = "cpu"):
        # Keep CPU to avoid MPS SIGSEGV; HF DINOv2 is fast enough on M-series
        self.device = device
        self._dinov2 = None
        self._proc = None
        self._mlp = None
        self._scaler = None

    def _load_models(self):
        if self._dinov2 is None:
            self._dinov2, self._proc = _load_dinov2()
        if self._mlp is None:
            self._mlp, self._scaler = _load_mlp()

    def analyze(self, frames: List[np.ndarray], fps: float = 25.0) -> AnalyzerResult:
        """
        Analyze frames using perceptual straightening + MLP.

        Args:
            frames: list of (H, W, 3) BGR uint8 arrays
            fps: original video FPS

        Returns:
            AnalyzerResult with score 0-100 (higher = more likely AI-generated)
        """
        if not frames:
            return AnalyzerResult(score=50.0, findings=[], error="No frames provided")

        if len(frames) < 3:
            return AnalyzerResult(
                score=50.0, findings=[], error="Too few frames for trajectory analysis"
            )

        self._load_models()

        if self._dinov2 is None or self._proc is None:
            return AnalyzerResult(
                score=50.0, findings=[], error="DINOv2 model unavailable"
            )

        # Extract DINOv2 embeddings
        embeddings = _frames_to_dinov2_embeddings(
            frames, self._dinov2, self._proc
        )
        if embeddings is None or len(embeddings) < 3:
            return AnalyzerResult(
                score=50.0, findings=[], error="DINOv2 embedding extraction failed"
            )

        # Compute 26-D trajectory features
        traj = _compute_features_26d(embeddings)
        mean_theta = traj["mean_theta"]
        max_theta = traj["max_theta"]
        var_theta = traj["var_theta"]
        mean_dist = traj["mean_dist"]
        mean_cos_sim = traj["mean_cos_sim"]
        straightness = traj["straightness"]
        features_26d = traj["features_26d"]

        # Score: MLP preferred, heuristic as fallback
        using_mlp = self._mlp is not None
        if using_mlp:
            score = _mlp_score(features_26d, self._mlp, self._scaler)
            if score < 0:
                score = _heuristic_score(mean_theta, max_theta, var_theta)
                using_mlp = False
        else:
            score = _heuristic_score(mean_theta, max_theta, var_theta)

        # Findings
        findings: List[Finding] = []

        if mean_theta > 0.35:
            findings.append(Finding(
                type="trajectory_curvature",
                confidence=round(min(95.0, mean_theta * 200), 1),
                description=(
                    f"DINOv2空間での軌跡が高い曲率を示す（平均回転角: {mean_theta:.3f} rad）。"
                    "AI生成動画の特徴的パターン（フレーム間の知覚的不連続性）を検出。"
                ),
                metadata={
                    "mean_theta_rad": round(mean_theta, 4),
                    "max_theta_rad": round(max_theta, 4),
                    "var_theta": round(var_theta, 5),
                    "mean_dist": round(mean_dist, 4),
                    "mean_cos_sim": round(mean_cos_sim, 4),
                    "straightness": round(straightness, 4),
                    "n_frames": len(frames),
                    "mode": "mlp" if using_mlp else "heuristic",
                },
            ))

        if max_theta > 1.2:
            findings.append(Finding(
                type="sharp_directional_reversal",
                confidence=round(min(90.0, max_theta * 60), 1),
                description=(
                    f"フレーム間に急激な方向転換を検出（最大角: {max_theta:.3f} rad = "
                    f"{np.degrees(max_theta):.1f}°）。通常の動画には見られない不自然な遷移。"
                ),
                metadata={"max_theta_rad": round(max_theta, 4)},
            ))

        if straightness < 0.3 and score > 60:
            findings.append(Finding(
                type="low_straightness",
                confidence=round(min(80.0, (1.0 - straightness) * 60 + 20), 1),
                description=(
                    f"軌跡の直線性が低い（straightness index: {straightness:.3f}）。"
                    "AI生成動画は特徴空間での軌跡が迂回することが多い。"
                ),
                metadata={"straightness_index": round(straightness, 4)},
            ))

        logger.info(
            f"ReStraV: {len(frames)} frames, mean_θ={mean_theta:.3f}, "
            f"straight={straightness:.3f}, cos_sim={mean_cos_sim:.3f}, "
            f"score={score} ({'MLP' if using_mlp else 'heuristic'})"
        )

        return AnalyzerResult(score=score, findings=findings)
