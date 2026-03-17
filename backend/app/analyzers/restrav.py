# © 2026 TimeWealthy Limited — FakeGuard
"""
ReStraVAnalyzer: AI-Generated Video Detection via Perceptual Straightening.

Based on the paper "AI-Generated Video Detection via Perceptual Straightening"
(NeurIPS 2025) by Christian Internò et al.
GitHub: https://github.com/ChristianInterno/ReStraV

Core idea:
  Natural (real) videos trace STRAIGHT trajectories in DINOv2 feature space.
  AI-generated videos trace CURVED/irregular trajectories.

  We encode frames with DINOv2 ViT-S/14 and measure the geometric properties
  (stepwise distances + turning angles) of the resulting trajectory.
  High curvature → high AI score.

21-D feature vector per video (matches ReStraV paper):
  d[0:7]        → 7 early stepwise L2 distances
  theta[0:6]    → 6 early turning angles (radians)
  mu_d, min_d, max_d, var_d       → 4 distance summary stats
  mu_theta, min_theta, max_theta, var_theta → 4 angle summary stats
  Total: 7 + 6 + 8 = 21
"""

import logging
import numpy as np
from typing import List, Optional

from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)

# Singleton DINOv2 model (shared across instances)
_DINOV2_MODEL = None
_DINOV2_LOADED = False

# Perceptual straightening normalization constants.
# Calibrated empirically on our 40-video benchmark dataset:
#   mean_theta_ai   ≈ 0.42 rad  (more curved)
#   mean_theta_real ≈ 0.15 rad  (straighter)
# We map [0.05, 0.60] → [0, 100] (clipped).
_THETA_LOW = 0.05    # rad — very straight (certainly real)
_THETA_HIGH = 0.60   # rad — very curved (certainly AI)


def _load_dinov2():
    """Lazy-load DINOv2 ViT-S/14 (singleton)."""
    global _DINOV2_MODEL, _DINOV2_LOADED
    if _DINOV2_LOADED:
        return _DINOV2_MODEL
    _DINOV2_LOADED = True
    try:
        import torch
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True, verbose=False
        )
        model.eval()
        _DINOV2_MODEL = model
        logger.info("DINOv2 ViT-S/14 loaded OK (embed_dim=384)")
    except Exception as e:
        logger.warning(f"Could not load DINOv2: {e}")
        _DINOV2_MODEL = None
    return _DINOV2_MODEL


def _frames_to_dinov2_embeddings(
    frames: List[np.ndarray],
    model,
    device: str = "cpu",
    batch_size: int = 8,
) -> Optional[np.ndarray]:
    """
    Convert a list of BGR frames to DINOv2 [CLS] token embeddings.

    Args:
        frames: list of (H,W,3) BGR uint8 arrays
        model: DINOv2 torch model
        device: 'cpu', 'mps', or 'cuda'
        batch_size: number of frames per forward pass

    Returns:
        (N, 384) float32 array of CLS embeddings, or None on failure
    """
    try:
        import torch
        from torchvision import transforms
        from PIL import Image

        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        model = model.to(device)
        all_embs = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            tensors = []
            for f in batch_frames:
                # BGR → RGB
                rgb = f[:, :, ::-1].astype(np.uint8)
                img = Image.fromarray(rgb)
                tensors.append(preprocess(img))

            batch_tensor = torch.stack(tensors).to(device)
            with torch.no_grad():
                emb = model(batch_tensor)  # (B, 384)
            all_embs.append(emb.cpu().numpy())

        return np.concatenate(all_embs, axis=0).astype(np.float32)

    except Exception as e:
        logger.warning(f"DINOv2 embedding failed: {e}")
        return None


def _compute_trajectory_features(embeddings: np.ndarray) -> dict:
    """
    Compute 21-D perceptual straightening features from frame embeddings.

    Args:
        embeddings: (N, D) float32 array, N >= 3

    Returns:
        dict with keys: distances, angles, features_21d,
                        mean_theta, mean_dist
    """
    N = len(embeddings)
    if N < 3:
        return {"mean_theta": 0.0, "mean_dist": 0.0, "features_21d": np.zeros(21)}

    # Stepwise L2 distances
    diffs = embeddings[1:] - embeddings[:-1]        # (N-1, D)
    distances = np.linalg.norm(diffs, axis=1)        # (N-1,)

    # Turning angles between consecutive displacement vectors
    # theta[i] = acos( diffs[i] · diffs[i+1] / (||diffs[i]|| * ||diffs[i+1]||) )
    angles = []
    for i in range(len(diffs) - 1):
        n1 = distances[i]
        n2 = distances[i + 1]
        if n1 < 1e-8 or n2 < 1e-8:
            angles.append(0.0)
            continue
        cos_theta = float(np.dot(diffs[i], diffs[i + 1]) / (n1 * n2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles.append(float(np.arccos(cos_theta)))

    angles = np.array(angles, dtype=np.float32)
    distances = distances.astype(np.float32)

    # Build 21-D feature vector (pad with mean if shorter than 7)
    def _pad7(arr):
        arr = arr.tolist()
        while len(arr) < 7:
            arr.append(float(np.mean(arr)) if arr else 0.0)
        return arr[:7]

    def _pad6(arr):
        arr = arr.tolist()
        while len(arr) < 6:
            arr.append(float(np.mean(arr)) if arr else 0.0)
        return arr[:6]

    d7 = _pad7(distances)
    t6 = _pad6(angles)

    # Summary stats
    mu_d = float(np.mean(distances))
    min_d = float(np.min(distances))
    max_d = float(np.max(distances))
    var_d = float(np.var(distances))

    mu_t = float(np.mean(angles)) if len(angles) > 0 else 0.0
    min_t = float(np.min(angles)) if len(angles) > 0 else 0.0
    max_t = float(np.max(angles)) if len(angles) > 0 else 0.0
    var_t = float(np.var(angles)) if len(angles) > 0 else 0.0

    features_21d = np.array(
        d7 + t6 + [mu_d, min_d, max_d, var_d, mu_t, min_t, max_t, var_t],
        dtype=np.float32,
    )

    return {
        "distances": distances,
        "angles": angles,
        "features_21d": features_21d,
        "mean_theta": mu_t,
        "mean_dist": mu_d,
        "max_theta": max_t,
        "var_theta": var_t,
    }


def _geometry_to_ai_score(mean_theta: float, max_theta: float, var_theta: float) -> float:
    """
    Convert trajectory geometry to AI probability [0, 100].

    Primary signal: mean_theta (mean turning angle).
    Secondary boost: max_theta and var_theta for extreme curvature events.

    Calibration (empirical on benchmark dataset):
      real avg mean_theta ≈ 0.10-0.20 rad → score ~10-35
      AI   avg mean_theta ≈ 0.35-0.55 rad → score ~65-90
    """
    # Primary score from mean turning angle
    score = (mean_theta - _THETA_LOW) / (_THETA_HIGH - _THETA_LOW)
    score = float(np.clip(score, 0.0, 1.0)) * 100.0

    # Secondary boost: if there are sharp directional reversals
    if max_theta > 1.2:          # > 68 degrees = likely AI frame jump
        score = min(100.0, score + 10.0)
    if var_theta > 0.05:         # high variance in turning = inconsistent motion
        score = min(100.0, score + 5.0)

    return round(score, 2)


class ReStraVAnalyzer(BaseAnalyzer):
    """
    Analyzes video frames using perceptual straightening in DINOv2 space.

    Natural videos trace straight paths; AI-generated videos trace curved paths.
    
    Replaces the SigLIP-based FrameAnalyzer for full-video AI detection
    (not just face-swap detection).
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        self._model = _load_dinov2()

    def analyze(self, frames: List[np.ndarray], fps: float = 25.0) -> AnalyzerResult:
        """
        Analyze a list of frames using perceptual straightening.

        Args:
            frames: list of (H, W, 3) BGR uint8 arrays
            fps: original video FPS (used for timestamp calculation)

        Returns:
            AnalyzerResult with score 0-100 (higher = more likely AI-generated)
        """
        if not frames:
            return AnalyzerResult(score=50.0, findings=[], error="No frames provided")

        if len(frames) < 3:
            return AnalyzerResult(
                score=50.0, findings=[], error="Too few frames for trajectory analysis"
            )

        self._load_model()

        if self._model is None:
            return AnalyzerResult(
                score=50.0, findings=[], error="DINOv2 model unavailable"
            )

        # Extract DINOv2 embeddings
        embeddings = _frames_to_dinov2_embeddings(frames, self._model, device=self.device)
        if embeddings is None or len(embeddings) < 3:
            return AnalyzerResult(
                score=50.0, findings=[], error="DINOv2 embedding extraction failed"
            )

        # Compute trajectory geometry
        traj = _compute_trajectory_features(embeddings)
        mean_theta = traj["mean_theta"]
        max_theta = traj["max_theta"]
        var_theta = traj["var_theta"]
        mean_dist = traj["mean_dist"]

        # Convert to AI score
        score = _geometry_to_ai_score(mean_theta, max_theta, var_theta)

        # Generate findings
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
                    "n_frames": len(frames),
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

        logger.info(
            f"ReStraV: {len(frames)} frames, "
            f"mean_θ={mean_theta:.3f}, max_θ={max_theta:.3f}, "
            f"score={score}"
        )

        return AnalyzerResult(score=score, findings=findings)
