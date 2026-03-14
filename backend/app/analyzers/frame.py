# © 2026 TimeWealthy Limited — FakeGuard
"""FrameAnalyzer: per-frame AI generation detection using SigLIP-based model."""

import logging
import numpy as np
from typing import List, Optional, Tuple
from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)

# Singleton model cache
_MODEL_CACHE: dict = {}


class FrameAnalyzer(BaseAnalyzer):
    """
    Analyzes individual frames to detect AI generation artifacts.

    Checks:
    - HuggingFace prithivMLmods/deepfake-detector-model-v1 (SigLIP-based)
      via AutoModelForImageClassification + AutoProcessor
    - FFT frequency-domain texture analysis (scipy.fft)
    - Face landmark anomalies (MediaPipe, optional)
    """

    def __init__(
        self,
        model_name: str = "prithivMLmods/deepfake-detector-model-v1",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._pipeline = None
        self._face_mesh = None

    def _load_model(self):
        """Lazy-load the HuggingFace SigLIP model (singleton per model_name)."""
        if self._model is not None:
            return

        # Check if pipeline was already injected (e.g., from tests)
        if self._pipeline is not None:
            self._model = "injected"
            return

        global _MODEL_CACHE
        if self.model_name in _MODEL_CACHE:
            cached = _MODEL_CACHE[self.model_name]
            self._model = cached["model"]
            self._processor = cached["processor"]
            self._pipeline = cached.get("pipeline")
            logger.info(f"Using cached model: {self.model_name}")
            return

        try:
            from transformers import AutoModelForImageClassification, AutoProcessor

            logger.info(f"Loading model: {self.model_name}")
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModelForImageClassification.from_pretrained(self.model_name)
            model.eval()

            self._model = model
            self._processor = processor

            # Cache for singleton access
            _MODEL_CACHE[self.model_name] = {
                "model": model,
                "processor": processor,
            }
            logger.info(f"Loaded model: {self.model_name}")

        except Exception as e:
            logger.warning(
                f"Could not load HuggingFace model {self.model_name}: {e}. "
                "Falling back to texture-only mode."
            )
            self._model = "failed"
            self._processor = None

    def _load_face_mesh(self):
        """Lazy-load MediaPipe face mesh (optional dependency)."""
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp

            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=4,
                min_detection_confidence=0.5,
            )
            logger.info("Loaded MediaPipe FaceMesh")
        except Exception as e:
            logger.debug(f"Could not load MediaPipe (optional): {e}")
            self._face_mesh = None

    def analyze(self, frames: List[np.ndarray], fps: float = 25.0) -> AnalyzerResult:
        """
        Analyze a list of frames sampled at 2fps from the original video.

        Args:
            frames: List of numpy arrays (H, W, C) in BGR
            fps: Original video frame rate (used for timestamp calculation)

        Returns:
            AnalyzerResult with score 0-100 (higher = more likely AI-generated)
        """
        if not frames:
            return AnalyzerResult(score=50.0, findings=[], error="No frames provided")

        self._load_model()
        self._load_face_mesh()

        scores: List[float] = []
        findings: List[Finding] = []

        for i, frame in enumerate(frames):
            # Approximate original frame number (assuming 2fps sampling)
            frame_num = int(i * fps / 2)
            timestamp = frame_num / fps if fps > 0 else 0.0

            frame_score = 50.0

            # 1. SigLIP-based AI detection
            cnn_score = self._cnn_score(frame)
            if cnn_score is not None:
                frame_score = cnn_score

            # 2. FFT Texture analysis
            texture_score, texture_finding = self._texture_analysis(frame, frame_num, timestamp)
            if texture_finding:
                findings.append(texture_finding)

            if cnn_score is not None:
                frame_score = frame_score * 0.7 + texture_score * 0.3
            else:
                # No ML model available → use texture-only
                frame_score = texture_score

            # 3. Face landmark analysis (optional, MediaPipe)
            face_findings = self._face_landmark_analysis(frame, frame_num, timestamp)
            if face_findings:
                findings.extend(face_findings)
                max_face_confidence = max(f.confidence for f in face_findings)
                frame_score = min(100.0, frame_score + max_face_confidence * 0.3)

            scores.append(min(100.0, max(0.0, frame_score)))

        final_score = float(np.mean(scores)) if scores else 50.0

        # High-confidence findings add a small boost
        for finding in findings:
            if finding.confidence > 85.0:
                final_score = min(100.0, final_score + 5.0)

        return AnalyzerResult(score=round(final_score, 2), findings=findings)

    def _cnn_score(self, frame: np.ndarray) -> Optional[float]:
        """
        Run the SigLIP-based classifier, return AI probability 0-100.
        
        Supports both:
        - injected pipeline (from tests) → pipeline(img) call
        - loaded AutoModelForImageClassification + AutoProcessor
        """
        # Case 1: Injected pipeline (e.g., tests)
        if self._pipeline is not None and self._model == "injected":
            return self._run_pipeline_score(frame)

        # Case 2: AutoModel + AutoProcessor
        if (
            self._model not in (None, "failed", "injected")
            and self._processor is not None
        ):
            return self._run_automodel_score(frame)

        # Case 3: Plain pipeline loaded via _load_model (fallback path)
        if hasattr(self, "_pipeline") and self._pipeline is not None:
            return self._run_pipeline_score(frame)

        return None

    def _run_pipeline_score(self, frame: np.ndarray) -> Optional[float]:
        """
        Run HuggingFace pipeline and extract AI score.
        Tries PIL Image first; falls back to raw numpy array if PIL unavailable.
        """
        try:
            # Try to convert to PIL Image for HuggingFace compatibility
            img = None
            try:
                from PIL import Image
                if frame.ndim == 3 and frame.shape[2] == 3:
                    img = Image.fromarray(frame.astype(np.uint8))
                else:
                    img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
            except ImportError:
                # PIL not available; pass numpy array directly (works with mocks)
                img = frame.astype(np.uint8)

            results = self._pipeline(img)

            # Look for AI/fake/artificial label
            for res in results:
                label = res["label"].lower()
                if any(kw in label for kw in ["artificial", "fake", "ai", "generated"]):
                    return float(res["score"]) * 100.0

            # Invert if only real/human label found
            for res in results:
                label = res["label"].lower()
                if any(kw in label for kw in ["real", "human", "authentic"]):
                    return (1.0 - float(res["score"])) * 100.0

        except Exception as e:
            logger.debug(f"Pipeline inference failed: {e}")
        return None

    def _run_automodel_score(self, frame: np.ndarray) -> Optional[float]:
        """
        Run AutoModelForImageClassification + AutoProcessor inference.
        Returns AI probability 0-100.
        """
        try:
            import torch
            from PIL import Image

            # Convert BGR → RGB for the processor
            rgb_frame = frame[:, :, ::-1].astype(np.uint8) if frame.shape[2] == 3 else frame
            img = Image.fromarray(rgb_frame)

            inputs = self._processor(images=img, return_tensors="pt")

            with torch.no_grad():
                outputs = self._model(**inputs)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

            # Map label id → probability
            id2label = self._model.config.id2label
            ai_prob = 0.0

            for idx, prob in enumerate(probs.tolist()):
                label = id2label.get(idx, "").lower()
                if any(kw in label for kw in ["artificial", "fake", "ai", "generated", "deepfake"]):
                    ai_prob = max(ai_prob, prob)
                elif any(kw in label for kw in ["real", "human", "authentic"]):
                    ai_prob = max(ai_prob, 1.0 - prob)

            return ai_prob * 100.0

        except Exception as e:
            logger.debug(f"AutoModel inference failed: {e}")
        return None

    def _texture_analysis(
        self, frame: np.ndarray, frame_num: int, timestamp: float
    ) -> Tuple[float, Optional[Finding]]:
        """
        FFT-based texture analysis using scipy.fft.
        AI-generated images tend to have too-uniform high-frequency components.

        Returns:
            (score 0-100, optional Finding)
        """
        try:
            import cv2
            from scipy.fft import fft2, fftshift

            gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            # scipy.fft for frequency analysis
            fft_result = fft2(gray.astype(np.float64))
            fft_shift = fftshift(fft_result)
            magnitude = np.abs(fft_shift)

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2

            # Radial distance mask
            y_idx, x_idx = np.ogrid[:h, :w]
            dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

            radius = min(h, w) // 4
            mask_hf = dist > radius  # high-frequency region

            hf_magnitudes = magnitude[mask_hf]
            uniformity_score = _compute_frequency_uniformity(hf_magnitudes)

            # Normalize to 0-100
            score = min(100.0, uniformity_score * 100.0)

            finding = None
            if score > 70.0:
                finding = Finding(
                    type="texture_anomaly",
                    confidence=round(score, 1),
                    description=f"背景テクスチャに反復パターンを検出（FFT均一度: {score:.1f}）",
                    frame_number=frame_num,
                    timestamp_sec=round(timestamp, 2),
                )
            return score, finding

        except Exception as e:
            logger.debug(f"Texture analysis (FFT) failed: {e}")
            return 50.0, None

    def _face_landmark_analysis(
        self, frame: np.ndarray, frame_num: int, timestamp: float
    ) -> List[Finding]:
        """Detect face landmark anomalies using MediaPipe (optional)."""
        findings: List[Finding] = []
        if self._face_mesh is None:
            return findings

        try:
            import cv2

            rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return findings

            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                asymmetry = _compute_face_asymmetry(landmarks)
                if asymmetry > 0.08:
                    confidence = min(95.0, asymmetry * 500)
                    findings.append(
                        Finding(
                            type="face_asymmetry",
                            confidence=round(confidence, 1),
                            description=f"顔の左右非対称性を検出（スコア: {asymmetry:.4f}）",
                            frame_number=frame_num,
                            timestamp_sec=round(timestamp, 2),
                        )
                    )
        except Exception as e:
            logger.debug(f"Face landmark analysis failed: {e}")

        return findings


# ─── Utility Functions ────────────────────────────────────────────────────────


def _compute_frequency_uniformity(hf_magnitudes: np.ndarray) -> float:
    """
    Measure how uniform the high-frequency energy is.
    High uniformity → likely AI generated (too-regular texture).

    Uses coefficient of variation (CV): low CV = uniform = AI-like.

    Special case:
    - Near-zero HF energy (perfectly flat/uniform image) → maximally AI-like → return 1.0
    - Real images have strong, varied high-frequency texture

    Returns:
        float in [0, 1], where 1 = maximally uniform (AI-like)
    """
    if hf_magnitudes.size == 0:
        return 0.5

    flat = hf_magnitudes.flatten()

    # Near-zero HF energy: perfectly flat image (no texture at all)
    # This is maximally AI-like (or a solid-color synthetic frame)
    if flat.max() < 1e-8:
        return 1.0

    normalized = flat / (flat.max() + 1e-10)
    mean = float(np.mean(normalized)) + 1e-10
    std = float(np.std(normalized))
    cv = std / mean

    # Low CV → high uniformity score (more AI-like)
    uniformity = max(0.0, 1.0 - min(1.0, cv))
    return float(uniformity)


def _compute_face_asymmetry(landmarks: list) -> float:
    """
    Compute left-right asymmetry from 468 face landmarks.

    Args:
        landmarks: List of (x, y, z) tuples

    Returns:
        Asymmetry score ≥ 0 (0 = perfectly symmetric)
    """
    if len(landmarks) < 10:
        return 0.0

    xs = [lm[0] for lm in landmarks]
    mid_x = float(np.mean(xs))

    left = [x for x in xs if x < mid_x]
    right = [x for x in xs if x >= mid_x]

    if not left or not right:
        return 0.0

    left_dist = float(np.mean([abs(x - mid_x) for x in left]))
    right_dist = float(np.mean([abs(x - mid_x) for x in right]))

    asymmetry = abs(left_dist - right_dist) / (left_dist + right_dist + 1e-8)
    return float(asymmetry)
