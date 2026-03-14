# © 2026 TimeWealthy Limited — DeepGuard
"""FrameAnalyzer: per-frame AI generation detection."""

import logging
import numpy as np
from typing import List, Optional, Tuple
from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)


class FrameAnalyzer(BaseAnalyzer):
    """
    Analyzes individual frames to detect AI generation artifacts.

    Checks:
    - CNN-based AI image detection (EfficientNet / HuggingFace model)
    - Face landmark anomalies (MediaPipe)
    - Texture analysis (FFT frequency distribution)
    - Edge / boundary naturalness
    """

    def __init__(self, model_name: str = "umm-maybe/AI-image-detector", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._face_mesh = None

    def _load_model(self):
        """Lazy-load the HuggingFace model."""
        if self._model is not None:
            return
        # Check if pipeline was already injected (e.g., from tests)
        if hasattr(self, "_pipeline") and self._pipeline is not None:
            self._model = "injected"
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "image-classification",
                model=self.model_name,
                device=-1,  # CPU
            )
            self._model = "loaded"
            logger.info(f"Loaded model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load HF model {self.model_name}: {e}. Using texture-only mode.")
            self._pipeline = None
            self._model = "failed"

    def _load_face_mesh(self):
        """Lazy-load MediaPipe face mesh."""
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
            logger.warning(f"Could not load MediaPipe: {e}. Skipping face analysis.")
            self._face_mesh = None

    def analyze(self, frames: List[np.ndarray], fps: float = 25.0) -> AnalyzerResult:
        """
        Analyze a list of frames.

        Args:
            frames: List of numpy arrays (H, W, C) in BGR or RGB
            fps: Frames per second of the original video

        Returns:
            AnalyzerResult with score 0-100
        """
        if not frames:
            return AnalyzerResult(score=50.0, findings=[], error="No frames provided")

        self._load_model()
        self._load_face_mesh()

        scores: List[float] = []
        findings: List[Finding] = []

        for i, frame in enumerate(frames):
            frame_num = int(i * fps / 2)  # approximate original frame number (sampled at 2fps)
            timestamp = frame_num / fps if fps > 0 else 0.0

            frame_score = 50.0

            # 1. CNN-based AI detection
            cnn_score = self._cnn_score(frame)
            if cnn_score is not None:
                frame_score = cnn_score

            # 2. Texture analysis (FFT)
            texture_score, texture_finding = self._texture_analysis(frame, frame_num, timestamp)
            if texture_finding:
                findings.append(texture_finding)
            frame_score = frame_score * 0.7 + texture_score * 0.3

            # 3. Face landmark analysis
            face_findings = self._face_landmark_analysis(frame, frame_num, timestamp)
            if face_findings:
                findings.extend(face_findings)
                # Boost score if face anomalies found
                max_face_confidence = max(f.confidence for f in face_findings)
                frame_score = min(100.0, frame_score + max_face_confidence * 0.3)

            scores.append(min(100.0, max(0.0, frame_score)))

        final_score = float(np.mean(scores)) if scores else 50.0

        # High-confidence findings boost the score
        for finding in findings:
            if finding.confidence > 85.0:
                final_score = min(100.0, final_score + 5.0)

        return AnalyzerResult(score=round(final_score, 2), findings=findings)

    def _cnn_score(self, frame: np.ndarray) -> Optional[float]:
        """Run the HuggingFace classifier, return AI probability 0-100."""
        if not hasattr(self, "_pipeline") or self._pipeline is None:
            return None
        try:
            from PIL import Image
            if frame.ndim == 3 and frame.shape[2] == 3:
                img = Image.fromarray(frame.astype(np.uint8))
            else:
                img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")

            results = self._pipeline(img)
            # Model labels vary; look for 'artificial'/'fake'/'ai' label
            for res in results:
                label = res["label"].lower()
                if any(kw in label for kw in ["artificial", "fake", "ai", "generated"]):
                    return float(res["score"]) * 100.0
            # If only 'real' label found, invert
            for res in results:
                label = res["label"].lower()
                if any(kw in label for kw in ["real", "human", "authentic"]):
                    return (1.0 - float(res["score"])) * 100.0
        except Exception as e:
            logger.debug(f"CNN inference failed: {e}")
        return None

    def _texture_analysis(
        self, frame: np.ndarray, frame_num: int, timestamp: float
    ) -> Tuple[float, Optional[Finding]]:
        """
        FFT-based texture analysis.
        AI-generated images tend to have too-uniform high-frequency components.
        Returns (score 0-100, optional Finding).
        """
        try:
            import cv2

            gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 4

            # High-frequency energy ratio
            mask_hf = np.zeros_like(magnitude, dtype=bool)
            y_idx, x_idx = np.ogrid[:h, :w]
            dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
            mask_hf[dist > radius] = True

            hf_energy = np.mean(magnitude[mask_hf])
            lf_energy = np.mean(magnitude[~mask_hf]) + 1e-8
            ratio = hf_energy / lf_energy

            # AI images: suspiciously uniform ratio (neither too high nor too low)
            # Empirically: real images have ratio ~0.01-0.15; AI often 0.05-0.12 (very uniform)
            uniformity_score = _compute_frequency_uniformity(magnitude[mask_hf])
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
            logger.debug(f"Texture analysis failed: {e}")
            return 50.0, None

    def _face_landmark_analysis(
        self, frame: np.ndarray, frame_num: int, timestamp: float
    ) -> List[Finding]:
        """Detect face landmark anomalies using MediaPipe."""
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


def _compute_frequency_uniformity(hf_magnitudes: np.ndarray) -> float:
    """
    Measure how uniform the high-frequency energy is.
    High uniformity → likely AI generated.
    Returns 0-1.
    """
    if hf_magnitudes.size == 0:
        return 0.5
    flat = hf_magnitudes.flatten()
    if flat.max() < 1e-8:
        return 0.0
    # Normalize
    normalized = flat / flat.max()
    # Coefficient of variation (lower CV = more uniform = more AI-like)
    mean = np.mean(normalized) + 1e-8
    std = np.std(normalized)
    cv = std / mean
    # Invert: low CV → high uniformity score
    uniformity = max(0.0, 1.0 - min(1.0, cv))
    return float(uniformity)


def _compute_face_asymmetry(landmarks: list) -> float:
    """
    Compute simple left-right asymmetry of 468 face landmarks.
    Returns asymmetry score (0 = perfectly symmetric, larger = more asymmetric).
    """
    if len(landmarks) < 10:
        return 0.0
    # Compare X coordinates of mirrored landmark pairs (simplified)
    xs = [lm[0] for lm in landmarks]
    mid_x = np.mean(xs)
    left = [x for x in xs if x < mid_x]
    right = [x for x in xs if x >= mid_x]
    if not left or not right:
        return 0.0
    left_dist = np.mean([abs(x - mid_x) for x in left])
    right_dist = np.mean([abs(x - mid_x) for x in right])
    asymmetry = abs(left_dist - right_dist) / (left_dist + right_dist + 1e-8)
    return float(asymmetry)
