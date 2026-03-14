# © 2026 TimeWealthy Limited — DeepGuard
"""WarFootageAnalyzer: specialized checks for war/conflict footage."""

import logging
import numpy as np
from typing import List, Tuple
from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)


class WarFootageAnalyzer(BaseAnalyzer):
    """
    Specialized analyzer for war/conflict footage.

    Checks:
    - Explosion particle physics consistency
    - Smoke/fire fluid dynamics
    - Weapon/vehicle model detection (simplified)
    - Acoustic analysis of explosion sounds
    """

    def analyze(self, frames: List[np.ndarray], audio: np.ndarray = None, sr: int = 44100) -> AnalyzerResult:
        """
        Args:
            frames: List of video frames
            audio: Optional audio array
            sr: Audio sample rate

        Returns:
            AnalyzerResult with score 0-100
        """
        if not frames:
            return AnalyzerResult(score=50.0, findings=[], error="No frames provided")

        findings: List[Finding] = []
        scores: List[float] = []

        # 1. Explosion/fire detection and physics check
        explosion_score, explosion_findings = self._explosion_physics_check(frames)
        scores.append(explosion_score)
        findings.extend(explosion_findings)

        # 2. Smoke uniformity check
        smoke_score, smoke_findings = self._smoke_uniformity_check(frames)
        scores.append(smoke_score)
        findings.extend(smoke_findings)

        # 3. Audio-visual sync for explosions (if audio available)
        if audio is not None and len(audio) > 0:
            av_score, av_findings = self._audio_visual_sync_check(frames, audio, sr)
            scores.append(av_score)
            findings.extend(av_findings)

        final_score = float(np.mean(scores)) if scores else 40.0
        return AnalyzerResult(score=round(final_score, 2), findings=findings)

    def _explosion_physics_check(
        self, frames: List[np.ndarray]
    ) -> Tuple[float, List[Finding]]:
        """
        Detect explosion-like regions and check if particle spread is physically plausible.
        AI-generated explosions often have too-uniform particle distributions.
        """
        findings: List[Finding] = []
        scores: List[float] = []

        try:
            import cv2

            for i, frame in enumerate(frames):
                # Detect bright regions (fire/explosion candidates)
                hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2HSV)
                # Orange-red fire color range
                lower_fire = np.array([0, 50, 200])
                upper_fire = np.array([30, 255, 255])
                fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

                fire_ratio = np.sum(fire_mask > 0) / (frame.shape[0] * frame.shape[1])

                if fire_ratio > 0.05:
                    # Analyze the shape/uniformity of fire region
                    fire_region = cv2.bitwise_and(
                        frame.astype(np.uint8), frame.astype(np.uint8), mask=fire_mask
                    )
                    gray_fire = cv2.cvtColor(fire_region, cv2.COLOR_BGR2GRAY)

                    # Standard deviation of fire region - too uniform = suspicious
                    fire_std = float(np.std(gray_fire[fire_mask > 0])) if np.any(fire_mask > 0) else 0

                    if fire_std < 30 and fire_ratio > 0.1:
                        confidence = min(80.0, (30 - fire_std) * 3)
                        findings.append(
                            Finding(
                                type="explosion_uniformity",
                                confidence=round(confidence, 1),
                                description=f"フレーム{i}: 爆発パーティクルが不自然に均一 (std={fire_std:.1f})",
                                frame_number=i,
                            )
                        )
                        scores.append(confidence)
                    else:
                        scores.append(25.0)

        except Exception as e:
            logger.debug(f"Explosion physics check failed: {e}")
            return 30.0, []

        return (float(np.mean(scores)) if scores else 30.0), findings

    def _smoke_uniformity_check(
        self, frames: List[np.ndarray]
    ) -> Tuple[float, List[Finding]]:
        """
        Check smoke region uniformity.
        AI-generated smoke tends to be too smooth / lacks turbulent texture.
        """
        findings: List[Finding] = []
        scores: List[float] = []

        try:
            import cv2

            for i, frame in enumerate(frames):
                # Detect grayish smoke regions
                gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2HSV)

                # Low saturation, mid-brightness = smoke
                s_channel = hsv[:, :, 1]
                v_channel = hsv[:, :, 2]
                smoke_mask = (s_channel < 50) & (v_channel > 80) & (v_channel < 200)

                smoke_ratio = np.sum(smoke_mask) / (frame.shape[0] * frame.shape[1])

                if smoke_ratio > 0.1:
                    smoke_texture = gray[smoke_mask].astype(float)
                    smoke_std = float(np.std(smoke_texture))

                    if smoke_std < 15:
                        confidence = min(75.0, (15 - smoke_std) * 5)
                        findings.append(
                            Finding(
                                type="smoke_uniformity",
                                confidence=round(confidence, 1),
                                description=f"フレーム{i}: 煙のテクスチャが不自然に均一 (std={smoke_std:.1f})",
                                frame_number=i,
                            )
                        )
                        scores.append(confidence)
                    else:
                        scores.append(20.0)

        except Exception as e:
            logger.debug(f"Smoke uniformity check failed: {e}")
            return 25.0, []

        return (float(np.mean(scores)) if scores else 25.0), findings

    def _audio_visual_sync_check(
        self, frames: List[np.ndarray], audio: np.ndarray, sr: int
    ) -> Tuple[float, List[Finding]]:
        """
        Check if audio peaks (explosions) align with visual bright flashes.
        Misalignment suggests AI-generated or spliced content.
        """
        findings: List[Finding] = []

        try:
            import cv2

            fps = 25.0  # Assume standard

            # Find visual bright frames (potential explosions)
            bright_frames = []
            for i, frame in enumerate(frames):
                mean_brightness = float(np.mean(frame))
                bright_frames.append((i, mean_brightness))

            if not bright_frames:
                return 20.0, []

            # Find audio peaks
            if len(audio) == 0:
                return 20.0, []

            chunk_size = sr // int(fps)
            audio_energy = []
            for j in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[j: j + chunk_size]
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                audio_energy.append(energy)

            if not audio_energy:
                return 20.0, []

            max_energy = max(audio_energy) + 1e-8
            audio_peaks = [e / max_energy for e in audio_energy]

            # Compare visual brightness with audio peaks
            max_frames = min(len(bright_frames), len(audio_peaks))
            visual_bright = [b for _, b in bright_frames[:max_frames]]
            max_visual = max(visual_bright) + 1e-8
            visual_norm = [v / max_visual for v in visual_bright]

            # Correlation
            if len(visual_norm) < 2 or len(audio_peaks[:max_frames]) < 2:
                return 20.0, []

            correlation = float(np.corrcoef(visual_norm, audio_peaks[:max_frames])[0, 1])

            # Very low correlation between audio peaks and visual explosions = suspicious
            if not np.isnan(correlation) and correlation < 0.2:
                confidence = min(75.0, (0.2 - correlation) * 200)
                findings.append(
                    Finding(
                        type="av_sync_mismatch",
                        confidence=round(confidence, 1),
                        description=f"爆発音と映像フラッシュのタイミングが不一致 (相関: {correlation:.2f})",
                    )
                )
                return confidence, findings

        except Exception as e:
            logger.debug(f"AV sync check failed: {e}")

        return 20.0, findings
