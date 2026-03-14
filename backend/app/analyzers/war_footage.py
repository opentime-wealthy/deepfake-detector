# © 2026 TimeWealthy Limited — FakeGuard
"""WarFootageAnalyzer: specialized checks for war/conflict footage."""

import logging
import numpy as np
from typing import List, Optional, Tuple
from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)

# Speed of sound at ~20°C (m/s) — used for audio-visual time lag estimation
SPEED_OF_SOUND_MS = 343.0


class WarFootageAnalyzer(BaseAnalyzer):
    """
    Specialized analyzer for war/conflict footage.

    Checks:
    - Explosion/fire particle pixel variance (too uniform = AI-generated)
    - Smoke texture uniformity (AI smoke lacks natural turbulence)
    - Audio-visual time lag analysis (speed-of-sound consistency)
    - Acoustic energy correlation with visual brightness flashes
    """

    def analyze(
        self,
        frames: List[np.ndarray],
        audio: Optional[np.ndarray] = None,
        sr: int = 44100,
    ) -> AnalyzerResult:
        """
        Args:
            frames: List of video frames (BGR numpy arrays)
            audio: Optional audio array (float32, mono)
            sr: Audio sample rate

        Returns:
            AnalyzerResult with score 0-100
        """
        if not frames:
            return AnalyzerResult(score=50.0, findings=[], error="No frames provided")

        findings: List[Finding] = []
        scores: List[float] = []

        # 1. Explosion/fire pixel variance check
        explosion_score, explosion_findings = self._explosion_physics_check(frames)
        scores.append(explosion_score)
        findings.extend(explosion_findings)

        # 2. Smoke uniformity check (pixel variance)
        smoke_score, smoke_findings = self._smoke_uniformity_check(frames)
        scores.append(smoke_score)
        findings.extend(smoke_findings)

        # 3. Audio-visual analysis (if audio available)
        if audio is not None and len(audio) > 0:
            # Time lag analysis (speed of sound)
            lag_score, lag_findings = self._audio_visual_time_lag_analysis(frames, audio, sr)
            scores.append(lag_score)
            findings.extend(lag_findings)

            # Acoustic-visual correlation
            av_score, av_findings = self._audio_visual_sync_check(frames, audio, sr)
            scores.append(av_score)
            findings.extend(av_findings)

        final_score = float(np.mean(scores)) if scores else 40.0
        return AnalyzerResult(score=round(final_score, 2), findings=findings)

    def _explosion_physics_check(
        self, frames: List[np.ndarray]
    ) -> Tuple[float, List[Finding]]:
        """
        Detect explosion-like regions and check pixel variance.

        AI-generated explosions have unnaturally uniform particle distributions
        (too little pixel variance within the explosion region).
        """
        findings: List[Finding] = []
        scores: List[float] = []

        try:
            import cv2

            for i, frame in enumerate(frames):
                hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2HSV)

                # Orange-red fire/explosion color range
                lower_fire = np.array([0, 50, 200])
                upper_fire = np.array([30, 255, 255])
                fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

                fire_ratio = float(np.sum(fire_mask > 0)) / (frame.shape[0] * frame.shape[1])

                if fire_ratio > 0.05:
                    # Pixel variance within explosion region
                    gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    fire_pixels = gray[fire_mask > 0]

                    if len(fire_pixels) == 0:
                        continue

                    fire_std = float(np.std(fire_pixels))
                    fire_variance = float(np.var(fire_pixels))

                    # AI explosions: very low variance (too uniform)
                    if fire_std < 30 and fire_ratio > 0.1:
                        confidence = min(80.0, (30 - fire_std) * 3)
                        findings.append(
                            Finding(
                                type="explosion_uniformity",
                                confidence=round(confidence, 1),
                                description=(
                                    f"フレーム{i}: 爆発パーティクルが不自然に均一 "
                                    f"(ピクセル標準偏差={fire_std:.1f}, 分散={fire_variance:.1f})"
                                ),
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
        Check smoke/dust region pixel variance.

        AI-generated smoke lacks natural turbulence → low pixel variance.
        Real smoke has fractal-like texture with high local variance.
        """
        findings: List[Finding] = []
        scores: List[float] = []

        try:
            import cv2

            for i, frame in enumerate(frames):
                hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

                # Low saturation, mid-brightness → smoke/dust region
                s_channel = hsv[:, :, 1]
                v_channel = hsv[:, :, 2]
                smoke_mask = (s_channel < 50) & (v_channel > 80) & (v_channel < 200)

                smoke_ratio = float(np.sum(smoke_mask)) / (frame.shape[0] * frame.shape[1])

                if smoke_ratio > 0.1:
                    smoke_pixels = gray[smoke_mask].astype(np.float64)
                    smoke_std = float(np.std(smoke_pixels))
                    smoke_variance = float(np.var(smoke_pixels))

                    # AI smoke: very low variance (too smooth)
                    if smoke_std < 15:
                        confidence = min(75.0, (15 - smoke_std) * 5)
                        findings.append(
                            Finding(
                                type="smoke_uniformity",
                                confidence=round(confidence, 1),
                                description=(
                                    f"フレーム{i}: 煙のテクスチャが不自然に均一 "
                                    f"(ピクセル標準偏差={smoke_std:.1f}, 分散={smoke_variance:.1f})"
                                ),
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

    def _audio_visual_time_lag_analysis(
        self, frames: List[np.ndarray], audio: np.ndarray, sr: int
    ) -> Tuple[float, List[Finding]]:
        """
        Analyze audio-visual time lag for physical plausibility.

        Real explosion footage: sound arrives slightly after the flash
        (speed of sound ~343 m/s). The expected lag depends on distance.

        AI-generated footage often has:
        - Zero lag (synchronized perfectly)
        - Implausible lag (off by more than a few seconds)
        - Random inconsistency between multiple explosions
        """
        findings: List[Finding] = []

        try:
            import cv2

            fps = 25.0  # Assume standard fps

            # Find visual bright frames (explosion/flash candidates)
            visual_brightness = []
            for i, frame in enumerate(frames):
                mean_b = float(np.mean(frame))
                visual_brightness.append(mean_b)

            if not visual_brightness:
                return 20.0, []

            # Normalize visual brightness
            vb_arr = np.array(visual_brightness)
            vb_norm = (vb_arr - vb_arr.min()) / (vb_arr.max() - vb_arr.min() + 1e-8)

            # Find audio energy per frame
            chunk_size = max(1, sr // int(fps))
            audio_energy = []
            for j in range(0, min(len(audio) - chunk_size, len(frames) * chunk_size), chunk_size):
                chunk = audio[j: j + chunk_size]
                energy = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                audio_energy.append(energy)

            if len(audio_energy) < 3:
                return 20.0, []

            ae_arr = np.array(audio_energy[:len(vb_norm)])
            ae_norm = (ae_arr - ae_arr.min()) / (ae_arr.max() - ae_arr.min() + 1e-8)

            # Find peaks in visual brightness
            visual_peaks = []
            for i in range(1, len(vb_norm) - 1):
                if vb_norm[i] > 0.7 and vb_norm[i] > vb_norm[i - 1] and vb_norm[i] > vb_norm[i + 1]:
                    visual_peaks.append(i)

            if not visual_peaks:
                return 20.0, []

            # For each visual peak, find nearest audio peak
            # Expected lag range: 0 to ~2 seconds (0-700m distance)
            expected_lag_frames_min = 0  # explosion right next to camera
            expected_lag_frames_max = int(2.0 * fps)  # ~686m away

            suspicious_lags = 0
            for peak_frame in visual_peaks:
                # Search for audio peak within expected lag window
                search_start = peak_frame
                search_end = min(len(ae_norm), peak_frame + expected_lag_frames_max + 5)

                if search_end <= search_start:
                    continue

                audio_window = ae_norm[search_start:search_end]
                if len(audio_window) == 0:
                    continue

                max_audio_in_window = float(np.max(audio_window))

                # If no audio peak after visual peak → suspicious (AI sync issue)
                if max_audio_in_window < 0.3:
                    suspicious_lags += 1

            if suspicious_lags > len(visual_peaks) // 2:
                confidence = min(70.0, suspicious_lags * 25)
                findings.append(
                    Finding(
                        type="av_time_lag_anomaly",
                        confidence=round(confidence, 1),
                        description=(
                            f"爆発フラッシュ後の音響エネルギーが不足 "
                            f"（{suspicious_lags}/{len(visual_peaks)} ピーク）"
                        ),
                    )
                )
                return confidence, findings

            return 20.0, []

        except Exception as e:
            logger.debug(f"AV time lag analysis failed: {e}")
            return 20.0, []

    def _audio_visual_sync_check(
        self, frames: List[np.ndarray], audio: np.ndarray, sr: int
    ) -> Tuple[float, List[Finding]]:
        """
        Check correlation between audio energy peaks and visual brightness flashes.
        Low correlation → likely AI-generated or spliced content.
        """
        findings: List[Finding] = []

        try:
            fps = 25.0

            # Visual brightness per frame
            visual_bright = [float(np.mean(frame)) for frame in frames]
            if not visual_bright:
                return 20.0, []

            # Audio energy per frame-sized chunk
            chunk_size = max(1, sr // int(fps))
            audio_energy = []
            for j in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[j: j + chunk_size]
                energy = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                audio_energy.append(energy)

            if not audio_energy:
                return 20.0, []

            max_frames = min(len(visual_bright), len(audio_energy))
            if max_frames < 2:
                return 20.0, []

            vb = visual_bright[:max_frames]
            ae = audio_energy[:max_frames]

            max_vb = max(vb) + 1e-8
            max_ae = max(ae) + 1e-8
            vb_norm = [v / max_vb for v in vb]
            ae_norm = [a / max_ae for a in ae]

            # Pearson correlation
            correlation = float(np.corrcoef(vb_norm, ae_norm)[0, 1])

            if np.isnan(correlation):
                return 20.0, []

            # Very low correlation → suspicious
            if correlation < 0.2:
                confidence = min(75.0, (0.2 - correlation) * 200)
                findings.append(
                    Finding(
                        type="av_sync_mismatch",
                        confidence=round(confidence, 1),
                        description=(
                            f"爆発音と映像フラッシュのタイミングが不一致 "
                            f"(相関係数: {correlation:.2f})"
                        ),
                    )
                )
                return confidence, findings

        except Exception as e:
            logger.debug(f"AV sync check failed: {e}")

        return 20.0, findings
