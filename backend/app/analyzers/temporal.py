# © 2026 TimeWealthy Limited — DeepGuard
"""TemporalAnalyzer: inter-frame consistency and motion analysis."""

import logging
import numpy as np
from typing import List
from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)


class TemporalAnalyzer(BaseAnalyzer):
    """
    Analyzes temporal consistency across frames.

    Checks:
    - Optical flow consistency
    - Flicker / edge instability detection
    - Inter-frame consistency score
    """

    def analyze(self, frames: List[np.ndarray], fps: float = 25.0) -> AnalyzerResult:
        """
        Args:
            frames: Ordered list of frames (BGR numpy arrays)
            fps: Video frame rate

        Returns:
            AnalyzerResult with score 0-100
        """
        if len(frames) < 2:
            return AnalyzerResult(score=50.0, findings=[], error="Insufficient frames for temporal analysis")

        findings: List[Finding] = []
        scores: List[float] = []

        # 1. Optical flow analysis
        flow_score, flow_findings = self._optical_flow_analysis(frames, fps)
        scores.append(flow_score)
        findings.extend(flow_findings)

        # 2. Flicker detection
        flicker_score, flicker_findings = self._flicker_detection(frames, fps)
        scores.append(flicker_score)
        findings.extend(flicker_findings)

        # 3. Inter-frame consistency
        consistency_score, consistency_findings = self._inter_frame_consistency(frames, fps)
        scores.append(consistency_score)
        findings.extend(consistency_findings)

        final_score = float(np.mean(scores)) if scores else 50.0
        return AnalyzerResult(score=round(final_score, 2), findings=findings)

    def _optical_flow_analysis(
        self, frames: List[np.ndarray], fps: float
    ) -> tuple:
        """
        Compute dense optical flow and detect unphysical motion patterns.
        AI-generated videos often have inconsistent or unrealistically smooth flow.
        """
        findings: List[Finding] = []
        flow_scores: List[float] = []

        try:
            import cv2

            prev_gray = cv2.cvtColor(frames[0].astype(np.uint8), cv2.COLOR_BGR2GRAY)

            for i in range(1, len(frames)):
                curr_gray = cv2.cvtColor(frames[i].astype(np.uint8), cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray,
                    None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )

                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = float(np.mean(magnitude))
                std_mag = float(np.std(magnitude))

                # AI videos sometimes have suspiciously uniform flow (std/mean very low)
                if mean_mag > 1e-3:
                    cv_flow = std_mag / mean_mag
                    # Low CV = too uniform = suspicious
                    if cv_flow < 0.3:
                        frame_num = i
                        ts = frame_num / fps
                        confidence = min(90.0, (0.3 - cv_flow) * 300)
                        findings.append(
                            Finding(
                                type="unnatural_flow",
                                confidence=round(confidence, 1),
                                description=f"フレーム{frame_num}: オプティカルフローが不自然に均一 (CV={cv_flow:.3f})",
                                frame_number=frame_num,
                                timestamp_sec=round(ts, 2),
                            )
                        )
                        flow_scores.append(min(100.0, confidence))
                    else:
                        flow_scores.append(30.0)
                else:
                    flow_scores.append(30.0)

                prev_gray = curr_gray

        except Exception as e:
            logger.debug(f"Optical flow analysis failed: {e}")
            return 40.0, []

        score = float(np.mean(flow_scores)) if flow_scores else 40.0
        return round(score, 2), findings

    def _flicker_detection(
        self, frames: List[np.ndarray], fps: float
    ) -> tuple:
        """Detect unnatural brightness flicker between consecutive frames."""
        findings: List[Finding] = []
        flicker_scores: List[float] = []

        try:
            import cv2

            brightness = []
            for frame in frames:
                gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                brightness.append(float(np.mean(gray)))

            for i in range(1, len(brightness) - 1):
                prev_b = brightness[i - 1]
                curr_b = brightness[i]
                next_b = brightness[i + 1]
                # Isolated spike (flicker)
                delta_prev = abs(curr_b - prev_b)
                delta_next = abs(curr_b - next_b)
                if delta_prev > 15 and delta_next > 15:
                    severity = (delta_prev + delta_next) / 2
                    confidence = min(90.0, severity * 2)
                    frame_num = i
                    ts = frame_num / fps
                    findings.append(
                        Finding(
                            type="flicker",
                            confidence=round(confidence, 1),
                            description=f"フレーム{frame_num}: 輝度フリッカー検出 (Δ={severity:.1f})",
                            frame_number=frame_num,
                            frames=[i - 1, i, i + 1],
                            timestamp_sec=round(ts, 2),
                        )
                    )
                    flicker_scores.append(confidence)
                else:
                    flicker_scores.append(20.0)

        except Exception as e:
            logger.debug(f"Flicker detection failed: {e}")
            return 30.0, []

        score = float(np.mean(flicker_scores)) if flicker_scores else 30.0
        return round(score, 2), findings

    def _inter_frame_consistency(
        self, frames: List[np.ndarray], fps: float
    ) -> tuple:
        """Check structural similarity (SSIM) between consecutive frames."""
        findings: List[Finding] = []
        consistency_scores: List[float] = []

        try:
            import cv2

            for i in range(1, len(frames)):
                prev = cv2.cvtColor(frames[i - 1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
                curr = cv2.cvtColor(frames[i].astype(np.uint8), cv2.COLOR_BGR2GRAY)

                # Simple MSE-based consistency (SSIM optional dep)
                diff = np.abs(prev.astype(float) - curr.astype(float))
                mse = float(np.mean(diff ** 2))

                # Very high MSE between adjacent frames = temporal inconsistency
                if mse > 2000:
                    confidence = min(85.0, mse / 50)
                    frame_num = i
                    ts = frame_num / fps
                    findings.append(
                        Finding(
                            type="temporal_inconsistency",
                            confidence=round(confidence, 1),
                            description=f"フレーム{frame_num}: フレーム間の急激な変化 (MSE={mse:.0f})",
                            frame_number=frame_num,
                            timestamp_sec=round(ts, 2),
                        )
                    )
                    consistency_scores.append(confidence)
                else:
                    consistency_scores.append(20.0)

        except Exception as e:
            logger.debug(f"Inter-frame consistency check failed: {e}")
            return 30.0, []

        score = float(np.mean(consistency_scores)) if consistency_scores else 30.0
        return round(score, 2), findings
