# © 2026 TimeWealthy Limited — FakeGuard
"""TemporalAnalyzer: inter-frame consistency, optical flow, and flicker detection."""

import logging
import numpy as np
from typing import List, Tuple
from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)


class TemporalAnalyzer(BaseAnalyzer):
    """
    Analyzes temporal consistency across video frames.

    Checks:
    - Optical flow: RAFT (torchvision) when available, else cv2 Farneback fallback
    - Inter-frame consistency: cosine similarity of flattened frames
    - Flicker detection: std deviation of adjacent frame brightness differences
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
            return AnalyzerResult(
                score=50.0,
                findings=[],
                error="Insufficient frames for temporal analysis",
            )

        findings: List[Finding] = []
        scores: List[float] = []

        # 1. Optical flow analysis (RAFT or Farneback)
        flow_score, flow_findings = self._optical_flow_analysis(frames, fps)
        scores.append(flow_score)
        findings.extend(flow_findings)

        # 2. Flicker detection
        flicker_score, flicker_findings = self._flicker_detection(frames, fps)
        scores.append(flicker_score)
        findings.extend(flicker_findings)

        # 3. Inter-frame consistency (cosine similarity)
        consistency_score, consistency_findings = self._inter_frame_consistency(frames, fps)
        scores.append(consistency_score)
        findings.extend(consistency_findings)

        final_score = float(np.mean(scores)) if scores else 50.0
        return AnalyzerResult(score=round(final_score, 2), findings=findings)

    def _optical_flow_analysis(
        self, frames: List[np.ndarray], fps: float
    ) -> Tuple[float, List[Finding]]:
        """
        Compute optical flow and detect unphysical motion patterns.

        Tries RAFT (torchvision) first; falls back to cv2 Farneback.
        AI-generated videos often have unnaturally smooth or uniform flow.
        """
        # Try RAFT first
        raft_result = self._optical_flow_raft(frames, fps)
        if raft_result is not None:
            return raft_result

        # Fallback to Farneback
        return self._optical_flow_farneback(frames, fps)

    def _optical_flow_raft(
        self, frames: List[np.ndarray], fps: float
    ) -> None:
        """
        RAFT optical flow via torchvision.models.optical_flow.raft_large.
        Returns (score, findings) or None if torch/torchvision unavailable.
        """
        try:
            import torch
            import torchvision.transforms.functional as F
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights).eval()

            findings: List[Finding] = []
            flow_scores: List[float] = []

            for i in range(1, len(frames)):
                prev_rgb = frames[i - 1][:, :, ::-1].astype(np.uint8)
                curr_rgb = frames[i][:, :, ::-1].astype(np.uint8)

                # RAFT requires 8x8 aligned images; resize if needed
                h, w = prev_rgb.shape[:2]
                h_pad = (8 - h % 8) % 8
                w_pad = (8 - w % 8) % 8
                if h_pad or w_pad:
                    prev_rgb = np.pad(prev_rgb, ((0, h_pad), (0, w_pad), (0, 0)))
                    curr_rgb = np.pad(curr_rgb, ((0, h_pad), (0, w_pad), (0, 0)))

                # HWC → BCHW tensor
                t1 = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).float()
                t2 = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).float()

                with torch.no_grad():
                    predicted_flows = model(t1, t2)
                flow = predicted_flows[-1][0].numpy()  # (2, H, W)

                magnitude = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
                mean_mag = float(np.mean(magnitude))
                std_mag = float(np.std(magnitude))

                # Cosine similarity between consecutive flow fields (if 2+ frames)
                score = self._flow_anomaly_score(mean_mag, std_mag, i, fps, findings)
                flow_scores.append(score)

            final = float(np.mean(flow_scores)) if flow_scores else 40.0
            return round(final, 2), findings

        except Exception as e:
            logger.debug(f"RAFT optical flow unavailable: {e}")
            return None

    def _optical_flow_farneback(
        self, frames: List[np.ndarray], fps: float
    ) -> Tuple[float, List[Finding]]:
        """
        Dense optical flow via cv2 Farneback (fallback when torch not available).
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
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = float(np.mean(magnitude))
                std_mag = float(np.std(magnitude))

                score = self._flow_anomaly_score(mean_mag, std_mag, i, fps, findings)
                flow_scores.append(score)

                prev_gray = curr_gray

        except Exception as e:
            logger.debug(f"Farneback optical flow failed: {e}")
            return 40.0, []

        final = float(np.mean(flow_scores)) if flow_scores else 40.0
        return round(final, 2), findings

    def _flow_anomaly_score(
        self,
        mean_mag: float,
        std_mag: float,
        frame_num: int,
        fps: float,
        findings: List[Finding],
    ) -> float:
        """
        Evaluate optical flow magnitude statistics for anomaly.
        Low coefficient of variation (CV) → suspiciously uniform flow.
        """
        if mean_mag > 1e-3:
            cv_flow = std_mag / mean_mag
            if cv_flow < 0.3:
                ts = frame_num / fps
                confidence = min(90.0, (0.3 - cv_flow) * 300)
                findings.append(
                    Finding(
                        type="unnatural_flow",
                        confidence=round(confidence, 1),
                        description=(
                            f"フレーム{frame_num}: オプティカルフローが不自然に均一 "
                            f"(CV={cv_flow:.3f})"
                        ),
                        frame_number=frame_num,
                        timestamp_sec=round(ts, 2),
                    )
                )
                return min(100.0, confidence)
            return 30.0
        return 30.0

    def _flicker_detection(
        self, frames: List[np.ndarray], fps: float
    ) -> Tuple[float, List[Finding]]:
        """
        Detect unnatural brightness flicker using std deviation of frame diffs.

        AI-generated videos often have isolated brightness spikes or temporal
        instability not present in real footage.
        """
        findings: List[Finding] = []
        flicker_scores: List[float] = []

        try:
            import cv2

            brightness = []
            for frame in frames:
                gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                brightness.append(float(np.mean(gray)))

            # Compute adjacent diffs
            diffs = np.abs(np.diff(brightness))
            diff_std = float(np.std(diffs))
            diff_mean = float(np.mean(diffs)) + 1e-8

            for i in range(1, len(brightness) - 1):
                delta_prev = abs(brightness[i] - brightness[i - 1])
                delta_next = abs(brightness[i] - brightness[i + 1])

                # Isolated spike: both neighbors differ by threshold
                if delta_prev > 15 and delta_next > 15:
                    severity = (delta_prev + delta_next) / 2
                    confidence = min(90.0, severity * 2)
                    ts = i / fps
                    findings.append(
                        Finding(
                            type="flicker",
                            confidence=round(confidence, 1),
                            description=(
                                f"フレーム{i}: 輝度フリッカー検出 "
                                f"(Δ={severity:.1f}, diff_std={diff_std:.2f})"
                            ),
                            frame_number=i,
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
    ) -> Tuple[float, List[Finding]]:
        """
        Check inter-frame consistency using cosine similarity.

        High cosine similarity between frames = consistent (expected for smooth video).
        Very low cosine similarity = abrupt change (temporal inconsistency).
        Very high cosine similarity across all frames = suspiciously static (AI-like).

        Also uses MSE for detecting extreme jumps.
        """
        findings: List[Finding] = []
        consistency_scores: List[float] = []

        try:
            import cv2

            for i in range(1, len(frames)):
                prev = cv2.cvtColor(frames[i - 1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
                curr = cv2.cvtColor(frames[i].astype(np.uint8), cv2.COLOR_BGR2GRAY)

                prev_flat = prev.flatten().astype(np.float64)
                curr_flat = curr.flatten().astype(np.float64)

                # Cosine similarity
                norm_prev = np.linalg.norm(prev_flat) + 1e-10
                norm_curr = np.linalg.norm(curr_flat) + 1e-10
                cosine_sim = float(np.dot(prev_flat, curr_flat) / (norm_prev * norm_curr))

                # MSE as cross-check for abrupt changes
                diff = np.abs(prev.astype(float) - curr.astype(float))
                mse = float(np.mean(diff ** 2))

                # Abrupt scene cut: low cosine similarity + high MSE
                if mse > 2000 or cosine_sim < 0.5:
                    confidence = min(85.0, mse / 50)
                    ts = i / fps
                    findings.append(
                        Finding(
                            type="temporal_inconsistency",
                            confidence=round(confidence, 1),
                            description=(
                                f"フレーム{i}: フレーム間の急激な変化 "
                                f"(MSE={mse:.0f}, cosine_sim={cosine_sim:.3f})"
                            ),
                            frame_number=i,
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
