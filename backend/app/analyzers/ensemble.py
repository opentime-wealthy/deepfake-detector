# © 2026 TimeWealthy Limited — FakeGuard
"""EnsembleScorer: combine analyzer results into final verdict."""

from typing import Dict, Optional
from app.analyzers.base import AnalyzerResult


VERDICT_THRESHOLDS = {
    "ai_generated": (91, 100),
    "likely_ai": (76, 90),
    "uncertain": (51, 75),
    "likely_human": (26, 50),
    "human_made": (0, 25),
}

VERDICT_MESSAGES = {
    "ai_generated": "この動画はAI生成の可能性が極めて高いです",
    "likely_ai": "この動画はAI生成の可能性が高いです",
    "uncertain": "判定困難です。一部AI加工の疑いがあります",
    "likely_human": "この動画は人間制作の可能性が高いです",
    "human_made": "この動画は人間制作と判断されます",
}

# War footage mode (includes war analyzer, redistributes weights)
DEFAULT_WEIGHTS = {
    "restrav": 0.40,   # ReStraV: DINOv2 perceptual straightening (primary)
    "temporal": 0.20,  # Optical flow
    "audio": 0.15,     # Audio MFCC/spectral
    "c2pa": 0.15,      # Content Credentials metadata
    "war": 0.10,       # War footage heuristics
}

# Standard mode (no war analyzer) — c2pa replaces metadata slot
STANDARD_WEIGHTS = {
    "restrav": 0.40,   # ReStraV: DINOv2 perceptual straightening (primary)
    "temporal": 0.20,  # Optical flow
    "audio": 0.15,     # Audio MFCC/spectral
    "c2pa": 0.15,      # Content Credentials metadata
    # metadata: folded into c2pa weight when not available
}

# Legacy weights (kept for backward compatibility — frame = old SigLIP slot)
LEGACY_WEIGHTS = {
    "frame": 0.40,
    "temporal": 0.20,
    "audio": 0.15,
    "c2pa": 0.15,
    "war": 0.10,
}


class EnsembleScorer:
    """
    Combines results from individual analyzers into a final AI-generation score.

    Weight configuration (v2 — ReStraV model):
    - Standard mode: restrav=0.40, temporal=0.20, audio=0.15, c2pa=0.15
    - War footage mode: restrav=0.40, temporal=0.20, audio=0.15, c2pa=0.15, war=0.10
    
    Boost logic:
    - finding.confidence > 90  → +8 points
    - finding.confidence > 80  → +4 points
    - finding.confidence > 70  → +2 points
    """

    def score(
        self,
        results: Dict[str, AnalyzerResult],
        mode: str = "standard",
    ) -> Dict:
        """
        Compute final verdict from analyzer results.

        Args:
            results: Dict mapping analyzer name → AnalyzerResult
            mode: 'standard' or 'war_footage'

        Returns:
            Dict with keys: score, verdict, summary, details
        """
        if not results:
            return {
                "score": 50.0,
                "verdict": "uncertain",
                "summary": "解析結果が不足しています",
                "details": {},
            }

        weights = self._get_weights(mode, list(results.keys()))
        total_weight = 0.0
        weighted_sum = 0.0
        details = {}

        for name, result in results.items():
            if result is None or result.has_error:
                continue

            w = weights.get(name, 0.0)
            weighted_sum += result.score * w
            total_weight += w

            # Serialize findings
            findings_list = []
            for f in result.findings:
                fd: Dict = {
                    "type": f.type,
                    "confidence": f.confidence,
                    "description": f.description,
                }
                if f.frame_number is not None:
                    fd["frame"] = f.frame_number
                if f.timestamp_sec is not None:
                    fd["timestamp"] = f.timestamp_sec
                if f.frames is not None:
                    fd["frames"] = f.frames
                if f.metadata is not None:
                    fd["metadata"] = f.metadata
                findings_list.append(fd)

            details[f"{name}_analysis"] = {
                "score": result.score,
                "findings": findings_list,
            }

        # Weighted base score
        base_score = (weighted_sum / total_weight) if total_weight > 0 else 50.0

        # Boost for high-confidence findings
        max_finding_confidence = 0.0
        for result in results.values():
            if result and not result.has_error:
                for finding in result.findings:
                    if finding.confidence > max_finding_confidence:
                        max_finding_confidence = finding.confidence

        boost = 0.0
        if max_finding_confidence > 90.0:
            boost = 8.0
        elif max_finding_confidence > 80.0:
            boost = 4.0
        elif max_finding_confidence > 70.0:
            boost = 2.0

        final_score = min(100.0, base_score + boost)
        verdict = self._score_to_verdict(final_score)
        summary = f"{VERDICT_MESSAGES[verdict]}（{final_score:.1f}%）"

        return {
            "score": round(final_score, 2),
            "verdict": verdict,
            "summary": summary,
            "details": details,
        }

    def _get_weights(self, mode: str, available_analyzers: list) -> Dict[str, float]:
        """Return weights normalized to available analyzers."""
        if mode == "war_footage":
            base = dict(DEFAULT_WEIGHTS)
        else:
            base = dict(STANDARD_WEIGHTS)

        # Filter to analyzers that actually provided results
        filtered = {k: v for k, v in base.items() if k in available_analyzers}

        if not filtered:
            # Uniform distribution fallback
            return {k: 1.0 / len(available_analyzers) for k in available_analyzers}

        total = sum(filtered.values())
        if total == 0:
            return {k: 1.0 / len(filtered) for k in filtered}

        return {k: v / total for k, v in filtered.items()}

    def _score_to_verdict(self, score: float) -> str:
        """Map numerical score [0-100] to verdict label."""
        for verdict, (low, high) in VERDICT_THRESHOLDS.items():
            if low <= score <= high:
                return verdict
        return "uncertain"
