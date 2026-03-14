# © 2026 TimeWealthy Limited — DeepGuard
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

DEFAULT_WEIGHTS = {
    "frame": 0.30,
    "temporal": 0.30,
    "audio": 0.15,
    "metadata": 0.15,
    "war": 0.10,
}

STANDARD_WEIGHTS = {
    "frame": 0.35,
    "temporal": 0.35,
    "audio": 0.175,
    "metadata": 0.125,
}


class EnsembleScorer:
    """
    Combines results from individual analyzers into a final AI-generation score and verdict.

    Weights:
    - Standard mode: frame=0.35, temporal=0.35, audio=0.175, metadata=0.125
    - War footage mode: adds war=0.10, redistributes proportionally
    """

    def score(
        self,
        results: Dict[str, AnalyzerResult],
        mode: str = "standard",
    ) -> Dict:
        """
        Compute final verdict.

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

        weights = self._get_weights(mode, results.keys())
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
                fd = {
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
                findings_list.append(fd)

            details[f"{name}_analysis"] = {
                "score": result.score,
                "findings": findings_list,
            }

        # Base score
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

    def _get_weights(self, mode: str, available_analyzers) -> Dict[str, float]:
        """Return normalized weights for available analyzers."""
        if mode == "war_footage":
            base = dict(DEFAULT_WEIGHTS)
        else:
            base = dict(STANDARD_WEIGHTS)

        # Filter to available analyzers only
        filtered = {k: v for k, v in base.items() if k in available_analyzers}
        total = sum(filtered.values())
        if total == 0:
            return {k: 1.0 / len(filtered) for k in filtered}
        return {k: v / total for k, v in filtered.items()}

    def _score_to_verdict(self, score: float) -> str:
        """Map numerical score to verdict label."""
        for verdict, (low, high) in VERDICT_THRESHOLDS.items():
            if low <= score <= high:
                return verdict
        return "uncertain"
