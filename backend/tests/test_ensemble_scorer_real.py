# © 2026 TimeWealthy Limited — FakeGuard
"""
tests/test_ensemble_scorer_real.py
TDD tests for EnsembleScorer (integrates real model outputs).
"""

import pytest
from app.analyzers.ensemble import EnsembleScorer, VERDICT_THRESHOLDS, DEFAULT_WEIGHTS, STANDARD_WEIGHTS
from app.analyzers.base import AnalyzerResult, Finding


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_result(score: float, findings=None, error=None) -> AnalyzerResult:
    """Create a mock AnalyzerResult."""
    return AnalyzerResult(score=score, findings=findings or [], error=error)


def make_finding(ftype: str, confidence: float, desc: str = "Test finding") -> Finding:
    """Create a mock Finding."""
    return Finding(type=ftype, confidence=confidence, description=desc)


# ─── Unit Tests ───────────────────────────────────────────────────────────────


class TestEnsembleScorerBase:
    """Unit tests for EnsembleScorer base functionality."""

    def test_empty_results_returns_uncertain(self):
        """Empty results should return 50.0 with uncertain verdict."""
        scorer = EnsembleScorer()
        output = scorer.score({})

        assert output["score"] == 50.0
        assert output["verdict"] == "uncertain"

    def test_single_analyzer_weighted_correctly(self):
        """Single analyzer result should produce correct weighted score."""
        scorer = EnsembleScorer()
        results = {"frame": make_result(80.0)}
        output = scorer.score(results, mode="standard")

        assert isinstance(output["score"], float)
        assert 0.0 <= output["score"] <= 100.0
        assert output["verdict"] in VERDICT_THRESHOLDS

    def test_all_analyzers_standard_mode(self):
        """Standard mode with all analyzers should produce valid output."""
        scorer = EnsembleScorer()
        results = {
            "frame": make_result(85.0),
            "temporal": make_result(75.0),
            "audio": make_result(60.0),
            "metadata": make_result(90.0),
        }
        output = scorer.score(results, mode="standard")

        assert isinstance(output["score"], float)
        assert 0.0 <= output["score"] <= 100.0
        assert output["verdict"] in VERDICT_THRESHOLDS
        assert "summary" in output
        assert "details" in output

    def test_war_footage_mode_includes_war_analyzer(self):
        """War footage mode should include war analyzer with its weight."""
        scorer = EnsembleScorer()
        results = {
            "frame": make_result(80.0),
            "temporal": make_result(75.0),
            "audio": make_result(60.0),
            "metadata": make_result(70.0),
            "war": make_result(95.0),
        }
        output = scorer.score(results, mode="war_footage")

        assert isinstance(output["score"], float)
        assert 0.0 <= output["score"] <= 100.0
        assert "war_analysis" in output["details"]

    def test_error_results_excluded_from_scoring(self):
        """Analyzers with errors should be excluded from weighted average."""
        scorer = EnsembleScorer()
        results = {
            "frame": make_result(85.0),
            "temporal": make_result(0.0, error="Some error"),
        }
        output = scorer.score(results, mode="standard")

        # Only frame result should contribute
        assert isinstance(output["score"], float)
        assert 0.0 <= output["score"] <= 100.0
        # Error analyzer should not appear in details
        assert "temporal_analysis" not in output["details"]

    def test_score_range_always_valid(self):
        """Output score must always be in [0, 100]."""
        scorer = EnsembleScorer()
        test_cases = [
            {"frame": make_result(0.0), "temporal": make_result(0.0)},
            {"frame": make_result(100.0), "temporal": make_result(100.0)},
            {"frame": make_result(50.0)},
            {"frame": make_result(99.9), "temporal": make_result(99.9), "audio": make_result(99.9)},
        ]
        for results in test_cases:
            output = scorer.score(results)
            assert 0.0 <= output["score"] <= 100.0, (
                f"Score out of range: {output['score']} for {results}"
            )


class TestVerdictMapping:
    """Test score-to-verdict mapping."""

    @pytest.mark.parametrize("score,expected_verdict", [
        (95.0, "ai_generated"),
        (85.0, "likely_ai"),
        (65.0, "uncertain"),
        (35.0, "likely_human"),
        (10.0, "human_made"),
    ])
    def test_verdict_from_score(self, score, expected_verdict):
        """Score should map to correct verdict."""
        scorer = EnsembleScorer()
        verdict = scorer._score_to_verdict(score)
        assert verdict == expected_verdict, (
            f"Score {score} should map to '{expected_verdict}', got '{verdict}'"
        )

    def test_all_verdict_thresholds_covered(self):
        """Every score from 0-100 should map to a verdict."""
        scorer = EnsembleScorer()
        for score in range(0, 101):
            verdict = scorer._score_to_verdict(float(score))
            assert verdict in VERDICT_THRESHOLDS, f"Score {score} has no verdict"


class TestHighConfidenceBoost:
    """Test score boost from high-confidence findings."""

    def test_high_confidence_finding_boosts_score(self):
        """Finding with confidence > 90 should add 8 points boost."""
        scorer = EnsembleScorer()
        high_conf_finding = make_finding("codec_signature", 95.0)
        results = {
            "frame": make_result(70.0),
            "metadata": make_result(70.0, findings=[high_conf_finding]),
        }
        output = scorer.score(results, mode="standard")

        # Base score ≈ 70, plus 8 boost = 78
        assert output["score"] >= 70.0

    def test_low_confidence_no_boost(self):
        """Finding with confidence < 70 should not add any boost."""
        scorer = EnsembleScorer()
        low_conf_finding = make_finding("texture_anomaly", 50.0)
        results = {
            "frame": make_result(60.0, findings=[low_conf_finding]),
        }
        output = scorer.score(results, mode="standard")

        # Score should be close to 60 (no boost)
        assert output["score"] <= 65.0


class TestWeights:
    """Test weight normalization."""

    def test_standard_weights_sum_to_one(self):
        """Standard weights must sum to 1.0."""
        total = sum(STANDARD_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Standard weights sum to {total}"

    def test_default_weights_sum_to_one(self):
        """Default (war) weights must sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Default weights sum to {total}"

    def test_get_weights_standard_mode(self):
        """Standard mode weights should be normalized correctly."""
        scorer = EnsembleScorer()
        weights = scorer._get_weights("standard", ["frame", "temporal", "audio", "metadata"])
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

    def test_get_weights_war_mode(self):
        """War mode weights should be normalized correctly."""
        scorer = EnsembleScorer()
        weights = scorer._get_weights("war_footage", ["frame", "temporal", "audio", "metadata", "war"])
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

    def test_missing_analyzers_weights_still_normalized(self):
        """Weights should normalize even if some analyzers are missing."""
        scorer = EnsembleScorer()
        weights = scorer._get_weights("standard", ["frame"])
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001


class TestDetailsOutput:
    """Test the details dict in output."""

    def test_details_include_analyzer_scores(self):
        """Details dict should include analyzer scores."""
        scorer = EnsembleScorer()
        results = {
            "frame": make_result(80.0),
            "audio": make_result(60.0),
        }
        output = scorer.score(results)

        assert "frame_analysis" in output["details"]
        assert "audio_analysis" in output["details"]
        assert output["details"]["frame_analysis"]["score"] == 80.0
        assert output["details"]["audio_analysis"]["score"] == 60.0

    def test_details_include_findings(self):
        """Details dict should include serialized findings."""
        scorer = EnsembleScorer()
        finding = make_finding("codec_signature", 95.0, "Test finding")
        finding.frame_number = 10
        finding.timestamp_sec = 0.4

        results = {"frame": make_result(90.0, findings=[finding])}
        output = scorer.score(results)

        frame_details = output["details"]["frame_analysis"]
        assert len(frame_details["findings"]) == 1
        f = frame_details["findings"][0]
        assert f["type"] == "codec_signature"
        assert f["confidence"] == 95.0
        assert f["frame"] == 10
        assert f["timestamp"] == 0.4

    def test_summary_includes_score(self):
        """Summary text should include the numerical score."""
        scorer = EnsembleScorer()
        results = {"frame": make_result(85.0)}
        output = scorer.score(results)

        assert "summary" in output
        assert isinstance(output["summary"], str)
        assert len(output["summary"]) > 0


class TestRealModelIntegration:
    """Tests specifically for real ML model output integration."""

    def test_siglip_output_format_compatible(self):
        """EnsembleScorer should handle FrameAnalyzer output with SigLIP scores."""
        scorer = EnsembleScorer()
        # Simulate a realistic SigLIP-based frame analyzer output
        siglip_finding = Finding(
            type="texture_anomaly",
            confidence=78.5,
            description="SigLIP detected AI-like texture patterns",
            frame_number=5,
            timestamp_sec=2.5,
        )
        results = {
            "frame": AnalyzerResult(score=82.0, findings=[siglip_finding]),
            "temporal": AnalyzerResult(score=65.0, findings=[]),
            "audio": AnalyzerResult(score=55.0, findings=[]),
            "metadata": AnalyzerResult(score=45.0, findings=[]),
        }
        output = scorer.score(results, mode="standard")

        assert isinstance(output["score"], float)
        assert 0.0 <= output["score"] <= 100.0
        assert output["verdict"] in VERDICT_THRESHOLDS

    def test_war_mode_with_all_real_analyzers(self):
        """War mode with all analyzer outputs should produce valid verdict."""
        scorer = EnsembleScorer()
        results = {
            "frame": AnalyzerResult(score=88.0, findings=[
                Finding(type="texture_anomaly", confidence=85.0, description="FFT anomaly")
            ]),
            "temporal": AnalyzerResult(score=72.0, findings=[
                Finding(type="unnatural_flow", confidence=70.0, description="Optical flow anomaly")
            ]),
            "audio": AnalyzerResult(score=60.0, findings=[]),
            "metadata": AnalyzerResult(score=95.0, findings=[
                Finding(type="codec_signature", confidence=95.0, description="Sora signature")
            ]),
            "war": AnalyzerResult(score=80.0, findings=[
                Finding(type="explosion_uniformity", confidence=75.0, description="Smoke uniformity")
            ]),
        }
        output = scorer.score(results, mode="war_footage")

        assert 0.0 <= output["score"] <= 100.0
        assert output["verdict"] in ("ai_generated", "likely_ai", "uncertain")
        assert "war_analysis" in output["details"]
