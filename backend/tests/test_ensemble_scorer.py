# © 2026 TimeWealthy Limited — DeepGuard
"""Tests for EnsembleScorer."""

import pytest
from app.analyzers.ensemble import EnsembleScorer
from app.analyzers.base import AnalyzerResult, Finding


def make_result(score: float, findings=None) -> AnalyzerResult:
    return AnalyzerResult(score=score, findings=findings or [])


class TestEnsembleScorer:

    def test_high_ai_score_returns_ai_generated(self):
        """全アナライザーが高スコア → ai_generated判定"""
        scorer = EnsembleScorer()
        results = {
            "frame": make_result(95.0),
            "temporal": make_result(93.0),
            "audio": make_result(91.0),
            "metadata": make_result(92.0),
        }
        verdict = scorer.score(results, mode="standard")
        assert verdict["verdict"] == "ai_generated"
        assert verdict["score"] >= 91.0

    def test_low_score_returns_human_made(self):
        """全アナライザーが低スコア → human_made判定"""
        scorer = EnsembleScorer()
        results = {
            "frame": make_result(10.0),
            "temporal": make_result(15.0),
            "audio": make_result(12.0),
            "metadata": make_result(8.0),
        }
        verdict = scorer.score(results, mode="standard")
        assert verdict["verdict"] == "human_made"
        assert verdict["score"] <= 25.0

    def test_mixed_scores_returns_uncertain(self):
        """スコアが混在 → uncertain判定"""
        scorer = EnsembleScorer()
        results = {
            "frame": make_result(70.0),
            "temporal": make_result(50.0),
            "audio": make_result(40.0),
            "metadata": make_result(60.0),
        }
        verdict = scorer.score(results, mode="standard")
        assert verdict["verdict"] in ("uncertain", "likely_ai", "likely_human")

    def test_single_high_confidence_finding_elevates_score(self):
        """1つでも高信頼度の異常 → スコア上昇"""
        scorer = EnsembleScorer()
        high_conf_finding = Finding(
            type="codec_signature",
            confidence=95.0,
            description="AI生成ツールのシグネチャ",
        )
        results = {
            "frame": make_result(60.0),
            "temporal": make_result(50.0),
            "metadata": make_result(80.0, findings=[high_conf_finding]),
        }
        verdict_with_finding = scorer.score(results, mode="standard")

        results_no_finding = {
            "frame": make_result(60.0),
            "temporal": make_result(50.0),
            "metadata": make_result(80.0),
        }
        verdict_no_finding = scorer.score(results_no_finding, mode="standard")

        assert verdict_with_finding["score"] >= verdict_no_finding["score"]

    def test_war_mode_includes_war_analyzer(self):
        """war_footageモードでwarアナライザーが含まれる"""
        scorer = EnsembleScorer()
        results = {
            "frame": make_result(70.0),
            "temporal": make_result(65.0),
            "audio": make_result(60.0),
            "metadata": make_result(55.0),
            "war": make_result(80.0),
        }
        verdict = scorer.score(results, mode="war_footage")
        assert "war_analysis" in verdict["details"]

    def test_empty_results_returns_uncertain(self):
        """解析結果がない → uncertain"""
        scorer = EnsembleScorer()
        verdict = scorer.score({})
        assert verdict["verdict"] == "uncertain"
        assert verdict["score"] == 50.0

    def test_error_results_are_excluded(self):
        """エラーのある結果は除外される"""
        scorer = EnsembleScorer()
        failed_result = AnalyzerResult(score=99.0, findings=[], error="Something broke")
        results = {
            "frame": failed_result,
            "temporal": make_result(20.0),
            "metadata": make_result(15.0),
        }
        verdict = scorer.score(results, mode="standard")
        # Failed frame result should be excluded; score should reflect temporal+metadata only
        assert verdict["score"] < 50.0

    def test_score_is_between_0_and_100(self):
        """スコアは常に0-100の範囲"""
        scorer = EnsembleScorer()
        for score_val in [0.0, 50.0, 100.0]:
            results = {"frame": make_result(score_val), "temporal": make_result(score_val)}
            verdict = scorer.score(results)
            assert 0.0 <= verdict["score"] <= 100.0

    def test_verdict_thresholds(self):
        """判定基準: スコアと verdict のマッピング確認"""
        scorer = EnsembleScorer()
        cases = [
            (10.0, "human_made"),
            (30.0, "likely_human"),
            (60.0, "uncertain"),
            (82.0, "likely_ai"),
            (95.0, "ai_generated"),
        ]
        for input_score, expected_verdict in cases:
            assert scorer._score_to_verdict(input_score) == expected_verdict, \
                f"Score {input_score} should map to {expected_verdict}"

    def test_summary_contains_score(self):
        """summaryにスコアが含まれる"""
        scorer = EnsembleScorer()
        results = {"frame": make_result(85.0), "temporal": make_result(82.0)}
        verdict = scorer.score(results)
        assert "%" in verdict["summary"]

    def test_details_keys_present(self):
        """detailsに各アナライザーのキーが含まれる"""
        scorer = EnsembleScorer()
        results = {
            "frame": make_result(70.0),
            "temporal": make_result(65.0),
        }
        verdict = scorer.score(results)
        assert "frame_analysis" in verdict["details"]
        assert "temporal_analysis" in verdict["details"]
