# © 2026 TimeWealthy Limited — DeepGuard
"""Tests for TemporalAnalyzer."""

import numpy as np
import pytest

from app.analyzers.temporal import TemporalAnalyzer


class TestTemporalAnalyzer:

    def test_analyze_returns_result(self, random_frames):
        """analyze() returns AnalyzerResult"""
        analyzer = TemporalAnalyzer()
        result = analyzer.analyze(random_frames, fps=25.0)
        assert result is not None
        assert 0.0 <= result.score <= 100.0

    def test_single_frame_returns_error(self, random_frame):
        """フレームが1枚 → error付きの結果"""
        analyzer = TemporalAnalyzer()
        result = analyzer.analyze([random_frame], fps=25.0)
        assert result.has_error

    def test_findings_is_list(self, random_frames):
        """findings は常にリスト"""
        analyzer = TemporalAnalyzer()
        result = analyzer.analyze(random_frames)
        assert isinstance(result.findings, list)

    def test_flickering_frames_detected(self):
        """急激な輝度変化 → フリッカー検出"""
        analyzer = TemporalAnalyzer()
        frames = []
        for i in range(10):
            # Alternating bright and dark frames
            val = 200 if i % 2 == 0 else 30
            frames.append(np.full((240, 320, 3), val, dtype=np.uint8))

        result = analyzer.analyze(frames, fps=25.0)
        # Should detect flicker
        flicker_findings = [f for f in result.findings if f.type == "flicker"]
        assert len(flicker_findings) > 0

    def test_stable_frames_low_flicker_score(self):
        """安定したフレーム → フリッカー少ない"""
        analyzer = TemporalAnalyzer()
        # Slowly varying frames
        frames = []
        for i in range(10):
            val = 128 + i  # very slow brightness increase
            frames.append(np.full((240, 320, 3), val, dtype=np.uint8))

        result = analyzer._flicker_detection(frames, fps=25.0)
        score, findings = result
        flicker_findings = [f for f in findings if f.type == "flicker"]
        assert len(flicker_findings) == 0

    def test_temporal_inconsistency_detected(self):
        """フレーム間の急激な変化 → 時間的不整合検出"""
        analyzer = TemporalAnalyzer()
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        frame2 = np.full((240, 320, 3), 255, dtype=np.uint8)  # Completely different

        result = analyzer._inter_frame_consistency([frame1, frame2], fps=25.0)
        score, findings = result
        assert score > 50.0 or len(findings) > 0

    def test_identical_frames_low_inconsistency(self):
        """同一フレーム → 時間的一貫性が高い"""
        analyzer = TemporalAnalyzer()
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        frames = [frame.copy() for _ in range(5)]

        score, findings = analyzer._inter_frame_consistency(frames, fps=25.0)
        inconsistency_findings = [f for f in findings if f.type == "temporal_inconsistency"]
        assert len(inconsistency_findings) == 0

    def test_score_always_0_to_100(self):
        """スコアは常に0-100"""
        analyzer = TemporalAnalyzer()
        rng = np.random.default_rng(0)
        for _ in range(3):
            frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(5)]
            result = analyzer.analyze(frames)
            assert 0.0 <= result.score <= 100.0

    def test_finding_has_required_fields(self):
        """Finding には type, confidence, description がある"""
        analyzer = TemporalAnalyzer()
        frames = []
        for i in range(10):
            val = 200 if i % 2 == 0 else 20
            frames.append(np.full((240, 320, 3), val, dtype=np.uint8))

        result = analyzer.analyze(frames, fps=25.0)
        for finding in result.findings:
            assert hasattr(finding, "type")
            assert hasattr(finding, "confidence")
            assert hasattr(finding, "description")
            assert 0.0 <= finding.confidence <= 100.0
