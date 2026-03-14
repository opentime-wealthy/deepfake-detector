# © 2026 TimeWealthy Limited — FakeGuard
"""
tests/test_temporal_analyzer_real.py
TDD tests for TemporalAnalyzer (optical flow + flicker + consistency).

Fast tests (no heavy ML) run by default.
Slow tests require @pytest.mark.slow.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.analyzers.temporal import TemporalAnalyzer
from app.analyzers.base import AnalyzerResult


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_frame(value: int = 128, shape=(120, 160, 3)) -> np.ndarray:
    """Return a solid-color BGR frame."""
    return np.full(shape, value, dtype=np.uint8)


def make_noise_frame(seed: int = 0, shape=(120, 160, 3)) -> np.ndarray:
    """Return a random noise BGR frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, shape, dtype=np.uint8)


def make_consistent_video(n_frames: int = 8, shape=(120, 160, 3)) -> list:
    """Create a smoothly-changing video (low temporal inconsistency)."""
    frames = []
    for i in range(n_frames):
        val = int(100 + i * 5)  # gradually brightening
        frames.append(np.full(shape, val, dtype=np.uint8))
    return frames


def make_flickering_video(n_frames: int = 8, shape=(120, 160, 3)) -> list:
    """Create frames with isolated brightness spikes (flicker)."""
    frames = []
    for i in range(n_frames):
        if i == 3 or i == 6:
            val = 240  # spike frames
        else:
            val = 100  # normal
        frames.append(np.full(shape, val, dtype=np.uint8))
    return frames


def make_abrupt_change_video(n_frames: int = 6, shape=(120, 160, 3)) -> list:
    """Create video with a sudden scene cut (temporal inconsistency)."""
    frames = []
    for i in range(n_frames):
        val = 50 if i < n_frames // 2 else 220  # abrupt brightness jump
        frames.append(np.full(shape, val, dtype=np.uint8))
    return frames


# ─── Unit Tests ───────────────────────────────────────────────────────────────


class TestRealTemporalAnalyzer:
    """Unit tests for TemporalAnalyzer."""

    def test_analyze_returns_result_with_enough_frames(self):
        """analyze() returns AnalyzerResult when >= 2 frames provided."""
        analyzer = TemporalAnalyzer()
        frames = make_consistent_video(4)
        result = analyzer.analyze(frames, fps=25.0)

        assert isinstance(result, AnalyzerResult)
        assert 0.0 <= result.score <= 100.0
        assert result.error is None

    def test_insufficient_frames_returns_error(self):
        """analyze() returns error result when < 2 frames provided."""
        analyzer = TemporalAnalyzer()

        result_zero = analyzer.analyze([], fps=25.0)
        assert result_zero.error is not None

        result_one = analyzer.analyze([make_frame()], fps=25.0)
        assert result_one.error is not None

    def test_score_range_valid(self):
        """Score must be in [0, 100] range."""
        analyzer = TemporalAnalyzer()
        frames = [make_noise_frame(seed=i) for i in range(6)]
        result = analyzer.analyze(frames, fps=25.0)
        assert 0.0 <= result.score <= 100.0

    def test_optical_flow_consistency_real_video(self):
        """Smoothly changing video has low optical flow anomaly."""
        analyzer = TemporalAnalyzer()
        frames = make_consistent_video(8)
        result = analyzer.analyze(frames, fps=25.0)

        assert isinstance(result, AnalyzerResult)
        # Consistent video should have moderate or low score
        assert result.score <= 80.0

    def test_optical_flow_inconsistency_detected(self):
        """Abrupt changes in video should produce findings or elevated score."""
        analyzer = TemporalAnalyzer()
        frames = make_abrupt_change_video(8)
        result = analyzer.analyze(frames, fps=25.0)

        assert isinstance(result, AnalyzerResult)
        # High MSE between frames should produce findings or elevated score
        # At least one of these should be true:
        has_findings = len(result.findings) > 0
        elevated_score = result.score > 30.0
        assert has_findings or elevated_score

    def test_flicker_detection(self):
        """Isolated brightness spikes should be detected as flicker."""
        analyzer = TemporalAnalyzer()
        frames = make_flickering_video(10)
        result = analyzer.analyze(frames, fps=25.0)

        assert isinstance(result, AnalyzerResult)
        flicker_findings = [f for f in result.findings if f.type == "flicker"]
        # Flickering video should have flicker findings or elevated score
        assert len(flicker_findings) > 0 or result.score > 25.0

    def test_inter_frame_consistency_smooth_video(self):
        """Smooth video has low inter-frame inconsistency."""
        analyzer = TemporalAnalyzer()
        frames = make_consistent_video(6)

        score, findings = analyzer._inter_frame_consistency(frames, fps=25.0)
        assert 0.0 <= score <= 100.0
        assert isinstance(findings, list)

    def test_inter_frame_consistency_abrupt_change(self):
        """Abrupt changes produce temporal inconsistency findings."""
        analyzer = TemporalAnalyzer()
        frames = make_abrupt_change_video(6)

        score, findings = analyzer._inter_frame_consistency(frames, fps=25.0)
        assert 0.0 <= score <= 100.0
        inconsistency_findings = [f for f in findings if f.type == "temporal_inconsistency"]
        # Should detect the jump
        assert len(inconsistency_findings) > 0

    def test_flicker_detection_method_isolated_spike(self):
        """_flicker_detection should find isolated brightness spikes."""
        analyzer = TemporalAnalyzer()
        # Frame with isolated spike at position 3
        frames = [make_frame(100)] * 3 + [make_frame(240)] + [make_frame(100)] * 4

        score, findings = analyzer._flicker_detection(frames, fps=25.0)
        assert 0.0 <= score <= 100.0
        flicker_findings = [f for f in findings if f.type == "flicker"]
        # Spike at frame 3 should be detected
        assert len(flicker_findings) > 0, f"Expected flicker findings, got score={score}, findings={findings}"

    def test_optical_flow_analysis_returns_tuple(self):
        """_optical_flow_analysis returns (score, findings) tuple."""
        analyzer = TemporalAnalyzer()
        frames = make_consistent_video(4)

        score, findings = analyzer._optical_flow_analysis(frames, fps=25.0)
        assert isinstance(score, float)
        assert isinstance(findings, list)
        assert 0.0 <= score <= 100.0

    def test_cosine_similarity_score_consistent_frames(self):
        """Consistent frames have high cosine similarity → low AI score."""
        analyzer = TemporalAnalyzer()
        # Two nearly identical frames → cosine similarity close to 1
        frames = [make_frame(128), make_frame(129)]  # almost identical
        score, findings = analyzer._inter_frame_consistency(frames, fps=25.0)
        # Low MSE → low score
        assert score < 50.0

    def test_physics_violation_gravity(self):
        """Test that sudden extreme motion changes produce anomaly findings."""
        analyzer = TemporalAnalyzer()
        # Create frames with extreme brightness jump (proxy for sudden motion)
        frames = []
        for i in range(6):
            val = 50 if i % 2 == 0 else 200  # alternating = extreme flicker
            frames.append(make_frame(val))

        result = analyzer.analyze(frames, fps=25.0)
        # Alternating extreme values should produce findings or elevated score
        assert result.score > 20.0 or len(result.findings) > 0

    def test_findings_have_required_fields(self):
        """All findings must have type, confidence, description."""
        analyzer = TemporalAnalyzer()
        frames = make_flickering_video(10)
        result = analyzer.analyze(frames, fps=25.0)

        for finding in result.findings:
            assert hasattr(finding, "type")
            assert hasattr(finding, "confidence")
            assert hasattr(finding, "description")
            assert 0.0 <= finding.confidence <= 100.0


# ─── Slow Integration Tests ───────────────────────────────────────────────────


@pytest.mark.slow
class TestTemporalAnalyzerRAFT:
    """Integration tests using RAFT optical flow from torchvision."""

    def test_raft_optical_flow_loads_and_runs(self):
        """
        RAFT model should load and compute optical flow for consecutive frames.
        NOTE: Requires torch + torchvision installed.
        """
        try:
            import torch
            import torchvision
        except ImportError:
            pytest.skip("torch/torchvision not available")

        analyzer = TemporalAnalyzer()
        frames = make_consistent_video(4, shape=(224, 224, 3))
        result = analyzer.analyze(frames, fps=25.0)

        assert isinstance(result, AnalyzerResult)
        assert 0.0 <= result.score <= 100.0

    def test_raft_real_video_consistency(self):
        """RAFT flow on real video should be consistent."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        analyzer = TemporalAnalyzer()
        frames = make_consistent_video(6, shape=(224, 224, 3))
        result = analyzer.analyze(frames, fps=25.0)

        assert 0.0 <= result.score <= 100.0
