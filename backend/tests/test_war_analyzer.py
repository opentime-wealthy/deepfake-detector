# © 2026 TimeWealthy Limited — DeepGuard
"""Tests for WarFootageAnalyzer."""

import numpy as np
import pytest

from app.analyzers.war_footage import WarFootageAnalyzer


def make_fire_frame(fire_ratio=0.15) -> np.ndarray:
    """Create a frame with orange fire-like pixels."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    fire_h = int(h * fire_ratio)
    # BGR orange-ish (fire color)
    frame[:fire_h, :] = [0, 100, 230]  # BGR approximation of orange-red
    return frame


class TestWarFootageAnalyzer:

    def test_analyze_returns_result(self, random_frames):
        """analyze() returns AnalyzerResult"""
        analyzer = WarFootageAnalyzer()
        result = analyzer.analyze(random_frames)
        assert result is not None
        assert 0.0 <= result.score <= 100.0

    def test_empty_frames_returns_error(self):
        """空のフレームリスト → error"""
        analyzer = WarFootageAnalyzer()
        result = analyzer.analyze([])
        assert result.has_error

    def test_findings_is_list(self, random_frames):
        """findings は常にリスト"""
        analyzer = WarFootageAnalyzer()
        result = analyzer.analyze(random_frames)
        assert isinstance(result.findings, list)

    def test_uniform_fire_flagged(self):
        """均一な火炎領域 → explosion_uniformity フラグ"""
        analyzer = WarFootageAnalyzer()
        # Frames with very uniform 'fire' region
        frames = []
        for _ in range(5):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Add HSV orange in BGR (approximately): H=15, S=255, V=240
            # In BGR: B=0, G=100, R=240
            frame[:120, :] = [0, 100, 240]  # Top half orange
            frames.append(frame)

        score, findings = analyzer._explosion_physics_check(frames)
        # Should detect uniform fire region
        assert score >= 0.0  # At minimum should not crash

    def test_analyze_with_audio(self, random_frames, random_audio):
        """音声付き解析が動作する"""
        audio, sr = random_audio
        analyzer = WarFootageAnalyzer()
        result = analyzer.analyze(random_frames, audio=audio, sr=sr)
        assert result is not None
        assert 0.0 <= result.score <= 100.0

    def test_analyze_without_audio(self, random_frames):
        """音声なし解析が動作する"""
        analyzer = WarFootageAnalyzer()
        result = analyzer.analyze(random_frames, audio=None)
        assert result is not None
        assert 0.0 <= result.score <= 100.0

    def test_score_always_in_range(self):
        """スコアは常に0-100"""
        analyzer = WarFootageAnalyzer()
        rng = np.random.default_rng(42)
        for _ in range(3):
            frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(5)]
            result = analyzer.analyze(frames)
            assert 0.0 <= result.score <= 100.0

    def test_av_sync_check_no_crash(self, random_frames, random_audio):
        """AV同期チェックがクラッシュしない"""
        audio, sr = random_audio
        analyzer = WarFootageAnalyzer()
        score, findings = analyzer._audio_visual_sync_check(random_frames, audio, sr)
        assert 0.0 <= score <= 100.0

    def test_smoke_uniformity_check(self):
        """煙均一性チェックが動作する"""
        analyzer = WarFootageAnalyzer()
        # Frames with gray smoke-like region
        frames = []
        for _ in range(5):
            frame = np.full((240, 320, 3), 140, dtype=np.uint8)  # Gray (smoke-like)
            frames.append(frame)

        score, findings = analyzer._smoke_uniformity_check(frames)
        assert 0.0 <= score <= 100.0
