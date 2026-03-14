# © 2026 TimeWealthy Limited — DeepGuard
"""Tests for AudioAnalyzer."""

import numpy as np
import pytest

from app.analyzers.audio import AudioAnalyzer


SR = 22050  # standard sample rate for tests


class TestAudioAnalyzer:

    def test_analyze_returns_result(self, random_audio):
        """analyze() returns AnalyzerResult"""
        audio, sr = random_audio
        analyzer = AudioAnalyzer()
        result = analyzer.analyze(audio, sr)
        assert result is not None
        assert 0.0 <= result.score <= 100.0

    def test_empty_audio_returns_error(self):
        """空の音声 → error付きの結果"""
        analyzer = AudioAnalyzer()
        result = analyzer.analyze(np.array([]), SR)
        assert result.has_error

    def test_score_is_float(self, random_audio):
        """score は float"""
        audio, sr = random_audio
        analyzer = AudioAnalyzer()
        result = analyzer.analyze(audio, sr)
        assert isinstance(result.score, float)

    def test_findings_is_list(self, random_audio):
        """findings は常にリスト"""
        audio, sr = random_audio
        analyzer = AudioAnalyzer()
        result = analyzer.analyze(audio, sr)
        assert isinstance(result.findings, list)

    def test_silence_detection(self, silent_audio):
        """大部分が無音 → unnatural_silence 検出"""
        audio, sr = silent_audio
        analyzer = AudioAnalyzer()
        score, findings = analyzer._silence_pattern_analysis(audio, sr)
        # All-silence should trigger finding
        assert len(findings) > 0 or score > 20.0

    def test_random_audio_silence_detection_no_finding(self):
        """ランダムノイズ → 無音フラグなし"""
        rng = np.random.default_rng(1)
        audio = rng.uniform(-0.5, 0.5, SR * 3).astype(np.float32)
        analyzer = AudioAnalyzer()
        score, findings = analyzer._silence_pattern_analysis(audio, sr=SR)
        silence_findings = [f for f in findings if f.type == "unnatural_silence"]
        assert len(silence_findings) == 0

    def test_spectrogram_analysis_returns_score(self, random_audio):
        """スペクトログラム解析がスコアを返す"""
        audio, sr = random_audio
        analyzer = AudioAnalyzer()
        score, findings = analyzer._spectrogram_analysis(audio, sr)
        assert 0.0 <= score <= 100.0

    def test_env_consistency_returns_score(self, random_audio):
        """環境整合性チェックがスコアを返す"""
        audio, sr = random_audio
        analyzer = AudioAnalyzer()
        score, findings = analyzer._environmental_consistency(audio, sr)
        assert 0.0 <= score <= 100.0

    def test_score_always_in_range(self):
        """スコアは常に0-100"""
        analyzer = AudioAnalyzer()
        rng = np.random.default_rng(0)
        for _ in range(3):
            audio = rng.uniform(-1.0, 1.0, SR * 5).astype(np.float32)
            result = analyzer.analyze(audio, SR)
            assert 0.0 <= result.score <= 100.0

    def test_finding_confidence_in_range(self, random_audio):
        """Finding の confidence は 0-100"""
        audio, sr = random_audio
        analyzer = AudioAnalyzer()
        result = analyzer.analyze(audio, sr)
        for finding in result.findings:
            assert 0.0 <= finding.confidence <= 100.0
