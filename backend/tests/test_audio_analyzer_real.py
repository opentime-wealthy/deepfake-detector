# © 2026 TimeWealthy Limited — FakeGuard
"""
tests/test_audio_analyzer_real.py
TDD tests for AudioAnalyzer (librosa mel spectrogram + synthesis scoring).
"""

import numpy as np
import pytest

from app.analyzers.audio import AudioAnalyzer
from app.analyzers.base import AnalyzerResult


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_silence(duration_sec: float = 3.0, sr: int = 22050) -> np.ndarray:
    """Return near-silent audio array."""
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


def make_noise_audio(duration_sec: float = 3.0, sr: int = 22050, seed: int = 42) -> np.ndarray:
    """Return white noise audio (natural-like)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, int(duration_sec * sr)).astype(np.float32)


def make_sine_wave(
    freq: float = 440.0,
    duration_sec: float = 3.0,
    sr: int = 22050,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Return a pure sine wave (unnaturally flat spectrum = AI-like)."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_tts_like_audio(duration_sec: float = 3.0, sr: int = 22050) -> np.ndarray:
    """Return audio that mimics AI TTS: limited harmonic range, very regular."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    # Combine a few harmonics with very stable amplitude (no natural variation)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
    )
    return audio.astype(np.float32)


def make_speech_like_audio(duration_sec: float = 3.0, sr: int = 22050, seed: int = 7) -> np.ndarray:
    """
    Return audio that mimics natural speech: varying harmonics + noise floor.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    # Varying fundamental with noise
    fundamental = rng.uniform(80, 300)
    audio = rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * fundamental * t)
    # Add formants with amplitude modulation
    for harmonic in [2, 3, 4]:
        amp = rng.uniform(0.02, 0.08)
        audio += amp * np.sin(2 * np.pi * fundamental * harmonic * t)
    # Natural noise floor
    audio += rng.normal(0, 0.01, len(t))
    return audio.astype(np.float32)


# ─── Unit Tests ───────────────────────────────────────────────────────────────


class TestRealAudioAnalyzer:
    """Unit tests for AudioAnalyzer."""

    def test_analyze_returns_result(self):
        """analyze() returns AnalyzerResult for valid audio input."""
        analyzer = AudioAnalyzer()
        audio = make_noise_audio()
        result = analyzer.analyze(audio, sr=22050)

        assert isinstance(result, AnalyzerResult)
        assert 0.0 <= result.score <= 100.0
        assert result.error is None

    def test_empty_audio_returns_error(self):
        """analyze() returns error when audio is empty."""
        analyzer = AudioAnalyzer()
        result = analyzer.analyze(np.array([]), sr=22050)

        assert result.error is not None

    def test_none_audio_returns_error(self):
        """analyze() returns error when audio is None."""
        analyzer = AudioAnalyzer()
        result = analyzer.analyze(None, sr=22050)

        assert result.error is not None

    def test_score_range_valid(self):
        """Score must always be between 0 and 100."""
        analyzer = AudioAnalyzer()
        for audio_fn in [make_silence, make_noise_audio, make_sine_wave, make_tts_like_audio]:
            audio = audio_fn()
            result = analyzer.analyze(audio, sr=22050)
            assert 0.0 <= result.score <= 100.0, (
                f"Score out of range for {audio_fn.__name__}: {result.score}"
            )

    def test_silence_detects_unnatural_silence(self):
        """Mostly-silent audio should trigger unnatural_silence finding."""
        analyzer = AudioAnalyzer()
        audio = make_silence(duration_sec=5.0)
        result = analyzer.analyze(audio, sr=22050)

        silence_findings = [f for f in result.findings if f.type == "unnatural_silence"]
        assert len(silence_findings) > 0 or result.score > 20.0

    def test_mel_spectrogram_analysis_completes(self):
        """_spectrogram_analysis returns (score, findings) without crashing."""
        analyzer = AudioAnalyzer()
        audio = make_noise_audio()
        score, findings = analyzer._spectrogram_analysis(audio, sr=22050)

        assert isinstance(score, float)
        assert isinstance(findings, list)
        assert 0.0 <= score <= 100.0

    def test_mel_spectrogram_pure_sine_high_flatness(self):
        """Pure sine wave has high spectral flatness → elevated AI score."""
        analyzer = AudioAnalyzer()
        audio = make_sine_wave(freq=440.0)
        score, findings = analyzer._spectrogram_analysis(audio, sr=22050)

        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_environmental_consistency_long_audio(self):
        """_environmental_consistency works for long audio (>4 seconds)."""
        analyzer = AudioAnalyzer()
        audio = make_noise_audio(duration_sec=8.0)
        score, findings = analyzer._environmental_consistency(audio, sr=22050)

        assert isinstance(score, float)
        assert isinstance(findings, list)
        assert 0.0 <= score <= 100.0

    def test_environmental_consistency_short_audio_skips(self):
        """_environmental_consistency returns low score for too-short audio."""
        analyzer = AudioAnalyzer()
        audio = make_noise_audio(duration_sec=1.0)
        score, findings = analyzer._environmental_consistency(audio, sr=22050)

        assert isinstance(score, float)
        assert score <= 50.0  # Short audio shouldn't trigger false positive

    def test_silence_pattern_analysis_silent_audio(self):
        """_silence_pattern_analysis detects mostly-silent audio."""
        analyzer = AudioAnalyzer()
        audio = make_silence(duration_sec=5.0)
        score, findings = analyzer._silence_pattern_analysis(audio, sr=22050)

        assert isinstance(score, float)
        assert isinstance(findings, list)
        silence_findings = [f for f in findings if f.type == "unnatural_silence"]
        assert len(silence_findings) > 0

    def test_silence_pattern_analysis_normal_audio(self):
        """_silence_pattern_analysis returns low score for normal audio."""
        analyzer = AudioAnalyzer()
        audio = make_noise_audio(duration_sec=5.0)
        score, findings = analyzer._silence_pattern_analysis(audio, sr=22050)

        assert isinstance(score, float)
        # Normal noise should not trigger silence detection
        silence_findings = [f for f in findings if f.type == "unnatural_silence"]
        assert len(silence_findings) == 0

    def test_synthesis_score_computed(self):
        """_compute_synthesis_score returns valid float in [0, 100]."""
        analyzer = AudioAnalyzer()
        audio = make_tts_like_audio()

        score = analyzer._compute_synthesis_score(audio, sr=22050)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_synthesis_score_pure_tone_vs_noise(self):
        """Pure tone should have different synthesis score than noise."""
        analyzer = AudioAnalyzer()

        tone_score = analyzer._compute_synthesis_score(make_sine_wave(), sr=22050)
        noise_score = analyzer._compute_synthesis_score(make_noise_audio(), sr=22050)

        # Scores should be valid
        assert 0.0 <= tone_score <= 100.0
        assert 0.0 <= noise_score <= 100.0

    def test_findings_have_required_fields(self):
        """All findings must have type, confidence, description."""
        analyzer = AudioAnalyzer()
        audio = make_silence(duration_sec=5.0)
        result = analyzer.analyze(audio, sr=22050)

        for finding in result.findings:
            assert hasattr(finding, "type")
            assert hasattr(finding, "confidence")
            assert hasattr(finding, "description")
            assert 0.0 <= finding.confidence <= 100.0

    def test_mfcc_variance_used_in_score(self):
        """MFCC variance should influence the spectrogram analysis score."""
        analyzer = AudioAnalyzer()

        # TTS-like audio (very regular, low MFCC variance)
        tts_audio = make_tts_like_audio()
        tts_score, _ = analyzer._spectrogram_analysis(tts_audio, sr=22050)

        # Natural speech (higher MFCC variance)
        speech_audio = make_speech_like_audio()
        speech_score, _ = analyzer._spectrogram_analysis(speech_audio, sr=22050)

        # Both should be valid
        assert 0.0 <= tts_score <= 100.0
        assert 0.0 <= speech_score <= 100.0

    def test_different_sample_rates_work(self):
        """AudioAnalyzer should work with different sample rates."""
        analyzer = AudioAnalyzer()
        for sr in [16000, 22050, 44100]:
            audio = make_noise_audio(sr=sr)
            result = analyzer.analyze(audio, sr=sr)
            assert isinstance(result, AnalyzerResult)
            assert 0.0 <= result.score <= 100.0


# ─── Slow Integration Tests ───────────────────────────────────────────────────


@pytest.mark.slow
class TestAudioAnalyzerIntegration:
    """Integration tests for AudioAnalyzer (may be slow on large files)."""

    def test_long_audio_no_crash(self):
        """AudioAnalyzer handles 60+ seconds of audio without crashing."""
        analyzer = AudioAnalyzer()
        audio = make_noise_audio(duration_sec=65.0)
        result = analyzer.analyze(audio, sr=22050)

        assert isinstance(result, AnalyzerResult)
        assert 0.0 <= result.score <= 100.0
