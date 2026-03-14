# © 2026 TimeWealthy Limited — FakeGuard
"""AudioAnalyzer: detect AI-synthesized audio via mel spectrogram + statistical analysis."""

import logging
import numpy as np
from typing import Optional, Tuple, List
from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)


class AudioAnalyzer(BaseAnalyzer):
    """
    Analyzes audio track for AI synthesis artifacts.

    Checks:
    - Mel spectrogram statistical features (MFCC variance, spectral flatness)
    - Frequency distribution statistical properties → synthesis score
    - Environmental sound consistency (reverb/room characteristics)
    - Silence pattern analysis (AI audio has unnaturally clean silence)
    """

    def analyze(self, audio: np.ndarray, sr: int) -> AnalyzerResult:
        """
        Args:
            audio: 1D numpy array of audio samples (mono, float32)
            sr: Sample rate (Hz)

        Returns:
            AnalyzerResult with score 0-100
        """
        if audio is None or (hasattr(audio, '__len__') and len(audio) == 0):
            return AnalyzerResult(score=50.0, findings=[], error="No audio data provided")

        findings: List[Finding] = []
        scores: List[float] = []

        # 1. Mel spectrogram + MFCC analysis
        spec_score, spec_findings = self._spectrogram_analysis(audio, sr)
        scores.append(spec_score)
        findings.extend(spec_findings)

        # 2. Frequency distribution synthesis score
        synth_score = self._compute_synthesis_score(audio, sr)
        scores.append(synth_score)
        if synth_score > 65.0:
            findings.append(
                Finding(
                    type="frequency_anomaly",
                    confidence=round(synth_score, 1),
                    description=(
                        f"周波数分布の統計的特性がAI合成音声と一致 "
                        f"（合成度スコア: {synth_score:.1f}）"
                    ),
                    timestamp_sec=0.0,
                )
            )

        # 3. Environmental consistency
        env_score, env_findings = self._environmental_consistency(audio, sr)
        scores.append(env_score)
        findings.extend(env_findings)

        # 4. Silence pattern
        silence_score, silence_findings = self._silence_pattern_analysis(audio, sr)
        scores.append(silence_score)
        findings.extend(silence_findings)

        final_score = float(np.mean(scores)) if scores else 50.0
        return AnalyzerResult(score=round(final_score, 2), findings=findings)

    def _spectrogram_analysis(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[float, List[Finding]]:
        """
        Analyze mel-spectrogram for AI synthesis artifacts.

        AI TTS characteristics:
        - High spectral flatness (noise-like, lacks natural resonances)
        - Low MFCC variance (overly smooth, robotic)
        """
        findings: List[Finding] = []
        try:
            import librosa

            audio_f = audio.astype(np.float32)

            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_f, sr=sr, n_mels=128, fmax=min(8000, sr // 2)
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Spectral flatness (high flatness → noise-like / AI-like)
            flatness = librosa.feature.spectral_flatness(y=audio_f)
            mean_flatness = float(np.mean(flatness))

            # MFCC variance (AI audio: too-smooth MFCCs, low variance)
            mfcc = librosa.feature.mfcc(y=audio_f, sr=sr, n_mfcc=20)
            mfcc_var = float(np.mean(np.var(mfcc, axis=1)))

            # Flatness score: high flatness → high AI probability
            flatness_score = min(100.0, mean_flatness * 300)

            # MFCC score: low variance → high AI probability
            mfcc_score = max(0.0, min(100.0, 100.0 - mfcc_var * 10))

            score = flatness_score * 0.5 + mfcc_score * 0.5

            if score > 65.0:
                findings.append(
                    Finding(
                        type="synthetic_audio",
                        confidence=round(score, 1),
                        description=(
                            f"スペクトログラムのパターンがAI合成音声と類似 "
                            f"（フラットネス: {mean_flatness:.3f}, MFCC分散: {mfcc_var:.2f}）"
                        ),
                        timestamp_sec=0.0,
                    )
                )

            return score, findings

        except Exception as e:
            logger.debug(f"Spectrogram analysis failed: {e}")
            return 40.0, []

    def _compute_synthesis_score(self, audio: np.ndarray, sr: int) -> float:
        """
        Compute synthesis likelihood from frequency distribution statistics.

        Real speech:
        - Irregular harmonic spacing
        - Amplitude modulation (natural prosody)
        - Non-zero noise floor with natural decay

        AI TTS:
        - Highly regular harmonics
        - Suspiciously flat amplitude envelope
        - Very low or zero noise floor

        Returns:
            float in [0, 100]
        """
        try:
            import librosa

            audio_f = audio.astype(np.float32)

            # Zero crossing rate: AI TTS has regular ZCR
            zcr = librosa.feature.zero_crossing_rate(y=audio_f)
            zcr_std = float(np.std(zcr))
            zcr_mean = float(np.mean(zcr)) + 1e-8
            zcr_cv = zcr_std / zcr_mean
            # Low ZCR CV → suspiciously regular → more AI-like
            zcr_score = max(0.0, min(100.0, (1.0 - min(1.0, zcr_cv * 2)) * 70))

            # Spectral bandwidth variance: AI TTS has stable bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio_f, sr=sr)
            bw_std = float(np.std(bandwidth))
            bw_mean = float(np.mean(bandwidth)) + 1e-8
            bw_cv = bw_std / bw_mean
            # Low BW CV → suspiciously stable → more AI-like
            bw_score = max(0.0, min(100.0, (1.0 - min(1.0, bw_cv * 3)) * 60))

            # RMS energy variance: real speech has natural dynamics
            rms = librosa.feature.rms(y=audio_f)
            rms_std = float(np.std(rms))
            rms_mean = float(np.mean(rms)) + 1e-8
            rms_cv = rms_std / rms_mean
            # Low RMS CV (very stable energy) → AI-like
            rms_score = max(0.0, min(100.0, (1.0 - min(1.0, rms_cv)) * 50))

            synthesis_score = zcr_score * 0.4 + bw_score * 0.35 + rms_score * 0.25
            return round(synthesis_score, 2)

        except Exception as e:
            logger.debug(f"Synthesis score computation failed: {e}")
            return 30.0

    def _environmental_consistency(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[float, List[Finding]]:
        """
        Check if reverb/room characteristics are consistent throughout the audio.
        Inconsistent reverb → likely spliced or AI-generated with inconsistent conditions.
        """
        findings: List[Finding] = []
        try:
            import librosa

            audio_f = audio.astype(np.float32)
            chunk_size = sr * 2  # 2-second chunks

            if len(audio_f) < chunk_size * 2:
                return 30.0, []

            chunks = [
                audio_f[i: i + chunk_size]
                for i in range(0, len(audio_f) - chunk_size, chunk_size)
            ]

            centroids = []
            for chunk in chunks:
                centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)
                centroids.append(float(np.mean(centroid)))

            if len(centroids) < 2:
                return 30.0, []

            centroid_std = float(np.std(centroids))
            centroid_mean = float(np.mean(centroids)) + 1e-8
            cv = centroid_std / centroid_mean

            # Flag extremes: too stable OR too variable
            if cv < 0.02:
                score = 65.0
                findings.append(
                    Finding(
                        type="env_mismatch",
                        confidence=65.0,
                        description="環境音の反響パターンが不自然に均一（AI生成の疑い）",
                        timestamp_sec=0.0,
                    )
                )
            elif cv > 0.4:
                score = 70.0
                findings.append(
                    Finding(
                        type="env_mismatch",
                        confidence=70.0,
                        description="環境音の反響パターンが不自然に変動（スプライス疑い）",
                        timestamp_sec=0.0,
                    )
                )
            else:
                score = 25.0

            return score, findings

        except Exception as e:
            logger.debug(f"Environmental consistency check failed: {e}")
            return 30.0, []

    def _silence_pattern_analysis(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[float, List[Finding]]:
        """
        Analyze silence patterns.
        AI TTS often has perfectly clean silence segments (RMS = 0).
        Real recordings have background noise even in "quiet" parts.
        """
        findings: List[Finding] = []
        try:
            import librosa

            audio_f = audio.astype(np.float32)
            frame_length = 2048
            hop_length = 512

            rms = librosa.feature.rms(
                y=audio_f, frame_length=frame_length, hop_length=hop_length
            )[0]

            threshold = 0.005
            is_silent = rms < threshold

            if not np.any(is_silent):
                return 20.0, []

            silent_ratio = float(np.sum(is_silent)) / len(is_silent)

            if silent_ratio > 0.3:
                confidence = min(80.0, silent_ratio * 200)
                findings.append(
                    Finding(
                        type="unnatural_silence",
                        confidence=round(confidence, 1),
                        description=(
                            f"音声の無音区間が不自然に多い "
                            f"（{silent_ratio * 100:.1f}%）"
                        ),
                        timestamp_sec=0.0,
                    )
                )
                return confidence, findings

            return 20.0, []

        except Exception as e:
            logger.debug(f"Silence pattern analysis failed: {e}")
            return 20.0, []
