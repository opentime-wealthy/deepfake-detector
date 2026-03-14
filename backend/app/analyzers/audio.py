# © 2026 TimeWealthy Limited — DeepGuard
"""AudioAnalyzer: detect AI-synthesized audio and environmental mismatch."""

import logging
import numpy as np
from typing import Optional, Tuple, List
from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)


class AudioAnalyzer(BaseAnalyzer):
    """
    Analyzes audio track for AI synthesis artifacts.

    Checks:
    - Spectrogram pattern analysis (mel spectrogram uniformity)
    - Environmental sound integrity (reverb consistency)
    - Silence pattern analysis (AI audio often has unnatural silence)
    """

    def analyze(self, audio: np.ndarray, sr: int) -> AnalyzerResult:
        """
        Args:
            audio: 1D numpy array of audio samples (mono)
            sr: Sample rate (Hz)

        Returns:
            AnalyzerResult with score 0-100
        """
        if audio is None or len(audio) == 0:
            return AnalyzerResult(score=50.0, findings=[], error="No audio data provided")

        findings: List[Finding] = []
        scores: List[float] = []

        # 1. Spectrogram analysis
        spec_score, spec_findings = self._spectrogram_analysis(audio, sr)
        scores.append(spec_score)
        findings.extend(spec_findings)

        # 2. Environmental consistency
        env_score, env_findings = self._environmental_consistency(audio, sr)
        scores.append(env_score)
        findings.extend(env_findings)

        # 3. Silence pattern
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
        AI TTS often has too-uniform spectral energy distribution.
        """
        findings: List[Finding] = []
        try:
            import librosa

            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio.astype(np.float32), sr=sr, n_mels=128, fmax=8000
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Spectral flatness (high flatness → more noise-like / AI-like)
            flatness = librosa.feature.spectral_flatness(y=audio.astype(np.float32))
            mean_flatness = float(np.mean(flatness))

            # MFCC variance (AI audio tends to have too-smooth MFCCs)
            mfcc = librosa.feature.mfcc(y=audio.astype(np.float32), sr=sr, n_mfcc=20)
            mfcc_var = float(np.mean(np.var(mfcc, axis=1)))

            # Combine signals
            flatness_score = min(100.0, mean_flatness * 300)
            # Low MFCC variance is suspicious
            mfcc_score = max(0.0, min(100.0, 100.0 - mfcc_var * 10))

            score = flatness_score * 0.5 + mfcc_score * 0.5

            if score > 65.0:
                findings.append(
                    Finding(
                        type="synthetic_audio",
                        confidence=round(score, 1),
                        description=f"スペクトログラムのパターンがAI合成音声と類似（フラットネス: {mean_flatness:.3f}）",
                        timestamp_sec=0.0,
                    )
                )

            return score, findings

        except Exception as e:
            logger.debug(f"Spectrogram analysis failed: {e}")
            return 40.0, []

    def _environmental_consistency(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[float, List[Finding]]:
        """
        Check if reverb/room characteristics are consistent throughout the audio.
        Inconsistent reverb → likely spliced or AI-generated.
        """
        findings: List[Finding] = []
        try:
            import librosa

            # Split into chunks and compare spectral centroids
            chunk_size = sr * 2  # 2-second chunks
            if len(audio) < chunk_size * 2:
                return 30.0, []

            chunks = [
                audio[i: i + chunk_size]
                for i in range(0, len(audio) - chunk_size, chunk_size)
            ]

            centroids = []
            for chunk in chunks:
                centroid = librosa.feature.spectral_centroid(
                    y=chunk.astype(np.float32), sr=sr
                )
                centroids.append(float(np.mean(centroid)))

            if len(centroids) < 2:
                return 30.0, []

            # High variance in centroid across chunks → inconsistent environment
            centroid_std = float(np.std(centroids))
            centroid_mean = float(np.mean(centroids)) + 1e-8
            cv = centroid_std / centroid_mean

            # AI-generated: sometimes unnaturally stable (low CV), sometimes
            # inconsistent when spliced (high CV). Flag both extremes.
            if cv < 0.02:
                score = 65.0
                findings.append(
                    Finding(
                        type="env_mismatch",
                        confidence=65.0,
                        description="環境音の反響パターンが不自然に均一",
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
        AI TTS often has perfectly clean silence segments.
        """
        findings: List[Finding] = []
        try:
            import librosa

            # Detect silence (energy below threshold)
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(
                y=audio.astype(np.float32),
                frame_length=frame_length,
                hop_length=hop_length,
            )[0]

            threshold = 0.005  # Silence threshold
            is_silent = rms < threshold

            if not np.any(is_silent):
                return 20.0, []

            # Count silent frames
            silent_ratio = float(np.sum(is_silent)) / len(is_silent)

            # Perfect silence in many segments → suspicious
            if silent_ratio > 0.3:
                confidence = min(80.0, silent_ratio * 200)
                findings.append(
                    Finding(
                        type="unnatural_silence",
                        confidence=round(confidence, 1),
                        description=f"音声の無音区間が不自然に多い（{silent_ratio*100:.1f}%）",
                        timestamp_sec=0.0,
                    )
                )
                return confidence, findings

            return 20.0, []

        except Exception as e:
            logger.debug(f"Silence pattern analysis failed: {e}")
            return 20.0, []
