# © 2026 TimeWealthy Limited — DeepGuard
"""MetadataAnalyzer: codec and metadata-based AI generation detection."""

import json
import logging
import subprocess
from typing import Dict, Any, List, Tuple
from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)

# Known AI video generation tool signatures in metadata
AI_TOOL_SIGNATURES = {
    "encoders": [
        "sora", "veo", "pika", "runway", "genmo", "luma", "kling",
        "dreamachine", "stable-video", "animatediff", "modelscope",
    ],
    "metadata_keys": [
        "ai_generated", "synthetic", "deepfake", "generated_by",
    ],
    "comment_patterns": [
        "sora", "veo", "pika", "runway", "genmo", "luma", "kling",
        "stable diffusion", "animatediff",
    ],
}

# Suspicious codec parameter ranges (typical AI video generation artifacts)
SUSPICIOUS_PARAMS = {
    "bit_rate_kbps": (0, 500),   # Unusually low bitrate for "authentic" footage
    "r_frame_rate_uniform": True,  # Perfectly uniform FPS (AI-typical)
}


class MetadataAnalyzer(BaseAnalyzer):
    """
    Analyzes video file metadata using ffprobe.

    Checks:
    - Known AI generation tool signatures in encoder metadata
    - Codec parameter anomalies
    - Suspiciously clean metadata (AI tools often produce minimal metadata)
    """

    def analyze(self, file_path: str) -> AnalyzerResult:
        """
        Args:
            file_path: Path to the video file

        Returns:
            AnalyzerResult with score 0-100
        """
        metadata = self._extract_metadata(file_path)
        if metadata is None:
            return AnalyzerResult(score=50.0, findings=[], error="Could not extract metadata (ffprobe not available?)")

        findings: List[Finding] = []
        scores: List[float] = []

        # 1. AI tool signature check
        sig_score, sig_findings = self._check_ai_signatures(metadata)
        scores.append(sig_score)
        findings.extend(sig_findings)

        # 2. Codec parameter analysis
        codec_score, codec_findings = self._analyze_codec_params(metadata)
        scores.append(codec_score)
        findings.extend(codec_findings)

        # 3. Metadata completeness check
        completeness_score, completeness_findings = self._check_metadata_completeness(metadata)
        scores.append(completeness_score)
        findings.extend(completeness_findings)

        final_score = sum(scores) / len(scores) if scores else 50.0
        return AnalyzerResult(score=round(final_score, 2), findings=findings)

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Run ffprobe and return parsed JSON metadata."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning(f"ffprobe returned {result.returncode}: {result.stderr}")
                return {}
            return json.loads(result.stdout)
        except FileNotFoundError:
            logger.warning("ffprobe not found. Install ffmpeg.")
            return {}
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}

    def _check_ai_signatures(
        self, metadata: Dict[str, Any]
    ) -> Tuple[float, List[Finding]]:
        """Check for known AI generation tool signatures."""
        findings: List[Finding] = []
        score = 10.0

        fmt = metadata.get("format", {})
        tags = fmt.get("tags", {})
        encoder = tags.get("encoder", "").lower()
        comment = tags.get("comment", "").lower()
        title = tags.get("title", "").lower()
        all_text = f"{encoder} {comment} {title}"

        # Check encoder signatures
        for sig in AI_TOOL_SIGNATURES["encoders"]:
            if sig in all_text:
                score = 95.0
                findings.append(
                    Finding(
                        type="codec_signature",
                        confidence=95.0,
                        description=f"AI動画生成ツールのシグネチャを検出: '{sig}'",
                        metadata={"encoder": encoder, "signature": sig},
                    )
                )
                break

        # Check metadata key patterns
        for key in AI_TOOL_SIGNATURES["metadata_keys"]:
            if key in tags:
                score = max(score, 90.0)
                findings.append(
                    Finding(
                        type="ai_metadata_tag",
                        confidence=90.0,
                        description=f"AI生成フラグのメタデータキーを検出: '{key}'",
                        metadata={"key": key, "value": tags[key]},
                    )
                )

        return score, findings

    def _analyze_codec_params(
        self, metadata: Dict[str, Any]
    ) -> Tuple[float, List[Finding]]:
        """Analyze codec parameters for AI generation artifacts."""
        findings: List[Finding] = []
        scores: List[float] = []

        streams = metadata.get("streams", [])
        for stream in streams:
            if stream.get("codec_type") != "video":
                continue

            codec_name = stream.get("codec_name", "").lower()
            bit_rate = int(stream.get("bit_rate", 0))

            # Check for unusually low bitrate (AI video often has low quality encoding)
            if 0 < bit_rate < 200_000:  # < 200 kbps for video
                confidence = min(80.0, (200_000 - bit_rate) / 2000)
                findings.append(
                    Finding(
                        type="low_bitrate",
                        confidence=round(confidence, 1),
                        description=f"動画のビットレートが異常に低い ({bit_rate // 1000} kbps)",
                        metadata={"bit_rate": bit_rate, "codec": codec_name},
                    )
                )
                scores.append(confidence)

            # Check for perfectly uniform frame rate (AI typical)
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            avg_frame_rate = stream.get("avg_frame_rate", "0/1")
            if r_frame_rate == avg_frame_rate and r_frame_rate not in ("0/0", "0/1"):
                # Perfectly uniform FPS is somewhat suspicious but not definitive
                scores.append(25.0)
            else:
                scores.append(15.0)

            # Check nb_frames vs duration consistency
            nb_frames = stream.get("nb_frames")
            duration = stream.get("duration")
            if nb_frames and duration:
                try:
                    actual_fps = int(nb_frames) / float(duration)
                    claimed_fps_str = r_frame_rate
                    num, den = claimed_fps_str.split("/")
                    claimed_fps = int(num) / int(den)
                    if abs(actual_fps - claimed_fps) > 2.0:
                        findings.append(
                            Finding(
                                type="fps_mismatch",
                                confidence=70.0,
                                description=f"フレームレートの不整合 (実際: {actual_fps:.1f} vs 申告: {claimed_fps:.1f})",
                                metadata={"actual_fps": actual_fps, "claimed_fps": claimed_fps},
                            )
                        )
                        scores.append(70.0)
                except Exception:
                    pass

        return (sum(scores) / len(scores) if scores else 20.0), findings

    def _check_metadata_completeness(
        self, metadata: Dict[str, Any]
    ) -> Tuple[float, List[Finding]]:
        """
        Check metadata completeness.
        Real cameras embed rich metadata; AI tools often produce minimal metadata.
        """
        findings: List[Finding] = []
        fmt = metadata.get("format", {})
        tags = fmt.get("tags", {})

        # Legitimate recordings often have: creation_time, location, device info
        real_camera_tags = {"creation_time", "location", "make", "model", "software"}
        present_tags = set(k.lower() for k in tags.keys())
        overlap = real_camera_tags & present_tags

        if len(overlap) == 0 and len(tags) < 3:
            confidence = 55.0
            findings.append(
                Finding(
                    type="minimal_metadata",
                    confidence=confidence,
                    description="カメラ固有のメタデータが存在しない（AI生成ツールはメタデータを含まないことが多い）",
                    metadata={"present_tags": list(present_tags)},
                )
            )
            return confidence, findings

        return 20.0, []
