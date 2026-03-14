# © 2026 TimeWealthy Limited — FakeGuard
"""MetadataAnalyzer: enhanced ffprobe parsing + AI tool signature database."""

import json
import logging
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)


# ─── AI Tool Signature Database ───────────────────────────────────────────────
# Known AI video generation tools and their metadata signatures.
# Covers: Sora (OpenAI), Runway (Gen-3), Pika, Kling (Kuaishou), Veo (Google),
#         Luma Dream Machine, Genmo, Stable Video Diffusion, AnimateDiff, etc.

AI_TOOL_SIGNATURES = {
    # Encoder strings commonly embedded by AI generation tools
    "encoders": [
        "sora",
        "veo",
        "pika",
        "pika labs",
        "runway",
        "runwayml",
        "kling",
        "kuaishou",
        "genmo",
        "luma",
        "luma dream machine",
        "dreamachine",
        "stable-video",
        "stable video diffusion",
        "animatediff",
        "modelscope",
        "zeroscope",
        "cogvideo",
        "open-sora",
        "opensora",
    ],
    # Metadata tag keys that indicate AI generation
    "metadata_keys": [
        "ai_generated",
        "ai_model",
        "synthetic",
        "deepfake",
        "generated_by",
        "generation_tool",
        "model_name",
    ],
    # Comment/title strings from AI tools
    "comment_patterns": [
        "sora",
        "veo",
        "pika",
        "runway",
        "kling",
        "genmo",
        "luma",
        "stable diffusion",
        "animatediff",
        "cogvideo",
        "ai generated",
        "ai-generated",
        "synthetically generated",
    ],
}

# Confidence levels per tool (well-known tools → higher confidence)
AI_TOOL_CONFIDENCE: Dict[str, float] = {
    "sora": 97.0,
    "veo": 97.0,
    "runway": 95.0,
    "runwayml": 95.0,
    "pika": 95.0,
    "pika labs": 95.0,
    "kling": 93.0,
    "kuaishou": 90.0,
    "genmo": 90.0,
    "luma": 90.0,
    "luma dream machine": 90.0,
    "dreamachine": 88.0,
    "stable-video": 85.0,
    "stable video diffusion": 85.0,
    "animatediff": 82.0,
    "modelscope": 80.0,
}

# Default confidence for unknown tools in the list
DEFAULT_AI_CONFIDENCE = 85.0


class MetadataAnalyzer(BaseAnalyzer):
    """
    Analyzes video file metadata using ffprobe.

    Checks:
    - Known AI generation tool signatures (Sora, Runway, Pika, Kling, Veo, …)
    - Codec parameter anomalies (bitrate, FPS inconsistency)
    - Metadata completeness (AI tools often produce minimal metadata)
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
            return AnalyzerResult(
                score=50.0,
                findings=[],
                error="Could not extract metadata (ffprobe not available?)",
            )

        findings: List[Finding] = []
        scores: List[float] = []

        # 1. AI tool signature check (highest priority)
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

        avg_score = sum(scores) / len(scores) if scores else 50.0

        # If a high-confidence AI signature was found, it dominates the final score.
        # A single definitive tool match is enough evidence: blend 70% signature + 30% avg.
        max_sig_confidence = max(
            (f.confidence for f in sig_findings if f.type == "codec_signature"),
            default=0.0,
        )
        if max_sig_confidence >= 90.0:
            final_score = max_sig_confidence * 0.70 + avg_score * 0.30
        else:
            final_score = avg_score

        return AnalyzerResult(score=round(final_score, 2), findings=findings)

    def _extract_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Run ffprobe and return parsed JSON metadata, or None on failure."""
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
                logger.warning(f"ffprobe returned {result.returncode}: {result.stderr[:200]}")
                return {}
            return json.loads(result.stdout)
        except FileNotFoundError:
            logger.warning("ffprobe not found. Install ffmpeg.")
            return {}
        except subprocess.TimeoutExpired:
            logger.warning("ffprobe timed out.")
            return {}
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}

    def _check_ai_signatures(
        self, metadata: Dict[str, Any]
    ) -> Tuple[float, List[Finding]]:
        """
        Check for known AI generation tool signatures in:
        - encoder tag
        - comment tag
        - title tag
        - explicit AI metadata keys
        """
        findings: List[Finding] = []
        score = 10.0

        fmt = metadata.get("format", {})
        tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}

        encoder = tags.get("encoder", "").lower()
        comment = tags.get("comment", "").lower()
        title = tags.get("title", "").lower()
        software = tags.get("software", "").lower()
        all_text = f"{encoder} {comment} {title} {software}"

        # Check encoder patterns
        for sig in AI_TOOL_SIGNATURES["encoders"]:
            if sig in all_text:
                confidence = AI_TOOL_CONFIDENCE.get(sig, DEFAULT_AI_CONFIDENCE)
                score = confidence
                findings.append(
                    Finding(
                        type="codec_signature",
                        confidence=confidence,
                        description=f"AI動画生成ツールのシグネチャを検出: '{sig}'",
                        metadata={
                            "encoder": encoder,
                            "comment": comment,
                            "signature": sig,
                        },
                    )
                )
                break  # First match wins

        # Check comment patterns independently
        for pattern in AI_TOOL_SIGNATURES["comment_patterns"]:
            if pattern in comment and pattern not in encoder:
                confidence = AI_TOOL_CONFIDENCE.get(pattern, DEFAULT_AI_CONFIDENCE)
                if confidence > score:
                    score = confidence
                    findings.append(
                        Finding(
                            type="comment_signature",
                            confidence=confidence,
                            description=f"コメントにAI生成ツールのパターンを検出: '{pattern}'",
                            metadata={"comment": comment, "pattern": pattern},
                        )
                    )

        # Check explicit AI metadata keys
        for key in AI_TOOL_SIGNATURES["metadata_keys"]:
            if key in tags:
                tag_score = 90.0
                if tag_score > score:
                    score = tag_score
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
        """
        Analyze codec parameters for AI generation artifacts:
        - Low bitrate (AI tools often produce low-quality encodes)
        - Perfectly uniform FPS (AI-typical)
        - Frame count / duration inconsistency
        """
        findings: List[Finding] = []
        scores: List[float] = []

        streams = metadata.get("streams", [])
        for stream in streams:
            if stream.get("codec_type") != "video":
                continue

            codec_name = stream.get("codec_name", "").lower()

            # Bitrate check
            bit_rate_str = stream.get("bit_rate", "0")
            try:
                bit_rate = int(bit_rate_str)
            except (ValueError, TypeError):
                bit_rate = 0

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

            # FPS uniformity check
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            avg_frame_rate = stream.get("avg_frame_rate", "0/1")
            if r_frame_rate == avg_frame_rate and r_frame_rate not in ("0/0", "0/1"):
                scores.append(25.0)
            else:
                scores.append(15.0)

            # Frame count vs duration consistency
            nb_frames = stream.get("nb_frames")
            duration = stream.get("duration")
            if nb_frames and duration:
                try:
                    actual_fps = int(nb_frames) / float(duration)
                    num, den = r_frame_rate.split("/")
                    claimed_fps = int(num) / (int(den) or 1)
                    if claimed_fps > 0 and abs(actual_fps - claimed_fps) > 2.0:
                        findings.append(
                            Finding(
                                type="fps_mismatch",
                                confidence=70.0,
                                description=(
                                    f"フレームレートの不整合 "
                                    f"(実際: {actual_fps:.1f} vs 申告: {claimed_fps:.1f})"
                                ),
                                metadata={
                                    "actual_fps": actual_fps,
                                    "claimed_fps": claimed_fps,
                                },
                            )
                        )
                        scores.append(70.0)
                except Exception:
                    pass

        final = sum(scores) / len(scores) if scores else 20.0
        return final, findings

    def _check_metadata_completeness(
        self, metadata: Dict[str, Any]
    ) -> Tuple[float, List[Finding]]:
        """
        Check metadata completeness.
        Real cameras embed rich metadata; AI tools often produce minimal metadata.
        """
        findings: List[Finding] = []
        fmt = metadata.get("format", {})
        tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}

        # Tags typically present in real camera recordings
        real_camera_tags = {"creation_time", "location", "make", "model", "software"}
        present_tags = set(tags.keys())
        overlap = real_camera_tags & present_tags

        if len(overlap) == 0 and len(tags) < 3:
            confidence = 55.0
            findings.append(
                Finding(
                    type="minimal_metadata",
                    confidence=confidence,
                    description=(
                        "カメラ固有のメタデータが存在しない "
                        "（AI生成ツールはメタデータを含まないことが多い）"
                    ),
                    metadata={"present_tags": sorted(present_tags)},
                )
            )
            return confidence, findings

        return 20.0, []
