# © 2026 TimeWealthy Limited — FakeGuard
"""
C2PAAnalyzer: Content Credentials (C2PA) metadata verification.

Uses c2patool CLI (https://github.com/contentauth/c2patool) to inspect
Content Authenticity Initiative (CAI) / C2PA metadata embedded in video files.

Scoring logic:
  - Has valid C2PA + camera/device claim  → very likely real → score ~5-15
  - Has C2PA + AI/generative assertion     → very likely AI  → score ~90-95
  - Has C2PA but claims unknown            → neutral          → score ~40
  - No C2PA metadata present              → slightly real   → score ~35
    (reasoning: most organic content lacks C2PA; AI tools increasingly embed it)
  - c2patool error / unreadable file      → neutral          → score ~45

AI tool keywords checked in C2PA assertions:
  sora, runway, pika, kling, luma, gen-2, gen-3, stable diffusion,
  midjourney, dall-e, openai, video generation, ai.generated
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from app.analyzers.base import BaseAnalyzer, AnalyzerResult, Finding

logger = logging.getLogger(__name__)

C2PA_TOOL = "c2patool"

# Keywords in C2PA assertions that indicate AI generation
_AI_ASSERTION_KEYWORDS = [
    "sora", "runway", "pika", "kling", "luma", "haiper", "gen-2", "gen-3",
    "stable diffusion", "midjourney", "dall-e", "openai",
    "video generation", "ai.generated", "text-to-video", "c2pa.ai",
    "adobe.generative", "adobefirefly",
]

# Keywords in C2PA assertions that indicate real camera capture
_CAMERA_ASSERTION_KEYWORDS = [
    "c2pa.capture", "c2pa.camera", "exif", "gps", "capture.camera",
    "adobe.capture", "nikon", "canon", "sony", "apple", "google",
    "iphone", "android", "gopro",
]


class C2PAAnalyzer(BaseAnalyzer):
    """
    Verifies Content Credentials (C2PA) metadata in video files.

    Provides a secondary authentication signal:
    - Valid camera provenance → real
    - AI generation claim    → AI-generated
    - Missing / no claim     → neutral (slight real bias)
    """

    def analyze(self, video_path: str, fps: float = 25.0) -> AnalyzerResult:
        """
        Run c2patool on the video file and interpret the result.

        Args:
            video_path: path to video file on disk
            fps: ignored (included for API compatibility)

        Returns:
            AnalyzerResult with score 0-100
        """
        if not Path(video_path).exists():
            return AnalyzerResult(score=45.0, error=f"File not found: {video_path}")

        c2pa_data = self._run_c2patool(video_path)
        return self._interpret(c2pa_data, video_path)

    def _run_c2patool(self, video_path: str) -> Optional[dict]:
        """Run c2patool and return parsed JSON output, or None on error."""
        try:
            result = subprocess.run(
                [C2PA_TOOL, video_path, "--output-format", "json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    logger.debug(f"c2patool non-JSON output: {result.stdout[:200]}")
            # "No claim found" is the most common case
            stderr = result.stderr.strip().lower()
            if "no claim" in stderr or "no c2pa" in stderr:
                return {"_no_claim": True}
            logger.debug(f"c2patool exit {result.returncode}: {result.stderr[:100]}")
        except FileNotFoundError:
            logger.warning("c2patool not found in PATH — C2PA analysis skipped")
        except subprocess.TimeoutExpired:
            logger.warning(f"c2patool timeout on {video_path}")
        except Exception as e:
            logger.warning(f"c2patool error: {e}")
        return None

    def _interpret(self, data: Optional[dict], video_path: str) -> AnalyzerResult:
        """Map c2patool output to AI probability score."""
        findings: list[Finding] = []

        # ── No claim ────────────────────────────────────────────────────────
        if data is None:
            return AnalyzerResult(
                score=45.0,
                findings=[],
                error="c2patool execution failed",
            )

        if data.get("_no_claim"):
            # Most videos lack C2PA; slight real bias
            return AnalyzerResult(score=35.0, findings=[])

        # ── Has C2PA data ────────────────────────────────────────────────────
        data_str = json.dumps(data).lower()

        # Check for AI generation assertions
        ai_match = [kw for kw in _AI_ASSERTION_KEYWORDS if kw in data_str]
        if ai_match:
            confidence = min(95.0, 75.0 + len(ai_match) * 5.0)
            findings.append(Finding(
                type="c2pa_ai_generation",
                confidence=round(confidence, 1),
                description=(
                    f"C2PAメタデータにAI生成ツールの署名を検出: {', '.join(ai_match[:3])}"
                ),
                metadata={"matched_keywords": ai_match},
            ))
            return AnalyzerResult(score=round(confidence, 2), findings=findings)

        # Check for camera/capture assertions → real
        camera_match = [kw for kw in _CAMERA_ASSERTION_KEYWORDS if kw in data_str]
        if camera_match:
            # Has real camera provenance → likely real
            confidence = min(90.0, 70.0 + len(camera_match) * 5.0)
            real_score = max(5.0, 15.0 - len(camera_match) * 2.0)
            findings.append(Finding(
                type="c2pa_camera_provenance",
                confidence=round(confidence, 1),
                description=(
                    f"C2PAにカメラ撮影の出処情報を確認: {', '.join(camera_match[:3])}"
                ),
                metadata={"matched_keywords": camera_match},
            ))
            return AnalyzerResult(score=round(real_score, 2), findings=findings)

        # Has C2PA but unknown claims
        return AnalyzerResult(score=40.0, findings=findings)
