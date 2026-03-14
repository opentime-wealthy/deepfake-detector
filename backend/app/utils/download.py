# © 2026 TimeWealthy Limited — DeepGuard
"""Video download utilities (yt-dlp)."""

import logging
import os
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_DOMAINS = [
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "tiktok.com",
    "instagram.com",
    "vimeo.com",
    "dailymotion.com",
]


def download_video(url: str, output_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Download a video from a URL using yt-dlp.

    Args:
        url: Video URL (YouTube, Twitter, TikTok, etc.)
        output_dir: Directory to save the file. Uses temp dir if None.

    Returns:
        (file_path, error_message) — file_path is None on failure
    """
    try:
        import yt_dlp
    except ImportError:
        return None, "yt-dlp not installed. Run: pip install yt-dlp"

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="deepguard_")

    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": output_template,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 500 * 1024 * 1024,  # 500 MB
        "socket_timeout": 30,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

            # yt-dlp may produce a different extension after merge
            for ext in [".mp4", ".webm", ".mkv", ".mov"]:
                candidate = os.path.splitext(filename)[0] + ext
                if os.path.exists(candidate):
                    return candidate, None

            if os.path.exists(filename):
                return filename, None

            return None, "Downloaded file not found"

    except Exception as e:
        logger.error(f"yt-dlp download failed for {url}: {e}")
        return None, str(e)


def is_supported_url(url: str) -> bool:
    """Check if the URL is from a supported platform."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in SUPPORTED_DOMAINS)
