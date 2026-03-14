# © 2026 TimeWealthy Limited — DeepGuard
"""Video utility functions using FFmpeg and OpenCV."""

import logging
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(
    file_path: str,
    fps: float = 2.0,
    max_frames: int = 120,
    resize: Optional[Tuple[int, int]] = (320, 240),
) -> Tuple[List[np.ndarray], float]:
    """
    Extract frames from a video file at the specified FPS.

    Args:
        file_path: Path to the video file
        fps: Target frame rate for extraction
        max_frames: Maximum number of frames to return
        resize: Optional (width, height) to resize frames

    Returns:
        (frames, original_fps) where frames is a list of BGR numpy arrays
    """
    try:
        import cv2

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {file_path}")
            return [], 0.0

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(original_fps / fps))

        frames = []
        frame_idx = 0

        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                if resize:
                    frame = cv2.resize(frame, resize)
                frames.append(frame)
            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {file_path} (original FPS: {original_fps})")
        return frames, original_fps

    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        return [], 0.0


def extract_audio(file_path: str, target_sr: int = 22050) -> Tuple[Optional[np.ndarray], int]:
    """
    Extract audio track from video file.

    Args:
        file_path: Path to the video file
        target_sr: Target sample rate

    Returns:
        (audio_array, sample_rate) or (None, 0) on failure
    """
    try:
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "ffmpeg", "-y",
            "-i", file_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(target_sr),
            "-ac", "1",  # mono
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)

        if result.returncode != 0:
            logger.warning(f"ffmpeg audio extraction failed: {result.stderr.decode()}")
            return None, 0

        audio, sr = sf.read(tmp_path)
        os.unlink(tmp_path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # to mono

        return audio.astype(np.float32), sr

    except FileNotFoundError:
        logger.warning("ffmpeg not found. Install ffmpeg.")
        return None, 0
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None, 0


def get_video_duration(file_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logger.debug(f"Could not get video duration: {e}")
    return 0.0


def validate_video_file(file_path: str) -> bool:
    """Check if the file is a valid video using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return result.returncode == 0 and "video" in result.stdout
    except Exception:
        return False
