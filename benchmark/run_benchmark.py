#!/usr/bin/env python3
"""
FakeGuard Benchmark — Real model, real data, no mocks.
Runs full detection pipeline on labeled videos and computes accuracy metrics.
"""

import os
import sys
import csv
import json
import time
import logging
import gc
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
import librosa
import soundfile as sf
from scipy.fft import fft2, fftshift
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("benchmark")

# ─── Configuration ───────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).parent
DATASET_DIR = BENCHMARK_DIR / "dataset"
RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
SAMPLE_FPS = 1  # extract 1 frame/sec for analysis (memory efficient)
MAX_FRAMES = 15  # cap frames per video
MAX_VIDEO_SECONDS = 60  # only process first 60s

# Device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")


# ─── Model Loading ────────────────────────────────────────────────────────────
_MODEL = None
_PROCESSOR = None
_MODEL_LOADED = False  # prevent repeated attempts

def load_model():
    global _MODEL, _PROCESSOR, _MODEL_LOADED
    if _MODEL_LOADED:
        return _MODEL, _PROCESSOR
    _MODEL_LOADED = True
    logger.info(f"Loading model: {MODEL_NAME}")
    try:
        # AutoImageProcessor works for vision-only SigLIP (no tokenizer needed)
        _PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        _MODEL.eval()
        # Keep on CPU for stability
        logger.info(f"Model loaded: {MODEL_NAME}")
        logger.info(f"Labels: {_MODEL.config.id2label}")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        _MODEL = None
        _PROCESSOR = None
    return _MODEL, _PROCESSOR


# ─── Video Processing ────────────────────────────────────────────────────────
def extract_frames(video_path: str, max_frames: int = MAX_FRAMES) -> tuple:
    """Extract frames at 1fps up to max_frames. Returns (frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps
    cap_duration = min(duration, MAX_VIDEO_SECONDS)

    # Sample at 1fps
    frame_interval = int(orig_fps)
    frames = []
    frame_count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_count / orig_fps
        if current_time > cap_duration:
            break
        if frame_count % frame_interval == 0:
            # Resize to reduce memory
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames, orig_fps


def extract_audio(video_path: str) -> Optional[tuple]:
    """Extract audio using ffmpeg, return (y, sr)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "22050", "-ac", "1", "-t", "60", tmp_path, "-y", "-loglevel", "error"],
            capture_output=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            y, sr = librosa.load(tmp_path, sr=22050, duration=60)
            os.unlink(tmp_path)
            return y, sr
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    except Exception as e:
        logger.debug(f"Audio extraction failed: {e}")
    return None


def get_metadata(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
             video_path],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        logger.debug(f"ffprobe failed: {e}")
    return {}


# ─── Analysis Layers ─────────────────────────────────────────────────────────
def analyze_frame_siglip(frames: list) -> float:
    """SigLIP model inference. Returns AI probability 0-100."""
    model, processor = load_model()
    if model is None or processor is None:
        return 50.0

    scores = []
    id2label = model.config.id2label
    logger.info(f"  SigLIP id2label: {id2label}")

    for frame in frames[:MAX_FRAMES]:
        try:
            rgb = frame[:, :, ::-1].astype(np.uint8)
            img = Image.fromarray(rgb)
            inputs = processor(images=img, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

            ai_prob = 0.0
            for idx, prob in enumerate(probs.tolist()):
                label = id2label.get(idx, "").lower()
                if any(kw in label for kw in ["artificial", "fake", "ai", "generated", "deepfake"]):
                    ai_prob = max(ai_prob, prob)  # prob is already 0-1
                elif any(kw in label for kw in ["real", "human", "authentic"]):
                    ai_prob = max(ai_prob, 1.0 - prob)

            scores.append(ai_prob * 100.0)  # convert to 0-100
        except Exception as e:
            logger.debug(f"SigLIP frame error: {e}")

    return float(np.mean(scores)) if scores else 50.0


def analyze_fft_texture(frames: list) -> float:
    """FFT-based texture uniformity analysis. High = AI-like."""
    scores = []
    for frame in frames[:MAX_FRAMES]:
        try:
            gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            fft_result = fft2(gray.astype(np.float64))
            fft_shift = fftshift(fft_result)
            magnitude = np.abs(fft_shift)

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            y_idx, x_idx = np.ogrid[:h, :w]
            dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
            mask_hf = dist > min(h, w) // 4

            hf_mag = magnitude[mask_hf].flatten()
            if hf_mag.max() < 1e-8:
                scores.append(100.0)
                continue

            norm = hf_mag / (hf_mag.max() + 1e-10)
            mean = float(np.mean(norm)) + 1e-10
            std = float(np.std(norm))
            cv = std / mean
            uniformity = max(0.0, 1.0 - min(1.0, cv))
            scores.append(uniformity * 100.0)
        except Exception as e:
            logger.debug(f"FFT error: {e}")

    return float(np.mean(scores)) if scores else 50.0


def analyze_optical_flow(frames: list) -> float:
    """Farneback optical flow. Low variance = AI-like (too smooth)."""
    if len(frames) < 2:
        return 50.0

    flow_magnitudes = []
    flow_variances = []

    for i in range(len(frames) - 1):
        try:
            g1 = cv2.cvtColor(frames[i].astype(np.uint8), cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames[i+1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(float(np.mean(mag)))
            flow_variances.append(float(np.var(mag)))
        except Exception as e:
            logger.debug(f"OF error: {e}")

    if not flow_variances:
        return 50.0

    # Low variance in flow magnitude = too uniform = AI-like
    mean_var = float(np.mean(flow_variances))
    mean_mag = float(np.mean(flow_magnitudes))

    # Normalize: low var relative to magnitude → AI-like
    if mean_mag < 0.01:
        # Nearly static video - no motion
        return 50.0

    cv_flow = np.std(flow_variances) / (np.mean(flow_variances) + 1e-10)
    # Low coefficient of variation in variance = too regular = AI
    smoothness = max(0.0, 1.0 - min(1.0, cv_flow * 0.5))
    return smoothness * 100.0


def analyze_interframe_consistency(frames: list) -> float:
    """Cosine similarity between consecutive frames. Too high = AI."""
    if len(frames) < 2:
        return 50.0

    similarities = []
    for i in range(len(frames) - 1):
        f1 = frames[i].astype(np.float32).flatten()
        f2 = frames[i+1].astype(np.float32).flatten()
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 > 0 and norm2 > 0:
            sim = float(np.dot(f1, f2) / (norm1 * norm2))
            similarities.append(sim)

    if not similarities:
        return 50.0

    mean_sim = float(np.mean(similarities))
    # Very high similarity (>0.99) = AI (too smooth transitions)
    # Very low similarity (<0.7) = scene cuts (not useful)
    ai_score = max(0.0, (mean_sim - 0.85) / 0.15) * 100.0
    return min(100.0, max(0.0, ai_score))


def analyze_audio_features(video_path: str) -> float:
    """Audio MFCC and spectral analysis. Returns AI probability 0-100."""
    audio = extract_audio(video_path)
    if audio is None:
        return 50.0  # no audio → uncertain

    y, sr = audio
    if len(y) < sr:  # < 1 second
        return 50.0

    scores = []

    try:
        # MFCC variance: AI audio tends to have less variance
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.var(mfcc))
        # Low MFCC variance → possibly AI generated
        # Typical real speech: var > 100; AI TTS: often < 50
        if mfcc_var < 20:
            scores.append(80.0)
        elif mfcc_var < 50:
            scores.append(60.0)
        elif mfcc_var < 100:
            scores.append(40.0)
        else:
            scores.append(20.0)
    except Exception as e:
        logger.debug(f"MFCC failed: {e}")

    try:
        # Spectral flatness: AI audio tends to be flatter
        flatness = librosa.feature.spectral_flatness(y=y)
        mean_flat = float(np.mean(flatness))
        # Flat spectrum → AI-like (synthesized)
        flat_score = min(100.0, mean_flat * 1000.0)
        scores.append(flat_score)
    except Exception as e:
        logger.debug(f"Spectral flatness failed: {e}")

    try:
        # Zero crossing rate: AI tends to be more periodic
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_std = float(np.std(zcr))
        # Low std = periodic = possibly AI
        periodic_score = max(0.0, 1.0 - min(1.0, zcr_std * 100)) * 70.0
        scores.append(periodic_score)
    except Exception as e:
        logger.debug(f"ZCR failed: {e}")

    return float(np.mean(scores)) if scores else 50.0


def analyze_metadata(video_path: str) -> float:
    """Metadata-based AI tool detection. Returns AI probability 0-100."""
    meta = get_metadata(video_path)
    if not meta:
        return 30.0

    score = 30.0  # default: somewhat real

    # AI tool signatures in metadata
    ai_keywords = [
        "sora", "runway", "pika", "kling", "haiper", "luma", "stable diffusion",
        "midjourney", "dalle", "openai", "gen-3", "gen3", "dreamachine",
        "artificial intelligence", "ai generated", "text-to-video"
    ]

    meta_str = json.dumps(meta).lower()

    # Check tags/comments for AI keywords
    for kw in ai_keywords:
        if kw in meta_str:
            score = min(95.0, score + 30.0)
            break

    # Check encoder/handler signatures
    format_tags = meta.get("format", {}).get("tags", {})
    for key, val in format_tags.items():
        val_lower = str(val).lower()
        for kw in ai_keywords:
            if kw in val_lower:
                score = min(95.0, score + 20.0)

    # Very short videos with no audio = AI demo clip
    fmt = meta.get("format", {})
    duration = float(fmt.get("duration", 0))
    streams = meta.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    has_video = any(s.get("codec_type") == "video" for s in streams)

    if has_video and not has_audio and duration < 30:
        score = min(90.0, score + 20.0)  # short silent clip → AI demo

    return score


# ─── Ensemble ────────────────────────────────────────────────────────────────
def compute_ensemble(scores: dict) -> float:
    """Weighted ensemble of layer scores."""
    weights = {
        "siglip": 0.35,
        "fft": 0.15,
        "optical_flow": 0.20,
        "interframe": 0.10,
        "audio": 0.10,
        "metadata": 0.10,
    }
    total = 0.0
    total_w = 0.0
    for k, w in weights.items():
        v = scores.get(k)
        if v is not None:
            total += v * w
            total_w += w

    return total / total_w if total_w > 0 else 50.0


# ─── Per-video Analysis ──────────────────────────────────────────────────────
def analyze_video(video_path: str, label: str) -> dict:
    """Run full pipeline on one video. Returns result dict."""
    logger.info(f"Analyzing: {Path(video_path).name} [{label}]")
    t0 = time.time()

    result = {
        "filename": Path(video_path).name,
        "label": label,  # "ai_generated" or "real"
        "duration_s": 0,
        "frames_analyzed": 0,
        "score_siglip": None,
        "score_fft": None,
        "score_optical_flow": None,
        "score_interframe": None,
        "score_audio": None,
        "score_metadata": None,
        "score_ensemble": None,
        "prediction": None,
        "correct": None,
        "processing_time_s": 0,
        "error": None,
    }

    try:
        # 1. Extract frames
        frames, fps = extract_frames(video_path)
        result["frames_analyzed"] = len(frames)
        if fps > 0:
            result["duration_s"] = round(
                int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)) / fps, 1
            )

        if not frames:
            result["error"] = "no_frames"
            result["score_ensemble"] = 50.0
            result["prediction"] = "uncertain"
            return result

        logger.info(f"  Frames: {len(frames)}, FPS: {fps:.1f}")

        # 2. SigLIP
        try:
            result["score_siglip"] = round(analyze_frame_siglip(frames), 1)
            logger.info(f"  SigLIP: {result['score_siglip']}")
        except Exception as e:
            logger.warning(f"  SigLIP failed: {e}")
            result["score_siglip"] = 50.0

        # Free model from memory after each video if on MPS
        if DEVICE == "mps":
            gc.collect()

        # 3. FFT texture
        try:
            result["score_fft"] = round(analyze_fft_texture(frames), 1)
            logger.info(f"  FFT: {result['score_fft']}")
        except Exception as e:
            logger.warning(f"  FFT failed: {e}")
            result["score_fft"] = 50.0

        # 4. Optical flow
        try:
            result["score_optical_flow"] = round(analyze_optical_flow(frames), 1)
            logger.info(f"  Optical Flow: {result['score_optical_flow']}")
        except Exception as e:
            logger.warning(f"  OF failed: {e}")
            result["score_optical_flow"] = 50.0

        # 5. Interframe consistency
        try:
            result["score_interframe"] = round(analyze_interframe_consistency(frames), 1)
            logger.info(f"  Interframe: {result['score_interframe']}")
        except Exception as e:
            logger.warning(f"  Interframe failed: {e}")
            result["score_interframe"] = 50.0

        # 6. Audio
        try:
            result["score_audio"] = round(analyze_audio_features(video_path), 1)
            logger.info(f"  Audio: {result['score_audio']}")
        except Exception as e:
            logger.warning(f"  Audio failed: {e}")
            result["score_audio"] = 50.0

        # 7. Metadata
        try:
            result["score_metadata"] = round(analyze_metadata(video_path), 1)
            logger.info(f"  Metadata: {result['score_metadata']}")
        except Exception as e:
            logger.warning(f"  Metadata failed: {e}")
            result["score_metadata"] = 30.0

        # 8. Ensemble
        scores = {
            "siglip": result["score_siglip"],
            "fft": result["score_fft"],
            "optical_flow": result["score_optical_flow"],
            "interframe": result["score_interframe"],
            "audio": result["score_audio"],
            "metadata": result["score_metadata"],
        }
        ensemble = compute_ensemble(scores)
        result["score_ensemble"] = round(ensemble, 1)

        # 9. Prediction: threshold at 55
        threshold = 55.0
        result["prediction"] = "ai_generated" if ensemble >= threshold else "real"
        result["correct"] = result["prediction"] == label

        logger.info(f"  Ensemble: {ensemble:.1f} → {result['prediction']} (correct={result['correct']})")

        # Free memory
        del frames
        gc.collect()

    except Exception as e:
        logger.error(f"  Error processing {video_path}: {e}")
        result["error"] = str(e)
        result["score_ensemble"] = 50.0
        result["prediction"] = "uncertain"

    result["processing_time_s"] = round(time.time() - t0, 1)
    return result


# ─── Metrics Computation ─────────────────────────────────────────────────────
def compute_metrics(results: list) -> dict:
    """Compute classification metrics from results."""
    valid = [r for r in results if r["prediction"] not in (None, "uncertain") and r["label"] in ("ai_generated", "real")]
    if not valid:
        return {}

    y_true = [1 if r["label"] == "ai_generated" else 0 for r in valid]
    y_pred = [1 if r["prediction"] == "ai_generated" else 0 for r in valid]
    y_score = [r["score_ensemble"] / 100.0 for r in valid]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = None

    # Per-layer metrics (using ensemble score as proxy, same threshold)
    layer_metrics = {}
    layers = ["siglip", "fft", "optical_flow", "interframe", "audio", "metadata"]
    for layer in layers:
        layer_scores = [r.get(f"score_{layer}") for r in valid]
        if all(s is not None for s in layer_scores):
            layer_pred = [1 if s >= 55.0 else 0 for s in layer_scores]
            layer_acc = accuracy_score(y_true, layer_pred)
            layer_f1 = f1_score(y_true, layer_pred, zero_division=0)
            layer_metrics[layer] = {
                "accuracy": round(layer_acc, 4),
                "f1": round(layer_f1, 4),
                "mean_ai_score": round(np.mean([s for s, t in zip(layer_scores, y_true) if t == 1]), 2),
                "mean_real_score": round(np.mean([s for s, t in zip(layer_scores, y_true) if t == 0]), 2),
            }

    return {
        "total_videos": len(results),
        "valid_predictions": len(valid),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "auc_roc": round(auc, 4) if auc else "N/A",
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "per_layer": layer_metrics,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("FakeGuard Benchmark — Start")
    logger.info("=" * 60)

    # Pre-load model
    load_model()

    all_results = []

    # AI-generated videos
    ai_dir = DATASET_DIR / "ai_generated"
    ai_videos = sorted(ai_dir.glob("*.mp4"))
    logger.info(f"\nAI-generated videos: {len(ai_videos)}")

    for vp in ai_videos:
        result = analyze_video(str(vp), label="ai_generated")
        all_results.append(result)

    # Real videos
    real_dir = DATASET_DIR / "real"
    real_videos = sorted(real_dir.glob("*.mp4"))
    logger.info(f"\nReal videos: {len(real_videos)}")

    for vp in real_videos:
        result = analyze_video(str(vp), label="real")
        all_results.append(result)

    # Save CSV
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    fieldnames = list(all_results[0].keys()) if all_results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    logger.info(f"\nResults saved: {csv_path}")

    # Compute metrics
    metrics = compute_metrics(all_results)
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 60)
    for k, v in metrics.items():
        if k != "per_layer":
            logger.info(f"  {k}: {v}")
    logger.info("\nPer-layer performance:")
    for layer, lm in metrics.get("per_layer", {}).items():
        logger.info(f"  {layer}: acc={lm['accuracy']:.3f}, f1={lm['f1']:.3f}, "
                    f"mean_ai={lm['mean_ai_score']:.1f}, mean_real={lm['mean_real_score']:.1f}")

    # Return metrics for report generation
    return all_results, metrics


if __name__ == "__main__":
    results, metrics = main()
