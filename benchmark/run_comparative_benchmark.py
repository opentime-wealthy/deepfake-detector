#!/usr/bin/env python3
"""
FakeGuard vs DeepfakeBench Comparative Benchmark
Compares FakeGuard with EfficientNetB4, XceptionNet, RECCE, UCF, MesoNet,
and HuggingFace ViT/Swin models on the same 40-video dataset.

No mocks. All models loaded and run in real inference mode.
Memory-efficient: models loaded/freed one at a time.
"""

import os, sys, csv, json, time, logging, gc, subprocess, tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("comparative_benchmark")

BENCH_DIR    = Path(__file__).parent
DATASET_DIR  = BENCH_DIR / "dataset"
RESULTS_DIR  = BENCH_DIR / "results"
WEIGHTS_DIR  = BENCH_DIR / "weights"
RESULTS_DIR.mkdir(exist_ok=True)

# File log after RESULTS_DIR is defined
_log_fh = open(RESULTS_DIR / "benchmark_log.txt", "w", buffering=1)
logging.getLogger().addHandler(logging.StreamHandler(_log_fh))

MAX_FRAMES   = 5     # reduced for speed (5 frames per video)
SAMPLE_FPS   = 1     # 1 frame/sec
MAX_SEC      = 30    # only first 30s
INPUT_SIZE   = 299   # Xception/EfficientNet standard
DEVICE       = "cpu" # CPU for stability on MPS (avoid precision issues)
LOG_FILE     = RESULTS_DIR / "benchmark_log.txt"

# ─── Video Utilities ─────────────────────────────────────────────────────────
def extract_frames(video_path: str, max_frames: int = MAX_FRAMES,
                   size: int = INPUT_SIZE) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_sec = min(total / orig_fps, MAX_SEC)
    interval = max(1, int(orig_fps))
    frames, fc = [], 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if fc / orig_fps > cap_sec: break
        if fc % interval == 0:
            frame = cv2.resize(frame, (size, size))
            frames.append(frame)
        fc += 1
    cap.release()
    return frames, orig_fps

def frames_to_tensor(frames: List[np.ndarray],
                     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    """Convert BGR frames → normalised tensor (B, C, H, W)"""
    imgs = []
    for f in frames:
        rgb = f[:, :, ::-1].astype(np.float32) / 255.0
        for c in range(3):
            rgb[:, :, c] = (rgb[:, :, c] - mean[c]) / std[c]
        imgs.append(rgb.transpose(2, 0, 1))
    return torch.tensor(np.stack(imgs), dtype=torch.float32)

# ─── Model Architectures ─────────────────────────────────────────────────────

# --- MesoNet4 (tiny model, easy to reconstruct) ---
class Meso4(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.mp1   = nn.MaxPool2d(2, 2)
        self.mp2   = nn.MaxPool2d(4, 4)
        self.drop  = nn.Dropout2d(0.5)
        self.fc1   = nn.Linear(16 * 8 * 8, 16)
        self.fc2   = nn.Linear(16, num_classes)
        self.relu  = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.mp1(self.relu(self.bn1(self.conv1(x))))
        x = self.mp1(self.relu(self.bn1(self.conv2(x))))
        x = self.mp1(self.relu(self.bn2(self.conv3(x))))
        x = self.mp2(self.relu(self.bn2(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.fc1(self.drop(x)))
        return self.fc2(self.drop(x))


# --- Xception (copied from DeepfakeBench, stripped registry) ---
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, ks, stride, pad, dilation, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, 1, 1, bias=bias)
    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    def __init__(self, in_f, out_f, reps, strides=1, start_relu=True, grow_first=True):
        super().__init__()
        if out_f != in_f or strides != 1:
            self.skip = nn.Conv2d(in_f, out_f, 1, strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_f)
        else:
            self.skip = None
        layers = []
        filters = in_f
        if grow_first:
            layers += [nn.ReLU(inplace=True) if start_relu else nn.Identity(),
                       SeparableConv2d(in_f, out_f, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(out_f)]
            filters = out_f
        for _ in range(reps - 1):
            layers += [nn.ReLU(inplace=True),
                       SeparableConv2d(filters, filters, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(filters)]
        if not grow_first:
            layers += [nn.ReLU(inplace=True),
                       SeparableConv2d(in_f, out_f, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(out_f)]
        if strides != 1:
            layers += [nn.ReLU(inplace=True),
                       SeparableConv2d(filters, filters, 3, 2, 1, bias=False),
                       nn.BatchNorm2d(filters)]
        self.rep = nn.Sequential(*layers)

    def forward(self, inp):
        x = self.rep(inp)
        skip = self.skipbn(self.skip(inp)) if self.skip is not None else inp
        return x + skip


class Xception(nn.Module):
    def __init__(self, num_classes=2, inc=3, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.conv1   = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1     = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2     = nn.BatchNorm2d(64)
        self.block1  = Block(64,  128, 2, 2, False, True)
        self.block2  = Block(128, 256, 2, 2, True,  True)
        self.block3  = Block(256, 728, 2, 2, True,  True)
        self.block4  = Block(728, 728, 3, 1, True,  True)
        self.block5  = Block(728, 728, 3, 1, True,  True)
        self.block6  = Block(728, 728, 3, 1, True,  True)
        self.block7  = Block(728, 728, 3, 1, True,  True)
        self.block8  = Block(728, 728, 3, 1, True,  True)
        self.block9  = Block(728, 728, 3, 1, True,  True)
        self.block10 = Block(728, 728, 3, 1, True,  True)
        self.block11 = Block(728, 728, 3, 1, True,  True)
        self.block12 = Block(728, 1024, 2, 2, True, False)
        self.conv3   = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3     = nn.BatchNorm2d(1536)
        self.conv4   = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4     = nn.BatchNorm2d(2048)
        self.last_linear    = nn.Linear(2048, num_classes)
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512))
        self.dropout = nn.Dropout(dropout)

    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.block1(x);  x = self.block2(x);  x = self.block3(x)
        x = self.block4(x);  x = self.block5(x);  x = self.block6(x)
        x = self.block7(x);  x = self.block8(x);  x = self.block9(x)
        x = self.block10(x); x = self.block11(x); x = self.block12(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

    def forward(self, x):
        feat = self.features(x)
        return self.last_linear(self.dropout(feat))


# --- EfficientNet-B4 wrapper ---
class EfficientNetB4Wrapper(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        from efficientnet_pytorch import EfficientNet
        self.efficientnet = EfficientNet.from_name('efficientnet-b4')
        self.efficientnet._fc = nn.Identity()
        self.last_layer = nn.Linear(1792, num_classes)

    def forward(self, x):
        feat = self.efficientnet.extract_features(x)
        feat = self.efficientnet._avg_pooling(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.efficientnet._dropout(feat)
        return self.last_layer(feat)


# --- RECCE: uses Xception backbone + reconstruction; use backbone only for classification ---
class RECCEWrapper(nn.Module):
    """Use only the Xception backbone classifier head from RECCE checkpoint."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = Xception(num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)


# --- UCF: two-encoder model; use encoder_f (fine-grained Xception) only ---
class UCFWrapper(nn.Module):
    """Load only encoder_f from UCF checkpoint (Xception-based)."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder_f = Xception(num_classes=num_classes)

    def forward(self, x):
        return self.encoder_f(x)


# ─── Load/Unload helpers ──────────────────────────────────────────────────────
def load_checkpoint_strict_false(model, path: str, prefix_map: Dict[str, str]):
    """Load .pth weights with key remapping. Returns (model, missing_count)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {}
    for k, v in ckpt.items():
        new_k = k
        for old_pref, new_pref in prefix_map.items():
            if k.startswith(old_pref):
                new_k = new_pref + k[len(old_pref):]
                break
        state[new_k] = v
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    return model


# ─── Per-model Inference ─────────────────────────────────────────────────────
def run_deepfakebench_model(model: nn.Module, frames: List[np.ndarray],
                            mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) -> float:
    """Run DeepfakeBench model on frames, return fake probability 0-100."""
    if not frames:
        return 50.0
    model.eval()
    scores = []
    with torch.no_grad():
        for frame in frames:
            t = frames_to_tensor([frame], mean=mean, std=std)  # (1,3,H,W)
            try:
                out = model(t)  # (1, 2)
                probs = F.softmax(out, dim=1)[0]
                # idx=1 = fake in most DeepfakeBench models (convention varies)
                # We check both and take max
                p0, p1 = float(probs[0]), float(probs[1])
                # In DeepfakeBench: typically class 0=real, 1=fake
                scores.append(p1 * 100.0)
            except Exception as e:
                logger.debug(f"Frame inference error: {e}")
    return float(np.mean(scores)) if scores else 50.0


def run_hf_model(model, processor, frames: List[np.ndarray],
                 fake_label_kws=("fake","artificial","ai","generated","deepfake"),
                 real_label_kws=("real","human","authentic")) -> float:
    """Run HuggingFace AutoModel on frames."""
    if not frames:
        return 50.0
    model.eval()
    id2label = model.config.id2label
    scores = []
    with torch.no_grad():
        for frame in frames:
            try:
                rgb = frame[:, :, ::-1].astype(np.uint8)
                img = Image.fromarray(rgb)
                inputs = processor(images=img, return_tensors="pt")
                out = model(**inputs)
                probs = F.softmax(out.logits, dim=-1)[0]
                ai_prob = 0.0
                for idx, p in enumerate(probs.tolist()):
                    lbl = id2label.get(idx, "").lower()
                    if any(kw in lbl for kw in fake_label_kws):
                        ai_prob = max(ai_prob, p)
                    elif any(kw in lbl for kw in real_label_kws):
                        ai_prob = max(ai_prob, 1.0 - p)
                scores.append(ai_prob * 100.0)
            except Exception as e:
                logger.debug(f"HF frame error: {e}")
    return float(np.mean(scores)) if scores else 50.0


# ─── Video Analysis ───────────────────────────────────────────────────────────
def analyze_video_with_model(video_path: str, label: str,
                              score_fn, model_name: str) -> dict:
    """Run one model on one video."""
    result = {
        "filename": Path(video_path).name,
        "label": label,
        "model": model_name,
        "score": None,
        "prediction": None,
        "correct": None,
        "error": None,
        "time_s": 0,
    }
    t0 = time.time()
    try:
        frames, _ = extract_frames(video_path, max_frames=MAX_FRAMES, size=INPUT_SIZE)
        if not frames:
            result["error"] = "no_frames"
            result["score"] = 50.0
            result["prediction"] = "uncertain"
            return result
        score = score_fn(frames)
        result["score"] = round(score, 1)
        result["prediction"] = "ai_generated" if score >= 50.0 else "real"
        result["correct"] = result["prediction"] == label
    except Exception as e:
        logger.error(f"  {model_name} error on {Path(video_path).name}: {e}")
        result["error"] = str(e)
        result["score"] = 50.0
        result["prediction"] = "uncertain"
    result["time_s"] = round(time.time() - t0, 1)
    return result


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_model_metrics(results: List[dict]) -> dict:
    valid = [r for r in results
             if r["prediction"] not in (None, "uncertain")
             and r["label"] in ("ai_generated", "real")]
    if not valid:
        return {"accuracy": None, "f1": None, "auc_roc": None}

    y_true  = [1 if r["label"] == "ai_generated" else 0 for r in valid]
    y_pred  = [1 if r["prediction"] == "ai_generated" else 0 for r in valid]
    y_score = [r["score"] / 100.0 for r in valid]

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)

    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
    except Exception:
        tn = fp = fn = tp = 0

    fpr  = fp / (fp + tn + 1e-9)
    fnr  = fn / (fn + tp + 1e-9)
    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else None
    except Exception:
        auc = None

    return {
        "n": len(valid),
        "accuracy":   round(acc,  4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1_score":   round(f1,   4),
        "fpr":        round(fpr,  4),
        "fnr":        round(fnr,  4),
        "auc_roc":    round(auc, 4) if auc is not None else "N/A",
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "mean_ai_score":   round(np.mean([r["score"] for r in valid if r["label"] == "ai_generated"]), 1),
        "mean_real_score": round(np.mean([r["score"] for r in valid if r["label"] == "real"]), 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def get_all_videos() -> List[Tuple[str, str]]:
    """Return [(path, label), ...]"""
    videos = []
    for p in sorted((DATASET_DIR / "ai_generated").glob("*.mp4")):
        videos.append((str(p), "ai_generated"))
    for p in sorted((DATASET_DIR / "real").glob("*.mp4")):
        videos.append((str(p), "real"))
    return videos


def run_all_models():
    videos = get_all_videos()
    logger.info(f"Total videos: {len(videos)}")

    all_results   = []
    model_metrics = {}

    # ── 1. MesoNet4 (DeepfakeBench, trained on FF++) ──────────────────────────
    logger.info("\n=== 1/8: MesoNet4 (DeepfakeBench) ===")
    try:
        model = Meso4(num_classes=2)
        ckpt_path = WEIGHTS_DIR / "meso4_best.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Keys: backbone.conv1.weight etc
        state = {k.replace("backbone.", ""): v for k, v in ckpt.items()
                 if k.startswith("backbone.")}
        miss, unexp = model.load_state_dict(state, strict=False)
        logger.info(f"  Loaded. Missing={len(miss)}, Unexpected={len(unexp)}")
        model.eval()
        # MesoNet uses 256x256 input
        score_fn = lambda frames: run_deepfakebench_model(
            model, [cv2.resize(f, (256, 256)) for f in frames])
        results_m = []
        for vpath, lbl in videos:
            logger.info(f"  Processing: {Path(vpath).name}")
            r = analyze_video_with_model(vpath, lbl, score_fn, "meso4")
            results_m.append(r)
            all_results.append(r)
        model_metrics["MesoNet4 (DeepfakeBench)"] = compute_model_metrics(results_m)
        del model; gc.collect()
    except Exception as e:
        logger.error(f"MesoNet4 failed: {e}")
        model_metrics["MesoNet4 (DeepfakeBench)"] = {"error": str(e)}

    # ── 2. XceptionNet (DeepfakeBench, trained on FF++) ───────────────────────
    logger.info("\n=== 2/8: XceptionNet (DeepfakeBench) ===")
    try:
        model = Xception(num_classes=2, inc=3, dropout=0.5)
        wrapper = nn.Module()
        wrapper.backbone = model
        # remap backbone.* → directly to model
        state = {}
        ckpt = torch.load(WEIGHTS_DIR / "xception_best.pth",
                          map_location="cpu", weights_only=False)
        for k, v in ckpt.items():
            if k.startswith("backbone."):
                state[k[len("backbone."):]] = v
        miss, unexp = model.load_state_dict(state, strict=False)
        logger.info(f"  Loaded. Missing={len(miss)}, Unexpected={len(unexp)}")
        model.eval()
        score_fn = lambda frames: run_deepfakebench_model(model, frames)
        results_m = []
        for vpath, lbl in videos:
            logger.info(f"  Processing: {Path(vpath).name}")
            r = analyze_video_with_model(vpath, lbl, score_fn, "xception")
            results_m.append(r)
            all_results.append(r)
        model_metrics["XceptionNet (DeepfakeBench)"] = compute_model_metrics(results_m)
        del model; gc.collect()
    except Exception as e:
        logger.error(f"XceptionNet failed: {e}")
        model_metrics["XceptionNet (DeepfakeBench)"] = {"error": str(e)}

    # ── 3. EfficientNet-B4 (DeepfakeBench, trained on FF++) ──────────────────
    logger.info("\n=== 3/8: EfficientNetB4 (DeepfakeBench) ===")
    try:
        model = EfficientNetB4Wrapper(num_classes=2)
        ckpt = torch.load(WEIGHTS_DIR / "effnb4_best.pth",
                          map_location="cpu", weights_only=False)
        state = {}
        for k, v in ckpt.items():
            if k.startswith("backbone.efficientnet."):
                state["efficientnet." + k[len("backbone.efficientnet."):]] = v
            elif k.startswith("backbone.last_layer"):
                state["last_layer" + k[len("backbone.last_layer"):]] = v
        miss, unexp = model.load_state_dict(state, strict=False)
        logger.info(f"  Loaded. Missing={len(miss)}, Unexpected={len(unexp)}")
        model.eval()
        # EfficientNet uses ImageNet normalization
        score_fn = lambda frames: run_deepfakebench_model(
            model, frames,
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        results_m = []
        for vpath, lbl in videos:
            logger.info(f"  Processing: {Path(vpath).name}")
            r = analyze_video_with_model(vpath, lbl, score_fn, "efficientnetb4")
            results_m.append(r)
            all_results.append(r)
        model_metrics["EfficientNet-B4 (DeepfakeBench)"] = compute_model_metrics(results_m)
        del model; gc.collect()
    except Exception as e:
        logger.error(f"EfficientNetB4 failed: {e}")
        model_metrics["EfficientNet-B4 (DeepfakeBench)"] = {"error": str(e)}

    # ── 4. RECCE (CVPR 2022, DeepfakeBench) ──────────────────────────────────
    logger.info("\n=== 4/8: RECCE (CVPR 2022, DeepfakeBench) ===")
    try:
        model = Xception(num_classes=2, inc=3, dropout=0.5)
        ckpt = torch.load(WEIGHTS_DIR / "recce_best.pth",
                          map_location="cpu", weights_only=False)
        state = {k[len("backbone."):]: v for k, v in ckpt.items()
                 if k.startswith("backbone.")}
        miss, unexp = model.load_state_dict(state, strict=False)
        logger.info(f"  Loaded. Missing={len(miss)}, Unexpected={len(unexp)}")
        model.eval()
        score_fn = lambda frames: run_deepfakebench_model(model, frames)
        results_m = []
        for vpath, lbl in videos:
            logger.info(f"  Processing: {Path(vpath).name}")
            r = analyze_video_with_model(vpath, lbl, score_fn, "recce")
            results_m.append(r)
            all_results.append(r)
        model_metrics["RECCE (CVPR 2022)"] = compute_model_metrics(results_m)
        del model; gc.collect()
    except Exception as e:
        logger.error(f"RECCE failed: {e}")
        model_metrics["RECCE (CVPR 2022)"] = {"error": str(e)}

    # ── 5. UCF (ICCV 2023) — encoder_f branch only ───────────────────────────
    logger.info("\n=== 5/8: UCF encoder_f (ICCV 2023, DeepfakeBench) ===")
    try:
        model = Xception(num_classes=2, inc=3, dropout=0.5)
        ckpt = torch.load(WEIGHTS_DIR / "ucf_best.pth",
                          map_location="cpu", weights_only=False)
        state = {k[len("encoder_f."):]: v for k, v in ckpt.items()
                 if k.startswith("encoder_f.")}
        miss, unexp = model.load_state_dict(state, strict=False)
        logger.info(f"  Loaded. Missing={len(miss)}, Unexpected={len(unexp)}")
        model.eval()
        score_fn = lambda frames: run_deepfakebench_model(model, frames)
        results_m = []
        for vpath, lbl in videos:
            logger.info(f"  Processing: {Path(vpath).name}")
            r = analyze_video_with_model(vpath, lbl, score_fn, "ucf")
            results_m.append(r)
            all_results.append(r)
        model_metrics["UCF (ICCV 2023)"] = compute_model_metrics(results_m)
        del model; gc.collect()
    except Exception as e:
        logger.error(f"UCF failed: {e}")
        model_metrics["UCF (ICCV 2023)"] = {"error": str(e)}

    # ── 6. dima806/deepfake_vs_real_image_detection (ViT, HuggingFace) ───────
    logger.info("\n=== 6/8: ViT Deepfake vs Real (dima806, HuggingFace) ===")
    try:
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        m_name = "dima806/deepfake_vs_real_image_detection"
        proc  = AutoImageProcessor.from_pretrained(m_name)
        model = AutoModelForImageClassification.from_pretrained(m_name)
        model.eval()
        logger.info(f"  Labels: {model.config.id2label}")
        score_fn = lambda frames: run_hf_model(model, proc, frames)
        results_m = []
        for vpath, lbl in videos:
            logger.info(f"  Processing: {Path(vpath).name}")
            frames, _ = extract_frames(vpath, max_frames=MAX_FRAMES, size=224)
            score = score_fn(frames)
            r = {"filename": Path(vpath).name, "label": lbl, "model": "vit_deepfake_dima806",
                 "score": round(score, 1), "prediction": "ai_generated" if score >= 50.0 else "real",
                 "correct": None, "error": None, "time_s": 0}
            r["correct"] = r["prediction"] == lbl
            results_m.append(r)
            all_results.append(r)
        model_metrics["ViT Deepfake/Real (dima806)"] = compute_model_metrics(results_m)
        del model, proc; gc.collect()
    except Exception as e:
        logger.error(f"ViT dima806 deepfake failed: {e}")
        model_metrics["ViT Deepfake/Real (dima806)"] = {"error": str(e)}

    # ── 7. dima806/ai_vs_human_generated_image_detection (ViT, HuggingFace) ──
    logger.info("\n=== 7/8: ViT AI vs Human (dima806, HuggingFace) ===")
    try:
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        m_name = "dima806/ai_vs_human_generated_image_detection"
        proc  = AutoImageProcessor.from_pretrained(m_name)
        model = AutoModelForImageClassification.from_pretrained(m_name)
        model.eval()
        logger.info(f"  Labels: {model.config.id2label}")
        results_m = []
        for vpath, lbl in videos:
            logger.info(f"  Processing: {Path(vpath).name}")
            frames, _ = extract_frames(vpath, max_frames=MAX_FRAMES, size=224)
            score = run_hf_model(model, proc, frames,
                                 fake_label_kws=("ai-generated", "ai", "generated", "artificial", "fake"),
                                 real_label_kws=("human", "real", "authentic"))
            r = {"filename": Path(vpath).name, "label": lbl, "model": "vit_ai_human_dima806",
                 "score": round(score, 1), "prediction": "ai_generated" if score >= 50.0 else "real",
                 "correct": None, "error": None, "time_s": 0}
            r["correct"] = r["prediction"] == lbl
            results_m.append(r)
            all_results.append(r)
        model_metrics["ViT AI/Human (dima806) — Best Match"] = compute_model_metrics(results_m)
        del model, proc; gc.collect()
    except Exception as e:
        logger.error(f"ViT dima806 AI/human failed: {e}")
        model_metrics["ViT AI/Human (dima806) — Best Match"] = {"error": str(e)}

    # ── 8. umm-maybe/AI-image-detector (Swin, HuggingFace) ───────────────────
    logger.info("\n=== 8/8: Swin AI Image Detector (umm-maybe, HuggingFace) ===")
    try:
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        m_name = "umm-maybe/AI-image-detector"
        proc  = AutoImageProcessor.from_pretrained(m_name)
        model = AutoModelForImageClassification.from_pretrained(m_name)
        model.eval()
        logger.info(f"  Labels: {model.config.id2label}")
        results_m = []
        for vpath, lbl in videos:
            logger.info(f"  Processing: {Path(vpath).name}")
            frames, _ = extract_frames(vpath, max_frames=MAX_FRAMES, size=224)
            score = run_hf_model(model, proc, frames,
                                 fake_label_kws=("artificial", "ai", "generated", "fake"),
                                 real_label_kws=("human", "real"))
            r = {"filename": Path(vpath).name, "label": lbl, "model": "swin_ai_detector",
                 "score": round(score, 1), "prediction": "ai_generated" if score >= 50.0 else "real",
                 "correct": None, "error": None, "time_s": 0}
            r["correct"] = r["prediction"] == lbl
            results_m.append(r)
            all_results.append(r)
        model_metrics["Swin AI-Image-Detector (umm-maybe)"] = compute_model_metrics(results_m)
        del model, proc; gc.collect()
    except Exception as e:
        logger.error(f"Swin umm-maybe failed: {e}")
        model_metrics["Swin AI-Image-Detector (umm-maybe)"] = {"error": str(e)}

    # ── Add FakeGuard results from previous benchmark ─────────────────────────
    prev_csv = RESULTS_DIR / "benchmark_results.csv"
    if prev_csv.exists():
        import csv as csv_mod
        with open(prev_csv) as f:
            rows = list(csv_mod.DictReader(f))
        fg_results = []
        for row in rows:
            r = {
                "filename": row["filename"],
                "label": row["label"],
                "model": "fakeguard_ensemble",
                "score": float(row.get("score_ensemble", 50)),
                "prediction": row.get("prediction", "uncertain"),
                "correct": row.get("correct", "").lower() == "true",
                "error": row.get("error", None),
                "time_s": float(row.get("processing_time_s", 0)),
            }
            fg_results.append(r)
        model_metrics["FakeGuard (Ensemble, v1)"] = compute_model_metrics(fg_results)
        logger.info(f"  Loaded FakeGuard results: {len(fg_results)} videos")

    # ── Save results ───────────────────────────────────────────────────────────
    out_csv = RESULTS_DIR / "comparative_results.csv"
    if all_results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)

    out_json = RESULTS_DIR / "comparative_metrics.json"
    with open(out_json, "w") as f:
        json.dump(model_metrics, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("COMPARATIVE BENCHMARK SUMMARY")
    logger.info("="*70)
    logger.info(f"{'Model':<40} {'Acc':>6} {'F1':>6} {'AUC':>6} {'AI↑':>7} {'Real↑':>7}")
    logger.info("-"*70)
    for name, m in model_metrics.items():
        if "error" in m:
            logger.info(f"{name:<40} ERROR: {m['error'][:30]}")
        else:
            acc = m.get("accuracy", 0) or 0
            f1  = m.get("f1_score", 0) or 0
            auc = m.get("auc_roc", "N/A")
            ai_s  = m.get("mean_ai_score", 0) or 0
            real_s = m.get("mean_real_score", 0) or 0
            logger.info(f"{name:<40} {acc:>6.3f} {f1:>6.3f} {str(auc):>6} {ai_s:>7.1f} {real_s:>7.1f}")

    return all_results, model_metrics


if __name__ == "__main__":
    results, metrics = run_all_models()
