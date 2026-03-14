# FakeGuard リアルモデル統合 — 要件定義書

## 1. 目的
モックエンジンを実際のAI検出モデルに置き換え、リアルな検出スコアを返す。

## 2. 使用モデル（リサーチ結果）

### 2.1 フレーム解析（FrameAnalyzer）

**Primary: SigLIP-based Deepfake Detector**
- Model: `prithivMLmods/deepfake-detector-model-v1` (HuggingFace)
- Base: `google/siglip-base-patch16-512` fine-tuned
- Task: Real vs AI-generated image binary classification
- 精度: 高い汎化性能（CLIP-ViT特徴空間利用）

**Secondary: EfficientNet-B0 (FaceForensics++)**
- Model: `TRahulsingh/DeepfakeDetector` approach
- Base: EfficientNet-B0 (ImageNet pre-trained)
- Training: DFDC, DFD, Celeb-DF datasets
- 精度: ~92% on known deepfakes

**Fallback: UniversalFakeDetect**
- Model: TrueMedia.org `UniversalFakeDetect`
- Base: CLIP-ViT pre-trained features
- 特徴: 未知のAI生成手法にも汎化

### 2.2 テクスチャ解析（FFT/周波数解析）
- 手法: 高速フーリエ変換（FFT）でフレームの周波数成分を分析
- 根拠: AI生成画像はGAN/Diffusionの特徴的な周波数パターンを持つ
- 実装: NumPy/SciPy FFT → 周波数分布の統計的特徴量抽出

### 2.3 時系列解析（TemporalAnalyzer）
- オプティカルフロー: RAFT (Recurrent All-Pairs Field Transforms)
  - `torchvision.models.optical_flow.raft_large`
- フレーム間一貫性: コサイン類似度 + L2距離
- フリッカー検出: 隣接フレームの差分の標準偏差

### 2.4 音声解析（AudioAnalyzer）
- メルスペクトログラム: librosa
- 音声合成度: wav2vec2 ベースの特徴量抽出
- 環境音整合性: スペクトログラムの統計的特性

### 2.5 メタデータ解析（MetadataAnalyzer）
- FFprobe: コーデック/ビットレート/エンコーダ情報
- AIツールシグネチャDB:
  ```json
  {
    "sora": {"encoder_patterns": ["libx264", "h264_nvenc"], "fps_common": [24, 30], "resolution_common": ["1920x1080"]},
    "runway": {"encoder_patterns": ["libx264"], "bitrate_range": [2000, 8000]},
    "pika": {"encoder_patterns": ["libx264"], "fps_common": [24]},
    "kling": {"encoder_patterns": ["h264"], "fps_common": [30]},
    "veo": {"encoder_patterns": ["vp9", "h264"], "fps_common": [24, 30]}
  }
  ```

### 2.6 戦争映像解析（WarFootageAnalyzer）
- 爆発検出: YOLOv8 + カスタムクラス（explosion, smoke, fire）
- 煙の均一性: ピクセル分散解析
- 音響-映像同期: 爆発の光と音のタイムラグ（340m/s で距離推定）

## 3. アーキテクチャ

```
入力動画
  │
  ├── FFmpeg → フレーム抽出（2fps）
  │     │
  │     ├── SigLIP Deepfake Detector → AI生成確率
  │     ├── FFT テクスチャ解析 → 周波数異常スコア
  │     ├── MediaPipe Face Mesh → 顔ランドマーク異常
  │     └── EasyOCR → テキスト可読性チェック
  │
  ├── RAFT → オプティカルフロー → 物理整合性スコア
  │
  ├── librosa → メルスペクトログラム → 音声合成度
  │
  ├── FFprobe → メタデータ → ツールシグネチャ照合
  │
  └── [war_footage mode]
        ├── YOLOv8 → 爆発/煙検出 → 物理整合性
        └── 音響-映像タイムラグ解析
  
  ↓ EnsembleScorer（重み付き統合）
  ↓
  最終判定: 0-100% + ラベル + 根拠リスト
```

## 4. 依存パッケージ

```
# ML Core
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
timm>=0.9.0

# Image/Video Processing
opencv-python-headless>=4.8.0
Pillow>=10.0.0
ffmpeg-python>=0.2.0

# Audio
librosa>=0.10.0
soundfile>=0.12.0

# Face Detection
mediapipe>=0.10.0

# Text Detection
easyocr>=1.7.0

# Frequency Analysis
scipy>=1.11.0
numpy>=1.24.0

# Optical Flow (included in torchvision)
# RAFT model from torchvision.models.optical_flow

# Object Detection (war footage)
ultralytics>=8.0.0  # YOLOv8

# API
fastapi>=0.104.0
uvicorn>=0.24.0
celery>=5.3.0
redis>=5.0.0
sqlalchemy>=2.0.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.0
python-multipart>=0.0.6
```

## 5. モデルダウンロード戦略

初回起動時に自動ダウンロード:
```python
from transformers import AutoModelForImageClassification, AutoProcessor

# SigLIP Deepfake Detector
model = AutoModelForImageClassification.from_pretrained(
    "prithivMLmods/deepfake-detector-model-v1"
)
processor = AutoProcessor.from_pretrained(
    "prithivMLmods/deepfake-detector-model-v1"
)

# RAFT Optical Flow
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
raft = raft_large(weights=Raft_Large_Weights.DEFAULT)
```

キャッシュ: `~/.cache/huggingface/` (デフォルト)

## 6. テスト計画

### 6.1 テスト戦略
- **TDD**: テストファースト
- **レイヤー**: Unit → Integration → E2E
- **モック方針**: MLモデルのロードが重いためconftest.pyでセッションスコープのfixture使用

### 6.2 テスト仕様

#### Unit Tests

**test_frame_analyzer_real.py**
```python
class TestRealFrameAnalyzer:
    def test_ai_generated_image_high_score(self):
        """AI生成画像(Stable Diffusion出力)に高スコアを返す"""
        # fixture: tests/fixtures/ai_generated/sd_sample.png
        
    def test_real_photo_low_score(self):
        """実写写真に低スコアを返す"""
        # fixture: tests/fixtures/real/photo_sample.jpg
        
    def test_fft_detects_gan_pattern(self):
        """FFTでGAN特有の周波数パターンを検出"""
        
    def test_fft_real_image_no_pattern(self):
        """実写画像のFFTは異常パターンなし"""
        
    def test_face_landmark_normal(self):
        """正常な顔のランドマークは異常なし"""
        
    def test_face_landmark_deepfake(self):
        """ディープフェイク顔のランドマーク異常を検出"""
```

**test_temporal_analyzer_real.py**
```python
class TestRealTemporalAnalyzer:
    def test_optical_flow_consistency_real_video(self):
        """実写動画のオプティカルフローは一貫性あり"""
        
    def test_optical_flow_inconsistency_ai_video(self):
        """AI生成動画のフロー不一致を検出"""
        
    def test_flicker_detection(self):
        """フレーム間フリッカーを検出"""
        
    def test_physics_violation_gravity(self):
        """重力に反する動きを検出"""
```

**test_audio_analyzer_real.py**
```python
class TestRealAudioAnalyzer:
    def test_synthetic_voice_detection(self):
        """合成音声を検出"""
        
    def test_natural_voice_low_score(self):
        """自然な音声は低スコア"""
        
    def test_environmental_sound_mismatch(self):
        """環境音と映像の不一致を検出"""
```

**test_metadata_analyzer_real.py**
```python
class TestRealMetadataAnalyzer:
    def test_sora_signature_detection(self):
        """Soraで生成された動画のシグネチャを検出"""
        
    def test_normal_video_no_signature(self):
        """通常のカメラ撮影動画はシグネチャなし"""
        
    def test_abnormal_bitrate(self):
        """異常なビットレートパターンを検出"""
```

**test_ensemble_scorer_real.py**
```python
class TestRealEnsembleScorer:
    def test_all_high_scores_ai_generated(self):
        """全アナライザー高スコア → ai_generated"""
        
    def test_all_low_scores_human_made(self):
        """全アナライザー低スコア → human_made"""
        
    def test_mixed_scores_uncertain(self):
        """混在スコア → uncertain"""
        
    def test_war_mode_weights(self):
        """war_footageモードの重み配分が正しい"""
        
    def test_high_confidence_finding_boost(self):
        """高信頼度の単一異常でスコア上昇"""
```

#### Integration Tests

**test_api_integration.py**
```python
class TestAPIIntegration:
    def test_upload_and_get_result(self):
        """アップロード → 解析 → 結果取得の一連の流れ"""
        
    def test_url_analysis_youtube(self):
        """YouTube URL → 解析の流れ"""
        
    def test_quota_enforcement(self):
        """クォータ超過で429"""
```

### 6.3 テストフィクスチャ

テスト用の小さな動画/画像を用意:
```
tests/fixtures/
  ├── ai_generated/
  │   ├── sd_sample.png      # Stable Diffusion生成（小さいサンプル）
  │   └── ai_video_3sec.mp4  # AI生成3秒動画
  ├── real/
  │   ├── photo_sample.jpg   # 実写写真
  │   └── real_video_3sec.mp4 # 実写3秒動画
  └── war_footage/
      └── mock_explosion.mp4 # テスト用爆発映像（フリー素材）
```

※ テストフィクスチャはGitに含めず、初回テスト時にダウンロードするconftest.pyフィクスチャを作る

## 7. 実装順序（TDD）

1. **テスト作成** → 全テストファイルを先に書く（RED）
2. **FrameAnalyzer** → SigLIPモデル統合（GREEN）
3. **FFT解析** → 周波数パターン検出（GREEN）
4. **TemporalAnalyzer** → RAFT統合（GREEN）
5. **AudioAnalyzer** → librosa統合（GREEN）
6. **MetadataAnalyzer** → シグネチャDB拡充（GREEN）
7. **WarFootageAnalyzer** → YOLOv8統合（GREEN）
8. **EnsembleScorer** → 実モデル出力の統合（GREEN）
9. **リファクタ** → コード品質向上（REFACTOR）

## 8. パフォーマンス要件
- 10秒動画: <60秒で解析完了
- 1分動画: <3分で解析完了
- GPU推論: CUDA対応（CPU fallback付き）
- モデルロード: 初回のみ（以後キャッシュ）

---
© 2026 TimeWealthy Limited — FakeGuard
