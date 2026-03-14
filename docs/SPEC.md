# 技術仕様書: DeepGuard MVP

## 1. システム概要

DeepGuardは動画のAI生成度を判定するSaaS。
REST APIとWebダッシュボードを提供する。

## 2. アーキテクチャ

```
[Client]
  ├── Web UI (Next.js)
  └── API Client
        │
        ▼
[API Gateway] FastAPI
  ├── POST /api/v1/analyze       (ファイルアップロード)
  ├── POST /api/v1/analyze-url   (URL指定)
  ├── GET  /api/v1/results/{id}  (結果取得)
  ├── GET  /api/v1/history       (履歴一覧)
  └── POST /api/v1/auth/signup   (ユーザー登録)
        │
        ▼
[Task Queue] Celery + Redis
        │
        ▼
[Analysis Workers]
  ├── FrameAnalyzer      (空間解析)
  ├── TemporalAnalyzer   (時系列解析)
  ├── AudioAnalyzer      (音声解析)
  ├── MetadataAnalyzer   (メタデータ解析)
  └── EnsembleScorer     (統合スコア算出)
        │
        ▼
[Storage]
  ├── PostgreSQL (結果/ユーザー)
  ├── Redis (キャッシュ/キュー)
  └── S3 (動画一時保存)
```

## 3. API仕様

### 3.1 POST /api/v1/analyze

動画ファイルをアップロードして解析を開始。

**Request:**
```
Content-Type: multipart/form-data
Authorization: Bearer {api_key}

Fields:
  file: (binary) MP4/MOV/WebM, max 500MB
  mode: "standard" | "war_footage" (default: "standard")
```

**Response (202 Accepted):**
```json
{
  "id": "analysis_abc123",
  "status": "processing",
  "estimated_seconds": 45,
  "poll_url": "/api/v1/results/analysis_abc123"
}
```

### 3.2 POST /api/v1/analyze-url

URL指定で解析。

**Request:**
```json
{
  "url": "https://youtube.com/watch?v=xxx",
  "mode": "standard"
}
```

**Response:** 同上

### 3.3 GET /api/v1/results/{id}

解析結果の取得。

**Response (200 OK):**
```json
{
  "id": "analysis_abc123",
  "status": "completed",
  "verdict": "ai_generated",
  "confidence": 87.3,
  "summary": "この動画はAI生成の可能性が高いです（87.3%）",
  "details": {
    "frame_analysis": {
      "score": 82.1,
      "findings": [
        {"type": "hand_anomaly", "frame": 142, "confidence": 91.2, "description": "左手の指が6本検出"},
        {"type": "texture_anomaly", "frame": 300, "confidence": 78.5, "description": "背景テクスチャに反復パターン"}
      ]
    },
    "temporal_analysis": {
      "score": 89.5,
      "findings": [
        {"type": "physics_violation", "frames": [200, 250], "confidence": 92.0, "description": "物体の落下速度が重力加速度と不一致"},
        {"type": "flicker", "frames": [150, 155], "confidence": 75.0, "description": "エッジフリッカリング検出"}
      ]
    },
    "audio_analysis": {
      "score": 71.3,
      "findings": [
        {"type": "env_mismatch", "timestamp": "00:12", "confidence": 68.0, "description": "室内環境だが反響パターンが不自然"}
      ]
    },
    "metadata_analysis": {
      "score": 95.0,
      "findings": [
        {"type": "codec_signature", "confidence": 95.0, "description": "H.264エンコードにAI生成ツール特有のパラメータ検出"}
      ]
    },
    "war_footage_analysis": null
  },
  "heatmap_url": "/api/v1/results/analysis_abc123/heatmap",
  "created_at": "2026-03-14T11:00:00Z",
  "completed_at": "2026-03-14T11:00:45Z"
}
```

### 3.4 判定基準

| スコア | 判定 | ラベル |
|---|---|---|
| 0-25 | 人間制作の可能性が高い | `human_made` |
| 26-50 | おそらく人間制作 | `likely_human` |
| 51-75 | 判定困難/一部AI加工の疑い | `uncertain` |
| 76-90 | AI生成の可能性が高い | `likely_ai` |
| 91-100 | ほぼ確実にAI生成 | `ai_generated` |

## 4. データモデル

### 4.1 Users
```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  plan VARCHAR(20) DEFAULT 'free',
  api_key VARCHAR(64) UNIQUE,
  monthly_quota INT DEFAULT 10,
  used_this_month INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW()
);
```

### 4.2 Analyses
```sql
CREATE TABLE analyses (
  id VARCHAR(64) PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  status VARCHAR(20) DEFAULT 'pending',
  mode VARCHAR(20) DEFAULT 'standard',
  source_type VARCHAR(10), -- 'upload' or 'url'
  source_url TEXT,
  file_path TEXT,
  verdict VARCHAR(20),
  confidence DECIMAL(5,2),
  summary TEXT,
  details JSONB,
  duration_seconds INT,
  video_duration_seconds DECIMAL(10,2),
  created_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP
);
```

### 4.3 Findings
```sql
CREATE TABLE findings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id VARCHAR(64) REFERENCES analyses(id),
  analyzer VARCHAR(30), -- 'frame', 'temporal', 'audio', 'metadata', 'war'
  type VARCHAR(50),
  confidence DECIMAL(5,2),
  description TEXT,
  frame_number INT,
  timestamp_sec DECIMAL(10,2),
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);
```

## 5. 解析パイプライン詳細

### 5.1 FrameAnalyzer
```python
class FrameAnalyzer:
    """フレームレベルのAI生成検出"""
    
    def analyze(self, frames: List[np.ndarray]) -> FrameResult:
        # 1. サンプリング（1秒あたり2フレーム）
        # 2. 各フレームをViT/EfficientNetに通す
        # 3. 顔検出 → ランドマーク異常チェック
        # 4. 手指検出 → 指の本数/関節チェック
        # 5. テキスト検出 → 可読性チェック
        # 6. テクスチャ解析 → FFT周波数分布
        # 7. エッジ解析 → 境界の自然さ
        pass
```

### 5.2 TemporalAnalyzer
```python
class TemporalAnalyzer:
    """時系列の一貫性チェック"""
    
    def analyze(self, frames: List[np.ndarray]) -> TemporalResult:
        # 1. オプティカルフロー計算（RAFT）
        # 2. 物体トラッキング → 消失/出現チェック
        # 3. 物理法則チェック（重力、運動量保存）
        # 4. フレーム間一貫性スコア
        # 5. フリッカリング検出
        pass
```

### 5.3 AudioAnalyzer
```python
class AudioAnalyzer:
    """音声の合成度チェック"""
    
    def analyze(self, audio: np.ndarray, sr: int) -> AudioResult:
        # 1. 音声分離（音声 vs 環境音）
        # 2. 音声合成度チェック（スペクトログラム解析）
        # 3. 環境音の整合性（反響パターン）
        # 4. リップシンクチェック（映像との同期）
        pass
```

### 5.4 MetadataAnalyzer
```python
class MetadataAnalyzer:
    """メタデータ/コーデック解析"""
    
    def analyze(self, file_path: str) -> MetadataResult:
        # 1. FFprobeでメタデータ抽出
        # 2. コーデックパラメータの異常検出
        # 3. 圧縮アーティファクトパターン
        # 4. 生成ツール痕跡の検出
        pass
```

### 5.5 WarFootageAnalyzer
```python
class WarFootageAnalyzer:
    """戦争映像特化解析"""
    
    def analyze(self, frames, audio) -> WarResult:
        # 1. 爆発検出 → パーティクル物理整合性
        # 2. 煙/炎検出 → 流体力学チェック
        # 3. 兵器/車両検出 → データベース照合
        # 4. 音響解析 → 爆発音の距離整合性
        # 5. 地理検証 → 衛星画像照合（利用可能な場合）
        pass
```

### 5.6 EnsembleScorer
```python
class EnsembleScorer:
    """各アナライザーの結果を統合"""
    
    def score(self, results: Dict[str, AnalysisResult]) -> FinalVerdict:
        weights = {
            'frame': 0.30,
            'temporal': 0.30,
            'audio': 0.15,
            'metadata': 0.15,
            'war': 0.10  # war_footageモードの場合のみ
        }
        # 重み付き平均 + 高信頼度の異常値は重み増加
        pass
```

## 6. テスト仕様

### 6.1 ユニットテスト
```
tests/
  ├── test_frame_analyzer.py
  ├── test_temporal_analyzer.py
  ├── test_audio_analyzer.py
  ├── test_metadata_analyzer.py
  ├── test_war_analyzer.py
  ├── test_ensemble_scorer.py
  ├── test_api.py
  └── fixtures/
      ├── ai_generated/     # AI生成動画サンプル
      ├── human_made/        # 実写動画サンプル
      └── war_footage/       # 戦争映像サンプル
```

### 6.2 テストケース（TDD）
```python
# test_ensemble_scorer.py
class TestEnsembleScorer:
    def test_high_ai_score_returns_ai_generated(self):
        """全アナライザーが高スコア → ai_generated判定"""
        
    def test_low_score_returns_human_made(self):
        """全アナライザーが低スコア → human_made判定"""
        
    def test_mixed_scores_returns_uncertain(self):
        """スコアが混在 → uncertain判定"""
        
    def test_single_high_confidence_finding_elevates_score(self):
        """1つでも高信頼度の異常 → スコア上昇"""
        
    def test_war_mode_includes_war_analyzer(self):
        """war_footageモードでWarAnalyzerが有効"""

# test_frame_analyzer.py
class TestFrameAnalyzer:
    def test_detects_six_fingers(self):
        """6本指の手を検出"""
        
    def test_detects_text_anomaly(self):
        """読めないテキストを検出"""
        
    def test_real_frame_low_score(self):
        """実写フレームは低スコア"""

# test_api.py
class TestAPI:
    def test_upload_returns_202(self):
        """ファイルアップロード → 202 Accepted"""
        
    def test_invalid_format_returns_400(self):
        """非動画ファイル → 400 Bad Request"""
        
    def test_quota_exceeded_returns_429(self):
        """クォータ超過 → 429 Too Many Requests"""
        
    def test_results_pending(self):
        """処理中 → status: processing"""
        
    def test_results_completed(self):
        """完了 → verdict + confidence"""
```

## 7. ディレクトリ構成

```
deepfake-detector/
├── docs/
│   ├── RFP.md
│   └── SPEC.md
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # 設定
│   │   ├── models/              # SQLAlchemy models
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── analysis.py
│   │   │   └── finding.py
│   │   ├── api/                 # APIルート
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── analyze.py
│   │   │   └── results.py
│   │   ├── analyzers/           # 解析エンジン
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── frame.py
│   │   │   ├── temporal.py
│   │   │   ├── audio.py
│   │   │   ├── metadata.py
│   │   │   ├── war_footage.py
│   │   │   └── ensemble.py
│   │   ├── tasks/               # Celeryタスク
│   │   │   ├── __init__.py
│   │   │   └── analyze.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── video.py         # FFmpeg操作
│   │       └── download.py      # URL→動画DL
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_api.py
│   │   ├── test_frame_analyzer.py
│   │   ├── test_temporal_analyzer.py
│   │   ├── test_audio_analyzer.py
│   │   ├── test_metadata_analyzer.py
│   │   ├── test_war_analyzer.py
│   │   ├── test_ensemble_scorer.py
│   │   └── fixtures/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── celery_worker.py
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # LP
│   │   │   ├── dashboard/
│   │   │   ├── analyze/
│   │   │   └── results/
│   │   └── components/
│   ├── package.json
│   └── tailwind.config.ts
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## 8. MVP実装の優先順位

### Must Have (Week 1-2)
1. FastAPI基盤 + DB + 認証
2. 動画アップロード → フレーム抽出
3. FrameAnalyzer (EfficientNet/ViTベースの分類)
4. MetadataAnalyzer (FFprobe解析)
5. EnsembleScorer
6. REST API (analyze + results)
7. テスト

### Should Have (Week 3)
8. TemporalAnalyzer (オプティカルフロー)
9. AudioAnalyzer (音声合成度)
10. URL解析 (yt-dlp)
11. Web UI (Next.js)

### Nice to Have (Week 4)
12. WarFootageAnalyzer
13. ヒートマップ可視化
14. ブラウザ拡張

---
© 2026 TimeWealthy Limited — DeepGuard
