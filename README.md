# DeepGuard — AI生成動画検出SaaS

> 動画がAI生成か人間制作かを判定するAPI/SaaS

© 2026 TimeWealthy Limited — DeepGuard

---

## 概要

DeepGuardは、動画のAI生成度をマルチシグナル解析によって判定するSaaSプロダクトです。
ジャーナリスト・ファクトチェック機関・メディアプラットフォームを主な対象とし、
フェイク動画の拡散防止に貢献します。

### 判定基準

| スコア | 判定 | ラベル |
|---|---|---|
| 0–25 | 人間制作の可能性が高い | `human_made` |
| 26–50 | おそらく人間制作 | `likely_human` |
| 51–75 | 判定困難/一部AI加工の疑い | `uncertain` |
| 76–90 | AI生成の可能性が高い | `likely_ai` |
| 91–100 | ほぼ確実にAI生成 | `ai_generated` |

---

## アーキテクチャ

```
Client → FastAPI → Celery Worker → [FrameAnalyzer, TemporalAnalyzer,
                                    AudioAnalyzer, MetadataAnalyzer,
                                    WarFootageAnalyzer] → EnsembleScorer
                ↓
         PostgreSQL + Redis
```

---

## セットアップ

### 前提条件

- Python 3.11+
- ffmpeg (`brew install ffmpeg` or `apt install ffmpeg`)
- Redis (optional; tasks run in-thread without it)

### ローカル開発 (SQLite)

```bash
cd backend

# 1. 仮想環境を作る
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 依存パッケージをインストール
pip install -r requirements.txt

# 3. 環境変数を設定
cp ../.env.example ../.env
# .env を編集して SECRET_KEY を変更

# 4. サーバー起動
uvicorn app.main:app --reload

# → http://localhost:8000/docs でSwagger UIが開く
```

### テスト実行

```bash
cd backend
pytest tests/ -v
```

### Docker Compose (PostgreSQL + Redis 込み)

```bash
# .env.example をコピーして設定
cp .env.example .env

# 起動
docker compose up --build

# API: http://localhost:8000
# Swagger: http://localhost:8000/docs
```

---

## API エンドポイント

### 認証

```bash
# ユーザー登録
POST /api/v1/auth/signup
{"email": "user@example.com", "password": "SecurePass123"}

# ログイン
POST /api/v1/auth/login
{"email": "user@example.com", "password": "SecurePass123"}
```

### 解析

```bash
# ファイルアップロード
POST /api/v1/analyze
Content-Type: multipart/form-data
Authorization: Bearer {token}

Fields: file (MP4/MOV/WebM), mode (standard|war_footage)

# URL指定
POST /api/v1/analyze-url
{"url": "https://youtube.com/...", "mode": "standard"}

# 結果取得
GET /api/v1/results/{analysis_id}

# 履歴
GET /api/v1/history
Authorization: Bearer {token}
```

### レスポンス例

```json
{
  "id": "analysis_abc123",
  "status": "completed",
  "verdict": "ai_generated",
  "confidence": 87.3,
  "summary": "この動画はAI生成の可能性が極めて高いです（87.3%）",
  "details": {
    "frame_analysis": {"score": 82.1, "findings": [...]},
    "temporal_analysis": {"score": 89.5, "findings": [...]},
    "audio_analysis": {"score": 71.3, "findings": [...]},
    "metadata_analysis": {"score": 95.0, "findings": [...]}
  }
}
```

---

## 解析エンジン

| アナライザー | 手法 | 重み |
|---|---|---|
| FrameAnalyzer | EfficientNet/HuggingFace + FFT テクスチャ + MediaPipe 顔ランドマーク | 35% |
| TemporalAnalyzer | オプティカルフロー + フリッカー検出 + フレーム間一貫性 | 35% |
| AudioAnalyzer | メルスペクトログラム + 環境音整合性 + 無音パターン | 17.5% |
| MetadataAnalyzer | ffprobe + AI生成ツールシグネチャDB | 12.5% |
| WarFootageAnalyzer | 爆発物理整合性 + 煙均一性 + AV同期 | 10%* |

\* war_footage モード時のみ有効

---

## 料金プラン

| プラン | 月額 | 解析回数 |
|---|---|---|
| Free | ¥0 | 10回/月 |
| Journalist | ¥2,980 | 100回/月 |
| Pro | ¥9,800 | 500回/月 |
| Enterprise | 要相談 | 無制限 |

---

## 開発

### ディレクトリ構成

```
deepfake-detector/
├── docs/               # RFP, SPEC
├── backend/
│   ├── app/
│   │   ├── main.py     # FastAPI アプリ
│   │   ├── config.py   # 設定
│   │   ├── database.py # SQLAlchemy
│   │   ├── auth.py     # JWT認証
│   │   ├── models/     # User, Analysis, Finding
│   │   ├── api/        # auth, analyze, results ルート
│   │   ├── analyzers/  # 解析エンジン
│   │   ├── tasks/      # Celery タスク
│   │   └── utils/      # video, download ユーティリティ
│   ├── tests/          # pytest テスト
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

### Celery Worker 起動

```bash
cd backend
celery -A celery_worker.celery_app worker --loglevel=info
```

---

## ライセンス

© 2026 TimeWealthy Limited — DeepGuard. All rights reserved.
