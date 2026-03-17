# FakeGuard Scientific Validation Plan

## 目的
FakeGuardの検出精度を科学的に証明し、顧客が信頼できるエビデンスを提供する

## 1. ベンチマークデータセット

### 必須（業界標準）
- **FaceForensics++** — 1,000本のディープフェイク動画、学術論文2,000+引用
  - Deepfakes, Face2Face, FaceSwap, NeuralTextures
  - URL: https://github.com/ondyari/FaceForensics
- **Celeb-DF v2** — セレブのディープフェイク5,639本、Li et al. 2020
  - URL: https://github.com/yuezunli/celeb-deepfakeforensics
- **DFDC (Deepfake Detection Challenge)** — Meta/Facebook提供、100,000+本
  - URL: https://ai.meta.com/datasets/dfdc/

### 追加（AI生成動画）
- **GenVideo** — Sora/Runway/Pika等の最新AI生成動画
  - 自前収集: 各ツールで100本生成 + YouTube/X上の公開AI動画
- **Real Videos Baseline** — YouTube/Vimeoからの確実に本物の動画1,000本

## 2. 測定指標

| 指標 | 目標 | 説明 |
|------|------|------|
| Accuracy | >95% | 全体の正答率 |
| AUC-ROC | >0.98 | 閾値非依存の分離性能 |
| F1-Score | >0.95 | 精度と再現率の調和平均 |
| False Positive Rate | <3% | 本物を偽物と誤判定する率 |
| False Negative Rate | <5% | 偽物を本物と見逃す率 |
| Detection Latency | <5sec | 1分動画の処理時間 |

## 3. 各検出層の学術的根拠

### Frame Analysis
- **SigLIP**: Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (2023)
- **FFT Texture**: Durall et al., "Unmasking DeepFakes with simple Features" (2020)
  - AI生成画像はフーリエ変換の高周波成分にスペクトル減衰の異常がある
- **Face Landmarks**: Matern et al., "Exploiting Visual Artifacts to Expose Deepfakes" (2019)

### Temporal Analysis
- **Optical Flow**: Amerini et al., "Deepfake Video Detection through Optical Flow" (2020)
  - AI生成動画はフレーム間の光学フローが不自然に滑らか
- **Inter-frame Consistency**: Guera & Delp, "Deepfake Video Detection Using RNN" (2018)

### Audio Analysis
- **MFCC**: Alzantot et al., "Deep Residual Neural Networks for Audio Spoofing Detection" (2019)
- **Spectral Flatness**: Wiener entropy — AI音声はスペクトルが均一すぎる

### Metadata
- AIツール署名DB（独自構築）— Sora, Runway, Pika, Kling等15ツールのメタデータパターン

## 4. 実行計画

### Phase 1: データセット取得 (Day 1-2)
- FaceForensics++ ダウンロード（要申請）
- Celeb-DF v2 ダウンロード
- Real videos baseline収集

### Phase 2: ベンチマーク実行 (Day 3-5)
- 全データセットに対してFakeGuard実行
- 結果をCSV出力
- AUC-ROC, F1, Accuracy計算

### Phase 3: モデル改善 (Day 6-10)
- 弱点分析（どのタイプで誤判定が多いか）
- Ensemble重みの最適化（グリッドサーチ）
- 必要ならファインチューニング

### Phase 4: レポート作成 (Day 11-12)
- ベンチマーク結果をLP/ドキュメントに掲載
- 技術ホワイトペーパー作成
- 「FakeGuard Detection Methodology」を公開
