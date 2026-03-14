"use client";

import { useState } from "react";

type AnalysisResult = {
  overall_score: number;
  verdict: "REAL" | "FAKE" | "SUSPICIOUS";
  confidence: number;
  layers: {
    name: string;
    score: number;
    status: "safe" | "warning" | "danger";
    detail: string;
  }[];
  processing_time: number;
};

const SAMPLE_URLS = [
  "https://www.youtube.com/watch?v=example1",
  "https://twitter.com/user/status/example",
  "https://tiktok.com/@user/video/example",
];

const MOCK_RESULTS: Record<string, AnalysisResult> = {
  default: {
    overall_score: 87,
    verdict: "FAKE",
    confidence: 92,
    processing_time: 18.4,
    layers: [
      { name: "Frame Analysis", score: 91, status: "danger", detail: "GAN指紋検出: 高確率でStyleGAN3生成" },
      { name: "Temporal Analysis", score: 76, status: "warning", detail: "フレーム間補間に不自然なパターン" },
      { name: "Audio Analysis", score: 89, status: "danger", detail: "音声クローン特徴を検出" },
      { name: "Metadata Analysis", score: 95, status: "danger", detail: "EXIF改ざん痕跡・タイムスタンプ矛盾" },
      { name: "War Footage", score: 12, status: "safe", detail: "紛争映像特徴なし" },
    ],
  },
  real: {
    overall_score: 8,
    verdict: "REAL",
    confidence: 96,
    processing_time: 14.2,
    layers: [
      { name: "Frame Analysis", score: 5, status: "safe", detail: "自然なノイズパターン・GAN指紋なし" },
      { name: "Temporal Analysis", score: 9, status: "safe", detail: "自然な動きパターン確認" },
      { name: "Audio Analysis", score: 11, status: "safe", detail: "音声・口元同期正常" },
      { name: "Metadata Analysis", score: 7, status: "safe", detail: "メタデータ整合性確認" },
      { name: "War Footage", score: 3, status: "safe", detail: "紛争映像特徴なし" },
    ],
  },
};

function ScoreBar({ score, color }: { score: number; color: string }) {
  return (
    <div className="relative h-2 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.1)" }}>
      <div
        className="absolute left-0 top-0 h-full rounded-full transition-all duration-1000"
        style={{ width: `${score}%`, backgroundColor: color }}
      />
    </div>
  );
}

export default function Demo() {
  const [url, setUrl] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentLayer, setCurrentLayer] = useState("");

  const analyze = async () => {
    if (!url.trim()) return;

    setAnalyzing(true);
    setResult(null);
    setProgress(0);

    const layers = [
      "Frame Analysis を実行中...",
      "Temporal Analysis を実行中...",
      "Audio Analysis を実行中...",
      "Metadata Analysis を実行中...",
      "War Footage Analysis を実行中...",
      "Ensemble判定を計算中...",
    ];

    for (let i = 0; i < layers.length; i++) {
      setCurrentLayer(layers[i]);
      setProgress(Math.round(((i + 1) / layers.length) * 100));
      await new Promise((r) => setTimeout(r, 600));
    }

    // Mock result — real backend would call /api/analyze
    const mockResult = url.includes("real") ? MOCK_RESULTS.real : MOCK_RESULTS.default;
    setResult(mockResult);
    setAnalyzing(false);
    setCurrentLayer("");
    setProgress(0);
  };

  const getVerdictColor = (verdict: string) => {
    if (verdict === "REAL") return "#22c55e";
    if (verdict === "FAKE") return "#ef4444";
    return "#f59e0b";
  };

  const getStatusColor = (status: string) => {
    if (status === "safe") return "#22c55e";
    if (status === "warning") return "#f59e0b";
    return "#ef4444";
  };

  return (
    <section
      id="demo"
      className="py-24 relative"
      style={{ background: "linear-gradient(to bottom, #030712, #0a0f1e)" }}
    >
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-4"
            style={{
              border: "1px solid rgba(34, 197, 94, 0.3)",
              background: "rgba(34, 197, 94, 0.05)",
              color: "#22c55e",
            }}
          >
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            デモ体験
          </div>
          <h2 className="text-3xl sm:text-4xl font-black text-white mb-4">
            動画URLを入力して<span className="gradient-text">解析</span>
          </h2>
          <p className="text-gray-400">
            YouTube・Twitter・TikTokなどのURLに対応。
            <span className="text-yellow-400">※デモはモック結果を返します</span>
          </p>
        </div>

        {/* Input area */}
        <div
          className="p-6 rounded-2xl border mb-6"
          style={{
            borderColor: "rgba(34, 197, 94, 0.2)",
            background: "rgba(10, 15, 30, 0.8)",
          }}
        >
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="flex-1 relative">
              <div className="absolute left-4 top-1/2 -translate-y-1/2">
                <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                </svg>
              </div>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && analyze()}
                placeholder="https://youtube.com/watch?v=..."
                className="w-full pl-12 pr-4 py-3 rounded-xl text-white placeholder-gray-600 outline-none focus:ring-2 font-mono text-sm"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(34,197,94,0.2)",
                }}
              />
            </div>
            <button
              onClick={analyze}
              disabled={analyzing || !url.trim()}
              className="px-6 py-3 rounded-xl font-bold text-white transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105"
              style={{
                background: analyzing ? "#374151" : "linear-gradient(135deg, #22c55e, #16a34a)",
                minWidth: "120px",
              }}
            >
              {analyzing ? "解析中..." : "解析する"}
            </button>
          </div>

          {/* Sample URLs */}
          <div className="mt-3 flex flex-wrap gap-2">
            <span className="text-gray-600 text-xs">サンプル:</span>
            {SAMPLE_URLS.map((sample) => (
              <button
                key={sample}
                onClick={() => setUrl(sample)}
                className="text-xs text-blue-400 hover:text-blue-300 font-mono truncate max-w-[200px]"
              >
                {sample}
              </button>
            ))}
          </div>
        </div>

        {/* Progress */}
        {analyzing && (
          <div
            className="p-6 rounded-2xl border mb-6"
            style={{
              borderColor: "rgba(34, 197, 94, 0.2)",
              background: "rgba(10, 15, 30, 0.8)",
            }}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-green-400 text-sm font-mono">{currentLayer}</span>
              <span className="text-green-400 text-sm font-mono">{progress}%</span>
            </div>
            <div className="h-2 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.1)" }}>
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${progress}%`,
                  background: "linear-gradient(90deg, #22c55e, #3b82f6)",
                }}
              />
            </div>
            <p className="text-gray-600 text-xs mt-2 font-mono">
              {"> "}解析中... AIモデルが動画を多角的に検証しています
            </p>
          </div>
        )}

        {/* Result */}
        {result && (
          <div
            className="p-6 rounded-2xl border"
            style={{
              borderColor: `${getVerdictColor(result.verdict)}40`,
              background: "rgba(10, 15, 30, 0.9)",
            }}
          >
            {/* Verdict header */}
            <div className="flex items-center justify-between mb-6 pb-6 border-b border-white/10">
              <div>
                <div className="text-gray-400 text-sm mb-1">総合判定</div>
                <div
                  className="text-4xl font-black"
                  style={{ color: getVerdictColor(result.verdict) }}
                >
                  {result.verdict}
                </div>
                <div className="text-gray-400 text-sm mt-1">
                  信頼度 {result.confidence}% · 処理時間 {result.processing_time}s
                </div>
              </div>
              <div className="text-center">
                <div
                  className="text-6xl font-black"
                  style={{ color: getVerdictColor(result.verdict) }}
                >
                  {result.overall_score}
                </div>
                <div className="text-gray-500 text-xs">AI生成スコア</div>
              </div>
            </div>

            {/* Layer results */}
            <div className="space-y-4">
              <h4 className="text-white font-bold text-sm mb-3">レイヤー別スコア</h4>
              {result.layers.map((layer) => (
                <div key={layer.name}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-gray-300 text-sm">{layer.name}</span>
                    <span
                      className="text-sm font-bold"
                      style={{ color: getStatusColor(layer.status) }}
                    >
                      {layer.score}
                    </span>
                  </div>
                  <ScoreBar score={layer.score} color={getStatusColor(layer.status)} />
                  <p className="text-gray-600 text-xs mt-1 font-mono">{layer.detail}</p>
                </div>
              ))}
            </div>

            {/* CTA */}
            <div
              className="mt-6 pt-6 border-t border-white/10 text-center"
            >
              <p className="text-gray-400 text-sm mb-3">
                詳細なPDFレポートや API連携はProプランから
              </p>
              <a
                href="#pricing"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-white text-sm transition-all hover:scale-105"
                style={{ background: "linear-gradient(135deg, #22c55e, #16a34a)" }}
              >
                Proにアップグレード
              </a>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
