const LAYERS = [
  {
    id: "01",
    name: "Frame Analysis",
    nameJa: "フレーム解析",
    color: "#22c55e",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
    description:
      "フレーム単位でGAN指紋・圧縮アーティファクト・メッシュ歪みを検出。1フレームの微細な不整合も見逃しません。",
    details: ["GAN指紋検出", "顔メッシュ整合性", "圧縮アーティファクト分析", "照明方向解析"],
  },
  {
    id: "02",
    name: "Temporal Analysis",
    nameJa: "時間軸解析",
    color: "#3b82f6",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    description:
      "フレーム間の時系列パターンを分析。自然な動きと人工的な補間の違いをLSTMモデルで識別します。",
    details: ["フレーム間一貫性", "自然な動き検証", "補間パターン検出", "時系列異常検知"],
  },
  {
    id: "03",
    name: "Audio Analysis",
    nameJa: "音声解析",
    color: "#06b6d4",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
      </svg>
    ),
    description:
      "音声と映像の口元同期精度・背景ノイズパターンを解析。AI音声クローンも高精度で検出します。",
    details: ["リップシンク整合性", "音声クローン検出", "背景ノイズ分析", "音声指紋照合"],
  },
  {
    id: "04",
    name: "Metadata Analysis",
    nameJa: "メタデータ解析",
    color: "#8b5cf6",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    description:
      "ファイルメタデータ・エンコード履歴・タイムスタンプの矛盾を検出。改ざん証跡を数値で証明します。",
    details: ["EXIF/XMP解析", "エンコード履歴", "タイムスタンプ検証", "改ざん証跡検出"],
  },
  {
    id: "05",
    name: "War Footage Analysis",
    nameJa: "紛争映像解析",
    color: "#f59e0b",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
    description:
      "紛争・戦場映像に特化した解析エンジン。地政学的整合性・武器・軍備の外観検証も実施します。",
    details: ["地政学的整合性検証", "武器・装備照合", "地形認識", "情報操作リスク評価"],
  },
];

export default function Features() {
  return (
    <section
      id="features"
      className="py-24 relative overflow-hidden"
      style={{ background: "linear-gradient(to bottom, #030712, #0a0f1e, #030712)" }}
    >
      {/* Background decoration */}
      <div
        className="absolute inset-0 opacity-30"
        style={{
          background:
            "radial-gradient(ellipse at 30% 50%, rgba(59,130,246,0.08) 0%, transparent 60%), radial-gradient(ellipse at 70% 50%, rgba(34,197,94,0.05) 0%, transparent 60%)",
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section header */}
        <div className="text-center mb-16">
          <div
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-4"
            style={{
              border: "1px solid rgba(59, 130, 246, 0.3)",
              background: "rgba(59, 130, 246, 0.05)",
              color: "#60a5fa",
            }}
          >
            5レイヤー検出エンジン
          </div>
          <h2 className="text-3xl sm:text-5xl font-black text-white mb-4">
            なぜFakeGuardは<span className="gradient-text">正確</span>なのか
          </h2>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            単一の手法ではなく、5つの独立したAIモデルが協調して判定。
            どれか一つでも異常を検出すれば、総合スコアに反映されます。
          </p>
        </div>

        {/* Layers grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {LAYERS.map((layer, index) => (
            <div
              key={layer.id}
              className="group relative p-6 rounded-2xl border transition-all duration-300 hover:scale-[1.02]"
              style={{
                borderColor: `${layer.color}20`,
                background: `linear-gradient(135deg, ${layer.color}05 0%, transparent 100%)`,
              }}
            >
              {/* Layer number */}
              <div
                className="absolute top-4 right-4 text-xs font-mono font-bold opacity-30"
                style={{ color: layer.color }}
              >
                LAYER {layer.id}
              </div>

              {/* Icon */}
              <div
                className="w-14 h-14 rounded-xl flex items-center justify-center mb-4"
                style={{
                  background: `${layer.color}15`,
                  color: layer.color,
                  border: `1px solid ${layer.color}30`,
                }}
              >
                {layer.icon}
              </div>

              {/* Name */}
              <div className="mb-2">
                <h3 className="text-white font-bold text-lg">{layer.name}</h3>
                <span
                  className="text-xs font-medium"
                  style={{ color: layer.color }}
                >
                  {layer.nameJa}
                </span>
              </div>

              {/* Description */}
              <p className="text-gray-400 text-sm leading-relaxed mb-4">
                {layer.description}
              </p>

              {/* Details list */}
              <ul className="space-y-1">
                {layer.details.map((detail) => (
                  <li
                    key={detail}
                    className="flex items-center gap-2 text-xs text-gray-500"
                  >
                    <span
                      className="w-1 h-1 rounded-full flex-shrink-0"
                      style={{ backgroundColor: layer.color }}
                    />
                    {detail}
                  </li>
                ))}
              </ul>

              {/* Hover glow */}
              <div
                className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                style={{
                  boxShadow: `inset 0 0 30px ${layer.color}08`,
                  border: `1px solid ${layer.color}40`,
                }}
              />
            </div>
          ))}

          {/* Ensemble card */}
          <div
            className="md:col-span-2 lg:col-span-full p-6 rounded-2xl border"
            style={{
              borderColor: "rgba(34,197,94,0.2)",
              background: "linear-gradient(135deg, rgba(34,197,94,0.05), rgba(59,130,246,0.05))",
            }}
          >
            <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
              <div
                className="w-14 h-14 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{
                  background: "linear-gradient(135deg, rgba(34,197,94,0.2), rgba(59,130,246,0.2))",
                  border: "1px solid rgba(34,197,94,0.3)",
                }}
              >
                <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
                </svg>
              </div>
              <div>
                <h3 className="text-white font-bold text-lg mb-1">
                  Ensemble Engine — 総合判定
                </h3>
                <p className="text-gray-400 text-sm leading-relaxed">
                  5つのレイヤーの結果を重み付きアンサンブルで統合し、最終的な
                  <span className="text-green-400 font-medium">信頼度スコア（0〜100）</span>
                  を算出。スコアが高いほどAI生成の可能性が高く、閾値に基づいて自動判定します。
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
