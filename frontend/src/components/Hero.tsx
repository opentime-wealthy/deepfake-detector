"use client";

import { useEffect, useState } from "react";

const STATS = [
  { value: "97.3%", label: "検出精度" },
  { value: "< 30秒", label: "解析時間" },
  { value: "5層", label: "AI検出レイヤー" },
  { value: "10万+", label: "解析済み動画" },
];

export default function Hero() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <section
      className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16"
      style={{ background: "linear-gradient(to bottom, #030712, #0a0f1e)" }}
    >
      {/* Grid background */}
      <div className="absolute inset-0 grid-bg opacity-60" />

      {/* Radial glow */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 50%, rgba(34,197,94,0.08) 0%, transparent 70%)",
        }}
      />

      {/* Animated scan line */}
      <div
        className="absolute left-0 right-0 h-px opacity-30 pointer-events-none"
        style={{
          background: "linear-gradient(90deg, transparent, #22c55e, transparent)",
          animation: "scanLine 4s linear infinite",
          top: 0,
        }}
      />

      {/* Corner decorations */}
      <div className="absolute top-20 left-8 w-20 h-20 border-l-2 border-t-2 border-green-500/30" />
      <div className="absolute top-20 right-8 w-20 h-20 border-r-2 border-t-2 border-green-500/30" />
      <div className="absolute bottom-20 left-8 w-20 h-20 border-l-2 border-b-2 border-blue-500/30" />
      <div className="absolute bottom-20 right-8 w-20 h-20 border-r-2 border-b-2 border-blue-500/30" />

      <div className="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        {/* Badge */}
        <div
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-full border mb-8 transition-all duration-700 ${
            mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
          }`}
          style={{
            borderColor: "rgba(34, 197, 94, 0.3)",
            background: "rgba(34, 197, 94, 0.05)",
          }}
        >
          <span
            className="w-2 h-2 rounded-full animate-pulse"
            style={{ backgroundColor: "#22c55e" }}
          />
          <span className="text-green-400 text-sm font-medium">
            5レイヤーAI検出エンジン 稼働中
          </span>
        </div>

        {/* Headline */}
        <h1
          className={`text-4xl sm:text-6xl lg:text-7xl font-black tracking-tight mb-6 transition-all duration-700 delay-100 ${
            mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
          }`}
        >
          <span className="text-white block mb-2">この動画は</span>
          <span className="gradient-text block mb-2">本物？</span>
          <span className="text-white block">
            AIが作った？
          </span>
        </h1>

        {/* Subtitle */}
        <p
          className={`text-gray-400 text-lg sm:text-xl max-w-3xl mx-auto mb-10 leading-relaxed transition-all duration-700 delay-200 ${
            mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
          }`}
        >
          FakeGuardは、Frame・Temporal・Audio・Metadata・War Footage の
          <strong className="text-white"> 5つのAIレイヤー</strong>で動画を多角的に解析。
          ディープフェイクやAI生成コンテンツを
          <strong className="text-green-400"> 97.3%の精度</strong>で検出します。
        </p>

        {/* CTA buttons */}
        <div
          className={`flex flex-col sm:flex-row items-center justify-center gap-4 mb-16 transition-all duration-700 delay-300 ${
            mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
          }`}
        >
          <a
            href="#demo"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-xl font-bold text-base text-white glow-green transition-all duration-300 hover:scale-105"
            style={{ background: "linear-gradient(135deg, #22c55e, #16a34a)" }}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            今すぐ動画を解析する
          </a>
          <a
            href="#features"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-xl font-bold text-base border transition-all duration-300 hover:scale-105"
            style={{
              borderColor: "rgba(59, 130, 246, 0.4)",
              color: "#60a5fa",
              background: "rgba(59, 130, 246, 0.05)",
            }}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            仕組みを見る
          </a>
        </div>

        {/* Stats */}
        <div
          className={`grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto transition-all duration-700 delay-500 ${
            mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
          }`}
        >
          {STATS.map((stat) => (
            <div
              key={stat.label}
              className="flex flex-col items-center p-4 rounded-xl border"
              style={{
                borderColor: "rgba(34, 197, 94, 0.15)",
                background: "rgba(34, 197, 94, 0.03)",
              }}
            >
              <span
                className="text-2xl sm:text-3xl font-black mb-1"
                style={{ color: "#22c55e" }}
              >
                {stat.value}
              </span>
              <span className="text-gray-500 text-sm">{stat.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div
        className="absolute bottom-0 left-0 right-0 h-32"
        style={{
          background: "linear-gradient(to top, #030712, transparent)",
        }}
      />
    </section>
  );
}
