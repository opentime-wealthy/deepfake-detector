"use client";

import { useState } from "react";
import { PLANS, PlanKey } from "@/lib/stripe";

export default function Pricing() {
  const [loading, setLoading] = useState<PlanKey | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCheckout = async (planKey: PlanKey) => {
    const plan = PLANS[planKey];

    if (planKey === "free") {
      // Free plan — just scroll to demo
      window.location.href = "#demo";
      return;
    }

    if (!plan.priceId || plan.priceId.includes("placeholder")) {
      setError(
        "Stripeが未設定です。.env.localにSTRIPE_SECRET_KEYを設定してください。"
      );
      setTimeout(() => setError(null), 5000);
      return;
    }

    setLoading(planKey);
    setError(null);

    try {
      const res = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ priceId: plan.priceId, plan: planKey }),
      });

      const data = await res.json();

      if (!res.ok || !data.url) {
        throw new Error(data.error || "チェックアウトURLの取得に失敗しました");
      }

      window.location.href = data.url;
    } catch (err) {
      const msg = err instanceof Error ? err.message : "エラーが発生しました";
      setError(msg);
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(null);
    }
  };

  const planKeys = Object.keys(PLANS) as PlanKey[];

  return (
    <section
      id="pricing"
      className="py-24 relative overflow-hidden"
      style={{ background: "linear-gradient(to bottom, #0a0f1e, #030712)" }}
    >
      {/* Background glow */}
      <div
        className="absolute inset-0 opacity-40"
        style={{
          background:
            "radial-gradient(ellipse at 50% 0%, rgba(34,197,94,0.1) 0%, transparent 60%)",
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <div
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-4"
            style={{
              border: "1px solid rgba(34, 197, 94, 0.3)",
              background: "rgba(34, 197, 94, 0.05)",
              color: "#22c55e",
            }}
          >
            料金プラン
          </div>
          <h2 className="text-3xl sm:text-5xl font-black text-white mb-4">
            シンプルな<span className="gradient-text">料金設定</span>
          </h2>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            無料プランで機能を体験し、必要に応じてアップグレード。
            すべてのプランで5レイヤー検出エンジンを使用します。
          </p>
        </div>

        {/* Error message */}
        {error && (
          <div
            className="max-w-2xl mx-auto mb-8 p-4 rounded-xl border text-red-400 text-sm text-center"
            style={{ borderColor: "rgba(239, 68, 68, 0.3)", background: "rgba(239, 68, 68, 0.05)" }}
          >
            ⚠️ {error}
          </div>
        )}

        {/* Plans grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {planKeys.map((planKey) => {
            const plan = PLANS[planKey];
            const isHighlighted = plan.highlighted;

            return (
              <div
                key={planKey}
                className={`relative flex flex-col p-8 rounded-2xl border transition-all duration-300 ${
                  isHighlighted ? "scale-105 shadow-2xl" : "hover:scale-[1.02]"
                }`}
                style={{
                  borderColor: isHighlighted
                    ? "rgba(34, 197, 94, 0.5)"
                    : "rgba(255,255,255,0.1)",
                  background: isHighlighted
                    ? "linear-gradient(135deg, rgba(34,197,94,0.08), rgba(59,130,246,0.05))"
                    : "rgba(10, 15, 30, 0.8)",
                  boxShadow: isHighlighted
                    ? "0 0 40px rgba(34, 197, 94, 0.15)"
                    : "none",
                }}
              >
                {/* Popular badge */}
                {isHighlighted && (
                  <div
                    className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full text-xs font-bold text-white"
                    style={{ background: "linear-gradient(135deg, #22c55e, #3b82f6)" }}
                  >
                    🔥 最人気
                  </div>
                )}

                {/* Plan name */}
                <div className="mb-6">
                  <h3 className="text-white font-bold text-xl mb-1">{plan.name}</h3>
                  <div className="flex items-baseline gap-1">
                    {plan.priceJPY === 0 ? (
                      <span className="text-4xl font-black text-white">無料</span>
                    ) : (
                      <>
                        <span className="text-gray-400 text-lg">¥</span>
                        <span className="text-4xl font-black text-white">
                          {plan.priceJPY.toLocaleString()}
                        </span>
                        <span className="text-gray-400">/{plan.period}</span>
                      </>
                    )}
                  </div>
                  <p className="text-gray-500 text-sm mt-1">
                    {plan.analyses === -1 ? "無制限の解析" : `${plan.analyses}回/月の解析`}
                  </p>
                </div>

                {/* Features */}
                <ul className="space-y-3 flex-1 mb-8">
                  {plan.features.map((feature) => (
                    <li key={feature} className="flex items-start gap-3">
                      <svg
                        className="w-5 h-5 flex-shrink-0 mt-0.5"
                        style={{ color: isHighlighted ? "#22c55e" : "#6b7280" }}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                      <span className="text-gray-300 text-sm">{feature}</span>
                    </li>
                  ))}
                </ul>

                {/* CTA button */}
                <button
                  onClick={() => handleCheckout(planKey)}
                  disabled={loading === planKey}
                  className="w-full py-3 px-6 rounded-xl font-bold text-sm transition-all duration-200 hover:scale-105 disabled:opacity-60 disabled:cursor-not-allowed"
                  style={
                    isHighlighted
                      ? {
                          background: "linear-gradient(135deg, #22c55e, #16a34a)",
                          color: "white",
                        }
                      : {
                          border: "1px solid rgba(255,255,255,0.2)",
                          color: "white",
                          background: "rgba(255,255,255,0.05)",
                        }
                  }
                >
                  {loading === planKey ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      処理中...
                    </span>
                  ) : (
                    plan.cta
                  )}
                </button>
              </div>
            );
          })}
        </div>

        {/* Footer note */}
        <p className="text-center text-gray-600 text-sm mt-10">
          すべてのプランはクレジットカード払い。いつでもキャンセル可能。
          <br />
          法人向けの特別プランはお問い合わせください。
        </p>
      </div>
    </section>
  );
}
