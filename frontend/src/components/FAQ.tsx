"use client";

import { useState } from "react";

const FAQS = [
  {
    q: "どんな形式の動画に対応していますか？",
    a: "YouTube、Twitter/X、TikTok、InstagramなどのSNSのURL、および直接的な動画ファイルのURL（MP4、MOV、WebM等）に対応しています。ファイルの最大サイズは500MBまでです。",
  },
  {
    q: "検出精度はどのくらいですか？",
    a: "現行の学習データセットにおいて97.3%の検出精度を達成しています。ただし、最新のAI生成モデルは日々進化しているため、FakeGuardも継続的にモデルの更新を行っています。100%の保証はできませんが、業界トップレベルの精度を維持しています。",
  },
  {
    q: "解析した動画はどのように扱われますか？",
    a: "アップロードされた動画コンテンツは解析後24時間以内に自動削除されます。解析結果のメタデータのみを保存し、動画の内容を第三者と共有することは一切ありません。プライバシーポリシーに基づいて適切に管理されます。",
  },
  {
    q: "APIアクセスはできますか？",
    a: "Proプラン以上でREST APIへのアクセスが可能です。APIキーの発行・管理はダッシュボードから行えます。Enterpriseプランでは専用エンドポイントとより高いレート制限が利用できます。",
  },
  {
    q: "解析にどのくらい時間がかかりますか？",
    a: "動画の長さによって異なりますが、一般的な動画（1〜5分）で15〜30秒程度です。Proプランでは優先処理キューに入るため、通常より高速に処理されます。",
  },
  {
    q: "無料プランからProプランへの移行方法は？",
    a: "料金プランセクションの「Proを始める」ボタンをクリックし、Stripeの安全な決済フローでクレジットカード情報を入力するだけです。移行は即時に完了し、既存の解析履歴も引き継がれます。",
  },
  {
    q: "法人・企業での利用を検討しています",
    a: "Enterpriseプランでは、専任サポート・SLA保証・カスタムAPI統合・バッチ処理対応など、企業ニーズに対応した機能を提供しています。また、特定の業界要件や大量処理の場合はカスタムプランのご相談も承っております。",
  },
  {
    q: "キャンセルはいつでも可能ですか？",
    a: "はい、いつでもキャンセル可能です。キャンセル後も現在の請求期間の終了まではProまたはEnterpriseの機能をご利用いただけます。返金については利用規約をご確認ください。",
  },
];

export default function FAQ() {
  const [open, setOpen] = useState<number | null>(null);

  return (
    <section
      id="faq"
      className="py-24 relative"
      style={{ background: "linear-gradient(to bottom, #030712, #0a0f1e)" }}
    >
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <div
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-4"
            style={{
              border: "1px solid rgba(59, 130, 246, 0.3)",
              background: "rgba(59, 130, 246, 0.05)",
              color: "#60a5fa",
            }}
          >
            よくある質問
          </div>
          <h2 className="text-3xl sm:text-4xl font-black text-white mb-4">
            FAQ
          </h2>
          <p className="text-gray-400">
            ご不明な点はお気軽にお問い合わせください
          </p>
        </div>

        {/* FAQ items */}
        <div className="space-y-3">
          {FAQS.map((faq, i) => (
            <div
              key={i}
              className="rounded-xl border overflow-hidden transition-all duration-200"
              style={{
                borderColor: open === i ? "rgba(34,197,94,0.3)" : "rgba(255,255,255,0.08)",
                background: open === i ? "rgba(34,197,94,0.04)" : "rgba(10, 15, 30, 0.6)",
              }}
            >
              <button
                onClick={() => setOpen(open === i ? null : i)}
                className="w-full flex items-center justify-between p-5 text-left"
              >
                <span className="text-white font-medium text-sm sm:text-base pr-4">
                  {faq.q}
                </span>
                <div
                  className="flex-shrink-0 w-6 h-6 flex items-center justify-center rounded-full transition-all duration-200"
                  style={{
                    background: open === i ? "rgba(34,197,94,0.2)" : "rgba(255,255,255,0.05)",
                    color: open === i ? "#22c55e" : "#6b7280",
                    transform: open === i ? "rotate(180deg)" : "rotate(0deg)",
                  }}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>

              {open === i && (
                <div className="px-5 pb-5">
                  <p className="text-gray-400 text-sm leading-relaxed">{faq.a}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
