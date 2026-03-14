export default function Footer() {
  return (
    <footer
      className="border-t py-12"
      style={{
        borderColor: "rgba(34, 197, 94, 0.1)",
        background: "#030712",
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ background: "linear-gradient(135deg, #22c55e, #3b82f6)" }}
              >
                <span className="text-white font-bold text-sm">F</span>
              </div>
              <span className="text-white font-bold text-lg tracking-wide">
                Fake<span style={{ color: "#22c55e" }}>Guard</span>
              </span>
            </div>
            <p className="text-gray-500 text-sm leading-relaxed max-w-sm">
              5レイヤーAI検出エンジンで、ディープフェイク動画・AI生成コンテンツを
              高精度で解析します。情報の真偽を科学的に判定。
            </p>
            <div className="flex items-center gap-3 mt-4">
              <a
                href="#"
                className="w-9 h-9 rounded-lg flex items-center justify-center border transition-colors hover:border-green-500/50"
                style={{ borderColor: "rgba(255,255,255,0.1)", color: "#6b7280" }}
                aria-label="Twitter"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.744l7.73-8.835L1.254 2.25H8.08l4.253 5.622zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
              </a>
              <a
                href="#"
                className="w-9 h-9 rounded-lg flex items-center justify-center border transition-colors hover:border-green-500/50"
                style={{ borderColor: "rgba(255,255,255,0.1)", color: "#6b7280" }}
                aria-label="GitHub"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                </svg>
              </a>
            </div>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-white font-semibold text-sm mb-4">プロダクト</h4>
            <ul className="space-y-2">
              {["機能", "料金プラン", "API ドキュメント", "変更履歴"].map((item) => (
                <li key={item}>
                  <a href="#" className="text-gray-500 hover:text-gray-300 text-sm transition-colors">
                    {item}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="text-white font-semibold text-sm mb-4">会社</h4>
            <ul className="space-y-2">
              {["会社概要", "プライバシーポリシー", "利用規約", "お問い合わせ"].map((item) => (
                <li key={item}>
                  <a href="#" className="text-gray-500 hover:text-gray-300 text-sm transition-colors">
                    {item}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div
          className="pt-8 border-t flex flex-col sm:flex-row items-center justify-between gap-4"
          style={{ borderColor: "rgba(255,255,255,0.06)" }}
        >
          <p className="text-gray-600 text-sm">
            © 2026 TimeWealthy Limited. All rights reserved.
          </p>
          <div className="flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full animate-pulse"
              style={{ backgroundColor: "#22c55e" }}
            />
            <span className="text-gray-600 text-xs font-mono">
              System Status: Operational
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
