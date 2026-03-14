"use client";

import { useState, useEffect } from "react";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-dark-900/95 backdrop-blur-md border-b border-green-500/20"
          : "bg-transparent"
      }`}
      style={{ backgroundColor: scrolled ? "rgba(3, 7, 18, 0.95)" : "transparent" }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: "linear-gradient(135deg, #22c55e, #3b82f6)" }}>
              <span className="text-white font-bold text-sm">F</span>
            </div>
            <span className="text-white font-bold text-lg tracking-wide">
              Fake<span style={{ color: "#22c55e" }}>Guard</span>
            </span>
          </div>

          {/* Nav links */}
          <div className="hidden md:flex items-center gap-8">
            {[
              { label: "機能", href: "#features" },
              { label: "デモ", href: "#demo" },
              { label: "料金", href: "#pricing" },
              { label: "FAQ", href: "#faq" },
            ].map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="text-gray-400 hover:text-green-400 transition-colors text-sm font-medium"
              >
                {link.label}
              </a>
            ))}
          </div>

          {/* CTA */}
          <div className="flex items-center gap-3">
            <a
              href="#pricing"
              className="hidden sm:inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-all"
              style={{
                background: "linear-gradient(135deg, #22c55e, #16a34a)",
                color: "white",
              }}
            >
              無料で始める
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
}
