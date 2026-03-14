import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FakeGuard — AIが作った動画を見破る",
  description:
    "5レイヤーAI検出エンジンで、ディープフェイク動画・AI生成コンテンツを瞬時に解析。Frame・Temporal・Audio・Metadata・War Footage解析で真偽を判定。",
  keywords: ["ディープフェイク", "AI検出", "フェイク動画", "deepfake detector", "AI生成動画"],
  openGraph: {
    title: "FakeGuard — AIが作った動画を見破る",
    description: "5レイヤーAI検出エンジンでディープフェイクを瞬時に解析",
    type: "website",
    locale: "ja_JP",
  },
  twitter: {
    card: "summary_large_image",
    title: "FakeGuard — AIが作った動画を見破る",
    description: "5レイヤーAI検出エンジンでディープフェイクを瞬時に解析",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
