import { loadStripe, Stripe } from "@stripe/stripe-js";

// Stripe publishable key from environment variable
const stripePublishableKey = process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY;

let stripePromise: Promise<Stripe | null>;

/**
 * Get a singleton Stripe.js instance (client-side only)
 */
export const getStripe = (): Promise<Stripe | null> => {
  if (!stripePromise) {
    if (!stripePublishableKey) {
      console.warn(
        "NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY is not set. " +
          "Set it in .env.local to enable Stripe payments."
      );
      return Promise.resolve(null);
    }
    stripePromise = loadStripe(stripePublishableKey);
  }
  return stripePromise;
};

/**
 * Pricing plan definitions
 */
export const PLANS = {
  free: {
    name: "Free",
    priceJPY: 0,
    analyses: 5,
    period: "月",
    features: [
      "5回/月の解析",
      "基本的なディープフェイク検出",
      "Frame解析",
      "結果レポート（基本）",
    ],
    cta: "無料で始める",
    priceId: null,
    highlighted: false,
  },
  pro: {
    name: "Pro",
    priceJPY: 2980,
    analyses: 100,
    period: "月",
    features: [
      "100回/月の解析",
      "5レイヤー完全検出",
      "優先処理キュー",
      "詳細レポートPDF",
      "APIアクセス",
      "メールサポート",
    ],
    cta: "Proを始める",
    priceId: process.env.NEXT_PUBLIC_STRIPE_PRICE_PRO || "price_pro_placeholder",
    highlighted: true,
  },
  enterprise: {
    name: "Enterprise",
    priceJPY: 29800,
    analyses: -1, // unlimited
    period: "月",
    features: [
      "無制限の解析",
      "5レイヤー完全検出",
      "専用処理インフラ",
      "カスタムレポート",
      "専用APIエンドポイント",
      "SLA 99.9%保証",
      "専任サポート",
      "カスタム統合対応",
    ],
    cta: "Enterpriseを始める",
    priceId: process.env.NEXT_PUBLIC_STRIPE_PRICE_ENTERPRISE || "price_enterprise_placeholder",
    highlighted: false,
  },
} as const;

export type PlanKey = keyof typeof PLANS;
