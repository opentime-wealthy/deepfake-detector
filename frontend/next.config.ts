import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Serverless deployment (Vercel)
  // Remove 'output: export' to enable API routes for Stripe
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
