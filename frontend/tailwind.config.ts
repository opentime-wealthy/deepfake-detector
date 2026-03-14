import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          green: {
            400: "#4ade80",
            500: "#22c55e",
            600: "#16a34a",
          },
          blue: {
            400: "#60a5fa",
            500: "#3b82f6",
            600: "#2563eb",
          },
          cyan: {
            400: "#22d3ee",
            500: "#06b6d4",
          },
        },
        dark: {
          900: "#030712",
          800: "#0a0f1e",
          700: "#0f1629",
          600: "#111827",
          500: "#1f2937",
          400: "#374151",
        },
      },
      backgroundImage: {
        "grid-pattern":
          "linear-gradient(rgba(34, 197, 94, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(34, 197, 94, 0.05) 1px, transparent 1px)",
      },
      backgroundSize: {
        "grid-size": "60px 60px",
      },
      animation: {
        "pulse-slow": "pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "scan-line": "scanLine 3s linear infinite",
        float: "float 6s ease-in-out infinite",
      },
      keyframes: {
        scanLine: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
