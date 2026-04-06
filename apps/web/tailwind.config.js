/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"DM Sans"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      colors: {
        ink: {
          950: "#070708",
          900: "#0c0d10",
          850: "#111218",
          800: "#16181f",
          700: "#1e212b",
        },
        accent: {
          DEFAULT: "#38bdf8",
          dim: "#0ea5e9",
          glow: "rgba(56, 189, 248, 0.12)",
        },
      },
      boxShadow: {
        panel: "0 0 0 1px rgba(255,255,255,0.06), 0 24px 80px -32px rgba(0,0,0,0.85)",
        soft: "0 8px 32px -12px rgba(0,0,0,0.55)",
      },
      backgroundImage: {
        "grid-fade":
          "linear-gradient(to bottom, rgba(12,13,16,0.2), rgba(7,7,8,0.92)), radial-gradient(ellipse 80% 50% at 50% -20%, rgba(56,189,248,0.14), transparent)",
      },
    },
  },
  plugins: [],
};
