/* Vite 构建配置。负责本地开发代理、构建输出与前端打包行为。 */

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = "http://127.0.0.1:8000";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === "production" ? "/ui/" : "/",
  server: {
    port: 5173,
    proxy: {
      "/healthz": backend,
      "/metrics": backend,
      "/chat": backend,
      "/ingest": backend,
      "/reindex": backend,
      "/eval": backend,
      "/jobs": backend,
    },
  },
}));
