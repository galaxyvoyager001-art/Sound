import { resolve } from "node:path";
import { defineConfig } from "vite";

// 多页面：会谈页（index）、设置页、调参试听页。“老家漫游”导出页单独用 vite.roam.config.ts 打包成单文件。
export default defineConfig({
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      input: {
        index: resolve(__dirname, "index.html"),
        setup: resolve(__dirname, "setup.html"),
        tuner: resolve(__dirname, "tuner.html"),
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/files": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
