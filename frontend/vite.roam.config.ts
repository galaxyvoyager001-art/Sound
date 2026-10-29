import { resolve } from "node:path";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// “老家漫游”：打包成一个自包含的 HTML 模板，导出时由后端把场景数据和素材内嵌进去。
export default defineConfig({
  plugins: [viteSingleFile()],
  build: {
    outDir: "dist-roam",
    emptyOutDir: true,
    rollupOptions: { input: resolve(__dirname, "roam.html") },
  },
});
