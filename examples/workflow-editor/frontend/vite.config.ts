import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Single-port production build is served by FastAPI at the same origin.
// During dev (vite on 5173) proxy /api to the backend on 7873.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:7873",
        changeOrigin: false,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
  },
});
