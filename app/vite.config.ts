import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The browser viewer's server (`calib-viewer --port 8765 --no-open <run>`)
// for `VITE_CALIB_BACKEND=http bun run dev`: Vite proxies its API so the
// page is served with hot reload against a real run.
const viewer = process.env.CALIB_VIEWER_URL ?? "http://127.0.0.1:8765";

// Vite serves the React UI; Tauri's Rust backend runs alongside on
// localhost:1420. The clearScreen / strictPort settings match the
// official Tauri 2 React template so `tauri dev` and `vite` agree on
// the dev URL.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    proxy: {
      "/api": {
        target: viewer,
        // The viewer answers only its own Host and Origin.
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("proxyReq", (request) => request.setHeader("Origin", viewer));
        },
      },
    },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: "es2021",
    sourcemap: true,
  },
});
