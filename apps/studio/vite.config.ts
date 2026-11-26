import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// `bun run dev` in a browser talks to a running calibration-studio-server
// (default address 127.0.0.1:8757; `CALIBRATION_STUDIO_URL` overrides it):
// Vite proxies `/api` to it, so the page gets hot reload against a real
// back end. Without one, the page says no back end answers.
const server = process.env.CALIBRATION_STUDIO_URL ?? "http://127.0.0.1:8757";

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
        target: server,
        // The server answers only its own Host and Origin.
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("proxyReq", (request) => request.setHeader("Origin", server));
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
