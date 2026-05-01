import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite serves the React UI; Tauri's Rust backend runs alongside on
// localhost:1420. The clearScreen / strictPort settings match the
// official Tauri 2 React template so `tauri dev` and `vite` agree on
// the dev URL.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: "es2021",
    sourcemap: true,
  },
});
