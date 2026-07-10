import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Pure-logic `*.test.ts` files run in a Node environment — no jsdom, no
// Tauri, no Vite plugins. The functions under test (inferExportKind,
// exportKindLabel, mergeConfig) are dependency-free, so this keeps
// `bun run test` fast and isolated from the app shell.
//
// `*.test.tsx` component tests (B-QUAL3) render through
// `@testing-library/react` and need a DOM — `environmentMatchGlobs`
// switches those files to `jsdom` while every `.test.ts` file keeps the
// original fast `node` environment. The React plugin is only needed for
// the jsdom project (JSX/Fast-Refresh transform); it's a no-op for the
// plain-TS unit tests.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "node",
    environmentMatchGlobs: [["src/**/*.test.tsx", "jsdom"]],
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["src/test/setupTests.ts"],
  },
});
