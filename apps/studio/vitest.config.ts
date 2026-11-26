import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Pure-logic `*.test.ts` files run in a Node environment — no jsdom, no
// Tauri, no Vite plugins. The functions under test (inferExportKind,
// exportKindLabel, mergeConfig) are dependency-free, so this keeps
// `bun run test` fast and isolated from the app shell.
//
// `*.test.tsx` component tests render through
// `@testing-library/react` and need a DOM — each one starts with a
// `// @vitest-environment jsdom` pragma, while every `.test.ts` file keeps
// the fast `node` environment. The React plugin is only needed for the
// jsdom tests (JSX/Fast-Refresh transform); it's a no-op for the plain-TS
// unit tests.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "node",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["src/test/setupTests.ts"],
  },
});
