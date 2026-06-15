import { defineConfig } from "vitest/config";

// Pure-logic unit tests run in a Node environment — no jsdom, no Tauri,
// no Vite plugins (React/Tailwind). The functions under test
// (inferExportKind, exportKindLabel, mergeConfig) are dependency-free,
// so this keeps `bun run test` fast and isolated from the app shell.
//
// This is the B-INFRA "Vitest unit" slice. The heavier B-INFRA items
// (ts-rs codegen + export discriminator tag, resource_dir presets,
// Playwright smoke tests) are deferred — see docs/backlog.md.
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
