// The shared vitavision flat config (@vitavision/config-eslint): type-aware
// typescript-eslint, @eslint-react, the hooks rules. This file adds only what
// is this app's own: its ignores, the untyped config/e2e files, Fast Refresh,
// and the advisory hooks rules explained below.
import js from "@eslint/js";
import { recommended } from "@vitavision/config-eslint";
import tseslint from "typescript-eslint";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";

export default [
  {
    // dist/ is the Vite build output, src-tauri/ is a separate Rust crate
    // (its `target/` and Tauri's codegen `gen/` are not JS at all),
    // tsconfig.tsbuildinfo is a tsc cache file, and src/types/generated is
    // machine-generated from the Rust wire types (B-QUAL2) — regenerate via
    // `bun run generate:types`, don't lint or hand-edit it.
    // Keep in sync with app/.gitignore / app/.prettierignore.
    ignores: [
      "dist/**",
      "src-tauri/**",
      "node_modules/**",
      "tsconfig.tsbuildinfo",
      "src/types/generated/**",
      "e2e/.screens/**",
      "test-results/**",
      // Node ESM build script outside tsconfig's `include: ["src"]`, so the
      // type-aware project service can't resolve it (B-QUAL2 codegen).
      "scripts/**",
    ],
  },
  js.configs.recommended,
  ...recommended({ tsconfigRootDir: import.meta.dirname }),
  {
    languageOptions: { globals: globals.browser },
  },
  {
    // Vite/Vitest/ESLint/Playwright config files and the Playwright e2e specs
    // (B-QUAL4) sit outside `tsconfig.json`'s `include: ["src"]`, so they get
    // no type-aware linting — plain syntactic rules only, with Node globals.
    files: ["*.config.{js,ts}", "e2e/**/*.{ts,tsx}"],
    ...tseslint.configs.disableTypeChecked,
    languageOptions: {
      ...tseslint.configs.disableTypeChecked.languageOptions,
      globals: globals.node,
    },
  },
  {
    files: ["**/*.{ts,tsx}"],
    plugins: { "react-refresh": reactRefresh },
    rules: {
      // Vite preset: component files export only components (helpers live in
      // sibling modules) so Fast Refresh keeps component state on edit.
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],

      // Advisory rather than blocking: refs and props read or written
      // during render (lab-ui PLAN L2-3). Each fix changes when a screen
      // renders, so they wait for the per-screen work.
      "react-hooks/refs": "warn",
      "react-hooks/immutability": "warn",
    },
  },
];
