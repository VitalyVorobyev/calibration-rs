// ESLint 9 flat config (B-QUAL1-LINT-CI). Type-aware typescript-eslint is
// enabled via `projectService` — the workspace is small (~40 files) so the
// extra type-checking pass stays fast (well under a second locally).
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";

export default tseslint.config(
  {
    // dist/ is the Vite build output, src-tauri/ is a separate Rust crate
    // (its `target/` and Tauri's codegen `gen/` are not JS at all), and
    // tsconfig.tsbuildinfo is a tsc cache file.
    ignores: ["dist/**", "src-tauri/**", "node_modules/**", "tsconfig.tsbuildinfo"],
  },
  js.configs.recommended,
  tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
      globals: globals.browser,
    },
  },
  {
    // Vite/Vitest/ESLint's own config files sit outside `tsconfig.json`'s
    // `include: ["src"]`, so they get no type-aware linting — plain
    // syntactic rules only, with Node (not browser) globals.
    files: ["*.config.{js,ts}"],
    extends: [tseslint.configs.disableTypeChecked],
    languageOptions: {
      globals: globals.node,
    },
  },
  {
    files: ["**/*.{ts,tsx}"],
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs["recommended-latest"].rules,
      // Vite preset, downgraded to `warn`: several widget files
      // intentionally co-locate small pure helpers/types with the
      // component they serve (e.g. TargetBoard.tsx + computeBoardBbox,
      // FrameCanvas.tsx + colorForError). Splitting every helper into
      // its own module purely to satisfy Fast Refresh is not worth the
      // churn; `warn` still surfaces the HMR papercut without blocking CI.
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
    },
  },
);
