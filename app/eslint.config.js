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
      // Node ESM build script outside tsconfig's `include: ["src"]`, so the
      // type-aware project service can't resolve it (B-QUAL2 codegen).
      "scripts/**",
    ],
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
    // Playwright e2e specs (B-QUAL4) live outside `tsconfig.json`'s
    // `include: ["src"]` too — same syntactic-only treatment as the
    // config files above. Node globals for the test/config code; `page`
    // callbacks execute in the browser but are still authored/typed as
    // plain TS closures, so no browser globals are needed here.
    files: ["e2e/**/*.{ts,tsx}", "playwright.config.ts"],
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

      // New in eslint-plugin-react-hooks 7 (adopted with the eslint 10
      // bump). Both are advisory here rather than blocking, for different
      // reasons — see B-QUAL-HOOKS7 in docs/backlog.md.
      //
      // `purity` cannot tell an event handler from render code, so it
      // reports the `Date.now()` that stamps a run's start time inside
      // RunWorkspace's async submit handler. That call is legitimate; the
      // rule is wrong about it, and there is no narrower suppression that
      // does not also blind the file to real purity violations.
      "react-hooks/purity": "warn",
      // `set-state-in-effect` flags the "reset derived state when the
      // input changes" effect in useImageData / DiagnoseWorkspace. The
      // pattern is correct, just one render pass more expensive than
      // keying the component. Rewriting five call sites is app work, not
      // part of a dependency bump.
      "react-hooks/set-state-in-effect": "warn",
    },
  },
);
