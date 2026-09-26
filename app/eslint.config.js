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
      // Vite preset, downgraded to `warn`: several widget files
      // intentionally co-locate small pure helpers/types with the
      // component they serve (e.g. TargetBoard.tsx + computeBoardBbox,
      // FrameCanvas.tsx + colorForError). Splitting every helper into
      // its own module purely to satisfy Fast Refresh is not worth the
      // churn; `warn` still surfaces the HMR papercut without blocking CI.
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],

      // New in eslint-plugin-react-hooks 7 (adopted with the eslint 10
      // bump). Advisory here rather than blocking — see B-QUAL-HOOKS7 in
      // docs/backlog.md.
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
      // Same family, same reason (lab-ui PLAN L2-3): refs and props read or
      // written during render. Each fix changes when a screen renders, which
      // a toolchain upgrade must not, so they wait for the per-screen work.
      "react-hooks/refs": "warn",
      "react-hooks/immutability": "warn",
    },
  },
];
