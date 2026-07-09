# B-INFRA (slice): Vitest unit tests for pure app logic

**Date:** 2026-06-15
**Task:** `B-INFRA` (docs/backlog.md → Track B) — **partial.** This is the
"Vitest unit" deliverable only. The rest of B-INFRA (ts-rs/specta codegen +
export discriminator tag, `resource_dir` presets, Playwright smoke tests) is
deferred; B-INFRA stays open.

## Scope (what changed)

Stand up a Vitest harness in `app/` and cover the two dependency-free,
correctness-critical helpers that currently have no tests:

- `inferExportKind` (`src/store/exportShape.ts`) — the shape-probe that maps a
  loaded calibration export's JSON to one of nine `ExportKind`s. Its **probe
  order is load-bearing**: rig exports also carry legacy top-level
  `camera`-prefixed fields, so rig shapes must be matched before single-cam
  shapes. The tests pin every branch *and* the order-precedence cases, so the
  eventual ts-rs discriminator swap (deferred) has a behavioral spec to match.
- `exportKindLabel` (same file) — exhaustive label coverage (non-empty + unique
  for all nine kinds).
- `mergeConfig` (`src/workspaces/RunWorkspace/presets.ts`) — the recursive
  deep-merge used to apply preset config/manifest overrides. Tests cover
  deep-merge, array-replace (not merge), type-disagreement replacement, the
  `null` explicit-override case (mirrors `upstream_calibration: null` in the
  rtv3d joint-laser preset), and base-object immutability.

Tests run in a **Node** environment (no jsdom / Tauri / Vite plugins) because
the functions are pure — `bun run test` is fast and isolated from the app shell.

## Files changed

- `app/vitest.config.ts` — new; node env, `include: ["src/**/*.test.ts"]`.
- `app/package.json` — `vitest` devDependency; `test` (`vitest run`) and
  `test:watch` scripts.
- `app/bun.lock` — vitest + transitive deps.
- `app/src/store/exportShape.test.ts` — new; 10 `inferExportKind` cases + 2
  `exportKindLabel` cases.
- `app/src/workspaces/RunWorkspace/presets.test.ts` — new; 6 `mergeConfig`
  cases.

## Validation run

- `bun run test` — pass (2 files, 18 tests).
- `bunx tsc -b` — pass (test files type-check under the strict app tsconfig:
  `strict`, `noUnusedLocals`, `noUnusedParameters`).

Note: the Rust CI workflow does not build `app/`, so these tests are a
local/dev safety net (run with `bun run test`), not a GitHub CI gate. Wiring an
app test job into CI is a separate, larger change (bun setup + Playwright) and
is out of this slice.

## Follow-ups / remaining risks

- **ts-rs/specta codegen + export discriminator tag (F6)** — DEFERRED, risky.
  Replaces the `inferExportKind` shape-probe with a Rust-emitted discriminator.
  These tests become the conformance spec for that swap. Needs supervised
  design: which `*Export` types gain the tag, how `AnyExport` narrows on it, and
  whether old (untagged) exports must still load.
- **`resource_dir` presets** — `RunWorkspace/presets.ts:16` still hard-codes the
  repo root; replace with Tauri bundle-asset resolution.
- **Playwright smoke tests** — need a Tauri/webview harness; outside the
  pure-logic Vitest scope.
- **No CI gate yet** — see the validation note; the tests only run locally until
  an app CI job is added.
