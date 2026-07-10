import { invoke } from "@tauri-apps/api/core";

/** True iff the page is running inside a Tauri webview (i.e. the IPC
 * internals have been injected). When the user runs `bun run dev` and
 * loads localhost:1420 in a regular browser, this is false and any
 * `invoke` / dialog call would throw `Cannot read properties of
 * undefined (reading 'invoke')`. We guard up-front so the failure mode
 * is a readable banner instead of an opaque TypeError. */
export function isTauriContext(): boolean {
  return (
    typeof window !== "undefined" &&
    "__TAURI_INTERNALS__" in window &&
    (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ != null
  );
}

/** Path helpers for absolute filesystem paths, normalized to `/`.
 *
 * `@tauri-apps/api/path`'s `join`/`dirname` would be the "native" choice,
 * but in Tauri 2 they're IPC calls (`invoke("plugin:path|join", …)`), which
 * would (a) turn every join/dirname site into an async round-trip and (b)
 * require every `mockIPC` test fixture to stub `plugin:path|*` commands on
 * top of the app's own commands. Instead we normalize once: `repoRoot()`
 * can return a backslash-separated path on Windows (from the Rust
 * `repo_root_cmd`), while preset/export manifest paths are always
 * POSIX-style relative paths (`presets.ts`, export JSON) — so converting
 * every input to `/` up front and joining/splitting on `/` from then on
 * is separator-correct on every platform. Windows accepts `/` as a path
 * separator in both its own APIs and Rust's `std::fs`/`std::path`, so the
 * resulting forward-slash paths work unchanged when sent back to Tauri
 * commands like `load_text_file`. */
function normalizeSeparators(path: string): string {
  return path.replace(/\\/g, "/");
}

/** Joins `dir` with `rest`, normalizing all segments to `/` first (see
 * `normalizeSeparators`). */
export function joinPath(dir: string, ...rest: string[]): string {
  let out = normalizeSeparators(dir).replace(/\/+$/, "");
  for (const segment of rest) {
    if (!segment) continue;
    out = `${out}/${normalizeSeparators(segment).replace(/^\/+/, "")}`;
  }
  return out;
}

/** Returns the parent directory of `path` (trailing separators ignored),
 * normalizing to `/` first (see `normalizeSeparators`). */
export function dirnamePath(path: string): string {
  const normalized = normalizeSeparators(path).replace(/\/+$/, "");
  const idx = normalized.lastIndexOf("/");
  return idx >= 0 ? normalized.slice(0, idx) : normalized;
}

let repoRootPromise: Promise<string | null> | null = null;

/** Resolves the `calibration-rs` workspace repo root, memoized for the
 * session. Backed by the `repo_root_cmd` Tauri command (see
 * `src-tauri/src/commands.rs`), which reports *this developer's own
 * checkout path* — `app/src-tauri` is rebuilt locally by every
 * `bun run tauri dev`/`build`, so `CARGO_MANIFEST_DIR` is never a
 * stale, hard-coded, or another developer's path.
 *
 * Used to resolve `RunWorkspace` preset manifests (repo-root-relative
 * in `presets.ts`) to absolute paths. Returns `null` outside a Tauri
 * runtime (e.g. `bun run dev` in a plain browser tab) or if the
 * command call fails for any reason — callers should treat that the
 * same as "presets need Tauri" (mirrors `isTauriContext`'s guard). */
export function repoRoot(): Promise<string | null> {
  if (!isTauriContext()) return Promise.resolve(null);
  repoRootPromise ??= invoke<string>("repo_root_cmd").catch(() => null);
  return repoRootPromise;
}
