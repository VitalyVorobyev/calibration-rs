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

/** Path joining for absolute filesystem paths. Tauri exposes a path
 * utility but for our manifest needs (POSIX-style relatives appended
 * to absolute roots) a hand-rolled join keeps the dependency surface
 * tiny. */
export function joinPath(dir: string, ...rest: string[]): string {
  let out = dir.replace(/[\\/]+$/, "");
  for (const segment of rest) {
    if (!segment) continue;
    const sep = out.includes("\\") && !out.includes("/") ? "\\" : "/";
    out = `${out}${sep}${segment.replace(/^[\\/]+/, "")}`;
  }
  return out;
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
