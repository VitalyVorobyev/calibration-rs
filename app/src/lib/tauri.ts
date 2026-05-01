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
    (window as unknown as { __TAURI_INTERNALS__?: unknown })
      .__TAURI_INTERNALS__ != null
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
