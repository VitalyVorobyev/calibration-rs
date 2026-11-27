/** Path helpers for absolute filesystem paths, normalized to `/`.
 *
 * `@tauri-apps/api/path`'s `join`/`dirname` would be the "native" choice,
 * but in Tauri 2 they're IPC calls (`invoke("plugin:path|join", …)`), which
 * would (a) turn every join/dirname site into an async round-trip, (b)
 * require every `mockIPC` test fixture to stub `plugin:path|*` commands on
 * top of the app's own commands and (c) not exist on the server.
 * Instead we normalize once: the back end can return backslash-separated
 * paths on Windows, while export manifest paths are always POSIX-style
 * relative paths (export JSON) — so converting every input to `/` up front
 * and joining/splitting on `/` from then on is separator-correct on every
 * platform. Windows accepts `/` as a path separator in both its own APIs
 * and Rust's `std::fs`/`std::path`, so the resulting forward-slash paths
 * work unchanged when sent back to commands like `load_text_file`. */
function normalizeSeparators(path: string): string {
  return path.replace(/\\/g, "/");
}

/** True for an absolute path: POSIX `/…`, a Windows drive `C:/…` or a UNC
 * `//server/…` (after `normalizeSeparators`). */
function isAbsolute(path: string): boolean {
  return path.startsWith("/") || /^[A-Za-z]:\//.test(path);
}

/** Joins `dir` with `rest` as the file system would: an absolute segment
 * replaces what came before it (an export's image root may be absolute),
 * and `.` / `..` segments are folded, so equal paths compare equal and the
 * back end's file scope sees a plain path. Separators are normalized to
 * `/` first (see `normalizeSeparators`). */
export function joinPath(dir: string, ...rest: string[]): string {
  let out = normalizeSeparators(dir);
  for (const segment of rest) {
    if (!segment) continue;
    const s = normalizeSeparators(segment);
    out = isAbsolute(s) ? s : `${out.replace(/\/+$/, "")}/${s}`;
  }
  return foldDots(out);
}

/** `path` with `.` segments dropped and `..` segments folded into their
 * parent (lexically; never above the root). Trailing separators go. */
function foldDots(path: string): string {
  const lead = /^(\/\/|\/|[A-Za-z]:\/)?/.exec(path)?.[0] ?? "";
  const parts: string[] = [];
  for (const part of path.slice(lead.length).split("/")) {
    if (part === "" || part === ".") continue;
    if (part === ".." && parts.length > 0 && parts[parts.length - 1] !== "..")
      parts.pop();
    else if (part !== ".." || lead === "") parts.push(part);
  }
  return lead + parts.join("/");
}

/** Returns the parent directory of `path` (trailing separators ignored),
 * normalizing to `/` first (see `normalizeSeparators`). */
export function dirnamePath(path: string): string {
  const normalized = normalizeSeparators(path).replace(/\/+$/, "");
  const idx = normalized.lastIndexOf("/");
  return idx >= 0 ? normalized.slice(0, idx) : normalized;
}

/** `to` as a path relative to the folder `from` (both absolute): `../data/set1`,
 * or `.` for the same folder. A `to` on another drive stays absolute. */
export function relativePath(from: string, to: string): string {
  const a = foldDots(normalizeSeparators(from)).split("/").filter(Boolean);
  const b = foldDots(normalizeSeparators(to)).split("/").filter(Boolean);
  const drive = (p: string) =>
    /^[A-Za-z]:\//.test(p) ? p.slice(0, 2).toLowerCase() : "";
  if (drive(normalizeSeparators(from)) !== drive(normalizeSeparators(to))) {
    return foldDots(normalizeSeparators(to));
  }
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  const up = a.slice(i).map(() => "..");
  const rest = [...up, ...b.slice(i)];
  return rest.length === 0 ? "." : rest.join("/");
}
