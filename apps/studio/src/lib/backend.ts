/** The app's one seam to its back end, and the only module that imports
 * `@tauri-apps/*`.
 *
 * The same UI runs on two back ends:
 *
 * - **Tauri** — the desktop app: commands over IPC (`invoke`), source
 *   images through the asset protocol, native dialogs, projects and
 *   calibration runs.
 * - **HTTP** — the browser viewer (`calib-viewer`): the same commands as
 *   `POST /api/<command>` with the same argument object and the same
 *   answer, images from `GET /api/file?path=…`, and one run served per
 *   session (`GET /api/session`). It is a viewer: no dialogs, projects or
 *   calibration runs.
 *
 * Without either (plain `bun run dev`) nothing answers; the UI shows its
 * empty states. Which back end answers is fixed by the page — the Tauri
 * webview injects its IPC, `calib-viewer` marks its `index.html` with
 * `<meta name="calib-backend" content="http">` (or `VITE_CALIB_BACKEND=http`
 * in Vite dev) — and read when used, so a test that installs a Tauri IPC
 * mock after this module loads still reaches it. */
import { Channel, convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

export type BackendKind = "tauri" | "http" | "none";

/** What the current back end offers. */
export interface Capabilities {
  kind: BackendKind;
  /** The viewer commands answer: loading an export, undistortion,
   * epipolar overlays, uncertainty, disparity. Tauri and HTTP. */
  viewer: boolean;
  /** The desktop's own features: native dialogs, projects, Quick Start
   * samples and calibration runs. Tauri only. */
  desktop: boolean;
}

/** Why a command cannot run on the page as it is. */
export const NO_BACKEND =
  "No back end: launch the desktop app with `bun run tauri dev`, or a run " +
  "in the browser with `bun run viewer -- <run-dir>`. Plain `bun run dev` " +
  "only starts Vite.";

/** The Tauri webview has injected its IPC. */
function isTauri(): boolean {
  if (typeof window === "undefined") return false;
  const internals = (window as unknown as { __TAURI_INTERNALS__?: { invoke?: unknown } })
    .__TAURI_INTERNALS__;
  return typeof internals?.invoke === "function";
}

function isHttp(): boolean {
  if (import.meta.env.VITE_CALIB_BACKEND === "http") return true;
  return (
    typeof document !== "undefined" &&
    document.querySelector('meta[name="calib-backend"][content="http"]') !== null
  );
}

/** The back end that answers this page. */
export function backendKind(): BackendKind {
  if (isTauri()) return "tauri";
  if (isHttp()) return "http";
  return "none";
}

export function capabilities(): Capabilities {
  const kind = backendKind();
  return { kind, viewer: kind !== "none", desktop: kind === "tauri" };
}

/** Run a back-end command by its Tauri name with its (camelCase) argument
 * object. Rejects with the command's error value, as Tauri does: usually
 * the error string, a typed object for the project commands. */
export async function call<T>(
  command: string,
  args: Record<string, unknown> = {},
): Promise<T> {
  switch (backendKind()) {
    case "tauri":
      return invoke<T>(command, args);
    case "http":
      return httpCall<T>(command, args);
    case "none":
      throw new Error(NO_BACKEND);
  }
}

async function httpCall<T>(command: string, args: Record<string, unknown>): Promise<T> {
  const response = await fetch(`/api/${command}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  const json = (response.headers.get("Content-Type") ?? "").startsWith(
    "application/json",
  );
  const body: unknown = json ? await response.json() : await response.arrayBuffer();
  // Reject with the command's own error value, exactly as a rejected
  // Tauri command does, so callers handle both back ends alike.
  if (!response.ok) throw body;
  return body as T;
}

/** Run a desktop command that streams messages over a channel (a
 * calibration run's progress), passed as its argument `channelArg`. */
export async function callStreaming<T, M>(
  command: string,
  args: Record<string, unknown>,
  channelArg: string,
  onMessage: (message: M) => void,
): Promise<T> {
  if (backendKind() !== "tauri") throw new Error(NO_BACKEND);
  const channel = new Channel<M>();
  channel.onmessage = onMessage;
  return invoke<T>(command, { ...args, [channelArg]: channel });
}

/** The URL the page loads the file at absolute `path` from: the asset
 * protocol in the desktop app (scoped to the folders the user opened), the
 * viewer's file route in the browser (scoped to the run and its images). */
export function fileUrl(path: string): string {
  const normalized = path.replace(/\\/g, "/");
  switch (backendKind()) {
    case "tauri":
      return convertFileSrc(normalized);
    case "http":
      return `/api/file?path=${encodeURIComponent(normalized)}`;
    case "none":
      return normalized;
  }
}

/** A native open dialog's options. */
export interface OpenDialog {
  title: string;
  directory?: boolean;
  filters?: { name: string; extensions: string[] }[];
}

/** Ask the user for one file or folder with a native dialog; `null` when
 * they cancel. Desktop only. */
export async function openDialog(options: OpenDialog): Promise<string | null> {
  if (backendKind() !== "tauri") throw new Error(NO_BACKEND);
  const chosen = await open({
    multiple: false,
    directory: options.directory ?? false,
    title: options.title,
    ...(options.filters ? { filters: options.filters } : {}),
  });
  return typeof chosen === "string" ? chosen : null;
}

/** The run a browser viewer serves (`GET /api/session`). */
export interface ViewerSession {
  /** The run folder's name. */
  name: string;
  layout: "vcal" | "project_run" | "export_file";
  run_dir: string;
  /** The export to load. */
  export_path: string;
}

/** The run this page serves, or `null` unless the back end is the browser
 * viewer. */
export async function viewerSession(): Promise<ViewerSession | null> {
  if (backendKind() !== "http") return null;
  const response = await fetch("/api/session");
  if (!response.ok) throw new Error(`session: ${response.status} ${response.statusText}`);
  return (await response.json()) as ViewerSession;
}
