/** The seam between the app and its back end: a transport carries the
 * command table's calls (`invoke`), its event stream (`subscribe`) and the
 * URLs of its resources (`resourceUrl`). The typed client
 * (`generated/client.ts`) is written over it, so a transport never knows a
 * command by name.
 *
 * - **Tauri** (`transport/tauri.ts`) — the desktop app's IPC.
 * - **HTTP** (`transport/http.ts`) — the browser viewer (`calib-viewer`):
 *   `POST /api/cmd/<name>`, `GET /api/r/<name>?q=…`, `GET /api/events`.
 * - **none** — plain `bun run dev`: every call fails with `NO_BACKEND`.
 *
 * Which one answers is fixed by the page: the Tauri webview injects its IPC,
 * `calib-viewer` marks its `index.html` with
 * `<meta name="calib-backend" content="http">` (or `VITE_CALIB_BACKEND=http`
 * in Vite dev). */
import type {
  ErrorKind,
  ServerEvent,
  StudioError as StudioErrorBody,
  ViewerSession,
} from "./generated/types";

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

/** A native open dialog's options. */
export interface OpenDialog {
  title: string;
  directory?: boolean;
  filters?: { name: string; extensions: string[] }[];
}

/** An open event stream. */
export interface Subscription {
  /** Resolves once the stream is registered: an event published after it
   * reaches `onEvent`. */
  ready: Promise<void>;
  close(): void;
}

export interface Transport {
  kind: BackendKind;
  /** Run a command; rejects with a `StudioError`. */
  invoke(command: string, args: unknown): Promise<unknown>;
  /** The event stream, from after event `after` (from each live job's
   * latest state without it). */
  subscribe(onEvent: (event: ServerEvent) => void, after?: number): Subscription;
  /** The URL of resource `resource` for `query`: the `studio:` scheme in
   * the desktop app, `/api/r/…` in the browser. A failed load answers the
   * `StudioError` envelope with a 4xx/5xx status. */
  resourceUrl(resource: string, query: unknown): string;
  /** A native open dialog; `null` when the user cancels. Desktop only. */
  openDialog?(options: OpenDialog): Promise<string | null>;
  /** The run the browser viewer serves. HTTP only. */
  session?(): Promise<ViewerSession>;
}

/** Why a command, job or resource failed: the back end's error envelope,
 * as both transports reject with it. */
export class StudioError extends Error implements StudioErrorBody {
  readonly kind: ErrorKind;
  readonly path?: string | null;

  constructor(body: StudioErrorBody) {
    super(body.message);
    this.name = "StudioError";
    this.kind = body.kind;
    if (body.path !== undefined && body.path !== null) this.path = body.path;
  }

  /** The message alone, as the UI shows a failure. */
  override toString(): string {
    return this.message;
  }
}

/** `e` as a `StudioError`: the envelope a transport rejected with, or any
 * other failure (a broken IPC, a network error) as `internal`. */
export function asStudioError(e: unknown): StudioError {
  if (e instanceof StudioError) return e;
  if (typeof e === "object" && e !== null) {
    const { kind, message, path } = e as Record<string, unknown>;
    if (typeof kind === "string" && typeof message === "string") {
      return new StudioError({
        kind: kind as ErrorKind,
        message,
        ...(typeof path === "string" ? { path } : {}),
      });
    }
  }
  const message = e instanceof Error ? e.message : String(e);
  return new StudioError({ kind: "internal", message });
}

/** Whether `e` is a cancelled job's rejection. */
export function isCancelled(e: unknown): boolean {
  return e instanceof StudioError && e.kind === "cancelled";
}

/** Why a command cannot run on the page as it is. */
export const NO_BACKEND =
  "No back end: launch the desktop app with `bun run tauri dev`, or a run " +
  "in the browser with `bun run viewer -- <run-dir>`. Plain `bun run dev` " +
  "only starts Vite.";

/** The transport of a page without a back end: nothing answers. */
export const noBackend: Transport = {
  kind: "none",
  invoke: () =>
    Promise.reject(new StudioError({ kind: "internal", message: NO_BACKEND })),
  subscribe: () => ({ ready: Promise.resolve(), close: () => {} }),
  resourceUrl: (name, query) =>
    `/api/r/${encodeURIComponent(name)}?q=${encodeURIComponent(JSON.stringify(query ?? {}))}`,
};
