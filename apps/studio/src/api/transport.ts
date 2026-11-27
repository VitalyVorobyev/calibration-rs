/** The seam between the app and its back end: a transport carries the
 * command table's calls (`invoke`), its event stream (`subscribe`) and the
 * URLs of its resources (`resourceUrl`). The typed client
 * (`generated/client.ts`) is written over it, so a transport never knows a
 * command by name.
 *
 * - **Tauri** (`transport/tauri.ts`) — the desktop app's IPC and its
 *   `studio:` URI scheme.
 * - **HTTP** (`transport/http.ts`) — `calibration-studio-server` on the
 *   page's own origin: `POST /api/cmd/<name>`, `GET /api/r/<name>?q=…`,
 *   `GET /api/events`, and its login.
 *
 * What differs between the back ends is data (`server_info`: the roots,
 * whether they can change, whether samples download), never which transport
 * carries it. */
import type {
  ErrorKind,
  ServerEvent,
  StudioError as StudioErrorBody,
} from "./generated/types";

/** An open event stream. */
export interface Subscription {
  /** Resolves once the stream is registered: an event published after it
   * reaches `onEvent`. */
  ready: Promise<void>;
  close(): void;
}

export interface Transport {
  /** Run a command; rejects with a `StudioError`. */
  invoke(command: string, args: unknown): Promise<unknown>;
  /** The event stream, from after event `after` (from each live job's
   * latest state without it). */
  subscribe(onEvent: (event: ServerEvent) => void, after?: number): Subscription;
  /** The URL of resource `resource` for `query`: the `studio:` scheme in
   * the desktop app, `/api/r/…` in the browser. A failed load answers the
   * `StudioError` envelope with a 4xx/5xx status. */
  resourceUrl(resource: string, query: unknown): string;
  /** The native folder dialog, the one way the desktop grants a root
   * folder (Settings → Add folder…); `null` when the user cancels. Tauri
   * only. */
  pickNativeFolder?(): Promise<string | null>;
  /** Exchange the server's token for a session cookie. HTTP only. */
  login?(token: string): Promise<void>;
  /** End the session. HTTP only. */
  logout?(): Promise<void>;
}

/** Why a command, job or resource failed: the back end's error envelope,
 * as both transports reject with it. */
export class StudioError extends Error implements StudioErrorBody {
  readonly kind: ErrorKind;
  readonly path?: string | null;
  /** The field it is about (a dotted path into the project or manifest). */
  readonly field?: string | null;
  /** The values to offer for `field` (`needs_input`). */
  readonly suggestions?: string[];

  constructor(body: StudioErrorBody) {
    super(body.message);
    this.name = "StudioError";
    this.kind = body.kind;
    if (body.path !== undefined && body.path !== null) this.path = body.path;
    if (body.field !== undefined && body.field !== null) this.field = body.field;
    if (body.suggestions !== undefined && body.suggestions.length > 0)
      this.suggestions = body.suggestions;
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
    const { kind, message, path, field, suggestions } = e as Record<string, unknown>;
    if (typeof kind === "string" && typeof message === "string") {
      return new StudioError({
        kind: kind as ErrorKind,
        message,
        ...(typeof path === "string" ? { path } : {}),
        ...(typeof field === "string" ? { field } : {}),
        ...(Array.isArray(suggestions)
          ? { suggestions: suggestions.filter((s): s is string => typeof s === "string") }
          : {}),
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
