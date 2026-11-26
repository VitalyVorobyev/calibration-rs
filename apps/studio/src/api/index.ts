/** The app's one way to its back end: `api`, the typed client over the
 * page's transport. Command names, argument and reply types all come from
 * the back end's command table (`generated/`); nothing outside `src/api/`
 * imports Tauri or names a command.
 *
 * The transport is chosen when used, not when this module loads (the Tauri
 * webview's IPC, the browser viewer's marked page, or none), and a test
 * replaces it with `setTransport(createMockTransport(…))`. */
import { createClient } from "./generated/client";
import { JobHub } from "./jobs";
import { httpTransport } from "./transport/http";
import { isTauri, tauriTransport } from "./transport/tauri";
import {
  type Capabilities,
  NO_BACKEND,
  noBackend,
  type OpenDialog,
  StudioError,
  type Transport,
} from "./transport";
import type { ViewerSession } from "./generated/types";

export type { Job } from "./jobs";
export type { BackendKind, Capabilities, OpenDialog, Transport } from "./transport";
export { asStudioError, isCancelled, NO_BACKEND, StudioError } from "./transport";
export { fetchBytes, fetchPlane } from "./resources";

function isHttp(): boolean {
  if (import.meta.env.VITE_CALIB_BACKEND === "http") return true;
  return (
    typeof document !== "undefined" &&
    document.querySelector('meta[name="calib-backend"][content="http"]') !== null
  );
}

/** The transport this page has. */
export function selectTransport(): Transport {
  if (isTauri()) return tauriTransport;
  if (isHttp()) return httpTransport;
  return noBackend;
}

let override: Transport | null = null;

function current(): Transport {
  return override ?? selectTransport();
}

/** The page's transport, whichever it is at the time of each call. */
const transport: Transport = {
  get kind() {
    return current().kind;
  },
  invoke: (name, args) => current().invoke(name, args),
  subscribe: (onEvent, after) => current().subscribe(onEvent, after),
  resourceUrl: (name, query) => current().resourceUrl(name, query),
};

const hub = new JobHub(transport);

/** The back end's API. */
export const api = createClient(transport, hub);

/** Replace the page's transport (tests); `null` restores it. */
export function setTransport(next: Transport | null): void {
  override = next;
  hub.reset();
}

/** What the current back end offers. */
export function capabilities(): Capabilities {
  const kind = current().kind;
  return { kind, viewer: kind !== "none", desktop: kind === "tauri" };
}

/** Ask the user for one file or folder with a native dialog; `null` when
 * they cancel. Desktop only. */
export function openDialog(options: OpenDialog): Promise<string | null> {
  const t = current();
  if (!t.openDialog) {
    return Promise.reject(new StudioError({ kind: "internal", message: NO_BACKEND }));
  }
  return t.openDialog(options);
}

/** The run this page serves, or `null` unless the back end is the browser
 * viewer. */
export async function viewerSession(): Promise<ViewerSession | null> {
  const t = current();
  return t.session ? t.session() : null;
}
