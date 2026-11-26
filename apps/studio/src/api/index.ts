/** The app's one way to its back end: `api`, the typed client over the
 * page's transport. Command names, argument and reply types all come from
 * the back end's command table (`generated/`); nothing outside `src/api/`
 * imports Tauri or names a command.
 *
 * The transport is the Tauri webview's IPC when the page runs in the
 * desktop app, else HTTP to the page's own origin
 * (`calibration-studio-server`, or Vite's proxy to one). A test replaces
 * it with `setTransport(createMockTransport(…))`. */
import { createClient } from "./generated/client";
import { JobHub } from "./jobs";
import { httpTransport } from "./transport/http";
import { isTauri, tauriTransport } from "./transport/tauri";
import { StudioError, type Transport } from "./transport";

export type { Job } from "./jobs";
export type { Transport } from "./transport";
export { asStudioError, isCancelled, StudioError } from "./transport";
export { fetchBytes, fetchPlane } from "./resources";

/** The transport this page has. */
export function selectTransport(): Transport {
  return isTauri() ? tauriTransport : httpTransport;
}

let override: Transport | null = null;

function current(): Transport {
  return override ?? selectTransport();
}

/** The page's transport, whichever it is at the time of each call. */
const transport: Transport = {
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

/** Whether this transport has a native folder dialog (the desktop app). */
export function hasNativeFolderDialog(): boolean {
  return current().pickNativeFolder !== undefined;
}

/** The native folder dialog: the folder the user picked to add as a root,
 * `null` when they cancel. */
export function pickNativeFolder(): Promise<string | null> {
  const t = current();
  if (!t.pickNativeFolder) {
    return Promise.reject(
      new StudioError({ kind: "roots_fixed", message: "no native folder dialog here" }),
    );
  }
  return t.pickNativeFolder();
}

/** Log in with the server's token (the browser); a no-op on a transport
 * without a login. Rejects with `unauthorized` for a wrong token. */
export async function login(token: string): Promise<void> {
  const t = current();
  if (t.login) await t.login(token);
}

/** End the browser's session. */
export async function logout(): Promise<void> {
  const t = current();
  if (t.logout) await t.logout();
}
