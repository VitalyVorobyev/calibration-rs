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
import { isNetworkError, StudioError, type Transport } from "./transport";

export type { Connection, Job } from "./jobs";
export type { Transport } from "./transport";
export { asStudioError, isCancelled, isNetworkError, StudioError } from "./transport";
export { fetchBytes, fetchPlane } from "./resources";

/** The transport this page has. */
export function selectTransport(): Transport {
  return isTauri() ? tauriTransport : httpTransport;
}

let override: Transport | null = null;

/** Who hears of a command that never reached the back end. */
const unreachable = new Set<() => void>();

function current(): Transport {
  return override ?? selectTransport();
}

/** The page's transport, whichever it is at the time of each call. */
const transport: Transport = {
  invoke: async (name, args) => {
    try {
      return await current().invoke(name, args);
    } catch (e) {
      if (isNetworkError(e)) for (const listener of [...unreachable]) listener();
      throw e;
    }
  },
  subscribe: (onEvent, after, onStatus) => current().subscribe(onEvent, after, onStatus),
  resourceUrl: (name, query) => current().resourceUrl(name, query),
};

const hub = new JobHub(transport);

/** The back end's API. */
export const api = createClient(transport, hub);

/** The back end's event stream, watched for good: every event (a job's
 * progress, a project that changed), and whether the stream is up. Jobs
 * started through `api.jobs` share the same stream. */
export const events = {
  connect: (): Promise<void> => hub.connect(),
  disconnect: (): void => hub.disconnect(),
  listen: (listener: Parameters<JobHub["listen"]>[0]): (() => void) =>
    hub.listen(listener),
  onConnection: (listener: Parameters<JobHub["onConnection"]>[0]): (() => void) =>
    hub.onConnection(listener),
  /** Hear each command that failed to reach the back end at all (the
   * network, not an answer). Returns the unsubscribe function. */
  onUnreachable: (listener: () => void): (() => void) => {
    unreachable.add(listener);
    return () => unreachable.delete(listener);
  },
};

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
