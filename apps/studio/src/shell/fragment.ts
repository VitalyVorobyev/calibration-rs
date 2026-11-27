/** The start-up fragment `calibration-studio-server` prints:
 * `#token=<token>&project=<dir>`. The fragment never reaches a server or a
 * `Referer`; the page reads it once and replaces it with the router's
 * `#/`, so the token does not stay in the address bar or the history. */

/** What a start-up fragment carries. */
export interface StartFragment {
  token: string | null;
  project: string | null;
}

/** The token and project of `hash`, or `null` when it is a route
 * (`#/diagnose`) or empty. */
export function parseStartFragment(hash: string): StartFragment | null {
  const body = hash.startsWith("#") ? hash.slice(1) : hash;
  if (body === "" || body.startsWith("/")) return null;
  const params = new URLSearchParams(body);
  const token = params.get("token");
  const project = params.get("project");
  if (token === null && project === null) return null;
  return { token, project };
}

/** Read the start-up fragment of the page's address and replace it with the
 * router's root, leaving no token in the history. */
export function takeStartFragment(): StartFragment | null {
  if (typeof window === "undefined") return null;
  const fragment = parseStartFragment(window.location.hash);
  if (fragment !== null) {
    const { pathname, search } = window.location;
    window.history.replaceState(null, "", `${pathname}${search}#/`);
  }
  return fragment;
}

let captured: StartFragment | null = null;

/** Take the start-up fragment now, before the hash router reads the
 * address (`main.tsx` imports `captureStart.ts` first). */
export function captureStartFragment(): void {
  captured = takeStartFragment();
}

/** The fragment captured at start-up, once. */
export function startFragment(): StartFragment | null {
  const fragment = captured;
  captured = null;
  return fragment;
}
