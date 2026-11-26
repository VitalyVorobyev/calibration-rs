/** The browser viewer's transport (`calib-viewer`, over the page's own
 * origin): `POST /api/cmd/<name>` with the argument object as JSON,
 * `GET /api/r/<name>?q=<query as JSON>` for a resource (a source image, a
 * blob), `GET /api/events` (server-sent events, resumed after the last event
 * seen) for the event stream and `GET /api/session` for the run it serves. A
 * failure is the `StudioError` envelope as the body, with a 4xx/5xx
 * status. */
import type { ServerEvent, ViewerSession } from "../generated/types";
import { asStudioError, StudioError, type Transport } from "../transport";

/** A failed response as a `StudioError`: its envelope, or its status. */
async function failure(response: Response): Promise<StudioError> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // Not JSON: the status says what went wrong.
  }
  if (typeof body === "object" && body !== null && "kind" in body)
    return asStudioError(body);
  const message =
    typeof body === "string" ? body : `${response.status} ${response.statusText}`.trim();
  return new StudioError({ kind: "internal", message });
}

async function send(url: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (e) {
    throw asStudioError(e);
  }
  if (!response.ok) throw await failure(response);
  return response;
}

/** The URL of resource `name` for `query`. */
export function resourceUrl(name: string, query: unknown): string {
  return `/api/r/${encodeURIComponent(name)}?q=${encodeURIComponent(JSON.stringify(query ?? {}))}`;
}

export const httpTransport: Transport = {
  kind: "http",

  async invoke(name, args) {
    const response = await send(`/api/cmd/${encodeURIComponent(name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(args ?? {}),
    });
    return (await response.json()) as unknown;
  },

  subscribe(onEvent, after) {
    let last = after;
    const url = after === undefined ? "/api/events" : `/api/events?after=${after}`;
    const source = new EventSource(url);
    // A reconnecting EventSource resumes by itself with `Last-Event-ID`.
    source.onmessage = (message: MessageEvent<string>) => {
      let event: ServerEvent;
      try {
        event = JSON.parse(message.data) as ServerEvent;
      } catch {
        return;
      }
      if (last !== undefined && event.seq < last) return;
      last = event.seq;
      onEvent(event);
    };
    const ready = new Promise<void>((resolve) => {
      source.addEventListener("open", () => resolve(), { once: true });
      source.addEventListener("error", () => resolve(), { once: true });
    });
    return { ready, close: () => source.close() };
  },

  resourceUrl,

  async session(): Promise<ViewerSession> {
    const response = await send("/api/session");
    return (await response.json()) as ViewerSession;
  },
};
