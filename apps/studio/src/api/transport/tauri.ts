/** The desktop app's transport: Tauri IPC to the shell's three commands
 * (`studio_invoke`, `studio_subscribe`, `studio_unsubscribe`), the
 * `studio:` URI scheme for resources, and the native folder dialog for
 * Settings → Add folder…. The only module that imports `@tauri-apps/*`. */
import { Channel, convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import type { ServerEvent } from "../generated/types";
import { asStudioError, type Transport } from "../transport";

/** The Tauri webview has injected its IPC. */
export function isTauri(): boolean {
  if (typeof window === "undefined") return false;
  const internals = (window as unknown as { __TAURI_INTERNALS__?: { invoke?: unknown } })
    .__TAURI_INTERNALS__;
  return typeof internals?.invoke === "function";
}

async function call<T>(command: string, args: Record<string, unknown>): Promise<T> {
  try {
    return await invoke<T>(command, args);
  } catch (e) {
    throw asStudioError(e);
  }
}

export const tauriTransport: Transport = {
  invoke: (name, args) => call("studio_invoke", { name, args: args ?? {} }),

  subscribe(onEvent, after) {
    const channel = new Channel<ServerEvent>();
    channel.onmessage = onEvent;
    let closed = false;
    const id = call<number>("studio_subscribe", {
      after: after ?? null,
      onEvent: channel,
    });
    return {
      ready: id.then(() => undefined),
      close() {
        if (closed) return;
        closed = true;
        channel.onmessage = () => {};
        void id.then((subscription) => call("studio_unsubscribe", { id: subscription }));
      },
    };
  },

  // `studio://localhost/r%2F<name>%3Fq%3D<json>` (`http://studio.localhost/…`
  // on Windows): Tauri's own URL builder encodes the whole target, and the
  // shell's scheme handler decodes it.
  resourceUrl: (name, query) =>
    convertFileSrc(`r/${name}?q=${JSON.stringify(query ?? {})}`, "studio"),

  async pickNativeFolder() {
    const chosen = await open({
      multiple: false,
      directory: true,
      title: "Add a folder Calibration Studio may read",
    });
    return typeof chosen === "string" ? chosen : null;
  },
};
