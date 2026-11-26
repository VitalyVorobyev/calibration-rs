/** The desktop app's transport: Tauri IPC to the shell's four commands
 * (`studio_invoke`, `studio_subscribe`, `studio_unsubscribe`,
 * `studio_resource`), the asset protocol for source images and the native
 * dialog. The only module that imports `@tauri-apps/*`. */
import { Channel, convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import type { ServerEvent } from "../generated/types";
import { asStudioError, type OpenDialog, type Transport } from "../transport";

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
  kind: "tauri",

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

  fetchResource: (name, query) => call<ArrayBuffer>("studio_resource", { name, query }),

  fileUrl: (path) => convertFileSrc(path.replace(/\\/g, "/")),

  async openDialog(options: OpenDialog) {
    const chosen = await open({
      multiple: false,
      directory: options.directory ?? false,
      title: options.title,
      ...(options.filters ? { filters: options.filters } : {}),
    });
    return typeof chosen === "string" ? chosen : null;
  },
};
