/** The session with the back end: whether it answers, whether it wants the
 * token, and what it serves (`server_info`: its roots, whether they can
 * change, whether samples download). The shell starts it once; a login, an
 * added or removed root refreshes it. */
import { create } from "zustand";
import { api, login, StudioError } from "../api";
import type { ServerInfo } from "../api/generated/types";
import { useProject } from "../project/projectStore";
import { startFragment } from "../shell/fragment";

export type SessionStatus =
  | { kind: "connecting" }
  | { kind: "ready"; info: ServerInfo }
  /** The server wants its token; `error` after a wrong one. */
  | { kind: "login"; error: string | null }
  /** No back end answers. */
  | { kind: "offline"; message: string };

export interface SessionState {
  status: SessionStatus;
  /** Read the start-up fragment, log in with its token, then connect and
   * open its project. */
  start: () => Promise<void>;
  /** Ask the back end what it serves. */
  refresh: () => Promise<void>;
  /** Log in with `token` and connect. */
  submitToken: (token: string) => Promise<void>;
  /** Replace the known server info (after `add_root`, `remove_root`). */
  setInfo: (info: ServerInfo) => void;
}

/** The roots a user added (not the samples' folder). */
export function userRoots(info: ServerInfo): string[] {
  return info.roots.filter((r) => !r.samples).map((r) => r.path);
}

export const useSession = create<SessionState>()((set, get) => {
  const refresh = async () => {
    try {
      const info = await api.serverInfo();
      set({ status: { kind: "ready", info } });
      void useProject.getState().refreshRecent();
    } catch (e) {
      if (e instanceof StudioError && e.kind === "unauthorized") {
        const current = get().status;
        set({
          status: {
            kind: "login",
            error: current.kind === "login" ? current.error : null,
          },
        });
      } else {
        set({
          status: {
            kind: "offline",
            message: e instanceof Error ? e.message : String(e),
          },
        });
      }
    }
  };

  const submitToken = async (token: string) => {
    try {
      await login(token.trim());
    } catch (e) {
      const wrong = e instanceof StudioError && e.kind === "unauthorized";
      set({
        status: {
          kind: "login",
          error: wrong ? "That is not this server's token." : String(e),
        },
      });
      return;
    }
    await refresh();
  };

  return {
    status: { kind: "connecting" },
    refresh,
    submitToken,
    setInfo: (info) => set({ status: { kind: "ready", info } }),
    // Once: a second call (React's strict mode mounts twice) joins the first.
    start: () => {
      if (starting) return starting;
      if (get().status.kind !== "connecting") return Promise.resolve();
      starting = (async () => {
        const fragment = startFragment();
        if (fragment?.token) await submitToken(fragment.token);
        else await refresh();
        if (fragment?.project && get().status.kind === "ready") {
          await useProject.getState().open(fragment.project);
        }
      })().finally(() => {
        starting = null;
      });
      return starting;
    },
  };
});

/** The start in flight, if any. */
let starting: Promise<void> | null = null;
