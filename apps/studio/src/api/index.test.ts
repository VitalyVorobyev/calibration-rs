// @vitest-environment jsdom
/** Which transport a page has — Tauri's IPC when the webview injected it,
 * else HTTP to the page's own origin — and the shell-only extras each
 * offers. */
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  api,
  hasNativeFolderDialog,
  login,
  pickNativeFolder,
  selectTransport,
  setTransport,
} from ".";
import { httpTransport } from "./transport/http";
import { createMockTransport } from "./transport/mock";
import { tauriTransport } from "./transport/tauri";

afterEach(() => {
  clearMocks();
  setTransport(null);
  vi.unstubAllGlobals();
});

describe("the page's back end", () => {
  it("is HTTP on a plain page, with a login and no native dialog", async () => {
    expect(selectTransport()).toBe(httpTransport);
    expect(hasNativeFolderDialog()).toBe(false);
    await expect(pickNativeFolder()).rejects.toMatchObject({ kind: "roots_fixed" });
    const seen: { url: string; init?: RequestInit }[] = [];
    vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
      seen.push({ url, ...(init ? { init } : {}) });
      return Promise.resolve(new Response(null, { status: 204 }));
    });
    await login("the-token");
    expect(seen[0]?.url).toBe("/api/login");
    expect(seen[0]?.init?.body).toBe(JSON.stringify({ token: "the-token" }));
  });

  it("is Tauri's IPC in the desktop webview, with the native dialog", () => {
    mockIPC(() => null);
    expect(selectTransport()).toBe(tauriTransport);
    expect(hasNativeFolderDialog()).toBe(true);
  });

  it("is the test's transport while one is set", async () => {
    const mock = createMockTransport({
      commands: { camera_parameter_values: (args) => args.labels.map(() => 1) },
      nativeFolder: () => "/picked",
    });
    setTransport(mock);
    expect(await api.cameraParameterValues({ labels: ["fx", "fy"] })).toEqual([1, 1]);
    expect(await pickNativeFolder()).toBe("/picked");
    expect(mock.calls).toEqual([
      { command: "camera_parameter_values", args: { labels: ["fx", "fy"] } },
    ]);
  });
});
