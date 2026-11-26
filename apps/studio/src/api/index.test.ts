// @vitest-environment jsdom
/** Which back end a page has: none in plain Vite, Tauri when the webview
 * injected its IPC, HTTP on the browser viewer's marked page; and what
 * each offers. */
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, describe, expect, it } from "vitest";
import {
  api,
  capabilities,
  NO_BACKEND,
  selectTransport,
  setTransport,
  viewerSession,
} from ".";
import { createMockTransport } from "./transport/mock";

afterEach(() => {
  clearMocks();
  setTransport(null);
  document.head.querySelector('meta[name="calib-backend"]')?.remove();
});

describe("the page's back end", () => {
  it("is none without one, and says how to start one", async () => {
    expect(capabilities()).toEqual({ kind: "none", viewer: false, desktop: false });
    await expect(api.loadExport({ path: "/x" })).rejects.toThrow(NO_BACKEND);
    expect(await viewerSession()).toBeNull();
  });

  it("is Tauri's IPC in the desktop webview", () => {
    mockIPC(() => null);
    expect(selectTransport().kind).toBe("tauri");
    expect(capabilities()).toEqual({ kind: "tauri", viewer: true, desktop: true });
  });

  it("is HTTP on the browser viewer's page", () => {
    const meta = document.createElement("meta");
    meta.name = "calib-backend";
    meta.content = "http";
    document.head.append(meta);
    expect(capabilities()).toEqual({ kind: "http", viewer: true, desktop: false });
  });

  it("is the test's transport while one is set", async () => {
    const mock = createMockTransport({
      kind: "http",
      commands: { camera_parameter_values: (args) => args.labels.map(() => 1) },
      session: { name: "r", layout: "vcal", run_dir: "/r", export_path: "/r/c.json" },
    });
    setTransport(mock);
    expect(await api.cameraParameterValues({ labels: ["fx", "fy"] })).toEqual([1, 1]);
    expect((await viewerSession())?.export_path).toBe("/r/c.json");
    expect(mock.calls).toEqual([
      { command: "camera_parameter_values", args: { labels: ["fx", "fy"] } },
    ]);
  });
});
