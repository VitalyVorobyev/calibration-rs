// @vitest-environment jsdom
/** The back-end seam: which back end a page has, and the HTTP back end's
 * wire (the same command names, argument objects and answers as Tauri's,
 * with the command's error value as the rejection). */
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  backendKind,
  call,
  callStreaming,
  capabilities,
  fileUrl,
  NO_BACKEND,
  openDialog,
  viewerSession,
} from "./backend";

function markHttp() {
  const meta = document.createElement("meta");
  meta.name = "calib-backend";
  meta.content = "http";
  document.head.append(meta);
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  clearMocks();
  vi.unstubAllGlobals();
  document.head.querySelector('meta[name="calib-backend"]')?.remove();
});

describe("without a back end", () => {
  it("offers nothing and says how to start one", async () => {
    expect(capabilities()).toEqual({ kind: "none", viewer: false, desktop: false });
    await expect(call("load_export", { path: "/x" })).rejects.toThrow(NO_BACKEND);
    expect(await viewerSession()).toBeNull();
  });
});

describe("the desktop back end", () => {
  it("is Tauri's IPC", async () => {
    mockIPC((cmd, args) => ({ cmd, args }));
    expect(capabilities()).toEqual({ kind: "tauri", viewer: true, desktop: true });
    expect(await call("undistort_points", { camera: 1 })).toEqual({
      cmd: "undistort_points",
      args: { camera: 1 },
    });
  });
});

describe("the browser viewer's back end", () => {
  it("posts each command with its argument object and answers with its JSON", async () => {
    markHttp();
    const fetchMock = vi.fn(() => Promise.resolve(jsonResponse([[1, 2]])));
    vi.stubGlobal("fetch", fetchMock);
    expect(backendKind()).toBe("http");
    expect(capabilities()).toEqual({ kind: "http", viewer: true, desktop: false });

    const args = { camera: 0, pointsPx: [[1, 2]] };
    expect(await call("undistort_points", args)).toEqual([[1, 2]]);
    expect(fetchMock).toHaveBeenCalledWith("/api/undistort_points", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(args),
    });
  });

  it("rejects with the command's error value, as Tauri does", async () => {
    markHttp();
    vi.stubGlobal("fetch", () =>
      Promise.resolve(jsonResponse("no export loaded yet", 422)),
    );
    await expect(call("undistort_points", {})).rejects.toBe("no export loaded yet");
  });

  it("answers a binary command with its bytes", async () => {
    markHttp();
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    vi.stubGlobal("fetch", () =>
      Promise.resolve(new Response(png, { headers: { "Content-Type": "image/png" } })),
    );
    const bytes = await call<ArrayBuffer>("load_undistorted_image", { path: "/r/a.png" });
    expect(new Uint8Array(bytes)).toEqual(png);
  });

  it("loads files through its file route and names the run it serves", async () => {
    markHttp();
    expect(fileUrl("C:\\runs\\a b.png")).toBe("/api/file?path=C%3A%2Fruns%2Fa%20b.png");
    const session = {
      name: "run-001",
      layout: "project_run",
      run_dir: "/p/runs/run-001",
      export_path: "/p/runs/run-001/export.json",
    };
    vi.stubGlobal("fetch", () => Promise.resolve(jsonResponse(session)));
    expect(await viewerSession()).toEqual(session);
  });

  it("refuses the desktop's dialogs and streams", async () => {
    markHttp();
    await expect(openDialog({ title: "x" })).rejects.toThrow(NO_BACKEND);
    await expect(
      callStreaming("run_calibration_cmd", {}, "onProgress", () => {}),
    ).rejects.toThrow(NO_BACKEND);
  });
});
