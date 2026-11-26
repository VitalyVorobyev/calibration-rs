// @vitest-environment jsdom
/** The Tauri transport's contract with the shell: every command goes
 * through `studio_invoke` with its name and argument object, a rejection
 * is the error envelope, resources through `studio_resource`, and the event
 * stream is a channel opened by `studio_subscribe` and closed by
 * `studio_unsubscribe`. */
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, describe, expect, it } from "vitest";
import { StudioError } from "../transport";
import { isTauri, tauriTransport } from "./tauri";

afterEach(() => {
  clearMocks();
});

describe("the Tauri transport", () => {
  it("carries a command's name and argument object", async () => {
    const seen: { cmd: string; args: unknown }[] = [];
    mockIPC((cmd, args) => {
      seen.push({ cmd, args });
      return [[1, 2]];
    });
    expect(isTauri()).toBe(true);
    const args = { camera: 1, points_px: [[3, 4]] };
    expect(await tauriTransport.invoke("undistort_points", args)).toEqual([[1, 2]]);
    expect(seen).toEqual([
      { cmd: "studio_invoke", args: { name: "undistort_points", args } },
    ]);
  });

  it("rejects with the error envelope as a StudioError", async () => {
    mockIPC(() => {
      // oxlint-disable-next-line typescript/only-throw-error -- Tauri rejects with the command's serialized error
      throw { kind: "not_a_project", message: "/tmp is no project" };
    });
    const error = await tauriTransport
      .invoke("open_project", {})
      .catch((e: unknown) => e);
    expect(error).toBeInstanceOf(StudioError);
    expect(error).toMatchObject({ kind: "not_a_project", message: "/tmp is no project" });
  });

  it("fetches a resource and opens and closes the event stream", async () => {
    const seen: { cmd: string; args: Record<string, unknown> }[] = [];
    mockIPC((cmd, args) => {
      seen.push({ cmd, args: args as Record<string, unknown> });
      if (cmd === "studio_subscribe") return 7;
      if (cmd === "studio_resource") return new Uint8Array([137, 80]).buffer;
      return null;
    });
    await tauriTransport.fetchResource("undistorted_image", {
      path: "/a.png",
      camera: 0,
    });
    expect(seen[0]).toEqual({
      cmd: "studio_resource",
      args: { name: "undistorted_image", query: { path: "/a.png", camera: 0 } },
    });

    const subscription = tauriTransport.subscribe(() => {}, 3);
    await subscription.ready;
    expect(seen[1]!.cmd).toBe("studio_subscribe");
    expect(seen[1]!.args.after).toBe(3);
    subscription.close();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(seen[2]).toEqual({ cmd: "studio_unsubscribe", args: { id: 7 } });
  });
});
