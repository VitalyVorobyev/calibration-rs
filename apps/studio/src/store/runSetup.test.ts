import { afterEach, describe, expect, it } from "vitest";
import planarDefault from "../schemas/planar_intrinsics_config.default.json";
import { DEFAULT_DATASET, resetRunSetup, topologyOf, useRunSetup } from "./runSetup";

// Outside Tauri the defaults come from the emitted planar default (other
// topologies get `{}`), so these tests need no IPC.

afterEach(() => resetRunSetup());

describe("useRunSetup", () => {
  it("starts from the planar default session", () => {
    const s = useRunSetup.getState();
    expect(s.manifest).toEqual(DEFAULT_DATASET);
    expect(s.config).toEqual(planarDefault);
    expect(s.manifestDir).toBeNull();
  });

  it("resets the config when the manifest's topology changes", async () => {
    useRunSetup.getState().setConfig({ edited: true });
    useRunSetup.getState().setManifest({ topology: "rig_extrinsics" });
    await Promise.resolve();
    await Promise.resolve();
    expect(useRunSetup.getState().config).toEqual({});
  });

  it("keeps the config for an edit within the same topology", async () => {
    useRunSetup.getState().setConfig({ edited: true });
    useRunSetup.getState().updateManifest((prev) => ({
      ...(prev as object),
      description: "x",
    }));
    await Promise.resolve();
    expect(useRunSetup.getState().config).toEqual({ edited: true });
  });

  it("lets an explicit config win over a pending topology reset", async () => {
    useRunSetup.getState().setManifest({ topology: "rig_extrinsics" });
    useRunSetup.getState().setConfig({ pasted: true });
    await Promise.resolve();
    await Promise.resolve();
    expect(useRunSetup.getState().config).toEqual({ pasted: true });
  });

  it("loads a whole setup with config overrides over the defaults", async () => {
    await useRunSetup.getState().loadSetup({
      manifest: { topology: "planar_intrinsics" },
      manifestDir: "/d",
      manifestPath: "/d/dataset.toml",
      presetId: "p",
      configOverrides: { solver: { max_iters: 3 } },
    });
    const s = useRunSetup.getState();
    expect(s.manifestDir).toBe("/d");
    expect(s.presetId).toBe("p");
    expect((s.config as { solver: { max_iters: number } }).solver.max_iters).toBe(3);
  });

  it("drops the preset when the paths are changed by hand", () => {
    useRunSetup.setState({ presetId: "p", manifestPath: "/d/dataset.toml" });
    useRunSetup.getState().setPaths({ manifestDir: "/e" });
    const s = useRunSetup.getState();
    expect(s.presetId).toBeNull();
    expect(s.manifestDir).toBe("/e");
    expect(s.manifestPath).toBe("/d/dataset.toml");
  });

  it("defaults a manifest without a topology to planar", () => {
    expect(topologyOf({})).toBe("planar_intrinsics");
    expect(topologyOf(null)).toBe("planar_intrinsics");
  });
});
