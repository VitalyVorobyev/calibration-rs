import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport, StudioError } from "../api";
import { createMockTransport, type MockTransport } from "../api/transport/mock";
import { resetRunSetup, topologyOf, useRunSetup } from "./runSetup";

// The defaults come from the back end; a mock stands in for it.
const DATASET = { version: 1, topology: "planar_intrinsics", cameras: [] };
const DEFAULTS: Record<string, unknown> = {
  planar_intrinsics: { solver: { max_iters: 50, verbosity: 0 } },
  rig_extrinsics: { rig: true },
};

let mock: MockTransport;

beforeEach(() => {
  mock = createMockTransport({
    commands: {
      dataset_form: () => ({ schema: {}, default: DATASET }),
      config_form: ({ topology }) => {
        if (!(topology in DEFAULTS)) {
          throw new StudioError({
            kind: "invalid_args",
            message: `no form for ${topology}`,
          });
        }
        return { schema: {}, default: DEFAULTS[topology] };
      },
      // The library's rule, for the one shape these tests patch.
      resolve_config: ({ topology, patch }) => {
        const base = DEFAULTS[topology] as { solver: Record<string, unknown> };
        const solver = (patch as { solver?: Record<string, unknown> }).solver ?? {};
        return { ...base, solver: { ...base.solver, ...solver } };
      },
    },
  });
  setTransport(mock);
});

afterEach(() => {
  setTransport(null);
  resetRunSetup();
});

/** Let the store's back-end calls settle. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("useRunSetup", () => {
  it("starts empty and fills from the back end's defaults", async () => {
    expect(useRunSetup.getState().manifest).toBeNull();
    expect(useRunSetup.getState().config).toBeNull();
    await useRunSetup.getState().ensureDefaults();
    const s = useRunSetup.getState();
    expect(s.manifest).toEqual(DATASET);
    expect(s.config).toEqual(DEFAULTS.planar_intrinsics);
    expect(s.manifestDir).toBeNull();
    // A setup already there is kept.
    useRunSetup.getState().setConfig({ edited: true });
    await useRunSetup.getState().ensureDefaults();
    expect(useRunSetup.getState().config).toEqual({ edited: true });
  });

  it("resets the config when the manifest's topology changes", async () => {
    useRunSetup.getState().setConfig({ edited: true });
    useRunSetup.getState().setManifest({ topology: "rig_extrinsics" });
    await settle();
    expect(useRunSetup.getState().config).toEqual({ rig: true });
    // A topology without a form leaves no config.
    useRunSetup.getState().setManifest({ topology: "no_such_workflow" });
    await settle();
    expect(useRunSetup.getState().config).toBeNull();
  });

  it("keeps the config for an edit within the same topology", async () => {
    useRunSetup.getState().setManifest({ topology: "planar_intrinsics" });
    await settle();
    useRunSetup.getState().setConfig({ edited: true });
    useRunSetup.getState().updateManifest((prev) => ({
      ...(prev as object),
      description: "x",
    }));
    await settle();
    expect(useRunSetup.getState().config).toEqual({ edited: true });
  });

  it("lets an explicit config win over a pending topology reset", async () => {
    useRunSetup.getState().setManifest({ topology: "rig_extrinsics" });
    useRunSetup.getState().setConfig({ pasted: true });
    await settle();
    expect(useRunSetup.getState().config).toEqual({ pasted: true });
  });

  it("loads a whole setup with config overrides resolved by the back end", async () => {
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
    expect(s.config).toEqual({ solver: { max_iters: 3, verbosity: 0 } });
    expect(mock.calls[mock.calls.length - 1]).toEqual({
      command: "resolve_config",
      args: { topology: "planar_intrinsics", patch: { solver: { max_iters: 3 } } },
    });
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
