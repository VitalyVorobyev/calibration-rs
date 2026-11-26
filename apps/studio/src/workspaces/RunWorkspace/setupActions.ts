/** The ways a Run setup is loaded: a Quick Start preset, a sniffed folder,
 * a picked folder or a picked manifest file. Each failure becomes a run
 * status the banner shows. */
import { api, capabilities } from "../../api";
import { dirnamePath } from "../../lib/paths";
import { useRunSetup } from "../../store/runSetup";
import { pickFolder, pickManifestFile } from "../../shell/dialogs";
import { mergeConfig, type EnabledPreset } from "./presets";
import type { RunStatus } from "./runStatus";

type Report = (status: RunStatus) => void;

const needsTauri = (what: string): RunStatus => ({
  kind: "error",
  category: "no_tauri",
  message: `${what} requires the desktop app (bun run tauri dev).`,
});

/** Sequence number of the latest setup load. Module-scoped so it outlives a
 * render: a slower load started earlier never overwrites a newer one. */
let activeLoadSeq = 0;

export interface SetupActions {
  applyPreset: (preset: EnabledPreset) => Promise<boolean>;
  sniffFolder: () => Promise<boolean>;
  pickDatasetFolder: () => Promise<void>;
  pickManifest: () => Promise<void>;
}

/** The setup loaders, reporting failures (and the idle state after a
 * successful load) through `report`. */
export function setupActions(report: Report): SetupActions {
  const desktop = capabilities().desktop;
  const { loadSetup, setPaths, setManifest } = useRunSetup.getState();

  return {
    applyPreset: async (preset) => {
      if (!desktop) {
        report(needsTauri("Preset loading"));
        return false;
      }
      const seq = ++activeLoadSeq;
      try {
        // A sample is downloaded on first use; its manifest path comes back.
        const path =
          "sample" in preset.source
            ? await (
                await api.jobs.fetchSample({ id: preset.source.sample })
              ).result
            : preset.source.manifestPath;
        if (seq !== activeLoadSeq) return false;
        // The back end reads the manifest (TOML or JSON) as the library does.
        const parsed = await api.readManifest({ path });
        if (seq !== activeLoadSeq) return false;
        await loadSetup({
          manifest: preset.manifestOverrides
            ? mergeConfig(parsed, preset.manifestOverrides)
            : parsed,
          manifestDir: dirnamePath(path),
          manifestPath: path,
          presetId: preset.id,
          configOverrides: preset.configOverrides,
        });
        if (seq !== activeLoadSeq) return false;
        report({ kind: "idle" });
        return true;
      } catch (e) {
        if (seq === activeLoadSeq) {
          report({
            kind: "error",
            category: "preset_load",
            message: `Failed to load preset "${preset.name}": ${String(e)}`,
          });
        }
        return false;
      }
    },

    // A foreign dataset folder: the sniffer infers a manifest and leaves
    // what it cannot determine in `_unresolved`, which blocks Run.
    sniffFolder: async () => {
      if (!desktop) {
        report(needsTauri("Folder sniffing"));
        return false;
      }
      const seq = ++activeLoadSeq;
      try {
        const folder = await pickFolder("Pick a dataset folder to sniff");
        if (folder === null || seq !== activeLoadSeq) return false;
        const spec = await api.sniffFolder({ folder });
        if (seq !== activeLoadSeq) return false;
        await loadSetup({
          manifest: spec,
          manifestDir: folder,
          manifestPath: null,
          presetId: null,
        });
        if (seq !== activeLoadSeq) return false;
        report({ kind: "idle" });
        return true;
      } catch (e) {
        if (seq === activeLoadSeq) {
          report({
            kind: "error",
            category: "sniff",
            message: `Sniff failed: ${String(e)}`,
          });
        }
        return false;
      }
    },

    pickDatasetFolder: async () => {
      if (!desktop) {
        report(needsTauri("The folder picker"));
        return;
      }
      const seq = ++activeLoadSeq;
      try {
        const folder = await pickFolder("Pick the dataset folder");
        if (folder !== null && seq === activeLoadSeq) {
          setPaths({ manifestDir: folder });
        }
      } catch (e) {
        if (seq === activeLoadSeq) {
          report({ kind: "error", category: "dialog", message: String(e) });
        }
      }
    },

    pickManifest: async () => {
      if (!desktop) {
        report(needsTauri("The file picker"));
        return;
      }
      const seq = ++activeLoadSeq;
      try {
        const path = await pickManifestFile();
        if (path === null || seq !== activeLoadSeq) return;
        setPaths({ manifestDir: dirnamePath(path), manifestPath: path });
        try {
          const manifest = await api.readManifest({ path });
          if (seq !== activeLoadSeq) return;
          setManifest(manifest);
          report({ kind: "idle" });
        } catch (e) {
          if (seq === activeLoadSeq) {
            report({
              kind: "error",
              category: "manifest_parse",
              message: `Could not load manifest "${path}": ${String(e)}`,
            });
          }
        }
      } catch (e) {
        if (seq === activeLoadSeq) {
          report({ kind: "error", category: "dialog", message: String(e) });
        }
      }
    },
  };
}
