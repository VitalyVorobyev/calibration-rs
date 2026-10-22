/** The ways a Run setup is loaded: a Quick Start preset, a sniffed folder,
 * a picked folder or a picked manifest file. Each failure becomes a run
 * status the banner shows. */
import { invoke } from "@tauri-apps/api/core";
import * as TOML from "toml";
import { dirnamePath, isTauriContext, joinPath, repoRoot } from "../../lib/tauri";
import { useRunSetup } from "../../store/runSetup";
import { pickFolder, pickManifestFile } from "../../shell/dialogs";
import { mergeConfig, type EnabledPreset } from "./presets";
import type { RunStatus } from "./runStatus";

type Report = (status: RunStatus) => void;

const needsTauri = (what: string): RunStatus => ({
  kind: "error",
  category: "no_tauri",
  message: `${what} requires the Tauri runtime (bun run tauri dev).`,
});

/** Parse a manifest file's text by its extension: TOML or JSON. */
export function parseManifest(path: string, text: string): unknown {
  const ext = path.split(".").pop()?.toLowerCase();
  // TOML.parse / JSON.parse return `any`; narrow to `unknown` immediately.
  const parsed: unknown = ext === "toml" ? TOML.parse(text) : JSON.parse(text);
  return parsed;
}

export interface SetupActions {
  applyPreset: (preset: EnabledPreset) => Promise<boolean>;
  sniffFolder: () => Promise<boolean>;
  pickDatasetFolder: () => Promise<void>;
  pickManifest: () => Promise<void>;
}

/** The setup loaders, reporting failures (and the idle state after a
 * successful load) through `report`. */
export function setupActions(report: Report): SetupActions {
  const inTauri = isTauriContext();
  const { loadSetup, setPaths, setManifest } = useRunSetup.getState();

  return {
    applyPreset: async (preset) => {
      if (!inTauri) {
        report(needsTauri("Preset loading"));
        return false;
      }
      try {
        // Preset manifest paths are repo-root-relative (presets.ts).
        const root = await repoRoot();
        if (!root) {
          report({
            kind: "error",
            category: "no_tauri",
            message: "Could not resolve the workspace repo root (repo_root_cmd failed).",
          });
          return false;
        }
        const path = joinPath(root, preset.manifestPath);
        const parsed = parseManifest(
          path,
          await invoke<string>("load_text_file", { path }),
        );
        await loadSetup({
          manifest: preset.manifestOverrides
            ? mergeConfig(parsed, preset.manifestOverrides)
            : parsed,
          manifestDir: dirnamePath(path),
          manifestPath: path,
          presetId: preset.id,
          configOverrides: preset.configOverrides,
        });
        report({ kind: "idle" });
        return true;
      } catch (e) {
        report({
          kind: "error",
          category: "preset_load",
          message: `Failed to load preset "${preset.name}": ${String(e)}`,
        });
        return false;
      }
    },

    // A foreign dataset folder: the sniffer infers a manifest and leaves
    // what it cannot determine in `_unresolved`, which blocks Run.
    sniffFolder: async () => {
      if (!inTauri) {
        report(needsTauri("Folder sniffing"));
        return false;
      }
      try {
        const folder = await pickFolder("Pick a dataset folder to sniff");
        if (folder === null) return false;
        const spec = await invoke<Record<string, unknown>>("sniff_folder", { folder });
        await loadSetup({
          manifest: spec,
          manifestDir: folder,
          manifestPath: null,
          presetId: null,
        });
        report({ kind: "idle" });
        return true;
      } catch (e) {
        report({
          kind: "error",
          category: "sniff",
          message: `Sniff failed: ${String(e)}`,
        });
        return false;
      }
    },

    pickDatasetFolder: async () => {
      if (!inTauri) {
        report(needsTauri("The folder picker"));
        return;
      }
      try {
        const folder = await pickFolder("Pick the dataset folder");
        if (folder !== null) setPaths({ manifestDir: folder });
      } catch (e) {
        report({ kind: "error", category: "dialog", message: String(e) });
      }
    },

    pickManifest: async () => {
      if (!inTauri) {
        report(needsTauri("The file picker"));
        return;
      }
      try {
        const path = await pickManifestFile();
        if (path === null) return;
        setPaths({ manifestDir: dirnamePath(path), manifestPath: path });
        try {
          setManifest(
            parseManifest(path, await invoke<string>("load_text_file", { path })),
          );
        } catch {
          // Unreadable or malformed: keep the form as it is for the user to fix.
        }
      } catch (e) {
        report({ kind: "error", category: "dialog", message: String(e) });
      }
    },
  };
}
