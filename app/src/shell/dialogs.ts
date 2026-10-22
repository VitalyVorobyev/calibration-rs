/** Native file dialogs the shell opens. Each resolves to the chosen path,
 * or `null` when the user cancels. */
import { open } from "@tauri-apps/plugin-dialog";

/** Choose one folder. */
export async function pickFolder(title: string): Promise<string | null> {
  const chosen = await open({ directory: true, multiple: false, title });
  return typeof chosen === "string" ? chosen : null;
}

/** Choose a calibration `export.json`. */
export async function pickExportFile(): Promise<string | null> {
  const chosen = await open({
    multiple: false,
    directory: false,
    title: "Open a calibration export",
    filters: [{ name: "Calibration export", extensions: ["json"] }],
  });
  return typeof chosen === "string" ? chosen : null;
}

/** Choose a dataset manifest file. */
export async function pickManifestFile(): Promise<string | null> {
  const chosen = await open({
    multiple: false,
    title: "Pick a dataset.toml or dataset.json manifest",
    filters: [
      { name: "Manifest", extensions: ["toml", "json"] },
      { name: "All files", extensions: ["*"] },
    ],
  });
  return typeof chosen === "string" ? chosen : null;
}
