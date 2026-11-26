/** Native file dialogs the shell opens (desktop only). Each resolves to the
 * chosen path, or `null` when the user cancels. */
import { openDialog } from "../api";

/** Choose one folder. */
export function pickFolder(title: string): Promise<string | null> {
  return openDialog({ directory: true, title });
}

/** Choose a calibration `export.json`. */
export function pickExportFile(): Promise<string | null> {
  return openDialog({
    title: "Open a calibration export",
    filters: [{ name: "Calibration export", extensions: ["json"] }],
  });
}

/** Choose a dataset manifest file. */
export function pickManifestFile(): Promise<string | null> {
  return openDialog({
    title: "Pick a dataset.toml or dataset.json manifest",
    filters: [
      { name: "Manifest", extensions: ["toml", "json"] },
      { name: "All files", extensions: ["*"] },
    ],
  });
}
