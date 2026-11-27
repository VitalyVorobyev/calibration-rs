/** The shell's path questions. Each opens the in-app path picker over the
 * back end's root folders (on the desktop and in the browser alike) and
 * resolves to the chosen path, or `null` when the user cancels. */
import { requestPath } from "../components/PathPicker/pickerStore";

/** Choose one folder. */
export function pickFolder(title: string): Promise<string | null> {
  return requestPath({ title, mode: "folder" });
}

/** Choose a project folder. */
export function pickProject(): Promise<string | null> {
  return requestPath({ title: "Open a project", mode: "project" });
}

/** Choose a calibration export (`export.json`, `calibration.json`). */
export function pickExportFile(): Promise<string | null> {
  return requestPath({ title: "Open a calibration export", mode: "export" });
}

/** Choose a dataset manifest file. */
export function pickManifestFile(): Promise<string | null> {
  return requestPath({
    title: "Pick a dataset.toml or dataset.json manifest",
    mode: "manifest",
  });
}

/** Choose any file (a robot pose file, an image list). */
export function pickFile(title: string): Promise<string | null> {
  return requestPath({ title, mode: "file" });
}
