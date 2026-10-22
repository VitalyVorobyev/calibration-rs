import { Button, ThemeToggle } from "@vitavision/ui";
import { Logo } from "../components/Logo";
import { isTauriContext } from "../lib/tauri";
import { THEME_STORAGE_KEY } from "../lib/theme";
import { folderName } from "../project/projectModel";
import { useStore } from "../store";
import { exportKindLabel } from "../store/exportKind";
import { pickExportFile } from "./dialogs";
import { ProjectMenu } from "./ProjectMenu";

const NO_TAURI =
  "Tauri runtime not detected. Launch the app with `bun run tauri dev` " +
  "(or the bundled binary). Plain `bun run dev` only starts Vite, so " +
  "the file-dialog and IPC commands aren't wired up.";

/** The top bar: the app's name and the loaded export's kind, the project
 * menu, the loaded export and Open Export…, and the theme toggle. */
export function ShellHeader() {
  const exportPath = useStore((s) => s.exportPath);
  const kind = useStore((s) => s.kind);
  const loadExport = useStore((s) => s.loadExport);
  const setLoadError = useStore((s) => s.setLoadError);
  const tauriOk = isTauriContext();

  const handleOpenExport = async () => {
    setLoadError(null);
    if (!tauriOk) {
      setLoadError(NO_TAURI);
      return;
    }
    let chosen: string | null;
    try {
      chosen = await pickExportFile();
    } catch (e) {
      setLoadError(`File dialog error: ${String(e)}`);
      return;
    }
    if (chosen) await loadExport(chosen);
  };

  return (
    <div className="flex items-center justify-between gap-4 px-4 py-2.5">
      <div className="flex items-center gap-3">
        <Logo size={24} className="text-fg" />
        <div className="flex flex-col leading-tight">
          <h1 className="m-0 text-sm font-semibold tracking-tight">calibration-rs</h1>
          <span className="text-[10px] tracking-wider text-fg-muted uppercase">
            {kind ? exportKindLabel(kind) : "diagnose · 3d · epipolar · depth · run"}
          </span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <ProjectMenu enabled={tauriOk} />
        <span
          className="hidden max-w-[28rem] truncate font-mono text-[11px] text-fg-muted sm:inline"
          title={exportPath ?? ""}
        >
          {exportPath ? folderName(exportPath) : "No export loaded"}
        </span>
        <Button size="sm" onClick={() => void handleOpenExport()} disabled={!tauriOk}>
          Open Export…
        </Button>
        <ThemeToggle storageKey={THEME_STORAGE_KEY} />
      </div>
    </div>
  );
}
