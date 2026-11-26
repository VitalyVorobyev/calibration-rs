import { Button, ThemeToggle } from "@vitavision/ui";
import { Logo } from "../components/Logo";
import { capabilities, NO_BACKEND } from "../api";
import { THEME_STORAGE_KEY } from "../lib/theme";
import { folderName } from "../project/projectModel";
import { useStore } from "../store";
import { exportKindLabel } from "../store/exportKind";
import { pickExportFile } from "./dialogs";
import { ProjectMenu } from "./ProjectMenu";

/** The top bar: the app's name and the loaded export's kind, then — in the
 * desktop app — the project menu, the loaded export and Open Export…; in
 * the browser viewer, the run it serves; and the theme toggle. */
export function ShellHeader() {
  const exportPath = useStore((s) => s.exportPath);
  const exportDir = useStore((s) => s.exportDir);
  const kind = useStore((s) => s.kind);
  const loadExport = useStore((s) => s.loadExport);
  const setLoadError = useStore((s) => s.setLoadError);
  const { desktop, viewer } = capabilities();

  const handleOpenExport = async () => {
    setLoadError(null);
    let chosen: string | null;
    try {
      chosen = await pickExportFile();
    } catch (e) {
      setLoadError(`File dialog error: ${String(e)}`);
      return;
    }
    if (chosen) await loadExport(chosen);
  };

  // The desktop names the export file; the browser viewer serves one run,
  // named by its folder.
  const loaded = desktop
    ? exportPath && folderName(exportPath)
    : exportDir && folderName(exportDir);
  const label = !viewer ? "No back end" : (loaded ?? "No export loaded");

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
        {desktop && <ProjectMenu />}
        <span
          className="hidden max-w-[28rem] truncate font-mono text-[11px] text-fg-muted sm:inline"
          title={viewer ? (exportPath ?? "") : NO_BACKEND}
        >
          {label}
        </span>
        {desktop && (
          <Button size="sm" onClick={() => void handleOpenExport()}>
            Open Export…
          </Button>
        )}
        <ThemeToggle storageKey={THEME_STORAGE_KEY} />
      </div>
    </div>
  );
}
