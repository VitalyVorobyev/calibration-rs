import { Button, ThemeToggle } from "@vitavision/ui";
import { Settings } from "lucide-react";
import { useState } from "react";
import { Logo } from "../components/Logo";
import { THEME_STORAGE_KEY } from "../lib/theme";
import { useStore } from "../store";
import { exportKindLabel } from "../store/exportKind";
import { importExport } from "../project/projectActions";
import { useProject } from "../project/projectStore";
import { ProjectMenu } from "./ProjectMenu";
import { useSession } from "./session";
import { SettingsDialog } from "./SettingsDialog";

/** The top bar: the app's name and the loaded export's kind, then the
 * project menu, the run shown and Import Export…, Settings (the root
 * folders) and the theme toggle. */
export function ShellHeader() {
  const run = useStore((s) => s.run);
  const kind = useStore((s) => s.kind);
  const setLoadError = useStore((s) => s.setLoadError);
  const project = useProject((s) => s.project);
  const offline = useSession((s) => s.status.kind === "offline");
  const [settings, setSettings] = useState(false);

  const handleImport = async () => {
    setLoadError(null);
    await importExport();
  };

  const label = offline
    ? "No back end"
    : run
      ? `${project?.dir === run.dir ? project.name : run.dir} · ${run.id}`
      : "No run open";

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
        <ProjectMenu />
        <span
          className="hidden max-w-[28rem] truncate font-mono text-[11px] text-fg-muted sm:inline"
          title={run ? `${run.dir} · ${run.id}` : ""}
        >
          {label}
        </span>
        <Button
          size="sm"
          disabled={!project}
          title={
            project
              ? "Add a calibration export to the project as its next run"
              : "Open a project first: an export is imported into one"
          }
          onClick={() => void handleImport()}
        >
          Import Export…
        </Button>
        <Button
          size="sm"
          variant="ghost"
          aria-label="Settings"
          title="Settings: the folders the app may read"
          onClick={() => setSettings(true)}
        >
          <Settings aria-hidden className="size-4" />
        </Button>
        <ThemeToggle storageKey={THEME_STORAGE_KEY} />
      </div>
      <SettingsDialog open={settings} onOpenChange={setSettings} />
    </div>
  );
}
