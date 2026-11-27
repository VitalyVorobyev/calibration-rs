import {
  Button,
  DropdownMenu,
  MenuItem,
  MenuLabel,
  MenuSeparator,
  toast,
} from "@vitavision/ui";
import {
  ChevronDown,
  CircuitBoard,
  FileInput,
  FolderOpen,
  FolderPlus,
  RefreshCw,
  Save,
  X,
} from "lucide-react";
import { isDirty, useDraft } from "../project/draftStore";
import { useProjectDialog } from "../project/dialogs";
import { openProject } from "../project/projectActions";
import { folderName } from "../project/projectModel";
import { useProject } from "../project/projectStore";
import { useLibraryScan } from "../project/useLibraryScan";
import { useReadOnly } from "../session/jobs";
import { pickManifestFile } from "./dialogs";

/** The project commands: open, new (from a dataset folder or a manifest),
 * save the unsaved edits, reload, import an export as a run, close, and
 * reopen a recent project (the back end's list). What writes is disabled
 * while the connection is lost. A failure lands in the project store's
 * `error`, which the shell shows. */
export function ProjectMenu() {
  const project = useProject((s) => s.project);
  const recent = useProject((s) => s.recent);
  const dirty = useDraft(isDirty);
  const readOnly = useReadOnly();
  const show = useProjectDialog((s) => s.show);
  const scan = useLibraryScan();
  // The open project's device's other projects: one click to switch.
  const device = project?.spec.device ?? null;
  const siblings = (scan.data?.projects ?? []).filter(
    (p) =>
      device !== null &&
      p.dir !== project?.dir &&
      p.device?.id === device.id &&
      (p.device?.product ?? null) === (device.product ?? null),
  );

  const handleSave = async () => {
    if (await useDraft.getState().save()) {
      toast({ title: "Project saved", tone: "success" });
    }
  };

  return (
    <DropdownMenu
      align="start"
      trigger={
        <Button
          size="sm"
          variant="ghost"
          aria-label="Project"
          title={project?.dir ?? "No project open"}
        >
          <span className="max-w-[16rem] truncate font-semibold">
            {project ? project.name : "No project"}
          </span>
          <ChevronDown aria-hidden className="size-3.5" />
        </Button>
      }
    >
      <MenuItem icon={<FolderOpen />} onSelect={() => void openProject()}>
        Open project…
      </MenuItem>
      <MenuItem
        icon={<FolderPlus />}
        disabled={readOnly}
        onSelect={() => show({ kind: "new" })}
      >
        New project…
      </MenuItem>
      <MenuItem
        icon={<FolderPlus />}
        disabled={readOnly}
        onSelect={() =>
          void pickManifestFile().then((manifest) => {
            if (manifest !== null) show({ kind: "new", manifest });
          })
        }
      >
        New project from a manifest…
      </MenuItem>
      <MenuSeparator />
      <MenuItem
        icon={<Save />}
        disabled={!project || !dirty || readOnly}
        onSelect={() => void handleSave()}
      >
        Save project
      </MenuItem>
      <MenuItem
        icon={<RefreshCw />}
        disabled={!project}
        onSelect={() => void useProject.getState().reload()}
      >
        Reload project
      </MenuItem>
      <MenuItem
        icon={<FileInput />}
        disabled={!project || readOnly}
        onSelect={() => project && show({ kind: "import", dir: project.dir })}
      >
        Import an export as a run…
      </MenuItem>
      <MenuItem
        icon={<CircuitBoard />}
        disabled={!project || readOnly}
        onSelect={() => project && show({ kind: "device", dir: project.dir })}
      >
        Set device…
      </MenuItem>
      <MenuItem
        icon={<X />}
        disabled={!project}
        onSelect={() => useProject.getState().close()}
      >
        Close project
      </MenuItem>
      {siblings.length > 0 && (
        <>
          <MenuSeparator />
          <MenuLabel>Same device</MenuLabel>
          {siblings.map((p) => (
            <MenuItem key={p.dir} onSelect={() => void openProject(p.dir)}>
              <span className="flex flex-col">
                <span>{p.name}</span>
                <span className="font-mono text-[10px] text-fg-muted">{p.dir}</span>
              </span>
            </MenuItem>
          ))}
        </>
      )}
      {recent.length > 0 && (
        <>
          <MenuSeparator />
          <MenuLabel>Recent</MenuLabel>
          {recent.map((dir) => (
            <MenuItem key={dir} onSelect={() => void openProject(dir)}>
              <span className="flex flex-col">
                <span>{folderName(dir)}</span>
                <span className="font-mono text-[10px] text-fg-muted">{dir}</span>
              </span>
            </MenuItem>
          ))}
        </>
      )}
    </DropdownMenu>
  );
}
