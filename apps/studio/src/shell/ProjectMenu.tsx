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
  FileInput,
  FolderOpen,
  FolderPlus,
  RefreshCw,
  Save,
  X,
} from "lucide-react";
import { isDirty, useDraft } from "../project/draftStore";
import {
  importExport,
  newProjectFromFolder,
  newProjectFromManifest,
  openProject,
} from "../project/projectActions";
import { folderName } from "../project/projectModel";
import { useProject } from "../project/projectStore";

/** The header's project commands: new (from a dataset folder or a
 * manifest), open, save the unsaved edits, reload, import an export as a
 * run, close, and reopen a recent project (the back end's list). A failure
 * lands in the project store's `error`, which the shell shows. */
export function ProjectMenu() {
  const project = useProject((s) => s.project);
  const recent = useProject((s) => s.recent);
  const dirty = useDraft(isDirty);

  const handleSave = async () => {
    if (await useDraft.getState().save()) {
      toast({ title: "Project saved", tone: "success" });
    }
  };

  return (
    <DropdownMenu
      align="end"
      trigger={
        <Button size="sm" title={project?.dir ?? "No project open"}>
          <span className="max-w-[14rem] truncate">
            {project ? project.name : "Project"}
            {dirty ? " •" : ""}
          </span>
          <ChevronDown aria-hidden className="size-3.5" />
        </Button>
      }
    >
      <MenuItem icon={<FolderOpen />} onSelect={() => void openProject()}>
        Open project…
      </MenuItem>
      <MenuItem icon={<FolderPlus />} onSelect={() => void newProjectFromFolder()}>
        New project from a dataset folder…
      </MenuItem>
      <MenuItem icon={<FolderPlus />} onSelect={() => void newProjectFromManifest()}>
        New project from a manifest…
      </MenuItem>
      <MenuSeparator />
      <MenuItem
        icon={<Save />}
        disabled={!project || !dirty}
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
        disabled={!project}
        onSelect={() => void importExport()}
      >
        Import an export as a run…
      </MenuItem>
      <MenuItem
        icon={<X />}
        disabled={!project}
        onSelect={() => useProject.getState().close()}
      >
        Close project
      </MenuItem>
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
