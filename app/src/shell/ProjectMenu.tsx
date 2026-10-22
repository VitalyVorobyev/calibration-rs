import {
  Button,
  DropdownMenu,
  MenuItem,
  MenuLabel,
  MenuSeparator,
  toast,
} from "@vitavision/ui";
import { ChevronDown, FolderOpen, FolderPlus, Save, X } from "lucide-react";
import { useProject } from "../project/projectStore";
import { folderName } from "../project/projectModel";
import { useRunSetup } from "../store/runSetup";
import { pickFolder } from "./dialogs";

interface ProjectMenuProps {
  /** False outside the Tauri shell: no dialogs, no file system. */
  enabled: boolean;
}

/** The header's project commands: new, open, save, close, and reopen a
 * recent project. A failure lands in the project store's `error`, which the
 * shell shows. */
export function ProjectMenu({ enabled }: ProjectMenuProps) {
  const project = useProject((s) => s.project);
  const recent = useProject((s) => s.recent);
  const hasDataset = useRunSetup((s) => s.manifestDir !== null);

  const handleNew = async () => {
    const dir = await pickFolder("Choose the folder for the new project");
    if (dir === null) return;
    if (await useProject.getState().create(dir)) {
      toast({ title: "Project created", description: dir, tone: "success" });
    }
  };

  const handleOpen = async (dir?: string) => {
    const chosen = dir ?? (await pickFolder("Open a project folder"));
    if (chosen === null) return;
    await useProject.getState().open(chosen);
  };

  const handleSave = async () => {
    if (await useProject.getState().saveManifest()) {
      toast({ title: "Manifest saved", tone: "success" });
    }
  };

  return (
    <DropdownMenu
      align="end"
      trigger={
        <Button size="sm" disabled={!enabled} title={project?.dir ?? "No project open"}>
          <span className="max-w-[14rem] truncate">
            {project ? folderName(project.dir) : "Project"}
          </span>
          <ChevronDown aria-hidden className="size-3.5" />
        </Button>
      }
    >
      <MenuItem
        icon={<FolderPlus />}
        disabled={!hasDataset}
        onSelect={() => void handleNew()}
      >
        New project from this dataset…
      </MenuItem>
      <MenuItem icon={<FolderOpen />} onSelect={() => void handleOpen()}>
        Open project…
      </MenuItem>
      <MenuItem icon={<Save />} disabled={!project} onSelect={() => void handleSave()}>
        Save manifest
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
            <MenuItem key={dir} onSelect={() => void handleOpen(dir)}>
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
