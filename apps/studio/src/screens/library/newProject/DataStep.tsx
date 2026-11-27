import { Button, ErrorBox, SegmentedControl } from "@vitavision/ui";
import { FileText, FolderOpen, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api, asStudioError } from "../../../api";
import { pickFolder, pickManifestFile } from "../../../shell/dialogs";
import { dataFolder, openFields, type DataSource } from "./newProjectModel";
import type { NewProjectApi } from "./useNewProject";

/** Step 1: where the data is. A folder of images (the library infers its
 * layout; the folder is only read) or an existing manifest (referenced, or
 * copied into the project to edit it). Choosing reads it at once, so the
 * next step starts with a draft. */
export function DataStep({
  project,
  initialManifest,
}: {
  project: NewProjectApi;
  /** A manifest to start from (the Library's "from a manifest"). */
  initialManifest?: string | undefined;
}) {
  const { state, patch } = project;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const read = async (source: DataSource) => {
    setBusy(true);
    setError(null);
    try {
      const manifest =
        source.kind === "folder"
          ? (await api.sniffFolder({ folder: source.folder })).spec
          : await api.readManifest({ path: source.path });
      patch({ source, manifest, name: null, projectDir: null, reached: 1 });
    } catch (e) {
      setError(asStudioError(e).message);
    } finally {
      setBusy(false);
    }
  };

  // Started from a manifest: read it at once, once.
  useEffect(() => {
    if (initialManifest && !state.source) {
      void read({ kind: "manifest", path: initialManifest, embed: false });
    }
    // Only on opening.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const chooseFolder = async () => {
    const folder = await pickFolder("Pick the folder of images");
    if (folder !== null) await read({ kind: "folder", folder });
  };
  const chooseManifest = async () => {
    const path = await pickManifestFile();
    if (path !== null) await read({ kind: "manifest", path, embed: false });
  };

  const source = state.source;
  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-fg-muted">
        Where is the calibration data? The folder is only read; the project file is
        written in step 5.
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Choice
          icon={<FolderOpen aria-hidden className="size-5" />}
          title="A folder of images"
          note="The layout is inferred: cameras, views, robot poses, target."
          active={source?.kind === "folder"}
          disabled={busy}
          onClick={() => void chooseFolder()}
        />
        <Choice
          icon={<FileText aria-hidden className="size-5" />}
          title="An existing dataset manifest"
          note="A dataset.toml or dataset.json that already describes the data."
          active={source?.kind === "manifest"}
          disabled={busy}
          onClick={() => void chooseManifest()}
        />
      </div>

      {busy && (
        <p role="status" className="flex items-center gap-2 text-xs text-fg-muted">
          <Loader2 aria-hidden className="size-3.5 animate-spin" />
          Reading the data…
        </p>
      )}
      {error && <ErrorBox>{error}</ErrorBox>}

      {source && state.manifest && !busy && (
        <div className="flex flex-col gap-3 rounded-control border border-line bg-surface p-3">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
            <dt className="text-fg-subtle">
              {source.kind === "folder" ? "Folder" : "Manifest"}
            </dt>
            <dd
              className="truncate font-mono text-fg"
              title={source.kind === "folder" ? source.folder : source.path}
            >
              {source.kind === "folder" ? source.folder : source.path}
            </dd>
            <dt className="text-fg-subtle">Found</dt>
            <dd className="text-fg">
              {state.manifest.cameras.length}{" "}
              {state.manifest.cameras.length === 1 ? "camera" : "cameras"}
              {openFields(state.manifest).length > 0
                ? `, ${openFields(state.manifest).length} to decide in the next step`
                : ", nothing left open"}
            </dd>
            {source.kind === "manifest" && (
              <>
                <dt className="text-fg-subtle">Data folder</dt>
                <dd className="truncate font-mono text-fg">{dataFolder(source)}</dd>
              </>
            )}
          </dl>
          {source.kind === "manifest" && (
            <div className="flex flex-col gap-1.5">
              <SegmentedControl
                aria-label="How the project holds the manifest"
                value={source.embed ? "embed" : "reference"}
                onValueChange={(v) =>
                  patch({ source: { ...source, embed: v === "embed" } })
                }
                options={[
                  { value: "reference", label: "Reference it" },
                  { value: "embed", label: "Copy into the project" },
                ]}
              />
              <p className="text-xs text-fg-muted">
                {source.embed
                  ? "The manifest is written into the project file, where it can be edited."
                  : "The project points at the file, which stays as it is. It is shown read-only."}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Choice({
  icon,
  title,
  note,
  active,
  disabled,
  onClick,
}: {
  icon: React.ReactNode;
  title: string;
  note: string;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <Button
      variant="secondary"
      disabled={disabled}
      onClick={onClick}
      aria-pressed={active}
      className="h-auto flex-col items-start gap-1.5 p-3 text-left whitespace-normal data-[active=true]:ring-2 data-[active=true]:ring-signal"
      data-active={active}
    >
      <span className="flex items-center gap-2 text-sm font-medium text-fg">
        {icon}
        {title}
      </span>
      <span className="text-xs font-normal text-fg-muted">{note}</span>
    </Button>
  );
}
