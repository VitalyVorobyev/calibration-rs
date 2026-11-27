import { Callout, ErrorBox, Field, Input, Switch } from "@vitavision/ui";
import { useEffect } from "react";
import { relativePath } from "../../../lib/paths";
import { PathField } from "../../../components/PathField";
import { deviceIdentity } from "../../../project/deviceDraft";
import type { ProjectError } from "../../../project/projectModel";
import { userRoots, useSession } from "../../../session/session";
import {
  dataFolder,
  defaultName,
  openFields,
  resolveProjectDir,
} from "./newProjectModel";
import type { NewProjectApi } from "./useNewProject";

/** Step 5: name the project and say where its file goes: by default
 * `<first library root>/<device type>/<name>`, or in the data folder
 * itself. The project file only references the data; it is the one thing
 * written. A folder that already holds a project is refused by the back
 * end and said so here, with what was entered kept. */
export function CreateStep({
  project,
  error,
}: {
  project: NewProjectApi;
  error: ProjectError | null;
}) {
  const { state, patch } = project;
  const { source, manifest } = state;
  const status = useSession((s) => s.status);
  const roots = status.kind === "ready" ? userRoots(status.info) : [];
  const name = state.name ?? (source ? defaultName(source) : "");
  const folder = source ? dataFolder(source) : "";
  const projectDir = source
    ? resolveProjectDir({ ...state, source }, roots[0] ?? null)
    : "";

  // The default follows the name and device type until the user types a path.
  useEffect(() => {
    if (state.name === null) patch({ name });
    // Only the first time in this step.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const open = manifest ? openFields(manifest) : [];
  const device = deviceIdentity(state.device);
  return (
    <div className="flex max-w-xl flex-col gap-4">
      <Field label="Project name" description="Shown in the Library and the context bar.">
        <Input
          value={name}
          onChange={(e) => patch({ name: e.target.value, projectDir: null })}
        />
      </Field>

      <Field
        as="group"
        label="Project folder"
        description="Created if it does not exist. It holds the project file and, later, the runs."
      >
        <PathField
          aria-label="Project folder"
          kind="folder"
          value={projectDir}
          disabled={state.inDataFolder}
          onValueChange={(v) => patch({ projectDir: v })}
        />
      </Field>
      <Switch
        checked={state.inDataFolder}
        onCheckedChange={(inDataFolder) => patch({ inDataFolder })}
        label="Keep the project in the data folder"
        description="The project file and runs are written beside the images."
      />

      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 rounded-control border border-line bg-surface p-3 text-xs">
        <dt className="text-fg-subtle">Data</dt>
        <dd className="font-mono text-fg">
          {state.source?.kind === "manifest" && !state.source.embed
            ? `references ${relativePath(projectDir, state.source.path)}`
            : `${relativePath(projectDir, folder)} (data folder, relative to the project)`}
        </dd>
        <dt className="text-fg-subtle">Device</dt>
        <dd className="text-fg">
          {device
            ? `${device.product ? `${device.product} · ` : ""}${device.id}`
            : "none: listed under “(no device)”"}
        </dd>
      </dl>

      {open.length > 0 && (
        <Callout tone="warning" title="Fields still open">
          {open.join(", ")} {open.length === 1 ? "is" : "are"} not decided. The project is
          created and the Dataset screen asks again; a calibration refuses to run until
          they are set.
        </Callout>
      )}
      {error && (
        <ErrorBox>
          {error.kind === "project_exists"
            ? "That folder already holds a project. Choose another folder, or open it from the Library."
            : error.message}
        </ErrorBox>
      )}
    </div>
  );
}
