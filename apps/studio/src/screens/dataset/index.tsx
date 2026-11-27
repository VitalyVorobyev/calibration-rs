import type { JsonSchema } from "@vitavision/forms";
import { Button, Callout, DensityProvider, ErrorBox, SkeletonRows } from "@vitavision/ui";
import { useEffect, useMemo } from "react";
import { api } from "../../api";
import type { ProjectView } from "../../api/generated/types";
import { dirnamePath, joinPath, relativePath } from "../../lib/paths";
import { useDraft } from "../../project/draftStore";
import { useDatasetProblems } from "../../project/problems";
import { withManifest } from "../../project/projectModel";
import { useProject } from "../../project/projectStore";
import { useProjectViews } from "../../project/useProjectViews";
import { useQuery } from "../../session/query";
import { useSelection } from "../../session/selection";
import { useStore } from "../../store";
import { countProblems, derivedWarnings, placeMessages } from "./datasetModel";
import { DatasetForm } from "./DatasetForm";
import { MatrixPane } from "./MatrixPane";
import { SaveBar } from "./SaveBar";
import { useViewKeys } from "./useViewKeys";
import { ViewDetail } from "./ViewDetail";

/** The Dataset screen: how the data is laid out, as a form on the left, the
 * camera × view matrix of what that description finds in the middle, and the
 * selected image on the right. The description is an explicit draft: every
 * edit asks the library again what the layout and its problems are (400 ms
 * after the last), Save writes the manifest into the project file. A
 * manifest that is a file of its own is shown, never written. */
export function DatasetScreen() {
  const project = useProject((s) => s.project);
  return project === null ? null : <DatasetBody project={project} />;
}

function DatasetBody({ project }: { project: ProjectView }) {
  const manifest = useDraft((s) => s.manifest);
  const baseline = useDraft((s) => s.baseManifest);
  const dirty = useDraft((s) => s.manifestDirty);
  const editable = useDraft((s) => s.editable);
  const setManifest = useDraft((s) => s.setManifest);
  const selection = useSelection((s) => s.selection);
  const setSelection = useSelection((s) => s.setSelection);
  const noRun = useStore((s) => s.data === null);
  const projectError = useProject((s) => s.error);
  const form = useQuery("dataset_form", {}, () => api.datasetForm());
  const schema = form.data?.schema as JsonSchema | undefined;

  // The draft judged by the library, in place of the saved project when the
  // manifest is the project's own.
  const spec = useMemo(
    () => (editable && manifest ? withManifest(project.spec, manifest) : null),
    [editable, manifest, project.spec],
  );
  const views = useProjectViews(
    project.dir,
    spec,
    true,
    // A referenced manifest changes with its file, not the project's.
    project.input_digest?.dataset ?? 0,
  );
  const reply = views.data;

  const messages = useMemo(
    () => [
      ...placeMessages(reply?.messages ?? [], manifest),
      ...derivedWarnings(reply, manifest),
    ],
    [reply, manifest],
  );
  const { errors, warnings } = countProblems(messages);
  const loose = messages.filter((m) => m.path === null);

  // The rail's badge counts what is wrong.
  useEffect(() => {
    useDatasetProblems.getState().set(project.dir, errors);
  }, [project.dir, errors]);

  // A save over a newer file is this screen's to explain (the draft keeps
  // the edits), not the shell's strip: take the project's error and read
  // the file again, which the draft answers with its own banner.
  useEffect(() => {
    if (projectError?.kind === "changed_on_disk") {
      useProject.getState().clearError();
      void useProject.getState().reload();
    }
  }, [projectError]);

  useViewKeys(noRun, reply?.views.length ?? 0, reply?.cameras.length ?? 0);

  const view = selection.view !== null ? (reply?.views[selection.view] ?? null) : null;
  const unreadable = project.dataset.kind === "unreadable" ? project.dataset.error : null;

  return (
    <DensityProvider value="compact">
      <div className="grid h-full min-h-0 grid-cols-[minmax(21rem,26rem)_minmax(0,1fr)_minmax(17rem,22rem)] gap-3">
        <section
          aria-label="Description of the dataset"
          className="flex min-h-0 flex-col overflow-hidden rounded-panel border border-line bg-surface"
        >
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            {unreadable && (
              <ErrorBox className="mb-3">
                The manifest cannot be read: {unreadable.message}
              </ErrorBox>
            )}
            {!editable && manifest && <ReferencedNotice project={project} />}
            {loose.length > 0 && (
              <ul
                aria-label="Problems with the dataset"
                className="mb-3 flex flex-col gap-1.5 rounded-control border border-defect/30 bg-defect/8 p-2.5 text-xs"
              >
                {loose.map((m) => (
                  <li key={m.text} className="flex gap-2 text-fg">
                    <span
                      aria-hidden
                      className="mt-1.5 size-1.5 shrink-0 rounded-full bg-defect"
                    />
                    <span className="min-w-0 break-words">{m.text}</span>
                  </li>
                ))}
              </ul>
            )}
            {manifest && schema ? (
              <DatasetForm
                schema={schema}
                manifest={manifest}
                baseline={baseline}
                onChange={setManifest}
                messages={messages}
                views={reply}
                dataRoot={project.data_root}
                disabled={!editable}
              />
            ) : unreadable ? null : (
              <div role="status" aria-busy aria-label="Loading the description">
                <SkeletonRows rows={6} />
              </div>
            )}
          </div>
          <SaveBar
            dirty={dirty}
            errors={errors}
            warnings={warnings}
            editable={editable}
          />
        </section>

        <section aria-label="Views" className="flex min-h-0 min-w-0 flex-col">
          <MatrixPane
            reply={reply}
            loading={views.loading}
            manifest={manifest}
            dataRoot={project.data_root}
            selection={selection}
            onSelect={(camera, v) => setSelection({ camera, view: v, feature: null })}
          />
        </section>

        <aside
          aria-label="Selected image"
          className="min-h-0 overflow-y-auto rounded-panel border border-line bg-surface p-3"
        >
          <ViewDetail
            view={view}
            cameraIndex={selection.camera}
            cameras={reply?.cameras ?? []}
            dataRoot={project.data_root}
            manifest={manifest}
            onCamera={(camera) =>
              setSelection({ camera, view: selection.view, feature: null })
            }
          />
        </aside>
      </div>
    </DensityProvider>
  );
}

/** The manifest is a file the project points at: shown as it is, and the
 * way to edit it here, a copy in the project file. */
function ReferencedNotice({ project }: { project: ProjectView }) {
  const manifest = useDraft((s) => s.manifest);
  const reference =
    typeof project.spec.dataset.manifest === "string"
      ? project.spec.dataset.manifest
      : null;
  if (manifest === null || reference === null) return null;
  const file = joinPath(project.dir, reference);
  const embed = () =>
    useProject.getState().save({
      ...project.spec,
      dataset: {
        root: relativePath(project.dir, dirnamePath(file)),
        manifest,
      },
    });
  return (
    <Callout
      tone="info"
      title="Read-only: the manifest is a file of its own"
      className="mb-3"
      actions={
        <Button size="sm" onClick={() => void embed()}>
          Copy into the project to edit
        </Button>
      }
    >
      <span className="font-mono text-[11px] break-all">{reference}</span> is shown as it
      is and never written. Edit the file, or copy it into the project file.
    </Callout>
  );
}
