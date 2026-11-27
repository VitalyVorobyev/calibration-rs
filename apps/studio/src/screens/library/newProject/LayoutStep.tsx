import type { JsonSchema } from "@vitavision/forms";
import { SchemaValueForm } from "@vitavision/forms";
import { Callout, Disclosure, Empty, ReadoutStrip, SkeletonRows } from "@vitavision/ui";
import { useMemo } from "react";
import { api } from "../../../api";
import type { DatasetSpec, StudioError } from "../../../api/generated/types";
import { ViewMatrix } from "../../../components/ViewMatrix/ViewMatrix";
import { useProjectViews } from "../../../project/useProjectViews";
import { useQuery } from "../../../session/query";
import { exportKindLabel } from "../../../store/exportKind";
import {
  answer,
  dataFolder,
  isEditable,
  openFields,
  previewSpec,
  questionFor,
  valueAt,
} from "./newProjectModel";
import { QuestionRow } from "./QuestionRow";
import type { NewProjectApi } from "./useNewProject";

const PREVIEW_VIEWS = 12;

/** Step 2: what the library inferred, as a summary and as the camera × view
 * matrix of the first views; what it could not decide, asked one question
 * a field; and what a run would refuse, in words. Answers go into the draft
 * manifest and the preview asks the library again. */
export function LayoutStep({ project }: { project: NewProjectApi }) {
  const { state, patch } = project;
  const { source, manifest } = state;
  const form = useQuery("dataset_form", {}, () => api.datasetForm());
  const schema = form.data?.schema as JsonSchema | undefined;
  const folder = source ? dataFolder(source) : null;
  const views = useProjectViews(
    folder,
    manifest ? previewSpec(manifest) : null,
    manifest !== null,
  );
  const editable = source !== null && isEditable(source);

  const questions = useMemo(
    () =>
      manifest
        ? openFields(manifest)
            .filter((p) => p !== "target")
            .map((p) => questionFor(p, manifest, schema))
        : [],
    [manifest, schema],
  );
  if (!manifest || folder === null) return null;

  const edit = (next: DatasetSpec) => patch({ manifest: next });
  const reply = views.data;
  const shown = reply?.views.slice(0, PREVIEW_VIEWS) ?? [];
  // The target is the next step's to settle; everything else is said here.
  const problems = (reply?.messages ?? []).filter(
    (m) => !isTargetMessage(m) || !openFields(manifest).includes("target"),
  );

  return (
    <div className="flex flex-col gap-4">
      <ReadoutStrip
        items={[
          { label: "workflow", value: exportKindLabel(manifest.topology) },
          { label: "cameras", value: manifest.cameras.map((c) => c.id).join(", ") },
          { label: "views", value: reply ? String(reply.views.length) : "…" },
          { label: "pairing", value: pairingWords(manifest) },
          ...(manifest.robot_poses
            ? [{ label: "robot poses", value: manifest.robot_poses.path }]
            : []),
          ...(manifest.cameras.some((c) => c.laser_images)
            ? [{ label: "laser", value: "frames declared" }]
            : []),
        ]}
      />

      <div
        className={
          questions.length > 0 || problems.length > 0
            ? "grid grid-cols-1 gap-5 md:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]"
            : "flex flex-col"
        }
      >
        {(questions.length > 0 || problems.length > 0) && (
          <div className="flex min-w-0 flex-col gap-4">
            {questions.length > 0 && (
              <Callout
                tone="warning"
                title={`${questions.length} ${questions.length === 1 ? "thing" : "things"} the layout does not say`}
              >
                {editable
                  ? "Answer each; the preview below updates as you go."
                  : "This manifest leaves fields open. Go back and copy it into the project to answer them here."}
              </Callout>
            )}
            <div className="flex flex-col gap-3">
              {questions.map((q) => (
                <QuestionRow
                  key={q.path}
                  question={q}
                  value={valueAt(manifest, q.path)}
                  dataRoot={folder}
                  disabled={!editable}
                  onAnswer={(v) => edit(answer(manifest, q.path, v))}
                />
              ))}
            </div>

            {problems.length > 0 && (
              <ul
                className="flex flex-col gap-1 text-xs"
                aria-label="What a run would refuse"
              >
                {problems.map((m) => (
                  <li
                    key={`${m.field ?? ""}:${m.message}`}
                    className="flex gap-2 text-fg-muted"
                  >
                    <span
                      aria-hidden
                      className="mt-1.5 size-1.5 shrink-0 rounded-full bg-warn"
                    />
                    <span>
                      {m.message}
                      {m.field && (
                        <span className="ml-1.5 font-mono text-fg-subtle">{m.field}</span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        <section
          aria-label="Preview of the views"
          className="flex min-w-0 flex-col gap-1.5"
        >
          <h3 className="text-[10px] font-semibold tracking-wider text-fg-muted uppercase">
            {reply && reply.views.length > shown.length
              ? `First ${shown.length} of ${reply.views.length} views`
              : "Views"}
          </h3>
          {reply === null ? (
            views.error ? (
              <Callout tone="error">{views.error.message}</Callout>
            ) : (
              <SkeletonRows rows={3} />
            )
          ) : shown.length === 0 ? (
            <Empty className="py-6">
              No image matches the camera patterns of this layout.
            </Empty>
          ) : (
            <ViewMatrix
              aria-label="First views of the new project"
              views={shown}
              cameras={reply.cameras}
              dataRoot={folder}
              fitRows={4}
            />
          )}
        </section>
      </div>

      {editable && schema && (
        <Disclosure summary="Advanced: every field of the manifest">
          <SchemaValueForm
            schema={schema}
            value={manifest}
            columns={1}
            onValueChange={(v) => edit(v as DatasetSpec)}
          />
        </Disclosure>
      )}
    </div>
  );
}

function isTargetMessage(m: StudioError): boolean {
  return m.field === "target" || (m.field ?? "").startsWith("target.");
}

function pairingWords(spec: DatasetSpec): string {
  const p = spec.pose_pairing;
  if (!p) return "not needed";
  return p.kind === "by_index" ? "by order" : `by filename token “${p.group}”`;
}
