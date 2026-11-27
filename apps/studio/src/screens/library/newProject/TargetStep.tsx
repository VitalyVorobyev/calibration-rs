import type { JsonSchema } from "@vitavision/forms";
import { SchemaValueForm } from "@vitavision/forms";
import { Callout, SkeletonRows } from "@vitavision/ui";
import { useMemo } from "react";
import { api } from "../../../api";
import type { TargetSpec } from "../../../api/generated/types";
import { dataFolder, close, isEditable, previewSpec } from "./newProjectModel";
import { useProjectViews } from "../../../project/useProjectViews";
import { useQuery } from "../../../session/query";
import { BoardGlyph } from "./BoardGlyph";
import type { NewProjectApi } from "./useNewProject";

/** Step 3: the calibration target. The form is the target's own schema
 * (kind and parameters), starting from what the sniffer guessed; the glyph
 * beside it draws the lattice the numbers describe, so a swapped rows and
 * columns is seen before a run. What the library says about the target
 * under the form is the draft's own validation. */
export function TargetStep({ project }: { project: NewProjectApi }) {
  const { state, patch } = project;
  const { source, manifest } = state;
  const form = useQuery("dataset_form", {}, () => api.datasetForm());
  const schema = form.data?.schema as JsonSchema | undefined;
  const folder = source ? dataFolder(source) : null;
  const editable = source !== null && isEditable(source);
  const views = useProjectViews(
    folder,
    manifest ? previewSpec(manifest) : null,
    manifest !== null,
  );

  // The target's own schema, with the manifest's definitions beside it.
  const targetSchema = useMemo<JsonSchema | undefined>(
    () =>
      schema
        ? ({
            $ref: "#/$defs/TargetSpec",
            $defs: (schema as { $defs?: unknown }).$defs,
          } as unknown as JsonSchema)
        : undefined,
    [schema],
  );
  if (!manifest) return null;

  const problems = (views.data?.messages ?? []).filter(
    (m) => m.field === "target" || (m.field ?? "").startsWith("target."),
  );
  const edit = (target: TargetSpec) =>
    patch({ manifest: close({ ...manifest, target }, "target") });

  return (
    <div className="grid grid-cols-1 gap-5 md:grid-cols-[1fr_14rem]">
      <div className="flex min-w-0 flex-col gap-3">
        {!editable && (
          <Callout tone="info">
            The manifest is referenced, so its target is shown as it is.
          </Callout>
        )}
        {targetSchema ? (
          <SchemaValueForm
            schema={targetSchema}
            value={manifest.target}
            columns={2}
            disabled={!editable}
            onValueChange={(v) => edit(v as TargetSpec)}
          />
        ) : (
          <SkeletonRows rows={4} />
        )}
        {problems.map((m) => (
          <p key={m.message} role="alert" className="text-xs text-defect">
            {m.message}
          </p>
        ))}
      </div>
      <figure className="flex flex-col items-center gap-2 self-start rounded-control border border-line bg-surface p-3">
        <BoardGlyph target={manifest.target} />
        <figcaption className="text-center font-mono text-[11px] text-fg-muted">
          {manifest.target.kind}
        </figcaption>
      </figure>
    </div>
  );
}
