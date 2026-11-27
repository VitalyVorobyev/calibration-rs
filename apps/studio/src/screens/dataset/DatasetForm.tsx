import type { JsonSchema, RenderField } from "@vitavision/forms";
import { SchemaValueForm } from "@vitavision/forms";
import { useCallback, useMemo } from "react";
import type { DatasetSpec, ProjectViews } from "../../api/generated/types";
import { CamerasEditor } from "./CamerasEditor";
import { datasetUi, formMessages, usesLaser, type DraftMessage } from "./datasetModel";

/** The manifest's description: the library's schema in five groups
 * (workflow, cameras, pairing, target, poses and motion, laser, a folded
 * detector), every changed field marked against the saved manifest with its
 * reset, the library's messages under the fields they are about. The cameras
 * are a list of cards (`CamerasEditor`): the schema's own control for a list
 * of structures is a JSON box. */
export function DatasetForm({
  schema,
  manifest,
  baseline,
  onChange,
  messages,
  views,
  dataRoot,
  disabled,
}: {
  schema: JsonSchema;
  manifest: DatasetSpec;
  baseline: DatasetSpec | null;
  onChange: (next: DatasetSpec) => void;
  messages: readonly DraftMessage[];
  views: ProjectViews | null;
  dataRoot: string;
  disabled: boolean;
}) {
  const ui = useMemo(
    () => datasetUi(manifest.topology, manifest.cameras.length),
    [manifest.topology, manifest.cameras.length],
  );
  const fieldMessages = useMemo(() => formMessages(messages), [messages]);
  const laser = usesLaser(manifest.topology);

  const renderField = useCallback<RenderField>(
    (ctx) =>
      ctx.path === "cameras" ? (
        <CamerasEditor
          cameras={manifest.cameras}
          baseline={baseline?.cameras ?? null}
          onChange={(cameras) => onChange({ ...manifest, cameras })}
          views={views}
          dataRoot={dataRoot}
          laser={laser}
          disabled={disabled}
          messages={fieldMessages}
        />
      ) : undefined,
    [manifest, baseline, onChange, views, dataRoot, laser, disabled, fieldMessages],
  );

  return (
    <SchemaValueForm
      schema={schema}
      value={manifest}
      baseline={baseline ?? undefined}
      messages={fieldMessages}
      ui={ui}
      columns={1}
      disabled={disabled}
      renderField={renderField}
      onValueChange={(v) => onChange(v as DatasetSpec)}
    />
  );
}
