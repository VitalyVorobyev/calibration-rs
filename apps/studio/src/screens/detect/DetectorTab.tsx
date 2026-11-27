import { SchemaValueForm, type JsonSchema } from "@vitavision/forms";
import {
  Badge,
  Button,
  Callout,
  ErrorBox,
  SegmentedControl,
  SkeletonRows,
} from "@vitavision/ui";
import { RotateCcw, ScanSearch } from "lucide-react";
import { useMemo } from "react";
import { api } from "../../api";
import type {
  DetectionDetail,
  DetectorSpec,
  ProjectView,
} from "../../api/generated/types";
import { useDraft } from "../../project/draftStore";
import { useProject } from "../../project/projectStore";
import { useReadOnly } from "../../session/jobs";
import { useQuery } from "../../session/query";
import {
  canApply,
  DETECTOR_UI,
  detectorSchema,
  sameDetector,
  savedDetector,
  specWithDetector,
} from "./detectorModel";
import { describeDiff, diffDetections } from "./overlayModel";
import type { PreviewState } from "./useDetectData";

export type ShownDetection = "saved" | "preview";

/** The detector's settings for trying out. Edit them (the form is the
 * library's own schema, changed fields marked against the saved settings),
 * Preview runs the detector on this image with them and nothing else is
 * written, and Apply to project saves them into the manifest, after which
 * the other images' detections are stale and Detect all reads them again. */
export function DetectorTab({
  project,
  working,
  onWorking,
  hasImage,
  preview,
  onPreview,
  shown,
  onShown,
  saved,
  detecting,
  onRedetect,
  applied,
  onApplied,
}: {
  project: ProjectView;
  /** The settings being edited, or `null` when untouched (the saved ones). */
  working: DetectorSpec | null;
  onWorking: (next: DetectorSpec | null) => void;
  hasImage: boolean;
  preview: PreviewState | null;
  onPreview: () => void;
  shown: ShownDetection;
  onShown: (next: ShownDetection) => void;
  /** The image's cached detection, to compare the preview with. */
  saved: DetectionDetail | undefined;
  /** A detect-all job is live. */
  detecting: boolean;
  onRedetect: () => void;
  /** The settings were just applied (offer to detect again). */
  applied: boolean;
  onApplied: (applied: boolean) => void;
}) {
  const readOnly = useReadOnly();
  const draftDirty = useDraft((s) => s.manifestDirty);
  const form = useQuery("dataset_form", {}, () => api.datasetForm());
  const schema = useMemo(
    () => detectorSchema(form.data?.schema as JsonSchema | undefined),
    [form.data],
  );
  const base = savedDetector(project);
  const value = working ?? base;
  const changed = working !== null && !sameDetector(working, base);
  const writable = canApply(project);

  const diff = useMemo(
    () => (preview?.detail && saved ? diffDetections(saved, preview.detail) : null),
    [preview, saved],
  );

  const apply = async () => {
    const spec = specWithDetector(project, value);
    if (spec === null) return;
    if (await useProject.getState().save(spec)) {
      onWorking(null);
      onApplied(true);
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
      <p className="text-xs text-fg-muted">
        Try other detector settings on this image. A preview writes nothing; the settings
        are saved only when you apply them to the project.
      </p>

      {!writable && (
        <Callout tone="info" title="Read-only: the manifest is a file of its own">
          The detector settings live in the manifest, which this project keeps in a file
          the app does not write. Preview works; to save the settings, copy the manifest
          into the project file on the Dataset screen.
        </Callout>
      )}
      {draftDirty && writable && (
        <Callout tone="warning" title="Unsaved dataset edits">
          Save or revert the edits on the Dataset screen first: applying the detector
          writes the project file.
        </Callout>
      )}
      {applied && !changed && (
        <Callout
          tone="success"
          title="Detector settings saved"
          actions={
            <>
              <Button size="sm" disabled={readOnly || detecting} onClick={onRedetect}>
                Detect all with the new settings
              </Button>
              <Button size="sm" variant="ghost" onClick={() => onApplied(false)}>
                Later
              </Button>
            </>
          }
        >
          The detections of the other images were made with the old settings. Detect all
          reads them again with the new ones.
        </Callout>
      )}

      {schema === null ? (
        form.error ? (
          <ErrorBox>The settings form is not available: {form.error.message}</ErrorBox>
        ) : (
          <div role="status" aria-busy aria-label="Loading the settings">
            <SkeletonRows rows={5} />
          </div>
        )
      ) : (
        <SchemaValueForm
          schema={schema}
          value={{ detector: value }}
          baseline={{ detector: base }}
          ui={DETECTOR_UI}
          columns={1}
          disabled={readOnly}
          onValueChange={(next) => {
            const detector = (next as { detector?: DetectorSpec | null }).detector ?? {};
            onWorking(sameDetector(detector, base) ? null : detector);
          }}
        />
      )}

      <div className="sticky bottom-0 flex flex-col gap-2 border-t border-line bg-surface pt-2">
        {preview && (
          <div className="flex flex-col gap-1.5" aria-label="Preview result">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="info">Preview (unsaved settings)</Badge>
              {preview.detail && saved && (
                <SegmentedControl
                  aria-label="Detection shown"
                  value={shown}
                  onValueChange={(v) => onShown(v === "preview" ? "preview" : "saved")}
                  options={[
                    { value: "saved", label: "Saved" },
                    { value: "preview", label: "Preview" },
                  ]}
                />
              )}
            </div>
            {preview.loading && <p className="text-xs text-fg-muted">Detecting…</p>}
            {preview.error && (
              <p role="alert" className="text-xs text-defect">
                The preview failed: {preview.error.message}
              </p>
            )}
            {preview.detail && (
              <p className="text-xs text-fg" role="status">
                {preview.detail.features.length} features.{" "}
                {diff ? describeDiff(diff) : "No saved detection to compare with."}
              </p>
            )}
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="secondary"
            disabled={!hasImage || preview?.loading === true}
            title={hasImage ? undefined : "Select an image first"}
            onClick={onPreview}
          >
            <ScanSearch aria-hidden className="size-3.5" />
            Preview on this image
          </Button>
          <Button
            size="sm"
            variant="primary"
            disabled={readOnly || !writable || draftDirty || !changed}
            onClick={() => void apply()}
          >
            Apply to project
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={!changed && preview === null}
            onClick={() => {
              onWorking(null);
              onShown("saved");
            }}
          >
            <RotateCcw aria-hidden className="size-3.5" />
            Discard
          </Button>
        </div>
      </div>
    </div>
  );
}
