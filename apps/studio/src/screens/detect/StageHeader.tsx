import { Badge, Button, SegmentedControl, Select } from "@vitavision/ui";
import { Ban, RotateCcw } from "lucide-react";
import type { DetectionDetail } from "../../api/generated/types";
import type { Cell } from "./cell";
import { outcomeSentence } from "./detectModel";

const SEGMENTED_MAX = 4;

/** Above the stage, two lines: which camera's image of the view this is
 * (stepped with `,` and `.`) beside the two exclusion gestures, the image
 * (`X`) and the whole view (`⇧X`); then the view, the image's path and the
 * detector's outcome. */
export function StageHeader({
  cell,
  cameras,
  onCamera,
  detail,
  imageOut,
  viewOut,
  onToggleImage,
  onToggleView,
  readOnly,
}: {
  cell: Cell | null;
  cameras: readonly string[];
  onCamera: (camera: number) => void;
  detail: DetectionDetail | null;
  /** This camera's image is excluded by its own entry. */
  imageOut: boolean;
  /** The whole view is excluded. */
  viewOut: boolean;
  onToggleImage: () => void;
  onToggleView: () => void;
  readOnly: boolean;
}) {
  const options = cameras.map((id, i) => ({ value: String(i), label: id }));
  return (
    <header className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-2">
        {cameras.length > 1 &&
          (cameras.length <= SEGMENTED_MAX ? (
            <SegmentedControl
              aria-label="Camera"
              value={cell ? String(cell.camera) : ""}
              options={options}
              onValueChange={(v) => v !== "" && onCamera(Number(v))}
            />
          ) : (
            <Select
              aria-label="Camera"
              className="w-32"
              value={cell ? String(cell.camera) : ""}
              options={options}
              placeholder="Camera"
              onValueChange={(v) => onCamera(Number(v))}
            />
          ))}
        {cell && cell.image && (
          <span className="ml-auto flex items-center gap-1.5">
            <Button
              size="sm"
              variant="secondary"
              disabled={readOnly || viewOut}
              title={viewOut ? "The whole view is excluded" : "Toggle this image (X)"}
              onClick={onToggleImage}
            >
              {imageOut ? (
                <RotateCcw aria-hidden className="size-3" />
              ) : (
                <Ban aria-hidden className="size-3" />
              )}
              {imageOut ? "Restore image" : "Exclude image"}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              disabled={readOnly}
              title="Toggle the view in every camera (⇧X)"
              onClick={onToggleView}
            >
              {viewOut ? (
                <RotateCcw aria-hidden className="size-3" />
              ) : (
                <Ban aria-hidden className="size-3" />
              )}
              {viewOut ? "Restore view" : "Exclude view"}
            </Button>
          </span>
        )}
      </div>
      {cell && (
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="shrink-0 text-sm font-semibold tracking-tight whitespace-nowrap text-fg">
            View <span className="font-mono">{cell.token}</span>
            {cameras.length === 1 ? ` · ${cell.cameraId}` : ""}
          </h2>
          {cell.image && (
            <p
              className="min-w-0 flex-1 truncate font-mono text-[11px] text-fg-muted"
              title={cell.image}
            >
              {cell.image}
            </p>
          )}
          {detail && (
            <Badge
              className="shrink-0"
              tone={
                detail.outcome.status === "detected"
                  ? "normal"
                  : detail.outcome.status === "not_found"
                    ? "defect"
                    : "warning"
              }
            >
              {outcomeSentence(detail.outcome)}
            </Badge>
          )}
        </div>
      )}
    </header>
  );
}
