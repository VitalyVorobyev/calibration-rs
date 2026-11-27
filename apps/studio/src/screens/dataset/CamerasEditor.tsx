import type { FieldMessages } from "@vitavision/forms";
import type { ReactNode } from "react";
import { diffPaths } from "@vitavision/forms";
import { Badge, Button, Field, Input, NumberInput, Switch } from "@vitavision/ui";
import { ArrowDown, ArrowUp, Copy, Plus, RotateCcw, Trash2 } from "lucide-react";
import type { CameraSource, ProjectViews } from "../../api/generated/types";
import { ImagePatternField } from "./ImagePatternField";
import { nextCameraId, duplicateCamera, newCamera, move } from "./camerasModel";
import { patternMatch } from "./datasetModel";

/** The manifest's cameras as cards: id, where its images are (with what the
 * pattern matches now), a crop region for tiled strips, laser frames for the
 * workflows that use them. Cameras are added, duplicated (the next id, its
 * pattern following), moved and removed; a card edited from the saved
 * version says so and resets. Messages on any field of a card show under
 * it: the form cannot place them on a card it did not draw. */
export function CamerasEditor({
  cameras,
  baseline,
  onChange,
  views,
  dataRoot,
  laser,
  disabled,
  messages,
}: {
  cameras: CameraSource[];
  /** The saved cameras, to mark what changed. */
  baseline: CameraSource[] | null;
  onChange: (next: CameraSource[]) => void;
  views: ProjectViews | null;
  dataRoot: string;
  /** Whether the workflow uses laser frames. */
  laser: boolean;
  disabled: boolean;
  messages: FieldMessages;
}) {
  const set = (i: number, camera: CameraSource) =>
    onChange(cameras.map((c, j) => (j === i ? camera : c)));
  const listMessage = messages["cameras"];
  const listText = typeof listMessage === "string" ? listMessage : listMessage?.text;

  return (
    <div className="flex flex-col gap-3">
      {listText && (
        <p role="alert" className="text-xs text-defect">
          {listText}
        </p>
      )}
      {cameras.map((camera, i) => {
        const saved = baseline?.[i];
        const changed = saved === undefined ? "new" : diffPaths(camera, saved).length > 0;
        return (
          // A camera is its place in the list: renaming it keeps the focus.
          <section
            // oxlint-disable-next-line react/no-array-index-key
            key={i}
            aria-label={`Camera ${camera.id || i + 1}`}
            className="flex flex-col gap-3 rounded-control border border-line bg-surface p-3"
            data-changed={changed === false ? undefined : ""}
          >
            <header className="flex items-center gap-1.5">
              <span className="font-mono text-xs font-medium text-fg">
                {camera.id || `camera ${i + 1}`}
              </span>
              {changed === "new" && <Badge tone="info">new</Badge>}
              {changed === true && <Badge tone="info">changed</Badge>}
              <span className="ml-auto flex items-center">
                {changed === true && saved && (
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={disabled}
                    aria-label={`Reset camera ${camera.id}`}
                    title="Back to the saved camera"
                    onClick={() => set(i, saved)}
                  >
                    <RotateCcw aria-hidden className="size-3.5" />
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={disabled || i === 0}
                  aria-label={`Move ${camera.id} up`}
                  onClick={() => onChange(move(cameras, i, -1))}
                >
                  <ArrowUp aria-hidden className="size-3.5" />
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={disabled || i === cameras.length - 1}
                  aria-label={`Move ${camera.id} down`}
                  onClick={() => onChange(move(cameras, i, 1))}
                >
                  <ArrowDown aria-hidden className="size-3.5" />
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={disabled}
                  aria-label={`Duplicate ${camera.id}`}
                  title="Duplicate with the next camera id"
                  onClick={() =>
                    onChange([
                      ...cameras.slice(0, i + 1),
                      duplicateCamera(camera, cameras),
                      ...cameras.slice(i + 1),
                    ])
                  }
                >
                  <Copy aria-hidden className="size-3.5" />
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={disabled || cameras.length === 1}
                  aria-label={`Remove ${camera.id}`}
                  onClick={() => onChange(cameras.filter((_, j) => j !== i))}
                >
                  <Trash2 aria-hidden className="size-3.5" />
                </Button>
              </span>
            </header>

            <Field label="Camera id" {...fieldMessage(messages, `cameras.${i}.id`)}>
              <Input
                value={camera.id}
                disabled={disabled}
                className="font-mono"
                onChange={(e) => set(i, { ...camera, id: e.target.value })}
              />
            </Field>

            <Field
              as="group"
              label="Images"
              {...fieldMessage(messages, `cameras.${i}.images`)}
            >
              <ImagePatternField
                label={`Camera ${camera.id} images`}
                value={camera.images}
                dataRoot={dataRoot}
                disabled={disabled}
                match={views ? patternMatch(views, i) : null}
                onChange={(images) => set(i, { ...camera, images })}
              />
            </Field>

            <div className="flex flex-col gap-2">
              <Switch
                checked={camera.roi_xywh != null}
                disabled={disabled}
                label="Crop to a region"
                description="For tiled strips: only this region of each image is used."
                onCheckedChange={(on) => {
                  const { roi_xywh: _drop, ...rest } = camera;
                  set(i, on ? { ...rest, roi_xywh: [0, 0, 100, 100] } : rest);
                }}
              />
              {camera.roi_xywh != null && (
                <Field
                  as="group"
                  label="Region (x, y, width, height)"
                  {...fieldMessage(messages, `cameras.${i}.roi_xywh`)}
                >
                  <div className="grid grid-cols-4 gap-2">
                    {(["x", "y", "width", "height"] as const).map((name, k) => (
                      <NumberInput
                        key={name}
                        aria-label={`Camera ${camera.id} region ${name}`}
                        unit="px"
                        min={0}
                        disabled={disabled}
                        value={camera.roi_xywh![k]}
                        onValueChange={(v) => {
                          const roi = [...camera.roi_xywh!] as [
                            number,
                            number,
                            number,
                            number,
                          ];
                          roi[k] = v;
                          set(i, { ...camera, roi_xywh: roi });
                        }}
                      />
                    ))}
                  </div>
                </Field>
              )}
            </div>

            {laser && (
              <div className="flex flex-col gap-2">
                <Switch
                  checked={camera.laser_images != null}
                  disabled={disabled}
                  label="Laser frames"
                  description="One image a view with the laser line on, from this camera."
                  onCheckedChange={(on) => {
                    const { laser_images: _drop, ...rest } = camera;
                    set(
                      i,
                      on
                        ? {
                            ...rest,
                            laser_images: {
                              kind: "glob",
                              pattern: `${camera.id}_laser/*.png`,
                            },
                          }
                        : rest,
                    );
                  }}
                />
                {camera.laser_images != null && (
                  <Field
                    as="group"
                    label="Laser images"
                    {...fieldMessage(messages, `cameras.${i}.laser_images`)}
                  >
                    <ImagePatternField
                      label={`Camera ${camera.id} laser images`}
                      value={camera.laser_images}
                      dataRoot={dataRoot}
                      disabled={disabled}
                      onChange={(laser_images) => set(i, { ...camera, laser_images })}
                    />
                  </Field>
                )}
              </div>
            )}
          </section>
        );
      })}
      <Button
        size="sm"
        variant="secondary"
        disabled={disabled}
        className="self-start"
        onClick={() => onChange([...cameras, newCamera(nextCameraId(cameras))])}
      >
        <Plus aria-hidden className="size-3.5" />
        Add a camera
      </Button>
    </div>
  );
}

/** The message on `path` as the props a `Field` takes: an error in its
 * error slot (announced, in the defect colour), a warning under the label
 * in the warn colour. */
function fieldMessage(
  messages: FieldMessages,
  path: string,
): { error: string } | { description: ReactNode } | Record<string, never> {
  const m = messages[path];
  if (m === undefined) return {};
  const message = typeof m === "string" ? { severity: "error", text: m } : m;
  return message.severity === "error"
    ? { error: message.text }
    : { description: <span className="text-warn">{message.text}</span> };
}
