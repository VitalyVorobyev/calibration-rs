import { Button, Field, SegmentedControl, Select } from "@vitavision/ui";
import { Plus, Trash2 } from "lucide-react";
import {
  DISTORTION_KINDS,
  fixSlots,
  KIND_LABELS,
  PROJECTION_KINDS,
  SENSOR_KINDS,
  type ModelKinds,
} from "./cameraParams";
import { FixMask } from "./FixChips";

/** One camera whose model is not the rig's (`CameraOverride`). */
export interface Override {
  camera: number;
  model: ModelKinds;
  fix?: Record<string, unknown>;
}

function asOverrides(value: unknown): Override[] {
  return Array.isArray(value) ? (value as Override[]) : [];
}

const options = (kinds: readonly string[]) =>
  kinds.map((k) => ({ value: k, label: KIND_LABELS[k] ?? k }));

/** The rig's per-camera model overrides as one row per camera: which
 * camera, its projection, distortion and sensor, and its fix mask, with
 * "Override a camera" to add one (a camera of the rig with none yet) and a
 * remove button per row. At most one row per camera, as the library asks. */
export function CameraOverrides({
  value,
  baseModel,
  cameras,
  baseline,
  disabled,
  onChange,
}: {
  value: unknown;
  /** The rig's own model, which a new override starts from. */
  baseModel: ModelKinds;
  /** The cameras' ids, in rig order. */
  cameras: string[];
  baseline: unknown;
  disabled: boolean;
  onChange: (next: Override[]) => void;
}) {
  const rows = asOverrides(value);
  const was = asOverrides(baseline);
  const used = new Set(rows.map((r) => r.camera));
  const free = cameras
    .map((id, index) => ({ id, index }))
    .filter((c) => !used.has(c.index));
  const name = (index: number) => cameras[index] ?? `camera ${index}`;

  const update = (at: number, patch: Partial<Override>) =>
    onChange(rows.map((row, i) => (i === at ? { ...row, ...patch } : row)));

  return (
    <div role="group" aria-label="Per-camera overrides" className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-xs font-medium text-fg">Per-camera overrides</p>
        {JSON.stringify(rows) !== JSON.stringify(was) && (
          <span className="inline-flex items-center gap-1 text-[11px] font-medium text-signal">
            <span aria-hidden className="size-1.5 rounded-full bg-signal" />
            changed
          </span>
        )}
      </div>
      <p className="text-xs text-fg-muted">
        Cameras whose model differs from the one above, such as a tilted laser camera
        beside a frontal one or a fisheye beside a narrow lens. Every other camera has the
        rig's model.
      </p>
      {rows.map((row, at) => {
        const slots = fixSlots(row.model);
        return (
          <fieldset
            key={row.camera}
            className="flex flex-col gap-3 rounded-panel border border-line p-3"
          >
            <legend className="px-1 text-xs font-semibold text-fg">
              {name(row.camera)}{" "}
              <span className="font-mono font-normal text-fg-subtle">#{row.camera}</span>
            </legend>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Field as="group" label="Projection">
                <SegmentedControl
                  aria-label={`Projection of ${name(row.camera)}`}
                  value={row.model.projection ?? ""}
                  disabled={disabled}
                  options={options(PROJECTION_KINDS)}
                  onValueChange={(projection) =>
                    update(at, { model: { ...row.model, projection } })
                  }
                />
              </Field>
              <Field as="group" label="Sensor">
                <SegmentedControl
                  aria-label={`Sensor of ${name(row.camera)}`}
                  value={row.model.sensor ?? ""}
                  disabled={disabled}
                  options={options(SENSOR_KINDS)}
                  onValueChange={(sensor) =>
                    update(at, { model: { ...row.model, sensor } })
                  }
                />
              </Field>
              <Field label="Distortion" className="sm:col-span-2">
                <Select
                  aria-label={`Distortion of ${name(row.camera)}`}
                  value={row.model.distortion ?? ""}
                  disabled={disabled}
                  options={options(DISTORTION_KINDS)}
                  onValueChange={(distortion) =>
                    update(at, { model: { ...row.model, distortion } })
                  }
                />
              </Field>
            </div>
            {slots !== null && (
              <FixMask
                model={row.model}
                slots={slots}
                value={row.fix ?? {}}
                baseline={was.find((w) => w.camera === row.camera)?.fix ?? {}}
                disabled={disabled}
                onChange={(fix) => update(at, { fix })}
              />
            )}
            <div className="flex justify-end">
              <Button
                size="sm"
                variant="ghost"
                disabled={disabled}
                onClick={() => onChange(rows.filter((_, i) => i !== at))}
              >
                <Trash2 aria-hidden className="size-3.5" />
                Remove the override of {name(row.camera)}
              </Button>
            </div>
          </fieldset>
        );
      })}
      {free.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <Plus aria-hidden className="size-3.5 text-fg-muted" />
          <span className="text-xs text-fg-muted">Override a camera:</span>
          {free.map(({ id, index }) => (
            <Button
              key={index}
              size="sm"
              variant="secondary"
              disabled={disabled}
              onClick={() =>
                onChange(
                  [...rows, { camera: index, model: { ...baseModel }, fix: {} }].sort(
                    (a, b) => a.camera - b.camera,
                  ),
                )
              }
            >
              {id}
            </Button>
          ))}
        </div>
      )}
      {cameras.length === 0 && rows.length === 0 && (
        <p className="text-xs text-fg-subtle">The dataset's cameras are not known yet.</p>
      )}
    </div>
  );
}
