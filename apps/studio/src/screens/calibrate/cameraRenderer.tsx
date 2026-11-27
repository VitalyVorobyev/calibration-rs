import { getAtPath, type RenderField } from "@vitavision/forms";
import { fixSlots, type ModelKinds } from "./cameraParams";
import { CameraOverrides } from "./CameraOverrides";
import { FixMask } from "./FixChips";

const FIX_PATH = /^(.*)\.fix$/;
const OVERRIDES_PATH = /^(.*?)camera_overrides$/;

function asTable(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

/** The form's `renderField` for the camera controls the schema's own cannot
 * do well: a camera's fix mask (a `fix` object beside a `model`) as chips,
 * and a rig's `camera_overrides` as a row per camera. Anywhere this table
 * does not know the model, the form's own controls stay. `config` is the
 * value the form edits, to read a camera's model from; `baseline` the value
 * compared with; `cameras` the rig's camera ids. */
export const cameraRenderer =
  (config: unknown, baseline: unknown, cameras: string[]): RenderField =>
  ({ path, value, onChange, disabled }) => {
    const overrides = OVERRIDES_PATH.exec(path);
    if (overrides) {
      const base = getAtPath(config, `${overrides[1]}camera.model`);
      if (typeof base !== "object" || base === null) return undefined;
      return (
        <CameraOverrides
          value={value}
          baseModel={base}
          cameras={cameras}
          baseline={getAtPath(baseline, path)}
          disabled={disabled}
          onChange={onChange}
        />
      );
    }
    const match = FIX_PATH.exec(path);
    if (!match) return undefined;
    const model = getAtPath(config, `${match[1]}.model`);
    if (typeof model !== "object" || model === null) return undefined;
    const kinds = model as ModelKinds;
    const slots = fixSlots(kinds);
    if (slots === null) return undefined;
    return (
      <FixMask
        model={kinds}
        slots={slots}
        value={asTable(value)}
        baseline={asTable(getAtPath(baseline, path))}
        disabled={disabled}
        onChange={onChange}
      />
    );
  };
