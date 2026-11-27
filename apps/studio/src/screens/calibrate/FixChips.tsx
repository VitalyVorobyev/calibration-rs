import { Field, ToggleChip, cn } from "@vitavision/ui";
import {
  defaultFixed,
  fixedNames,
  toggledFix,
  type FixSlot,
  type FixSlotChoices,
  type ModelKinds,
} from "./cameraParams";

const SLOT_LABEL: Record<FixSlot, string> = {
  intrinsics: "Intrinsics",
  projection: "Projection parameters",
  distortion: "Distortion terms",
  sensor: "Sensor tilt",
};

/** The names a fix list holds fixed in effect: its own, or the model's
 * default when it is left out. */
function effective(value: unknown, slot: FixSlot, model: ModelKinds): string[] {
  return Array.isArray(value) ? fixedNames(value) : (defaultFixed(slot, model) ?? []);
}

function sameNames(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((n) => b.includes(n));
}

/** A camera's whole fix mask as rows of chips, one row per slot that has
 * parameters: pressed is held fixed, released is estimated. A list left
 * out means "the model's default", which the chips show as it applies and
 * the first press makes explicit. */
export function FixMask({
  model,
  slots,
  value,
  baseline,
  disabled,
  onChange,
}: {
  model: ModelKinds;
  slots: FixSlotChoices[];
  value: Record<string, unknown>;
  baseline: Record<string, unknown>;
  disabled: boolean;
  onChange: (next: Record<string, unknown>) => void;
}) {
  const set = (slot: FixSlot, next: string[] | undefined) => {
    const copy = { ...value };
    if (next === undefined) delete copy[slot];
    else copy[slot] = next;
    onChange(copy);
  };
  return (
    <div role="group" aria-label="Fixed parameters" className="flex flex-col gap-3">
      <p className="text-xs font-medium text-fg">Fixed in the solve</p>
      {slots.map(({ slot, choices }) => {
        const fixed = effective(value[slot], slot, model);
        const was = effective(baseline[slot], slot, model);
        const changed = !sameNames(fixed, was);
        const isDefault = !Array.isArray(value[slot]);
        return (
          <div
            key={slot}
            className={cn(
              changed &&
                "relative before:absolute before:top-0.5 before:bottom-0.5 before:-left-2.5 before:w-0.5 before:rounded-full before:bg-signal",
            )}
          >
            <Field
              as="group"
              label={SLOT_LABEL[slot]}
              annotation={isDefault ? "model default" : "explicit"}
              description={
                changed ? (
                  <span className="flex items-center gap-2 text-[11px]">
                    <span className="inline-flex items-center gap-1 font-medium text-signal">
                      <span aria-hidden className="size-1.5 rounded-full bg-signal" />
                      changed
                    </span>
                    <span className="truncate text-fg-muted">
                      was{" "}
                      <span className="font-mono">
                        {was.length > 0 ? was.join(" ") : "none"}
                      </span>
                    </span>
                  </span>
                ) : undefined
              }
            >
              <div className="flex flex-wrap items-center gap-1.5">
                {choices.map((name) => (
                  <ToggleChip
                    key={name}
                    checked={fixed.includes(name)}
                    disabled={disabled}
                    title={
                      fixed.includes(name)
                        ? `${name} is held fixed`
                        : `${name} is estimated`
                    }
                    onCheckedChange={(next) =>
                      set(slot, toggledFix(slot, model, fixed, name, next))
                    }
                  >
                    <span className="font-mono">{name}</span>
                  </ToggleChip>
                ))}
                {!isDefault && (
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => set(slot, undefined)}
                    className="ml-1 rounded-control px-1 py-0.5 text-[11px] font-medium text-fg-muted hover:text-fg focus-visible:outline-2 focus-visible:outline-signal disabled:opacity-50"
                  >
                    Use model default
                  </button>
                )}
              </div>
            </Field>
          </div>
        );
      })}
    </div>
  );
}
