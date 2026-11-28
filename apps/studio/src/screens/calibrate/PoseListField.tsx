import { Field, Input } from "@vitavision/ui";
import { useState } from "react";
import { asPoseList, formatPoseList, parsePoseList } from "./poseList";

/** `fix_poses`: which views' target poses the refinement holds at their
 * initial value, as a short text (`0, 3, 5-7`) kept in step with the list.
 * The text may be half-typed; only a valid one writes the list. `viewCount`
 * (the dataset's views, when known) warns about a number beyond them. */
export function PoseListField({
  value,
  label,
  baseline,
  viewCount,
  disabled,
  onChange,
}: {
  value: unknown;
  /** The field's name as its neighbours show it. */
  label: string;
  baseline: unknown;
  viewCount: number | null;
  disabled: boolean;
  onChange: (next: number[]) => void;
}) {
  const list = asPoseList(value);
  const was = asPoseList(baseline);
  const shown = formatPoseList(list);
  const [draft, setDraft] = useState<string | null>(null);
  const text = draft ?? shown;
  const parsed = parsePoseList(text);
  const changed = formatPoseList(was) !== shown;
  const beyond =
    viewCount !== null && "value" in parsed
      ? parsed.value.filter((i) => i >= viewCount)
      : [];
  const error = "problem" in parsed ? parsed.problem : undefined;

  return (
    <Field
      label={label}
      annotation={list.length === 0 ? "none held" : `${list.length} held`}
      error={error}
      description={
        <span className="flex flex-col gap-0.5 text-[11px]">
          <span>
            Views whose target pose stays at its initial value, by position among the
            views (counting from 0): <span className="font-mono">0, 3, 5-7</span>. Empty
            holds none; a held pose makes the reported uncertainty conditional on it.
          </span>
          {beyond.length > 0 && (
            <span className="text-warn">
              The dataset has {viewCount} view{viewCount === 1 ? "" : "s"};{" "}
              {beyond.join(", ")} {beyond.length === 1 ? "is" : "are"} beyond them.
            </span>
          )}
          {changed && (
            <span className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1 font-medium text-signal">
                <span aria-hidden className="size-1.5 rounded-full bg-signal" />
                changed
              </span>
              <span className="text-fg-muted">
                was <span className="font-mono">{formatPoseList(was) || "none"}</span>
              </span>
            </span>
          )}
        </span>
      }
    >
      <Input
        className="font-mono"
        placeholder="none"
        value={text}
        disabled={disabled}
        onChange={(e) => {
          setDraft(e.target.value);
          const next = parsePoseList(e.target.value);
          if ("value" in next) onChange(next.value);
        }}
        onBlur={() => setDraft(null)}
      />
    </Field>
  );
}
