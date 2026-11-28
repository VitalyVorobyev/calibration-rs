/** `fix_poses`: the indices of the target poses a refinement holds fixed,
 * edited as text (`0, 3, 5-7`). Pure rules; the control is `PoseList`. */

/** The indices `text` names, sorted and without repeats, or what is wrong.
 * Accepts numbers and `a-b` ranges separated by commas or spaces; an empty
 * text is no pose. */
export function parsePoseList(text: string): { value: number[] } | { problem: string } {
  const out = new Set<number>();
  for (const token of text.split(/[\s,;]+/).filter((t) => t !== "")) {
    const m = /^(\d+)(?:[-–](\d+))?$/.exec(token);
    if (!m) return { problem: `“${token}” is not a view number or a range like 2-4.` };
    const from = Number(m[1]);
    const to = m[2] === undefined ? from : Number(m[2]);
    if (to < from) return { problem: `The range ${token} runs backwards.` };
    if (to - from > 100_000) return { problem: `The range ${token} is too long.` };
    for (let i = from; i <= to; i++) out.add(i);
  }
  return { value: [...out].sort((a, b) => a - b) };
}

/** `indices` as text, collapsing runs of three or more into ranges. */
export function formatPoseList(indices: readonly number[]): string {
  const sorted = [...new Set(indices)].sort((a, b) => a - b);
  const parts: string[] = [];
  for (let i = 0; i < sorted.length;) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j]! + 1) j++;
    parts.push(
      j - i >= 2 ? `${sorted[i]}-${sorted[j]}` : sorted.slice(i, j + 1).join(", "),
    );
    i = j + 1;
  }
  return parts.join(", ");
}

/** The valid indices of `value`, or `[]` when it is not a list of them. */
export function asPoseList(value: unknown): number[] {
  return Array.isArray(value)
    ? value.filter((v): v is number => Number.isInteger(v) && v >= 0)
    : [];
}

/** Whether `path` is a `fix_poses` field. */
export const isPoseListPath = (path: string): boolean => /(^|\.)fix_poses$/.test(path);
