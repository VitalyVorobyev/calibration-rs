/** Writing the project's `[exclude]`: each operation is one `save_project`
 * over the revision the project was read at, and is remembered so it can be
 * undone (`⌘Z`). A save refused because the file changed on disk is tried
 * once more on what the disk now holds; a second refusal is the shell's
 * to report. */
import { toast } from "@vitavision/ui";
import { create } from "zustand";
import type { Exclusions, ProjectSpec } from "../../api/generated/types";
import { useProject } from "../../project/projectStore";
import { describeChange } from "./exclusions";

/** `exclude` without its empty lists, so the file says nothing it need not. */
export function compactExclusions(exclude: Exclusions): Exclusions {
  const out: Exclusions = {};
  if ((exclude.views ?? []).length > 0) out.views = exclude.views!;
  if ((exclude.images ?? []).length > 0) out.images = exclude.images!;
  if ((exclude.features ?? []).length > 0) out.features = exclude.features!;
  return out;
}

interface HistoryState {
  /** The project the entries are about. */
  dir: string | null;
  /** The `[exclude]` before each operation, oldest first. */
  before: Exclusions[];
  push: (dir: string, before: Exclusions) => void;
  pop: () => Exclusions | null;
  clear: () => void;
}

const MAX_UNDO = 50;

export const useExclusionHistory = create<HistoryState>()((set, get) => ({
  dir: null,
  before: [],
  push: (dir, before) =>
    set({
      dir,
      before: [...(get().dir === dir ? get().before : []), before].slice(-MAX_UNDO),
    }),
  pop: () => {
    const { before } = get();
    const last = before[before.length - 1];
    if (last === undefined) return null;
    set({ before: before.slice(0, -1) });
    return last;
  },
  clear: () => set({ dir: null, before: [] }),
}));

const same = (a: Exclusions, b: Exclusions) => JSON.stringify(a) === JSON.stringify(b);

function currentSpec(): { dir: string; spec: ProjectSpec } | null {
  const project = useProject.getState().project;
  return project ? { dir: project.dir, spec: project.spec } : null;
}

/** Save `spec` with `exclude`; once more on the disk's version when the
 * file changed under the first attempt. Returns whether it was written. */
async function writeExclusions(
  op: (exclude: Exclusions) => Exclusions,
): Promise<{ before: Exclusions; after: Exclusions } | null> {
  for (let attempt = 0; attempt < 2; attempt++) {
    const now = currentSpec();
    if (now === null) return null;
    const before = now.spec.exclude ?? {};
    const after = compactExclusions(op(before));
    if (same(compactExclusions(before), after)) return { before, after };
    const saved = await useProject.getState().save({ ...now.spec, exclude: after });
    if (saved) return { before, after };
    if (useProject.getState().error?.kind !== "changed_on_disk") return null;
    useProject.getState().clearError();
    if (!(await useProject.getState().reload())) return null;
  }
  // A second refusal: the file changed again; the shell reports it.
  return null;
}

/** Apply `op` to the project's exclusions and write them; on success the
 * change is remembered for `undoExclusion` and said in a toast. */
export async function changeExclusions(
  op: (exclude: Exclusions) => Exclusions,
  options: { quiet?: boolean } = {},
): Promise<boolean> {
  const dir = currentSpec()?.dir;
  const done = await writeExclusions(op);
  if (!done || dir === undefined) return false;
  if (!same(compactExclusions(done.before), done.after)) {
    useExclusionHistory.getState().push(dir, done.before);
    if (!options.quiet)
      toast({ title: describeChange(done.before, done.after), tone: "info" });
  }
  return true;
}

/** Put the exclusions back as they were before the last operation. */
export async function undoExclusion(): Promise<boolean> {
  const dir = currentSpec()?.dir;
  const history = useExclusionHistory.getState();
  if (dir === undefined || history.dir !== dir) return false;
  const before = history.pop();
  if (before === null) return false;
  const done = await writeExclusions(() => before);
  if (!done) {
    // Not written: keep it to try again.
    useExclusionHistory.getState().push(dir, before);
    return false;
  }
  toast({ title: describeChange(done.before, done.after), tone: "info" });
  return true;
}
