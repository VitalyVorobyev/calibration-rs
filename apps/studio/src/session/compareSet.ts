/** The runs ticked in the run list to be compared: the Compare screen's
 * input, kept in the order they were ticked (the first is the baseline).
 * A set belongs to one project; another project, or a run that is gone,
 * empties it of what no longer exists. */
import { create } from "zustand";

/** `ids` with `id` toggled: appended when ticked, removed when unticked. */
export function toggled(ids: readonly string[], id: string): string[] {
  return ids.includes(id) ? ids.filter((i) => i !== id) : [...ids, id];
}

/** `ids` without the runs `existing` no longer holds. */
export function pruned(ids: readonly string[], existing: readonly string[]): string[] {
  const kept = ids.filter((i) => existing.includes(i));
  return kept.length === ids.length ? (ids as string[]) : kept;
}

/** `ids` with `id` moved `by` places (negative: earlier); the same array
 * when it cannot move. Moving to the front makes it the reference. */
export function moved(ids: readonly string[], id: string, by: number): string[] {
  const from = ids.indexOf(id);
  const to = Math.min(ids.length - 1, Math.max(0, from + by));
  if (from < 0 || to === from) return ids as string[];
  const next = ids.filter((i) => i !== id);
  next.splice(to, 0, id);
  return next;
}

export interface CompareSetState {
  /** The project the ticks are of. */
  dir: string | null;
  ids: string[];
  toggle: (dir: string, id: string) => void;
  /** Replace the set of project `dir` (an address that names it). */
  setIds: (dir: string, ids: string[]) => void;
  /** Move run `id` by `by` places in the set (the first is the reference). */
  move: (id: string, by: number) => void;
  /** Take run `id` out of the set. */
  remove: (id: string) => void;
  /** Keep the ticks of runs `existing` of project `dir` only. */
  follow: (dir: string | null, existing: readonly string[]) => void;
  clear: () => void;
}

export const useCompareSet = create<CompareSetState>()((set, get) => ({
  dir: null,
  ids: [],
  toggle: (dir, id) => {
    const base = get().dir === dir ? get().ids : [];
    set({ dir, ids: toggled(base, id) });
  },
  setIds: (dir, ids) => set({ dir, ids }),
  move: (id, by) => {
    const ids = moved(get().ids, id, by);
    if (ids !== get().ids) set({ ids });
  },
  remove: (id) => set({ ids: get().ids.filter((i) => i !== id) }),
  follow: (dir, existing) => {
    const state = get();
    if (state.dir !== dir) {
      if (state.ids.length > 0 || state.dir !== null) set({ dir, ids: [] });
      return;
    }
    const ids = pruned(state.ids, existing);
    if (ids !== state.ids) set({ ids });
  },
  clear: () => set({ ids: [] }),
}));
