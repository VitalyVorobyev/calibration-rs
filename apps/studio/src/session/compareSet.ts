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

export interface CompareSetState {
  /** The project the ticks are of. */
  dir: string | null;
  ids: string[];
  toggle: (dir: string, id: string) => void;
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
