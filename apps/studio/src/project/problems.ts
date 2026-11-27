/** How many errors the open project's dataset draft has, as the Dataset
 * screen last judged it, for the rail's badge to show from any screen. The
 * count is only the Dataset screen's to set; it belongs to one project and
 * says nothing about another. */
import { create } from "zustand";

interface ProblemsState {
  /** The project folder the count is about. */
  dir: string | null;
  errors: number;
  set: (dir: string, errors: number) => void;
  clear: () => void;
}

export const useDatasetProblems = create<ProblemsState>()((set) => ({
  dir: null,
  errors: 0,
  set: (dir, errors) => set({ dir, errors }),
  clear: () => set({ dir: null, errors: 0 }),
}));
