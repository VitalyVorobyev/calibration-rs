/** Which project dialog is open: a new project, an export to import, a
 * device to set. One store so the Library, the project menu and the context
 * bar open the same dialogs (`shell/ProjectDialogs.tsx` hosts them). */
import { create } from "zustand";

export type ProjectDialog =
  /** A new project; `manifest` starts it from that dataset manifest. */
  | { kind: "new"; manifest?: string }
  /** Import an export as a run; `dir` pre-selects the project. */
  | { kind: "import"; dir?: string }
  /** Set the device of the project in `dir`. */
  | { kind: "device"; dir: string };

interface DialogState {
  open: ProjectDialog | null;
  show: (dialog: ProjectDialog) => void;
  close: () => void;
}

export const useProjectDialog = create<DialogState>()((set) => ({
  open: null,
  show: (open) => set({ open }),
  close: () => set({ open: null }),
}));
