/** Which of the shell's dialogs is open: Settings, the shortcut list and
 * the command palette. One store so a key, a button and the palette can
 * all open them. */
import { create } from "zustand";

export type ShellDialog = "settings" | "shortcuts" | "palette";

export interface UiState {
  dialog: ShellDialog | null;
  open: (dialog: ShellDialog) => void;
  close: () => void;
  /** Open `dialog`, or close it when it is the one open. */
  toggle: (dialog: ShellDialog) => void;
}

export const useUi = create<UiState>()((set, get) => ({
  dialog: null,
  open: (dialog) => set({ dialog }),
  close: () => set({ dialog: null }),
  toggle: (dialog) => set({ dialog: get().dialog === dialog ? null : dialog }),
}));
