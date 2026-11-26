/** Asking the user for a path: `requestPath` opens the shell's one
 * `PathPicker` dialog and resolves with the chosen path, or `null` when the
 * user cancels. The picker browses the back end's roots (`list_dir`) on
 * both transports, so it never offers a path the back end would refuse. */
import { create } from "zustand";
import type { PickMode } from "./pickRules";

export interface PathRequest {
  title: string;
  mode: PickMode;
}

interface PathPickerState {
  request: PathRequest | null;
  resolve: ((path: string | null) => void) | null;
  /** Answer the open request (and close the dialog). */
  answer: (path: string | null) => void;
}

export const usePathPicker = create<PathPickerState>()((set, get) => ({
  request: null,
  resolve: null,
  answer: (path) => {
    const resolve = get().resolve;
    set({ request: null, resolve: null });
    resolve?.(path);
  },
}));

/** Open the picker for `request`; a request already open is cancelled. */
export function requestPath(request: PathRequest): Promise<string | null> {
  usePathPicker.getState().answer(null);
  return new Promise((resolve) => usePathPicker.setState({ request, resolve }));
}
