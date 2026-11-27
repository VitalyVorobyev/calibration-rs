/** The app-wide selection store: `{ camera, view, feature }` and a
 * transient `hover` of the same shape. The rules are `lib/selection.ts`.
 *
 * The workspaces built before the shell read and write the export store's
 * own pose and camera slots (`cameraA`, `selectedPose`, `selectedFeature`).
 * `startSelectionBridge` keeps the two in step, in both directions, so
 * those workspaces follow the shell's keys and the shell follows their
 * clicks; it goes with them when each screen moves to this store. */
import { create } from "zustand";
import {
  escapeSelection,
  NO_SELECTION,
  sameSelection,
  withCamera,
  withFeature,
  withView,
  type Selection,
} from "../lib/selection";
import { useStore } from "../store";

export interface SelectionState {
  selection: Selection;
  /** What the pointer is over; never the selection itself. */
  hover: Selection;
  selectCamera: (camera: number | null) => void;
  selectView: (view: number | null) => void;
  selectFeature: (feature: number | null) => void;
  /** Replace the whole selection (the bridge, deep links). */
  setSelection: (selection: Selection) => void;
  setHover: (hover: Selection) => void;
  /** `Esc`: clear one level; `false` when nothing was selected. */
  escape: () => boolean;
}

export const useSelection = create<SelectionState>()((set, get) => ({
  selection: NO_SELECTION,
  hover: NO_SELECTION,
  selectCamera: (camera) => set({ selection: withCamera(get().selection, camera) }),
  selectView: (view) => set({ selection: withView(get().selection, view) }),
  selectFeature: (feature) => set({ selection: withFeature(get().selection, feature) }),
  setSelection: (selection) => {
    if (!sameSelection(get().selection, selection)) set({ selection });
  },
  setHover: (hover) => {
    if (!sameSelection(get().hover, hover)) set({ hover });
  },
  escape: () => {
    const before = get().selection;
    const after = escapeSelection(before);
    if (after === before) return false;
    set({ selection: after });
    return true;
  },
}));

/** What the export store's slots say, as a selection; none without a run
 * shown (its slots then hold defaults, not choices). */
function legacySelection(): Selection {
  const s = useStore.getState();
  if (s.data === null) return NO_SELECTION;
  return { camera: s.cameraA, view: s.selectedPose, feature: s.selectedFeature };
}

/** Keep the selection and the export store's slots one: returns the stop
 * function. The selection wins where they differ at the start. */
export function startSelectionBridge(): () => void {
  const toLegacy = (selection: Selection) => {
    const legacy = useStore.getState();
    if (legacy.data === null) return;
    if (selection.camera !== null && selection.camera !== legacy.cameraA) {
      legacy.setCamera(selection.camera);
    }
    if (selection.view !== null && selection.view !== legacy.selectedPose) {
      legacy.setSelectedPose(selection.view);
    }
    if (selection.feature !== legacy.selectedFeature) {
      legacy.setSelectedFeature(selection.feature);
    }
  };

  const stopSelection = useSelection.subscribe((state, prev) => {
    if (state.selection !== prev.selection) toLegacy(state.selection);
  });
  const stopLegacy = useStore.subscribe((state, prev) => {
    if (
      state.data === prev.data &&
      state.cameraA === prev.cameraA &&
      state.selectedPose === prev.selectedPose &&
      state.selectedFeature === prev.selectedFeature
    ) {
      return;
    }
    useSelection.getState().setSelection(legacySelection());
  });
  useSelection.getState().setSelection(legacySelection());
  return () => {
    stopSelection();
    stopLegacy();
  };
}
