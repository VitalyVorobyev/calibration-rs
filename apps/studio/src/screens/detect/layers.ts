/** Which layers the stage draws, and the menu that switches them. */
export interface LayerState {
  /** The photograph. */
  image: boolean;
  /** The accepted features and the target's marks (markers, edge bits, rings). */
  accepted: boolean;
  /** The candidates the detector left out, dimmed. */
  rejected: boolean;
  /** The lattice between neighbouring corners. */
  edges: boolean;
  /** Lattice indices, marker and ring ids. */
  labels: boolean;
  /** The 1σ position ellipses (magnified: the stage prints the gain). */
  uncertainty: boolean;
  /** A chessboard corner's two edge directions. */
  axes: boolean;
}

export const DEFAULT_LAYERS: LayerState = {
  image: true,
  accepted: true,
  rejected: true,
  edges: true,
  labels: true,
  uncertainty: false,
  axes: false,
};

export const LAYER_MENU: { id: keyof LayerState; label: string; shortcut?: string }[] = [
  { id: "image", label: "Image" },
  { id: "accepted", label: "Detected features" },
  { id: "rejected", label: "Rejected candidates" },
  { id: "edges", label: "Lattice edges" },
  { id: "labels", label: "Labels" },
  { id: "uncertainty", label: "1σ ellipses" },
  { id: "axes", label: "Corner axes" },
];
