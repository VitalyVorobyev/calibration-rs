import type { ValuePlane } from "@vitavision/stage2d";
import type { MapGrid, PlaneChannel } from "../../../api/generated/types";

/** A decoded map plane with what describes it: the grid its nodes sit on and
 * its channels, in the plane's order. */
export interface MapSource {
  plane: ValuePlane;
  grid: MapGrid;
  channels: PlaneChannel[];
}
