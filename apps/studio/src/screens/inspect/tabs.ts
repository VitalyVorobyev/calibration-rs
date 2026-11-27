/** Which of Inspect's tabs a run has. Absent, not disabled: a tab the run
 * cannot have is not offered. With no run shown yet (loading, or none
 * chosen) every built tab is offered, each with its own empty state. */
import { INSPECT_TABS, type InspectTab } from "../../shell/screens";

/** `cameras`: the shown run's camera count, `null` while no run is shown. */
export function tabsFor(cameras: number | null): InspectTab[] {
  if (cameras === null) return [...INSPECT_TABS];
  return INSPECT_TABS.filter(
    (tab) => (tab !== "epipolar" && tab !== "depth") || cameras >= 2,
  );
}
