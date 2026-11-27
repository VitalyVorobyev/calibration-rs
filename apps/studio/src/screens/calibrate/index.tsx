import { RunWorkspace } from "../../workspaces/RunWorkspace";

/** Calibrate: for now the Run workspace, which edits the dataset and the
 * configuration and starts the `calibrate` job. It goes when this screen's
 * own editor lands. */
export function CalibrateScreen() {
  return <RunWorkspace />;
}
