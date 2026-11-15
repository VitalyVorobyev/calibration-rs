/** The browser viewer's start: load the run the server serves. */
import { viewerSession } from "../lib/backend";
import { useStore } from "./index";

/** Load the run the browser viewer serves into the store; does nothing on
 * the other back ends. A failure lands in the store's `loadError`. */
export async function openServedRun(): Promise<void> {
  let session;
  try {
    session = await viewerSession();
  } catch (e) {
    useStore.getState().setLoadError(`Could not reach the viewer: ${String(e)}`);
    return;
  }
  if (session) await useStore.getState().loadExport(session.export_path);
}
