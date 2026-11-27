/** What the shell keeps in step with the back end while a session is
 * ready: the standing event stream (jobs, the connection), the project
 * another tab or program changed, and the selection bridge. One call
 * starts them all; the returned function stops them.
 *
 * `project_changed { dir }` marks the cached answers about `dir` stale and
 * re-reads the open project when it is `dir` (coalesced: a save and the
 * run it started arrive together). After the stream comes back from a drop
 * everything is read again, since events may have been missed. */
import { events } from "../api";
import { useProject } from "../project/projectStore";
import { useSession } from "./session";
import { startJobWatch } from "./jobs";
import { queryCache } from "./query";
import { startSelectionBridge } from "./selection";

/** How long project changes are gathered before the project is re-read. */
const COALESCE_MS = 150;

export function startSync(): () => void {
  const pending = new Set<string>();
  let timer: ReturnType<typeof setTimeout> | null = null;

  const flush = () => {
    timer = null;
    const open = useProject.getState().project?.dir;
    const dirs = [...pending];
    pending.clear();
    if (open !== undefined && dirs.includes(open)) void useProject.getState().reload();
  };

  const stopEvents = events.listen((event) => {
    if (event.body.kind !== "project_changed") return;
    const { dir } = event.body;
    queryCache.invalidateProject(dir);
    // Any project's change moves the library (its runs, its verdicts).
    queryCache.invalidate((command) => command === "scan_library");
    pending.add(dir);
    if (timer === null) timer = setTimeout(flush, COALESCE_MS);
  });

  const stopJobs = startJobWatch(() => {
    queryCache.invalidate(() => true);
    // Asks the back end again: a server that restarted may want the token.
    void useSession.getState().refresh();
    if (useProject.getState().project) void useProject.getState().reload();
  });
  const stopSelection = startSelectionBridge();

  return () => {
    stopEvents();
    stopJobs();
    stopSelection();
    if (timer !== null) clearTimeout(timer);
  };
}
