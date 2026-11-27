/** The library scan, shared: every project under the back end's roots with
 * its runs, fetched once for whoever asks (the Library, the breadcrumb's
 * sibling lists, the device and import dialogs), marked stale by any
 * `project_changed` (`session/sync.ts`) so the next read is fresh. */
import { useEffect } from "react";
import { api } from "../api";
import type { LibraryScan } from "../api/generated/types";
import { useQuery, type Query } from "../session/query";
import { useSession } from "../session/session";

const scan = (): Promise<LibraryScan> => api.scanLibrary();

/** The scan; `rescanOnFocus` asks again when the window regains focus (a
 * project changed by another program shows without a click). */
export function useLibraryScan(rescanOnFocus = false): Query<LibraryScan> {
  const ready = useSession((s) => s.status.kind === "ready");
  const query = useQuery("scan_library", {}, scan, ready);
  const { refetch } = query;
  useEffect(() => {
    if (!rescanOnFocus) return;
    window.addEventListener("focus", refetch);
    return () => window.removeEventListener("focus", refetch);
  }, [rescanOnFocus, refetch]);
  return query;
}
