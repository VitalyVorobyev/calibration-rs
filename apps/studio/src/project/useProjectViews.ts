/** The layout of a dataset as the library pairs it, live: `project_views`
 * of the saved project, or of a draft spec (a project's unsaved manifest,
 * or a new project's in its data folder), asked again 400 ms after the
 * last edit. The last good answer stays while the next is fetched or
 * fails, so the matrix never goes blank under a keystroke. */
import { useEffect, useRef, useState } from "react";
import { api, asStudioError, type StudioError } from "../api";
import type { ProjectSpec, ProjectViews } from "../api/generated/types";

export interface ViewsState {
  data: ProjectViews | null;
  /** The request in flight, or waiting out the debounce. */
  loading: boolean;
  /** The last failure of the command itself (not a message about the
   * dataset: those are `data.messages`). */
  error: StudioError | null;
}

export const VIEWS_DEBOUNCE_MS = 400;

/** The views of `dir` (a project, or with `spec` a folder that is not one
 * yet) under `spec`; `enabled: false` waits. */
export function useProjectViews(
  dir: string | null,
  spec: ProjectSpec | null,
  enabled = true,
  /** Bumped by the caller to ask again with the same arguments (the saved
   * project changed). */
  refresh: string | number = 0,
): ViewsState {
  const [state, setState] = useState<ViewsState>({
    data: null,
    loading: false,
    error: null,
  });
  const seq = useRef(0);
  const answered = useRef(false);
  const key = JSON.stringify([dir, spec, refresh]);

  useEffect(() => {
    if (!enabled || dir === null) return;
    const mine = ++seq.current;
    setState((s) => (s.loading ? s : { ...s, loading: true }));
    const timer = setTimeout(
      () => {
        void api
          .projectViews(spec === null ? { dir } : { dir, spec })
          .then(
            (data) => ({ data, error: null }),
            (e: unknown) => ({ data: undefined, error: asStudioError(e) }),
          )
          .then((result) => {
            if (mine !== seq.current) return;
            if (result.data) answered.current = true;
            setState((s) => ({
              data: result.data ?? s.data,
              loading: false,
              error: result.error,
            }));
          });
        // The first answer is not an edit's: no need to wait for one.
      },
      answered.current ? VIEWS_DEBOUNCE_MS : 0,
    );
    return () => clearTimeout(timer);
    // `key` is the content of dir, spec and refresh.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);

  return state;
}
