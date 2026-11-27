import { useEffect, useState } from "react";
import { api, asStudioError } from "../../api";
import type { ProblemKind, StudioError } from "../../api/generated/types";
import { errorMessages } from "../../project/configCheck";
import { useRunEditor } from "../../project/runEditor";

/** How long an edit rests before the library is asked about it. */
export const CHECK_DELAY_MS = 300;

export type CheckState =
  /** The configuration is valid: the project's own, or one the library accepted. */
  | { status: "ok" }
  /** An edit the library has not answered yet. */
  | { status: "checking" }
  | { status: "error"; error: StudioError };

export interface EditorCheck {
  state: CheckState;
  /** The library's complaint as form messages, by field. */
  messages: Record<string, string>;
}

/** What the library answered for one configuration. */
interface Answer {
  config: unknown;
  error: StudioError | null;
}

/** Asks the library, once an edit has rested, whether `config` is a valid
 * configuration of `topology`: `config_patch` (does it read) then
 * `resolve_config` (does it check). `edited` is false while the editor
 * holds the project's own configuration, which is valid already. A valid
 * answer leaves the editor's patch from the defaults in the store, what a
 * recorded run's patch is compared with. */
export function useEditorCheck(
  topology: ProblemKind | null,
  config: unknown,
  edited: boolean,
  projectPatch: Record<string, unknown>,
): EditorCheck {
  const [answer, setAnswer] = useState<Answer | null>(null);

  useEffect(() => {
    const setPatch = useRunEditor.getState().setEditorPatch;
    if (!edited || topology === null || config === null) {
      setPatch(projectPatch);
      return;
    }
    let live = true;
    const timer = setTimeout(() => {
      void (async () => {
        try {
          const patch = await api.configPatch({ topology, config });
          await api.resolveConfig({ topology, patch });
          if (!live) return;
          setAnswer({ config, error: null });
          setPatch(patch);
        } catch (e) {
          if (!live) return;
          setAnswer({ config, error: asStudioError(e) });
          setPatch(null);
        }
      })();
    }, CHECK_DELAY_MS);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [topology, config, edited, projectPatch]);

  // The answer is for one configuration: any other is still being checked.
  const state: CheckState = !edited
    ? { status: "ok" }
    : answer === null || answer.config !== config
      ? { status: "checking" }
      : answer.error
        ? { status: "error", error: answer.error }
        : { status: "ok" };

  const messages = errorMessages(state.status === "error" ? state.error : null, config);
  return { state, messages };
}
