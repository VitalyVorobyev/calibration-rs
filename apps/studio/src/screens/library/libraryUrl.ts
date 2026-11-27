/** The Library's state in the address, so a link reopens it: the search
 * text (`q`), the verdicts kept (`v`, comma-separated, `none` for a project
 * with no run), the workflow (`w`) and the selected tree node (`sel`). The
 * project and run keep their own parameters, which this never touches. */
import type { ProblemKind, Verdict } from "../../api/generated/types";
import { NO_FILTER, type LibraryFilter } from "./libraryModel";

const VERDICTS: readonly (Verdict | "none")[] = ["pass", "warn", "fail", "none"];

export interface LibraryView {
  filter: LibraryFilter;
  /** The selected tree node; `null` is the whole library. */
  selected: string | null;
}

export function fromParams(params: URLSearchParams): LibraryView {
  const verdicts = new Set(
    (params.get("v") ?? "")
      .split(",")
      .filter((v): v is Verdict | "none" => (VERDICTS as readonly string[]).includes(v)),
  );
  const w = params.get("w");
  return {
    filter: {
      ...NO_FILTER,
      query: params.get("q") ?? "",
      verdicts,
      topology: w ? (w as ProblemKind) : null,
    },
    selected: params.get("sel") || null,
  };
}

/** `params` with the Library's keys replaced by `view`'s (absent when at
 * their defaults). */
export function toParams(params: URLSearchParams, view: LibraryView): URLSearchParams {
  const next = new URLSearchParams(params);
  const set = (key: string, value: string | null) => {
    if (value) next.set(key, value);
    else next.delete(key);
  };
  set("q", view.filter.query === "" ? null : view.filter.query);
  set(
    "v",
    view.filter.verdicts.size > 0
      ? VERDICTS.filter((v) => view.filter.verdicts.has(v)).join(",")
      : null,
  );
  set("w", view.filter.topology);
  set("sel", view.selected);
  return next;
}
