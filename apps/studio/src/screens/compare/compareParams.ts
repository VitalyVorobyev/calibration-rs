/** What the Compare screen keeps in its address: the runs compared (in
 * order, the first is the reference) and the tab. A link reopens the same
 * comparison. */

export type CompareTab = "parameters" | "fit" | "maps";

export const COMPARE_TABS: readonly CompareTab[] = ["parameters", "fit", "maps"];

export function tabOf(params: URLSearchParams): CompareTab {
  const t = params.get("tab");
  return COMPARE_TABS.find((c) => c === t) ?? "parameters";
}

/** The run ids named by `runs=a,b,c`, without repeats. */
export function runsOf(params: URLSearchParams): string[] {
  const list = (params.get("runs") ?? "").split(",").filter((s) => s.length > 0);
  return list.filter((id, i) => list.indexOf(id) === i);
}

/** `params` with the runs and the tab written (the default tab and an
 * empty set are left out). */
export function withCompare(
  params: URLSearchParams,
  runs: readonly string[],
  tab: CompareTab,
): URLSearchParams {
  const next = new URLSearchParams(params);
  if (runs.length > 0) next.set("runs", runs.join(","));
  else next.delete("runs");
  if (tab === "parameters") next.delete("tab");
  else next.set("tab", tab);
  return next;
}
