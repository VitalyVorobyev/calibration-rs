/** Pure project-model rules: run names and folder labels. */

/** The next free `run-NNN` name: one past the highest numbered run, so a
 * deleted run's number is not reused while later runs exist. */
export function nextRunName(existing: readonly string[]): string {
  let highest = 0;
  for (const name of existing) {
    const match = /^run-(\d+)$/.exec(name);
    if (match) highest = Math.max(highest, Number(match[1]));
  }
  return `run-${String(highest + 1).padStart(3, "0")}`;
}

/** The last path segment, for labels. */
export function folderName(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  const idx = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return idx >= 0 ? trimmed.slice(idx + 1) : trimmed;
}
