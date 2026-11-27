import { TreeView, type TreeNode } from "@vitavision/workbench";
import { Boxes, CircuitBoard, FolderKanban, FolderTree } from "lucide-react";
import { useMemo } from "react";
import {
  NO_DEVICE,
  NO_DEVICE_ID,
  projectNodeId,
  worstVerdict,
  type LibraryGroups,
  type ProjectRow,
} from "./libraryModel";
import { VerdictDot } from "./VerdictMark";

/** `product → device → project`, from the (filtered) groups. A group shows
 * the worst verdict among its projects and how many there are; the
 * ungrouped projects are the last group. Arrow keys and Enter are the
 * tree's own. */
export function LibraryTree({
  groups,
  selected,
  onSelect,
  expandAll,
}: {
  groups: LibraryGroups;
  selected: string | null;
  onSelect: (id: string) => void;
  /** Open every group (a search is on, so what matched is not hidden). */
  expandAll: boolean;
}) {
  const nodes = useMemo(() => buildNodes(groups), [groups]);
  return (
    <TreeView
      // A search re-opens the tree on what it found.
      key={expandAll ? "all" : "default"}
      aria-label="Projects by device"
      nodes={nodes}
      selectedId={selected}
      onSelect={(id) => onSelect(id)}
      defaultExpandAll={expandAll}
      defaultExpanded={expandAll ? undefined : groups.products.map((p) => p.id)}
    />
  );
}

const count = (n: number) => (
  <span className="font-mono text-[11px] text-fg-subtle">{n}</span>
);

function groupMeta(rows: readonly ProjectRow[]) {
  return (
    <span className="flex items-center gap-2">
      <VerdictDot verdict={worstVerdict(rows.map((r) => r.verdict))} />
      {count(rows.length)}
    </span>
  );
}

function buildNodes(groups: LibraryGroups): TreeNode[] {
  const nodes: TreeNode[] = groups.products.map((product) => ({
    id: product.id,
    label: product.label,
    icon: <Boxes aria-hidden className="size-3.5" />,
    meta: groupMeta(product.devices.flatMap((d) => d.rows)),
    children: product.devices.map((device) => ({
      id: device.id,
      label: device.device.label
        ? `${device.device.label} · ${device.device.id}`
        : device.device.id,
      icon: <CircuitBoard aria-hidden className="size-3.5" />,
      meta: groupMeta(device.rows),
      children: device.rows.map(projectNode),
    })),
  }));
  if (groups.ungrouped.length > 0) {
    nodes.push({
      id: NO_DEVICE_ID,
      label: NO_DEVICE,
      icon: <FolderTree aria-hidden className="size-3.5" />,
      meta: groupMeta(groups.ungrouped),
      children: groups.ungrouped.map(projectNode),
    });
  }
  return nodes;
}

function projectNode(row: ProjectRow): TreeNode {
  return {
    id: projectNodeId(row.entry.dir),
    label: row.entry.name,
    icon: <FolderKanban aria-hidden className="size-3.5" />,
    meta: <VerdictDot verdict={row.verdict} />,
  };
}
