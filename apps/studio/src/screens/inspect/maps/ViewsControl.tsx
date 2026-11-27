import { Button, DropdownMenu, MenuCheckboxItem, Select } from "@vitavision/ui";
import { ToolbarGroup } from "../../../components/ViewerToolbar";
import type { ViewScope } from "../residualModel";

/** Which views a residual map is over: all, the selected one, or a chosen
 * few (a menu of the camera's views). */
export function ViewsControl({
  scope,
  onScope,
  views,
  selected,
}: {
  scope: ViewScope;
  onScope: (scope: ViewScope) => void;
  /** The views the camera has. */
  views: readonly number[];
  /** The selection's view, if any. */
  selected: number | null;
}) {
  const chosen = scope.kind === "subset" ? scope.views : [];
  const toggle = (view: number, on: boolean) =>
    onScope({
      kind: "subset",
      views: on ? [...chosen, view] : chosen.filter((v) => v !== view),
    });
  return (
    <ToolbarGroup label="Views">
      <Select
        aria-label="Views"
        value={scope.kind}
        onValueChange={(v) => {
          if (v === "all") onScope({ kind: "all" });
          else if (v === "view") onScope({ kind: "view" });
          else onScope({ kind: "subset", views: chosen.length ? chosen : [...views] });
        }}
        options={[
          { value: "all", label: "All views", note: String(views.length) },
          {
            value: "view",
            label: selected === null ? "Selected view" : `View ${selected}`,
            disabled: selected === null,
            ...(selected === null ? { note: "select a view first" } : {}),
          },
          { value: "subset", label: "Chosen views" },
        ]}
        className="w-36"
      />
      {scope.kind === "subset" && (
        <DropdownMenu
          trigger={
            <Button size="sm">
              {chosen.length} of {views.length} views
            </Button>
          }
        >
          {views.map((v) => (
            <MenuCheckboxItem
              key={v}
              checked={chosen.includes(v)}
              onCheckedChange={(on) => toggle(v, on)}
            >
              View {v}
            </MenuCheckboxItem>
          ))}
        </DropdownMenu>
      )}
    </ToolbarGroup>
  );
}
