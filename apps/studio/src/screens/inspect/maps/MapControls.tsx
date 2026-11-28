import { NumberInput, Select, ToggleChip } from "@vitavision/ui";
import { ToolbarGroup } from "../../../components/ViewerToolbar";
import { SEQUENTIAL_CHOICES, type ColormapChoice } from "./viewerModel";

/** One switch of the layers group: whether it is on, and why it is off the
 * table when it cannot be. */
export interface LayerSwitch {
  checked: boolean;
  onChange: (on: boolean) => void;
  /** The switch is there but cannot be turned on. */
  disabled?: boolean;
  title: string;
  /** The switch's own words, where "Fade not significant" does not fit. */
  label?: string;
}

/** The viewer's own toolbar groups, after the tab's: the colour map, the
 * range (automatic or a fixed top, so two maps share a scale) and the layers
 * a map offers. A signed quantity has the diverging map only. */
export function MapControls({
  signed,
  colormap,
  onColormap,
  fixedMax,
  autoTop,
  onFixed,
  unit,
  layers,
}: {
  signed: boolean;
  colormap: ColormapChoice;
  onColormap: (choice: ColormapChoice) => void;
  /** The fixed top of the range, `null` for automatic. */
  fixedMax: number | null;
  /** The top the automatic range has now, where a fixed one starts. */
  autoTop: number;
  onFixed: (max: number | null) => void;
  unit: string;
  layers: {
    iso?: LayerSwitch;
    arrows?: LayerSwitch;
    mask?: LayerSwitch;
    image?: LayerSwitch;
  };
}) {
  const chip = (label: string, layer: LayerSwitch | undefined) =>
    layer && (
      <ToggleChip
        checked={layer.checked}
        onCheckedChange={layer.onChange}
        disabled={layer.disabled ?? false}
        title={layer.title}
      >
        {label}
      </ToggleChip>
    );
  return (
    <>
      <ToolbarGroup label="Colours">
        <Select
          aria-label="Colour map"
          value={signed ? "blueOrange" : colormap}
          onValueChange={(v) => onColormap((v || "auto") as ColormapChoice)}
          disabled={signed}
          options={
            signed
              ? [{ value: "blueOrange", label: "Blue–orange (signed)" }]
              : SEQUENTIAL_CHOICES.map((c) => ({
                  value: c,
                  label: c === "auto" ? "Automatic" : c,
                }))
          }
          className="w-40"
        />
      </ToolbarGroup>
      <ToolbarGroup label="Range">
        <Select
          aria-label="Range"
          value={fixedMax === null ? "auto" : "fixed"}
          onValueChange={(v) => onFixed(v === "fixed" ? autoTop : null)}
          options={[
            { value: "auto", label: "Automatic" },
            { value: "fixed", label: "Fixed top" },
          ]}
          className="w-32"
        />
        {fixedMax !== null && (
          <NumberInput
            aria-label="Top of the range"
            value={fixedMax}
            min={0}
            unit={unit}
            onValueChange={(max) => max > 0 && onFixed(max)}
            className="w-28"
          />
        )}
      </ToolbarGroup>
      <ToolbarGroup label="Layers">
        {chip("Iso-lines", layers.iso)}
        {chip("Arrows", layers.arrows)}
        {chip(layers.mask?.label ?? "Fade not significant", layers.mask)}
        {chip("Image", layers.image)}
      </ToolbarGroup>
    </>
  );
}
