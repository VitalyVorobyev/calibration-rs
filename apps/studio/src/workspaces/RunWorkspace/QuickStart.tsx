import { Button } from "@vitavision/ui";
import { Check } from "lucide-react";
import { PresetCard } from "./PresetCard";
import { PRESETS, type EnabledPreset } from "./presets";

interface QuickStartGridProps {
  activePresetId: string | null;
  onUse: (preset: EnabledPreset) => void;
  onCollapse: () => void;
}

/** The preset card grid. */
export function QuickStartGrid({
  activePresetId,
  onUse,
  onCollapse,
}: QuickStartGridProps) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-[12px] font-semibold tracking-tight text-fg">Quick start</h3>
        {activePresetId && (
          <Button variant="ghost" size="sm" onClick={onCollapse}>
            collapse
          </Button>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {PRESETS.map((preset) => (
          <PresetCard
            key={preset.id}
            preset={preset}
            isActive={preset.id === activePresetId}
            onUse={(p) => onUse(p)}
          />
        ))}
      </div>
    </section>
  );
}

interface ActivePresetBarProps {
  /** The preset in use; `undefined` for a custom setup. */
  preset: EnabledPreset | undefined;
  onChangePreset: () => void;
}

/** The collapsed quick start: the preset in use (or "Custom") and a way
 * back to the grid. */
export function ActivePresetBar({ preset, onChangePreset }: ActivePresetBarProps) {
  return (
    <div className="flex items-center justify-between rounded-control border border-signal/40 bg-signal/[0.05] px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="flex size-4 shrink-0 items-center justify-center rounded-full bg-signal/20 text-signal">
          <Check aria-hidden className="size-2.5" />
        </span>
        <span className="text-[12px] font-medium text-fg">
          {preset ? preset.name : "Custom"}
        </span>
        {preset && (
          <span className="font-mono text-[11px] text-fg-muted">
            {preset.targetSummary}
          </span>
        )}
      </div>
      <Button variant="ghost" size="sm" onClick={onChangePreset}>
        change
      </Button>
    </div>
  );
}
