/** Preset card for the Run workspace quick-start grid.
 *
 * Enabled cards show the dataset metadata and a "Use preset" button.
 * Disabled cards are visually faded, carry a milestone badge, and explain
 * why they are not yet available. Disabled cards are non-interactive;
 * pointer-events are suppressed via Tailwind so no click handler is
 * needed.
 */
import type { Preset } from "./presets";

interface PresetCardProps {
  preset: Preset;
  isActive: boolean;
  onUse: (preset: Preset & { disabled?: false }) => void;
}

export function PresetCard({ preset, isActive, onUse }: PresetCardProps) {
  const disabled = preset.disabled === true;

  return (
    <div
      className={[
        "flex flex-col gap-3 rounded-lg border p-4 transition-colors",
        disabled
          ? "border-border opacity-40 cursor-not-allowed select-none"
          : isActive
            ? "border-brand bg-brand/[0.06]"
            : "border-border bg-bg-soft hover:border-brand/50 cursor-pointer",
      ].join(" ")}
    >
      {/* Card header: name + badges */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex flex-col gap-1 min-w-0">
          <span
            className={[
              "text-[13px] font-semibold leading-snug",
              disabled ? "text-muted-foreground" : "text-foreground",
            ].join(" ")}
          >
            {preset.name}
          </span>
          <span className="text-[11px] text-muted-foreground truncate">
            {preset.group}
          </span>
        </div>

        <div className="flex shrink-0 flex-col items-end gap-1.5">
          {/* Topology badge */}
          <TopologyBadge topology={preset.topology} />

          {/* Milestone badge for disabled cards */}
          {disabled && (
            <span
              className="rounded-full border border-border px-1.5 py-px font-mono text-[9px] uppercase tracking-widest text-muted-foreground"
              title={preset.disabledReason}
            >
              {preset.milestone}
            </span>
          )}

          {/* Active indicator for the currently selected preset */}
          {!disabled && isActive && (
            <span className="font-mono text-[10px] uppercase tracking-widest text-brand">
              active
            </span>
          )}
        </div>
      </div>

      {/* Card body: target info + image count */}
      <div className="flex flex-col gap-1">
        <MetaRow icon="target" label={preset.targetSummary} />
        {preset.imageCount != null && (
          <MetaRow icon="images" label={`${preset.imageCount} images`} />
        )}
        {disabled && (
          <p className="mt-1 text-[11px] text-muted-foreground">
            {preset.disabledReason}
          </p>
        )}
      </div>

      {/* Action button — only for enabled cards */}
      {!disabled && (
        <button
          type="button"
          onClick={() => onUse(preset as Preset & { disabled?: false })}
          className={[
            "mt-auto h-8 rounded-md border px-3 text-[12px] font-medium transition-colors",
            isActive
              ? "border-brand bg-brand text-white"
              : "border-border bg-bg text-foreground hover:border-brand/60 hover:bg-brand/[0.05]",
          ].join(" ")}
        >
          {isActive ? "Preset active" : "Use preset"}
        </button>
      )}
    </div>
  );
}

// ── Sub-components ───────────────────────────────────────────────────────────

function TopologyBadge({ topology }: { topology: string }) {
  return (
    <span
      className={[
        "rounded-full px-2 py-px text-[10px] font-medium",
        topologyColor(topology),
      ].join(" ")}
    >
      {topology}
    </span>
  );
}

/** Deterministic color mapping for known topology names. Unknown names fall
 * back to a neutral muted style so future topologies never cause a render
 * error. */
function topologyColor(topology: string): string {
  switch (topology) {
    case "PlanarIntrinsics":
      return "bg-brand/[0.12] text-brand";
    case "ScheimpflugIntrinsics":
      return "bg-[var(--color-accent,_#a78bfa)]/[0.12] text-[var(--color-accent,_#a78bfa)]";
    case "RigExtrinsics":
    case "RigHandeye":
    case "RigLaserlineDevice":
      return "bg-[var(--color-success,_#22c55e)]/[0.12] text-[var(--color-success,_#22c55e)]";
    case "SingleCamHandeye":
      return "bg-[var(--color-warning,_#f59e0b)]/[0.12] text-[var(--color-warning,_#f59e0b)]";
    default:
      return "bg-bg text-muted-foreground border border-border";
  }
}

function MetaRow({ icon, label }: { icon: "target" | "images"; label: string }) {
  return (
    <div className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground">
      {icon === "target" ? (
        // Crosshair icon
        <svg
          width="10"
          height="10"
          viewBox="0 0 10 10"
          fill="none"
          aria-hidden="true"
          className="shrink-0 opacity-60"
        >
          <circle cx="5" cy="5" r="3.5" stroke="currentColor" strokeWidth="1" />
          <line x1="5" y1="0" x2="5" y2="2.5" stroke="currentColor" strokeWidth="1" />
          <line x1="5" y1="7.5" x2="5" y2="10" stroke="currentColor" strokeWidth="1" />
          <line x1="0" y1="5" x2="2.5" y2="5" stroke="currentColor" strokeWidth="1" />
          <line x1="7.5" y1="5" x2="10" y2="5" stroke="currentColor" strokeWidth="1" />
        </svg>
      ) : (
        // Photo stack icon
        <svg
          width="10"
          height="10"
          viewBox="0 0 10 10"
          fill="none"
          aria-hidden="true"
          className="shrink-0 opacity-60"
        >
          <rect x="1" y="2.5" width="8" height="6" rx="1" stroke="currentColor" strokeWidth="1" />
          <path d="M2.5 1.5h5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
          <path d="M3.5 0.5h3" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
        </svg>
      )}
      <span>{label}</span>
    </div>
  );
}
