import {
  Callout,
  Checkbox,
  Empty,
  SegmentedControl,
  Select,
  SkeletonRows,
} from "@vitavision/ui";
import { useMemo, useState } from "react";
import type { DeviceHistoryView, DeviceRunRow } from "../../api/generated/types";
import { fixed } from "../../lib/format";
import { runKey, unitKey, whenText } from "./deviceModel";
import { SeedActions } from "./SeedActions";
import { seedable, seedRequest, type SeedChoice, type SourceKind } from "./seedModel";
import { SeedPreviewView } from "./SeedPreviewView";
import { useSeedPreview } from "./useDeviceData";

/** One run as a source offers it. */
interface Source {
  key: string;
  unit: string;
  run: DeviceRunRow;
}

const label = (s: Source, many: boolean): string =>
  `${many ? `${s.unit} · ` : ""}${s.run.project_name} · ${s.run.id}${s.run.starred ? " ★" : ""}${s.run.representative ? " (representative)" : ""}`;

/** Seed the next calibration: choose where the values come from (a run,
 * chosen runs, or the units of the type), see what the seed will contain,
 * write it to a file, or make it the open project's seed. */
export function SeedPanel({
  views,
  units,
  hasType,
}: {
  /** The history shown: one device's, or every unit's of the type. */
  views: DeviceHistoryView[];
  /** The type's units, for "the type's units" (empty without a type). */
  units: DeviceHistoryView[];
  hasType: boolean;
}) {
  const many = views.length > 1;
  const sources = useMemo<Source[]>(
    () =>
      views.flatMap((v) =>
        v.runs
          .filter(seedable)
          .map((run) => ({ key: runKey(run.project, run.id), unit: v.device.id, run })),
      ),
    [views],
  );
  const ready = useMemo(
    () => units.filter((u) => u.runs.some((r) => r.representative)),
    [units],
  );
  const [choice, setChoice] = useState<SeedChoice>(() => ({
    kind: "run",
    run: (sources.find((s) => s.run.representative) ?? sources[0])?.key ?? null,
    runs: [],
    units: ready.map((u) => unitKey(u.device)),
  }));

  const request = useMemo(
    () =>
      seedRequest(
        choice,
        views.flatMap((v) => v.runs),
        units,
      ),
    [choice, views, units],
  );
  const from = "from" in request ? request.from : null;
  const preview = useSeedPreview(from);

  const toggle = (list: readonly string[], key: string, on: boolean): string[] =>
    on ? [...list, key] : list.filter((k) => k !== key);

  return (
    <div className="grid min-h-0 grid-cols-1 gap-6 xl:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
      <div className="flex flex-col gap-4">
        <Callout tone="info" title="A seed is a starting value">
          It gives the next calibration of this device something to start from; the
          calibration still estimates its values from its own images. One exception: a
          tilted (Scheimpflug) sensor is refined within bounds around the seed, and a
          laserline device holds the seeded tilt.
        </Callout>
        <section aria-label="Seed source" className="flex flex-col gap-2">
          <h3 className="text-sm font-semibold text-fg">Seed from</h3>
          <SegmentedControl
            aria-label="Seed source"
            value={choice.kind}
            options={[
              { value: "run", label: "This run" },
              { value: "runs", label: "Chosen runs" },
              ...(hasType ? [{ value: "units", label: "The type’s units" }] : []),
            ]}
            onValueChange={(kind) => setChoice({ ...choice, kind: kind as SourceKind })}
          />
          {choice.kind === "run" && (
            <Select
              aria-label="Run to seed from"
              value={choice.run ?? ""}
              placeholder="Choose a run"
              options={sources.map((s) => ({ value: s.key, label: label(s, many) }))}
              onValueChange={(run) => setChoice({ ...choice, run })}
            />
          )}
          {choice.kind === "runs" && (
            <ul className="flex max-h-72 flex-col gap-1 overflow-y-auto">
              {sources.map((s) => (
                <li key={s.key}>
                  <Checkbox
                    checked={choice.runs.includes(s.key)}
                    label={label(s, many)}
                    description={whenText(s.run.timestamp)}
                    onCheckedChange={(on) =>
                      setChoice({ ...choice, runs: toggle(choice.runs, s.key, on) })
                    }
                  />
                </li>
              ))}
              {sources.length === 0 && (
                <li className="text-xs text-fg-muted">No run to choose from.</li>
              )}
            </ul>
          )}
          {choice.kind === "units" && (
            <>
              <p className="text-xs text-fg-muted">
                Each unit contributes its representative run.
              </p>
              <ul className="flex flex-col gap-1">
                {units.map((u) => {
                  const rep = u.runs.find((r) => r.representative);
                  return (
                    <li key={unitKey(u.device)}>
                      <Checkbox
                        checked={choice.units.includes(unitKey(u.device))}
                        disabled={!rep}
                        label={u.device.label ?? u.device.id}
                        description={
                          rep
                            ? `${rep.id} of ${rep.project_name}${rep.verdict ? ` · ${rep.verdict}` : ""}`
                            : "no representative run: no starred or passing calibrated run"
                        }
                        onCheckedChange={(on) =>
                          setChoice({
                            ...choice,
                            units: toggle(choice.units, unitKey(u.device), on),
                          })
                        }
                      />
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </section>
        <SeedActions from={from} />
      </div>

      <section aria-label="Seed preview" className="flex min-w-0 flex-col gap-3">
        <h3 className="text-sm font-semibold text-fg">What the seed will contain</h3>
        {"problem" in request ? (
          <Empty>{request.problem}</Empty>
        ) : preview.error && preview.data === undefined ? (
          <Callout tone="error" title="No seed can be made from this">
            {preview.error.message}
          </Callout>
        ) : preview.data === undefined ? (
          <div role="status" aria-busy aria-label="Making the seed">
            <SkeletonRows rows={6} />
          </div>
        ) : (
          <div className={preview.loading ? "opacity-60" : undefined}>
            {preview.error && (
              <Callout tone="error" title="No seed can be made from this">
                {preview.error.message}
              </Callout>
            )}
            <SeedPreviewView preview={preview.data} />
          </div>
        )}
        {from !== null && preview.data !== undefined && (
          <p className="text-[11px] text-fg-muted">
            {fixed(preview.data.seed.cameras.length, 0)}{" "}
            {preview.data.seed.cameras.length === 1 ? "camera" : "cameras"} · nothing of
            the capture (target poses, residuals) and no uncertainty is carried.
          </p>
        )}
      </section>
    </div>
  );
}
