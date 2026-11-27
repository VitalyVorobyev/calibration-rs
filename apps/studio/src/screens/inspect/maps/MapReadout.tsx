import type { PlaneChannel } from "../../../api/generated/types";
import { readoutNumber, unitLabel, type Readout } from "./mapModel";

/** What is under the pointer on a map: its image position and the value of
 * each listed channel with its unit, `no data` where the node has none.
 * It is also the map's text alternative. */
export function MapReadout({
  reading,
  channels,
  readout,
}: {
  reading: Readout | null;
  channels: readonly PlaneChannel[];
  readout: { channel: string; label: string }[];
}) {
  return (
    <section aria-label="Value under the pointer" data-testid="map-readout">
      <h3 className="mb-1 text-[11px] uppercase tracking-wider text-fg-muted">Pointer</h3>
      {reading === null ? (
        <p className="text-xs text-fg-muted">Move over the map to read a value.</p>
      ) : (
        <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 font-mono text-xs tabular-nums">
          <dt className="text-fg-muted">u, v</dt>
          <dd className="text-right">
            {Math.round(reading.x)}, {Math.round(reading.y)} px
          </dd>
          {readout.map(({ channel, label }) => {
            const at = channels.findIndex((c) => c.name === channel);
            const unit = unitLabel(channels[at]?.unit ?? "");
            return (
              <div key={channel} className="contents">
                <dt className="text-fg-muted">{label}</dt>
                <dd className="text-right">
                  {readoutNumber(reading.values[at])}
                  {unit && Number.isFinite(reading.values[at]) ? ` ${unit}` : ""}
                </dd>
              </div>
            );
          })}
        </dl>
      )}
    </section>
  );
}
