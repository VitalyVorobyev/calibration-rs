import { Select } from "@vitavision/ui";
import { useState } from "react";
import type { ResidualMap, RunRef } from "../../../api/generated/types";
import { ToolbarGroup } from "../../../components/ViewerToolbar";
import { useSelection } from "../../../session/selection";
import { useStore } from "../../../store";
import type { ViewScope } from "../residualModel";
import { MapPanel, type MapDescription } from "./MapPanel";
import { channelIndex, finiteValues, SIGNIFICANCE_P01 } from "./mapModel";
import type { MapSource } from "./mapSource";
import { residualProvenance, residualStatement } from "./mapText";
import { RESIDUAL_QUANTITIES, type ResidualQuantity } from "./residualQuantities";
import type { MapState } from "./useMaps";
import { useUnderlay } from "./useUnderlay";
import { ViewsControl } from "./ViewsControl";

interface Context {
  run: string;
  camera: number;
  quantity: ResidualQuantity;
  map: ResidualMap | undefined;
  underlay: { url: string | null; reason: string };
}

const READOUT = [
  { channel: "mean_magnitude", label: "|μ|" },
  { channel: "mean_dx", label: "μx" },
  { channel: "mean_dy", label: "μy" },
  { channel: "rms", label: "RMS" },
  { channel: "std_error", label: "std. err." },
  { channel: "significance", label: "χ" },
  { channel: "effective_samples", label: "samples" },
  { channel: "effective_views", label: "views" },
];

function build(source: MapSource, c: Context): MapDescription {
  const { map, quantity } = c;
  const tested =
    finiteValues(source.plane, channelIndex(source.channels, "significance")).length > 0;
  return {
    quantity,
    readout: READOUT,
    arrows: { dx: "mean_dx", dy: "mean_dy", on: true },
    isoLines: true,
    mask: { channel: "significance", minimum: SIGNIFICANCE_P01 },
    ...(quantity.channel === "significance"
      ? { threshold: { value: SIGNIFICANCE_P01, label: "p = 0.01" } }
      : {}),
    underlay: c.underlay,
    name: `Residual map of camera ${c.camera}`,
    provenance: map
      ? residualProvenance(map, c.run)
      : `${c.run} · camera ${c.camera} · residual map`,
    statement: residualStatement(source),
    children: (
      <>
        {!tested && (
          <p>
            Too few views for a significance test: the map includes the pose error of the
            view{map?.views === 1 ? "" : "s"}, and nothing is faded.
          </p>
        )}
        {map?.chain_level && (
          <p>
            Chain-level residuals: this run&apos;s board poses come through a rig or robot
            chain, so the residuals include its pose error as well as the camera
            model&apos;s misfit.
          </p>
        )}
        {map && map.outliers.length > 0 && (
          <p>
            {map.outliers.length} records were left out as outliers; the table below marks
            them.
          </p>
        )}
        <p>
          Unsupported areas are blank, not zero. A significant field is not a reason to
          add a field term, and a flat map does not prove the model.
        </p>
      </>
    ),
  };
}

function failure(message: string, kind: string, camera: number): string | undefined {
  return kind === "invalid"
    ? `This run has no per-feature residuals for camera ${camera}, so there is no residual map (${message}).`
    : undefined;
}

/** Camera `camera`'s residual map (`map`, over the views of `scope`): the
 * smooth part of `observed − projected` across the sensor. */
export function ResidualMapPanel({
  run,
  camera,
  scope,
  onScope,
  map,
  state,
}: {
  run: RunRef;
  camera: number;
  scope: ViewScope;
  onScope: (scope: ViewScope) => void;
  map: ResidualMap | undefined;
  state: MapState;
}) {
  const selectedView = useSelection((s) => s.selection.view);
  const views = useStore((s) => s.posesByCamera.get(camera) ?? s.poseValues);
  const [channel, setChannel] = useState(RESIDUAL_QUANTITIES[0]!.channel);
  const quantity = RESIDUAL_QUANTITIES.find((q) => q.channel === channel)!;
  const underlay = useUnderlay(camera);

  const controls = (
    <>
      <ToolbarGroup label="Show">
        <Select
          aria-label="Quantity"
          value={channel}
          onValueChange={(v) => v && setChannel(v)}
          options={RESIDUAL_QUANTITIES.map((q) => ({ value: q.channel, label: q.label }))}
          className="w-44"
        />
      </ToolbarGroup>
      <ViewsControl
        scope={scope}
        onScope={onScope}
        views={views}
        selected={selectedView}
      />
    </>
  );

  return (
    <MapPanel
      state={state}
      identity={String(camera)}
      controls={controls}
      build={build}
      context={{ run: run.id, camera, quantity, map, underlay }}
      failure={(message, kind) => failure(message, kind, camera)}
    />
  );
}
