import { Callout } from "@vitavision/ui";
import type { RunRef } from "../../../api/generated/types";
import { CameraFrame } from "../CameraFrame";
import { describeSigmaUnavailable } from "../parameterModel";
import type { MapSource } from "./mapSource";
import { gridText, sigmaStatement } from "./mapText";
import { MapPanel, type MapDescription } from "./MapPanel";
import { useSigmaMap } from "./useMaps";
import { useUnderlay } from "./useUnderlay";

interface Context {
  run: string;
  camera: number;
  underlay: { url: string | null; reason: string };
}

function build(source: MapSource, c: Context): MapDescription {
  return {
    quantity: { channel: "sigma", label: "σ", kind: "magnitude" },
    readout: [{ channel: "sigma", label: "σ" }],
    isoLines: true,
    underlay: c.underlay,
    name: `Projection uncertainty of camera ${c.camera}`,
    provenance: `${c.run} · camera ${c.camera} · projection σ · ${gridText(source.grid)}`,
    statement: sigmaStatement(source),
    children: (
      <p>
        The major-axis standard deviation of where a ray lands, from the camera parameters
        alone; the lens layer of the distortion map has none of its own.
      </p>
    ),
  };
}

/** How well each pixel's projection is determined: the 1σ of where the ray
 * through it lands, from the covariance of the camera's own parameters. */
export function SigmaTab() {
  return (
    <CameraFrame title="Projection σ">
      {({ run, camera }) => <Body key={camera} run={run} camera={camera} />}
    </CameraFrame>
  );
}

function Body({ run, camera }: { run: RunRef; camera: number }) {
  const { map, state } = useSigmaMap(run, camera);
  const underlay = useUnderlay(camera);
  if (map?.status === "unavailable") {
    return (
      <Callout tone="info" title="No projection uncertainty for this camera">
        {describeSigmaUnavailable(map.reason)}
      </Callout>
    );
  }
  return (
    <MapPanel
      state={state}
      identity={String(camera)}
      controls={null}
      build={build}
      context={{ run: run.id, camera, underlay }}
    />
  );
}
