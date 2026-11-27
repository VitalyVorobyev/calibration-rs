import { SegmentedControl } from "@vitavision/ui";
import { useState } from "react";
import type { RunRef } from "../../../api/generated/types";
import { ToolbarGroup } from "../../../components/ViewerToolbar";
import { CameraFrame } from "../CameraFrame";
import type { MapSource } from "./mapSource";
import { distortionStatement, gridText } from "./mapText";
import { MapPanel, type MapDescription } from "./MapPanel";
import { useDistortionMap } from "./useMaps";
import { useUnderlay } from "./useUnderlay";

type Layer = "distortion" | "sensor" | "total";
type Component = "magnitude" | "dx" | "dy";

const LAYER_WORDS: Record<Layer, string> = {
  distortion: "lens",
  sensor: "sensor tilt",
  total: "lens and sensor tilt together",
};

const COMPONENTS: Record<Component, { label: string; signed: boolean }> = {
  magnitude: { label: "|d|", signed: false },
  dx: { label: "dx", signed: true },
  dy: { label: "dy", signed: true },
};

interface Context {
  run: string;
  camera: number;
  layer: Layer;
  component: Component;
  underlay: { url: string | null; reason: string };
}

function build(source: MapSource, c: Context): MapDescription {
  const quantity = COMPONENTS[c.component];
  return {
    quantity: {
      channel: c.component,
      label: quantity.label,
      kind: quantity.signed ? "signed" : "magnitude",
    },
    readout: [
      { channel: "dx", label: "dx" },
      { channel: "dy", label: "dy" },
      { channel: "magnitude", label: "|d|" },
    ],
    arrows: { dx: "dx", dy: "dy" },
    isoLines: true,
    underlay: c.underlay,
    name: `Distortion map of camera ${c.camera}`,
    provenance: `${c.run} · camera ${c.camera} · ${LAYER_WORDS[c.layer]} · ${gridText(source.grid)} · ${source.grid.image_size[0]} × ${source.grid.image_size[1]} px`,
    statement: distortionStatement(source, LAYER_WORDS[c.layer]),
    children: (
      <p>
        Displacement is where a ray lands minus where it would land without the stage;
        arrows point the way the pixel moves. The lens layer depends only on the
        intrinsics and coefficients. It has no uncertainty of its own: it can be large
        while the projection stays precise (see Projection σ).
      </p>
    ),
  };
}

/** How the camera model moves a pixel: the displacement of where a ray lands
 * against the camera without that stage, as a map over the sensor. */
export function DistortionTab() {
  return (
    <CameraFrame title="Distortion">
      {({ run, camera }) => <Body key={camera} run={run} camera={camera} />}
    </CameraFrame>
  );
}

function Body({ run, camera }: { run: RunRef; camera: number }) {
  const [layer, setLayer] = useState<Layer>("distortion");
  const [component, setComponent] = useState<Component>("magnitude");
  const { map, state } = useDistortionMap(run, camera, layer);
  const underlay = useUnderlay(camera);
  const hasTilt = map?.has_sensor_layer ?? false;

  const controls = (
    <>
      <ToolbarGroup label="Layer">
        <SegmentedControl
          aria-label="Layer"
          value={layer}
          onValueChange={(v) => setLayer((v || "distortion") as Layer)}
          options={[
            { value: "distortion", label: "Lens" },
            ...(hasTilt ? [{ value: "sensor", label: "Tilt" }] : []),
            { value: "total", label: "Total" },
          ]}
        />
      </ToolbarGroup>
      <ToolbarGroup label="Component">
        <SegmentedControl
          aria-label="Component"
          value={component}
          onValueChange={(v) => setComponent((v || "magnitude") as Component)}
          options={(Object.keys(COMPONENTS) as Component[]).map((c) => ({
            value: c,
            label: COMPONENTS[c].label,
          }))}
        />
      </ToolbarGroup>
    </>
  );

  return (
    <MapPanel
      state={state}
      identity={String(camera)}
      controls={controls}
      build={build}
      context={{ run: run.id, camera, layer, component, underlay }}
    />
  );
}
