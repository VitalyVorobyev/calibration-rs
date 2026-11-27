/** The one viewer of the three maps (distortion, residual, projection σ):
 * a zoomable stage over the camera's image extent with the heatmap, its
 * iso-lines and arrows and an optional image under it; a vertical legend
 * with the unit, its clipping and its marked limit; the reading under the
 * pointer; and the toolbar that sets the colour map, the range and the
 * layers. What each map says of itself comes in as props; nothing here
 * knows which map it draws. */
import { ColormapLegend, colormapRgb } from "@vitavision/charts";
import {
  ContourLayer,
  HeatmapLayer,
  ImageLayer,
  ImageStage,
  StageReadout,
  StageToolbar,
  VectorField,
  VectorFieldKey,
  type Point,
  type StageView,
} from "@vitavision/stage2d";
import { useMemo, useState, type ReactNode } from "react";
import { ViewerToolbar } from "../../../components/ViewerToolbar";
import { LegendKey } from "./LegendKey";
import { MapControls } from "./MapControls";
import { MapReadout } from "./MapReadout";
import type { MapSource } from "./mapSource";
import {
  autoDomain,
  channelIndex,
  clipsHigh,
  clipsLow,
  finiteValues,
  maskedBelow,
  planeRect,
  quiver,
  quiverEvery,
  readoutAt,
  unitLabel,
} from "./mapModel";
import {
  chosenDomain,
  effectiveColormap,
  heatOpacity,
  isoStep,
  legendClipped,
  legendMarkers,
  type ColormapChoice,
  type MapKind,
} from "./viewerModel";

/** The zoom from which a heatmap shows its nodes as blocks: never, so a
 * coarse plane is interpolated across the image, not blocked. */
const SMOOTH = 1e6;

/** The arrows' scale: the 95th-percentile one spans 26 screen pixels, so a
 * few large ones do not shrink the rest; the key prints it. */
const arrowScale = { length: 26, percentile: 0.95 };

export interface MapQuantity {
  /** The channel's name in the plane. */
  channel: string;
  /** What the legend calls it (`|d|`, `σ`). */
  label: string;
  kind: MapKind;
}

export interface MapViewerProps {
  source: MapSource;
  quantity: MapQuantity;
  /** The channels the pointer reading lists, in order. */
  readout: { channel: string; label: string }[];
  /** Arrows of `(dx, dy)`, on at first when `on`; the switch is there when
   * this is. */
  arrows?: { dx: string; dy: string; on?: boolean } | null | undefined;
  /** The tab's own controls (which layer, which quantity): the first groups
   * of the toolbar. */
  controls?: ReactNode;
  /** Whether iso-lines of the quantity make sense (a smooth field). */
  isoLines?: boolean | undefined;
  /** Fade the nodes whose `channel` is below `minimum` (a significance
   * test); the switch is there when this is, and some node has a test. */
  mask?: { channel: string; minimum: number } | null | undefined;
  /** A limit on the quantity's scale, marked on the legend with its name. */
  threshold?: { value: number; label: string } | undefined;
  /** The image under the map (`url`), or why there is none (`reason`);
   * `null` for a map with no such choice. */
  underlay?: { url: string | null; reason: string } | null | undefined;
  /** The map's accessible name. */
  name: string;
  /** The provenance strip: what this map is of. */
  provenance: string;
  /** What the map says, in a line. */
  statement?: string | null | undefined;
  /** Further captions under the strip. */
  children?: ReactNode;
}

export function MapViewer({
  source,
  quantity,
  readout,
  arrows,
  controls,
  isoLines = false,
  mask,
  threshold,
  underlay,
  name,
  provenance,
  statement,
  children,
}: MapViewerProps) {
  const { plane, grid, channels } = source;
  const channel = channelIndex(channels, quantity.channel);
  const unit = unitLabel(channels[channel]?.unit ?? "");
  const signed = quantity.kind === "signed";

  const [view, setView] = useState<StageView | null>(null);
  const [cursor, setCursor] = useState<Point | null>(null);
  const [colormapChoice, setColormapChoice] = useState<ColormapChoice>("auto");
  // The person's top value for a channel; another channel is automatic again.
  const [fixed, setFixed] = useState<{ channel: string; max: number } | null>(null);
  const [isoOn, setIsoOn] = useState(true);
  const [arrowsOn, setArrowsOn] = useState(arrows?.on ?? false);
  const [maskOn, setMaskOn] = useState(true);
  const [imageOn, setImageOn] = useState(false);

  const auto = useMemo(
    () => autoDomain(plane, channel, signed),
    [plane, channel, signed],
  );
  const fixedMax = fixed?.channel === quantity.channel ? fixed.max : null;
  const domain = chosenDomain(quantity.kind, auto, fixedMax);
  const showImage = imageOn && !!underlay?.url;
  const map = effectiveColormap(quantity.kind, colormapChoice, showImage);
  const colormap = useMemo(() => (t: number) => colormapRgb(map, t), [map]);

  const maskName = mask?.channel;
  const maskMinimum = mask?.minimum ?? 0;
  const maskChannel = maskName ? channelIndex(channels, maskName) : -1;
  const maskUsable = useMemo(
    () => maskChannel >= 0 && finiteValues(plane, maskChannel).length > 0,
    [plane, maskChannel],
  );
  const masking = !!mask && maskOn && maskUsable;
  const bright = useMemo(
    () => (masking ? maskedBelow(plane, channel, maskChannel, maskMinimum) : null),
    [masking, plane, channel, maskChannel, maskMinimum],
  );

  const every = quiverEvery(grid.columns);
  const arrowDx = arrows?.dx;
  const arrowDy = arrows?.dy;
  const arrowChannels = useMemo(
    () =>
      arrowDx && arrowDy && arrowsOn
        ? [channelIndex(channels, arrowDx), channelIndex(channels, arrowDy)]
        : null,
    [arrowDx, arrowDy, arrowsOn, channels],
  );
  const test = useMemo(
    () => (masking ? { channel: maskChannel, minimum: maskMinimum } : undefined),
    [masking, maskChannel, maskMinimum],
  );
  // With a significance mask, the arrows where it holds are drawn in full and
  // the others faint, as the map under them is.
  const arrowData = useMemo(
    () =>
      arrowChannels
        ? quiver(plane, grid, arrowChannels[0]!, arrowChannels[1]!, every, test)
        : null,
    [arrowChannels, plane, grid, every, test],
  );
  const faintArrows = useMemo(
    () =>
      arrowChannels && test
        ? quiver(plane, grid, arrowChannels[0]!, arrowChannels[1]!, every, {
            ...test,
            below: true,
          })
        : null,
    [arrowChannels, plane, grid, every, test],
  );

  const reading = useMemo(
    () => (cursor ? readoutAt(plane, grid, cursor.x, cursor.y) : null),
    [cursor, plane, grid],
  );
  const cursorValue = reading?.values[channel];
  const highClip = useMemo(
    () => clipsHigh(plane, channel, domain),
    [plane, channel, domain],
  );
  const lowClip = useMemo(
    () => signed && clipsLow(plane, channel, domain),
    [signed, plane, channel, domain],
  );

  const rect = useMemo(() => planeRect(grid), [grid]);
  const levels = useMemo(() => ({ step: isoStep(domain) }), [domain]);
  const imageSize = { width: grid.image_size[0], height: grid.image_size[1] };
  const opacity = heatOpacity(showImage);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <ViewerToolbar label="Map display">
        {controls}
        <MapControls
          signed={signed}
          colormap={colormapChoice}
          onColormap={setColormapChoice}
          fixedMax={fixedMax}
          autoTop={domain[1]}
          onFixed={(max) =>
            setFixed(max === null ? null : { channel: quantity.channel, max })
          }
          unit={unit}
          layers={{
            ...(isoLines
              ? {
                  iso: {
                    checked: isoOn,
                    onChange: setIsoOn,
                    title: "Lines of equal value",
                  },
                }
              : {}),
            ...(arrows
              ? {
                  arrows: {
                    checked: arrowsOn,
                    onChange: setArrowsOn,
                    title: "Arrows along the direction of the displacement",
                  },
                }
              : {}),
            ...(mask
              ? {
                  mask: {
                    checked: masking,
                    onChange: setMaskOn,
                    disabled: !maskUsable,
                    title: maskUsable
                      ? "Fade where the value is not distinguishable from zero"
                      : "Too few views for a significance test: nothing to fade",
                  },
                }
              : {}),
            ...(underlay
              ? {
                  image: {
                    checked: showImage,
                    onChange: setImageOn,
                    disabled: !underlay.url,
                    title: underlay.url
                      ? "Show the frame under the map"
                      : underlay.reason,
                  },
                }
              : {}),
          }}
        />
      </ViewerToolbar>

      <div className="flex min-h-0 flex-1 gap-3">
        <div className="relative min-h-[20rem] min-w-0 flex-1 overflow-hidden rounded-control bg-canvas">
          <ImageStage
            image={imageSize}
            view={view}
            onView={(next) => setView(next)}
            onHover={setCursor}
            initialView="fit"
            label={name}
            className="bg-canvas"
            toolbar={<StageToolbar />}
            readout={
              <StageReadout
                cursor={cursor}
                extra={
                  arrowData ? (
                    <VectorFieldKey vectors={arrowData.vectors} unit={unit} bare />
                  ) : undefined
                }
              />
            }
          >
            {showImage && underlay?.url && (
              <ImageLayer src={underlay.url} alt="" pixelatedAbove={4} />
            )}
            <div className="pointer-events-none absolute inset-0 overflow-hidden">
              {bright ? (
                <>
                  <HeatmapLayer
                    plane={plane}
                    channel={channel}
                    colormap={colormap}
                    range={{ low: domain[0], high: domain[1] }}
                    rect={rect}
                    pixelatedAbove={SMOOTH}
                    opacity={opacity * 0.25}
                  />
                  <HeatmapLayer
                    plane={bright}
                    channel={0}
                    colormap={colormap}
                    range={{ low: domain[0], high: domain[1] }}
                    rect={rect}
                    pixelatedAbove={SMOOTH}
                    opacity={opacity}
                  />
                </>
              ) : (
                <HeatmapLayer
                  plane={plane}
                  channel={channel}
                  colormap={colormap}
                  range={{ low: domain[0], high: domain[1] }}
                  rect={rect}
                  pixelatedAbove={SMOOTH}
                  opacity={opacity}
                />
              )}
              {isoLines && isoOn && (
                <ContourLayer
                  plane={plane}
                  channel={channel}
                  levels={levels}
                  dashedLevels={signed ? [0] : undefined}
                  labels
                  rect={rect}
                  label={`Iso-lines of ${quantity.label}`}
                />
              )}
            </div>
            {faintArrows && faintArrows.anchors.length > 0 && (
              <div className="pointer-events-none absolute inset-0 opacity-35">
                <VectorField
                  anchors={faintArrows.anchors}
                  vectors={faintArrows.vectors}
                  ids={faintArrows.ids}
                  role="structure"
                  scale={arrowScale}
                  pickable={false}
                  label="Arrows where the value is not distinguishable from zero"
                />
              </div>
            )}
            {arrowData && (
              <VectorField
                anchors={arrowData.anchors}
                vectors={arrowData.vectors}
                ids={arrowData.ids}
                role="structure"
                scale={arrowScale}
                pickable={false}
                label="Arrows"
              />
            )}
          </ImageStage>
        </div>

        <aside className="flex w-44 shrink-0 flex-col gap-4 overflow-y-auto">
          <ColormapLegend
            map={map}
            domain={domain}
            symmetric={signed}
            unit={unit}
            label={quantity.label}
            orientation="vertical"
            length={200}
            clipped={legendClipped(highClip, lowClip)}
            marker={legendMarkers(threshold, cursorValue, unit)}
          />
          <LegendKey faded={masking} />
          <MapReadout reading={reading} channels={channels} readout={readout} />
        </aside>
      </div>

      <div className="flex flex-col gap-0.5 text-xs text-fg-muted">
        <p className="font-mono">{provenance}</p>
        {statement && <p className="text-fg">{statement}</p>}
        {children}
      </div>
    </div>
  );
}
