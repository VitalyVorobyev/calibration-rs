import { Button, ErrorBox, Skeleton } from "@vitavision/ui";
import { useState, type ReactNode } from "react";
import { ViewerToolbar } from "../../../components/ViewerToolbar";
import { MapViewer, type MapViewerProps } from "./MapViewer";
import type { MapState } from "./useMaps";
import type { MapSource } from "./mapSource";

/** What a tab says about its map: the viewer's props but the plane and the
 * controls, which the panel supplies. */
export type MapDescription = Omit<MapViewerProps, "source" | "controls">;

/** A map in its tab: the viewer once the plane is decoded, the tab's own
 * `controls` always (a switch that vanished while its plane loads could not
 * be switched back), and plain words for loading and failure. While a new
 * plane of the same `identity` (the camera) loads, the last one stays,
 * marked busy, and the viewer keeps its colour and layer choices.
 *
 * `build` makes the viewer's props of a plane and `context`; it is a plain
 * function of its arguments, so a tab defines it once, outside itself. */
export function MapPanel<C>({
  state,
  identity,
  controls,
  build,
  context,
  failure,
}: {
  state: MapState;
  /** What a retained plane belongs to: another identity drops it. */
  identity: string;
  controls: ReactNode;
  build: (source: MapSource, context: C) => MapDescription;
  context: C;
  /** Words for an error, when the library's own are not the ones to show. */
  failure?: (message: string, kind: string) => string | undefined;
}) {
  const [kept, setKept] = useState<{ identity: string; source: MapSource } | null>(null);
  if (state.source && (kept?.source !== state.source || kept.identity !== identity)) {
    setKept({ identity, source: state.source });
  }
  const shown = state.source ?? (kept?.identity === identity ? kept.source : null);

  if (shown) {
    return (
      <div
        className="flex min-h-0 flex-1 flex-col"
        aria-busy={state.loading || undefined}
      >
        <MapViewer
          key={identity}
          source={shown}
          controls={controls}
          {...build(shown, context)}
        />
      </div>
    );
  }
  const words = state.error
    ? failure?.(state.error.message, state.error.kind)
    : undefined;
  return (
    <div className="flex flex-col gap-3">
      <ViewerToolbar label="Map display">{controls}</ViewerToolbar>
      {state.error ? (
        <ErrorBox>
          {words ?? `Could not make this map: ${state.error.message}`}{" "}
          <Button size="sm" onClick={state.refetch}>
            Retry
          </Button>
        </ErrorBox>
      ) : (
        <div className="flex flex-col gap-2" aria-busy="true">
          <Skeleton className="h-72 w-full" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      )}
    </div>
  );
}
