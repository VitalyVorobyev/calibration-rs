import { Empty, StatusDot } from "@vitavision/ui";
import { ImageLayer, ImageStage, type StageView } from "@vitavision/stage2d";
import { useState } from "react";
import { api } from "../../api";
import type { DatasetSpec, ViewRow } from "../../api/generated/types";
import { imagePath } from "../../components/ViewMatrix/matrixModel";
import { useImageSize, type ImageSize } from "../../hooks/useImageSize";

/** The image in a zoomable stage, opened fitted (a new image, a new key). */
function Viewer({ url, size, label }: { url: string; size: ImageSize; label: string }) {
  const [view, setView] = useState<StageView | null>(null);
  return (
    <ImageStage
      image={size}
      view={view}
      onView={(next) => setView(next)}
      initialView="fit"
      label={label}
      className="bg-canvas"
    >
      <ImageLayer src={url} alt="" pixelatedAbove={1} />
    </ImageStage>
  );
}

/** The selected cell of the matrix: its image in a zoomable stage with its
 * path and size, and the other cameras' images of the same view to move
 * across to. A cell with no image says which camera lacks it. */
export function ViewDetail({
  view,
  cameraIndex,
  cameras,
  dataRoot,
  manifest,
  onCamera,
}: {
  view: ViewRow | null;
  cameraIndex: number | null;
  cameras: readonly string[];
  dataRoot: string;
  manifest: DatasetSpec | null;
  onCamera: (camera: number) => void;
}) {
  const rel = view && cameraIndex !== null ? (view.images[cameraIndex] ?? null) : null;
  const path =
    view && cameraIndex !== null ? imagePath(dataRoot, view, cameraIndex) : null;
  const url = path !== null ? api.urls.image({ path }) : null;
  const size = useImageSize(url);

  if (view === null || cameraIndex === null) {
    return (
      <Empty className="py-10">
        Select a view and a camera in the matrix to see its image here.
      </Empty>
    );
  }
  const camera = cameras[cameraIndex] ?? `camera ${cameraIndex + 1}`;
  return (
    <div className="flex min-h-0 flex-col gap-3">
      <header className="flex flex-col gap-0.5">
        <h2 className="text-sm font-semibold tracking-tight text-fg">
          View <span className="font-mono">{view.token}</span> · {camera}
        </h2>
        {rel === null ? (
          <p className="text-xs text-defect">{camera} has no image for this view.</p>
        ) : (
          <p className="truncate font-mono text-[11px] text-fg-muted" title={rel}>
            {rel}
          </p>
        )}
      </header>

      <div className="h-72 shrink-0 overflow-hidden rounded-control border border-line bg-canvas">
        {url === null ? (
          <div className="grid h-full place-items-center text-xs text-fg-subtle">
            Nothing to show.
          </div>
        ) : size === "error" ? (
          <div
            role="alert"
            className="grid h-full place-items-center px-4 text-center text-xs text-defect"
          >
            The image does not open.
          </div>
        ) : size === null ? (
          <div aria-hidden className="size-full animate-pulse bg-raised" />
        ) : (
          <Viewer key={url} url={url} size={size} label={`Image ${rel ?? ""}`} />
        )}
      </div>

      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
        <dt className="text-fg-subtle">Size</dt>
        <dd className="font-mono text-fg">
          {size !== null && size !== "error" ? `${size.width} × ${size.height} px` : "–"}
        </dd>
        <dt className="text-fg-subtle">Pairing token</dt>
        <dd className="font-mono text-fg">{view.token}</dd>
        {manifest?.robot_poses && (
          <>
            <dt className="text-fg-subtle">Robot pose</dt>
            <dd className="truncate font-mono text-fg" title={manifest.robot_poses.path}>
              {manifest.robot_poses.path}
            </dd>
          </>
        )}
      </dl>

      {cameras.length > 1 && (
        <section
          aria-label="The same view in the other cameras"
          className="flex flex-col gap-1"
        >
          <h3 className="text-[10px] font-semibold tracking-wider text-fg-muted uppercase">
            This view in
          </h3>
          <ul className="flex flex-col">
            {cameras.map((id, i) => {
              const other = view.images[i] ?? null;
              return (
                <li key={id}>
                  <button
                    type="button"
                    aria-current={i === cameraIndex ? "true" : undefined}
                    onClick={() => onCamera(i)}
                    className="flex w-full items-center gap-2 rounded-control px-2 py-1 text-left text-xs outline-none hover:bg-raised focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-signal aria-[current=true]:bg-signal/12"
                  >
                    <span className="w-14 shrink-0 font-mono text-fg">{id}</span>
                    {other === null ? (
                      <StatusDot tone="defect">missing</StatusDot>
                    ) : (
                      <span className="truncate font-mono text-fg-muted">{other}</span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
