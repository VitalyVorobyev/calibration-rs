import type { StageHandle } from "@vitavision/stage2d";
import { DensityProvider } from "@vitavision/ui";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DetectorSpec, ProjectView } from "../../api/generated/types";
import { useProject } from "../../project/projectStore";
import { useReadOnly } from "../../session/jobs";
import { useSelection } from "../../session/selection";
import { useStore } from "../../store";
import { useViewKeys } from "../dataset/useViewKeys";
import { currentCell } from "./cell";
import { savedDetector } from "./detectorModel";
import type { ShownDetection } from "./DetectorTab";
import { ExclusionsTray } from "./ExclusionsTray";
import {
  isFeatureExcluded,
  isImageExcluded,
  isImageOut,
  isViewExcluded,
  toggleFeatures,
  toggleImage,
  toggleView,
  unmatchedPoints,
  type Point3,
} from "./exclusions";
import { FeatureReadout } from "./FeatureReadout";
import { Inspector } from "./Inspector";
import { MatrixPane } from "./MatrixPane";
import { featureRows, rejectedItems, type RejectedItem } from "./overlayModel";
import { StageHeader } from "./StageHeader";
import { StagePane } from "./StagePane";
import { useDetectKeys } from "./useDetectKeys";
import { useDetail, useDetectAll, useOverview, usePreview } from "./useDetectData";
import { changeExclusions } from "./useExclusions";
import { usePicking } from "./usePicking";

/** The Detect screen: what the detector found in each image. The matrix of
 * feature counts on the left finds a bad image among hundreds, the stage in
 * the middle draws the detection over the image (what was kept, what was
 * left out and why), and the inspector reads it, tries other detector
 * settings on this image, and writes the project's exclusions. */
export function DetectScreen() {
  const project = useProject((s) => s.project);
  return project === null ? null : <DetectBody project={project} />;
}

const NO_EXCLUSIONS = {};

function DetectBody({ project }: { project: ProjectView }) {
  const dir = project.dir;
  const readOnly = useReadOnly();
  const exclude = project.spec.exclude ?? NO_EXCLUSIONS;
  const overview = useOverview(dir);
  const detectAll = useDetectAll(dir);
  const selection = useSelection((s) => s.selection);
  const setSelection = useSelection((s) => s.setSelection);
  const noRun = useStore((s) => s.data === null);
  const stageRef = useRef<StageHandle>(null);

  const cell = useMemo(
    () => currentCell(overview.data, selection, project.data_root),
    [overview.data, selection, project.data_root],
  );
  useViewKeys(
    noRun,
    overview.data?.views.length ?? 0,
    overview.data?.cameras.length ?? 0,
  );

  // An image's detail is read from the cache when it has one; for one that
  // has none it is detected only when asked.
  const imageKey = cell?.image ? `${cell.camera}|${cell.image}` : null;
  const [requested, setRequested] = useState<ReadonlySet<string>>(new Set());
  const state = cell?.detection?.state ?? null;
  const wantDetail =
    state?.kind === "detected" || (imageKey !== null && requested.has(imageKey));
  const detail = useDetail(dir, cell?.camera ?? null, cell?.image ?? null, wantDetail);

  // A detection made on request is a cache entry now.
  const refetchOverview = overview.refetch;
  useEffect(() => {
    if (detail.data && state?.kind === "not_detected") refetchOverview();
  }, [detail.data, state?.kind, refetchOverview]);

  // Trying other settings: the edit, the preview of one image, which of the
  // two the stage shows. A preview and the choice belong to one image.
  const { preview: lastPreview, run: runPreview, clear: clearPreview } = usePreview(dir);
  const [working, setWorking] = useState<DetectorSpec | null>(null);
  const [mode, setMode] = useState<{ key: string | null; shown: ShownDetection }>({
    key: null,
    shown: "saved",
  });
  const [applied, setApplied] = useState(false);
  const preview =
    imageKey !== null && lastPreview?.key.startsWith(`${imageKey}|`) ? lastPreview : null;
  const shownMode: ShownDetection =
    mode.key === imageKey && preview?.detail ? mode.shown : "saved";
  const onWorking = useCallback(
    (next: DetectorSpec | null) => {
      setWorking(next);
      if (next === null) clearPreview();
    },
    [clearPreview],
  );
  const shown =
    shownMode === "preview" && preview?.detail ? preview.detail : (detail.data ?? null);

  // Picking and hovering features of what is shown.
  const picking = usePicking(`${imageKey}|${shownMode}`);
  const { picked, hovered } = picking;

  // What the project excludes of this image.
  const camId = cell?.cameraId ?? "";
  const imagePath = cell?.image ?? "";
  const identity = useMemo(
    () =>
      cell
        ? { token: cell.token, images: cell.view.images.map((i) => i?.image ?? null) }
        : null,
    [cell],
  );
  const rows = useMemo(() => (shown ? featureRows(shown) : []), [shown]);
  const excluded = useMemo<ReadonlySet<number>>(
    () =>
      new Set(
        rows
          .filter((r) => isFeatureExcluded(exclude, camId, imagePath, r.world))
          .map((r) => r.index),
      ),
    [rows, exclude, camId, imagePath],
  );
  const rejected = useMemo(() => (shown ? rejectedItems(shown) : []), [shown]);
  const unmatched = useMemo(
    () =>
      shown
        ? unmatchedPoints(
            exclude,
            camId,
            imagePath,
            rows.map((r): Point3 => r.world),
          ).length
        : 0,
    [shown, exclude, camId, imagePath, rows],
  );
  const viewOut = identity !== null && isViewExcluded(exclude, identity);
  const imageOut = cell?.image ? isImageExcluded(exclude, camId, cell.image) : false;
  const anyOut =
    identity !== null && cell?.image
      ? isImageOut(exclude, identity, camId, cell.image)
      : false;

  const excludePicked = useCallback(() => {
    const points = [...picked].flatMap((i) => (rows[i] ? [rows[i].world] : []));
    if (points.length > 0)
      void changeExclusions((e) => toggleFeatures(e, camId, imagePath, points));
  }, [picked, rows, camId, imagePath]);
  const toggleImageOut = useCallback(() => {
    if (imagePath !== "") void changeExclusions((e) => toggleImage(e, camId, imagePath));
  }, [camId, imagePath]);
  const toggleViewOut = useCallback(() => {
    if (identity) void changeExclusions((e) => toggleView(e, identity));
  }, [identity]);

  useDetectKeys(
    {
      exclude: () => (picked.size > 0 ? excludePicked() : toggleImageOut()),
      excludeView: toggleViewOut,
    },
    !readOnly && cell !== null,
  );

  const focusItem = useCallback((item: RejectedItem) => {
    stageRef.current?.frame({ x: item.x - 24, y: item.y - 24, width: 48, height: 48 });
  }, []);

  const hoveredRow = rows[hovered ?? -1] ?? rows[selection.feature ?? -1] ?? null;
  const onSelect = useCallback(
    (camera: number, view: number) => setSelection({ camera, view, feature: null }),
    [setSelection],
  );
  const detecting = imageKey !== null && requested.has(imageKey);

  return (
    <DensityProvider value="compact">
      <div className="grid h-full min-h-0 grid-cols-[minmax(18rem,26rem)_minmax(0,1fr)_minmax(20rem,26rem)] gap-3">
        <section
          aria-label="Images"
          className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-panel border border-line bg-surface p-3"
        >
          <MatrixPane
            overview={overview.data}
            loading={overview.loading}
            error={overview.error?.message ?? null}
            exclude={exclude}
            dataRoot={project.data_root}
            selection={selection}
            onSelect={onSelect}
            detectAll={detectAll}
          />
        </section>

        <section aria-label="Image" className="flex min-h-0 min-w-0 flex-col gap-2">
          <StageHeader
            cell={cell}
            cameras={overview.data?.cameras ?? []}
            onCamera={(camera) =>
              setSelection({ camera, view: selection.view, feature: null })
            }
            detail={shown}
            imageOut={imageOut}
            viewOut={viewOut}
            onToggleImage={toggleImageOut}
            onToggleView={toggleViewOut}
            readOnly={readOnly}
          />
          <div className="min-h-0 flex-1 overflow-hidden rounded-panel border border-line bg-canvas">
            <StagePane
              cell={cell}
              detail={detail}
              shown={shown}
              rejected={rejected}
              detecting={detecting}
              needsDetect={state?.kind === "not_detected" && !wantDetail}
              onDetect={() =>
                imageKey !== null && setRequested(new Set([...requested, imageKey]))
              }
              excluded={excluded}
              imageExcluded={anyOut}
              picked={picked}
              hovered={hovered}
              onHover={picking.hoverHit}
              onPick={picking.pick}
              onBox={picking.box}
              onClear={picking.clear}
              stageRef={stageRef}
              readOnly={readOnly}
            />
          </div>
          <FeatureReadout
            kind={shown?.target.kind ?? null}
            row={hoveredRow}
            excluded={hoveredRow !== null && excluded.has(hoveredRow.index)}
          />
          <ExclusionsTray exclude={exclude} />
        </section>

        <aside
          aria-label="Inspector"
          className="flex min-h-0 flex-col rounded-panel border border-line bg-surface p-3"
        >
          <Inspector
            detection={cell?.detection ?? null}
            shown={shown}
            detailQuery={detail}
            expected={overview.data?.expected ?? null}
            rejected={rejected}
            onFocus={focusItem}
            features={{
              excluded,
              picked,
              hovered,
              onPick: picking.pick,
              onHover: picking.hoverIndex,
              onExclude: excludePicked,
              readOnly,
              unmatched,
            }}
            detector={{
              project,
              working,
              onWorking,
              hasImage: imageKey !== null,
              preview,
              onPreview: () => {
                if (cell?.image) {
                  void runPreview(
                    cell.camera,
                    cell.image,
                    working ?? savedDetector(project),
                  );
                  setMode({ key: imageKey, shown: "preview" });
                }
              },
              shown: shownMode,
              onShown: (next) => setMode({ key: imageKey, shown: next }),
              saved: detail.data,
              detecting: detectAll.running,
              onRedetect: () => {
                setApplied(false);
                void detectAll.start(true);
              },
              applied,
              onApplied: setApplied,
            }}
          />
        </aside>
      </div>
    </DensityProvider>
  );
}
