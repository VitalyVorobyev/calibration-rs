import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import { invoke } from "@tauri-apps/api/core";
import { isTauriContext, joinPath } from "../lib/tauri";
import type { FrameKey } from "../types";
import type { AnyExport, ExportKind, LoadExportResult } from "./types";
import { inferExportKind } from "./exportShape";

/** Wrap-around index lookup over a sorted array. Returns the value
 * at position `(idx(current) + delta) mod arr.length`, with a
 * fallback to the first/last element when `current` is missing.
 * Lifted from the original ResidualViewer so the store can drive
 * keyboard navigation independently of the diagnose component. */
function nextInWrap(arr: number[], current: number, delta: number): number {
  if (arr.length === 0) return current;
  const idx = arr.indexOf(current);
  if (idx < 0) return delta >= 0 ? arr[0] : arr[arr.length - 1];
  const n = arr.length;
  return arr[(((idx + delta) % n) + n) % n];
}

interface MaterializedExport {
  frames: FrameKey[];
  poseValues: number[];
  cameraValues: number[];
  posesByCamera: Map<number, number[]>;
  camerasByPose: Map<number, number[]>;
  first: FrameKey;
  second: FrameKey;
}

function materializeExport(data: AnyExport, exportDir: string):
  | { ok: true; value: MaterializedExport }
  | { ok: false; error: string } {
  const manifest = data.image_manifest;
  if (!manifest) {
    return {
      ok: false,
      error:
        "This export has no image_manifest field. v0 requires the manifest to render the source images alongside the residuals.",
    };
  }
  const root = joinPath(exportDir, manifest.root);
  const frames: FrameKey[] = manifest.frames.map((f) => ({
    pose: f.pose,
    camera: f.camera,
    label: `pose ${f.pose} · cam ${f.camera}`,
    abs_path: joinPath(root, f.path),
    roi: f.roi,
  }));
  if (frames.length === 0) {
    return {
      ok: false,
      error:
        "This export's image_manifest is empty. v0 needs at least one (pose, camera) frame to render.",
    };
  }

  const poseSet = new Set<number>();
  const cameraSet = new Set<number>();
  const posesByCameraSet = new Map<number, Set<number>>();
  const camerasByPoseSet = new Map<number, Set<number>>();
  for (const f of frames) {
    poseSet.add(f.pose);
    cameraSet.add(f.camera);
    if (!posesByCameraSet.has(f.camera)) {
      posesByCameraSet.set(f.camera, new Set());
    }
    posesByCameraSet.get(f.camera)!.add(f.pose);
    if (!camerasByPoseSet.has(f.pose)) {
      camerasByPoseSet.set(f.pose, new Set());
    }
    camerasByPoseSet.get(f.pose)!.add(f.camera);
  }
  const sortNum = (s: Set<number>) => [...s].sort((a, b) => a - b);
  const poseValues = sortNum(poseSet);
  const cameraValues = sortNum(cameraSet);
  const posesByCamera = new Map(
    [...posesByCameraSet].map(([k, v]) => [k, sortNum(v)]),
  );
  const camerasByPose = new Map(
    [...camerasByPoseSet].map(([k, v]) => [k, sortNum(v)]),
  );

  const first = frames[0];
  const second =
    frames.find((f) => f.pose !== first.pose || f.camera !== first.camera) ??
    first;

  return {
    ok: true,
    value: {
      frames,
      poseValues,
      cameraValues,
      posesByCamera,
      camerasByPose,
      first,
      second,
    },
  };
}

export interface ExportSlice {
  exportPath: string | null;
  exportDir: string | null;
  data: AnyExport | null;
  kind: ExportKind | null;
  frames: FrameKey[];
  poseValues: number[];
  cameraValues: number[];
  posesByCamera: Map<number, number[]>;
  camerasByPose: Map<number, number[]>;
  loadError: string | null;
  loadExport: (path: string) => Promise<void>;
  acceptLiveRunExport: (exportValue: unknown, exportDir: string) => void;
  resetExport: () => void;
  setLoadError: (msg: string | null) => void;
}

export interface SelectionSlice {
  /** Primary pose — used by Diagnose's left pane, Viewer3D, Epipolar. */
  selectedPose: number;
  /** Right-pane pose (Diagnose compare mode only). Most workspaces ignore. */
  selectedPoseB: number;
  /** Primary camera — left pane / active frustum / pane A. */
  cameraA: number;
  /** Secondary camera — right pane / pane B. */
  cameraB: number;
  /** Highlighted feature index (Epipolar crosshair). */
  selectedFeature: number | null;
  hoverFeature: number | null;
  setSelectedPose: (pose: number, which?: "A" | "B") => void;
  setCamera: (camera: number, which?: "A" | "B") => void;
  setSelectedFeature: (f: number | null) => void;
  setHoverFeature: (f: number | null) => void;
  /** Step the active pane's pose (wraps around the camera's
   * available pose set). */
  stepPose: (delta: number, which?: "A" | "B") => void;
  /** Step the active pane's camera (wraps around the pose's
   * available camera set). */
  stepCamera: (delta: number, which?: "A" | "B") => void;
}

export type AppState = ExportSlice & SelectionSlice;

export const useStore = create<AppState>()(
  subscribeWithSelector((set, get) => ({
    // --- Export slice ---
    exportPath: null,
    exportDir: null,
    data: null,
    kind: null,
    frames: [],
    poseValues: [],
    cameraValues: [],
    posesByCamera: new Map(),
    camerasByPose: new Map(),
    loadError: null,

    loadExport: async (path: string) => {
      if (!isTauriContext()) {
        set({
          loadError:
            "Tauri runtime not detected. Launch the app with `bun run tauri dev`.",
        });
        return;
      }
      let result: LoadExportResult;
      try {
        result = await invoke<LoadExportResult>("load_export", { path });
      } catch (e) {
        set({ loadError: `Could not load export: ${e}` });
        return;
      }
      const data = result.export;
      const materialized = materializeExport(data, result.export_dir);
      if (!materialized.ok) {
        set({ loadError: materialized.error });
        return;
      }

      // Promote the validated export into the backend cache *before*
      // updating the frontend state, so `compute_epipolar_overlay` and
      // any future math command sees the same dataset the user is
      // about to interact with. If the commit fails the frontend
      // surfaces the error and stays on the previous export.
      try {
        await invoke("set_active_export", { path, export: data });
      } catch (e) {
        set({ loadError: `Could not commit export to backend cache: ${e}` });
        return;
      }

      set({
        exportPath: path,
        exportDir: result.export_dir,
        data,
        kind: inferExportKind(data),
        frames: materialized.value.frames,
        poseValues: materialized.value.poseValues,
        cameraValues: materialized.value.cameraValues,
        posesByCamera: materialized.value.posesByCamera,
        camerasByPose: materialized.value.camerasByPose,
        loadError: null,
        selectedPose: materialized.value.first.pose,
        selectedPoseB: materialized.value.second.pose,
        cameraA: materialized.value.first.camera,
        cameraB: materialized.value.second.camera,
        selectedFeature: null,
        hoverFeature: null,
      });
    },

    acceptLiveRunExport: (exportValue, exportDir) => {
      const data = exportValue as AnyExport;
      const materialized = materializeExport(data, exportDir);
      if (!materialized.ok) {
        set({ loadError: materialized.error });
        return;
      }
      set({
        exportPath: "<live-run>",
        exportDir,
        data,
        kind: inferExportKind(data),
        frames: materialized.value.frames,
        poseValues: materialized.value.poseValues,
        cameraValues: materialized.value.cameraValues,
        posesByCamera: materialized.value.posesByCamera,
        camerasByPose: materialized.value.camerasByPose,
        loadError: null,
        selectedPose: materialized.value.first.pose,
        selectedPoseB: materialized.value.second.pose,
        cameraA: materialized.value.first.camera,
        cameraB: materialized.value.second.camera,
        selectedFeature: null,
        hoverFeature: null,
      });
    },

    resetExport: () =>
      set({
        exportPath: null,
        exportDir: null,
        data: null,
        kind: null,
        frames: [],
        poseValues: [],
        cameraValues: [],
        posesByCamera: new Map(),
        camerasByPose: new Map(),
        loadError: null,
      }),

    setLoadError: (msg) => set({ loadError: msg }),

    // --- Selection slice ---
    selectedPose: 0,
    selectedPoseB: 0,
    cameraA: 0,
    cameraB: 1,
    selectedFeature: null,
    hoverFeature: null,

    setSelectedPose: (pose, which = "A") =>
      set(which === "A" ? { selectedPose: pose } : { selectedPoseB: pose }),

    setCamera: (camera, which = "A") =>
      set(which === "A" ? { cameraA: camera } : { cameraB: camera }),

    setSelectedFeature: (f) => set({ selectedFeature: f }),
    setHoverFeature: (f) => set({ hoverFeature: f }),

    stepPose: (delta, which = "A") => {
      const s = get();
      if (s.frames.length === 0) return;
      const cam = which === "A" ? s.cameraA : s.cameraB;
      const poses = s.posesByCamera.get(cam) ?? s.poseValues;
      const cur = which === "A" ? s.selectedPose : s.selectedPoseB;
      const next = nextInWrap(poses, cur, delta);
      set(which === "A" ? { selectedPose: next } : { selectedPoseB: next });
    },

    stepCamera: (delta, which = "A") => {
      const s = get();
      if (s.frames.length === 0) return;
      const pose = which === "A" ? s.selectedPose : s.selectedPoseB;
      const cams = s.camerasByPose.get(pose) ?? s.cameraValues;
      const cur = which === "A" ? s.cameraA : s.cameraB;
      const next = nextInWrap(cams, cur, delta);
      set(which === "A" ? { cameraA: next } : { cameraB: next });
    },
  })),
);
