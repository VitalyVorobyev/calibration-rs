import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Canvas } from '@react-three/fiber';
import { Html, OrbitControls } from '@react-three/drei';
import {
  AlertCircle,
  BarChart3,
  Box,
  Camera,
  FolderOpen,
  Image as ImageIcon,
  Layers,
  Loader2,
  MousePointer2,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import * as THREE from 'three';
import { composeTransforms, formatMm, formatPx, transformToPose, vec3 } from './geometry';
import {
  parseManifest,
  looksLikeBenchRecord,
  parseBenchRecord,
  resolveAssetUrl,
  type BenchRecord,
  type CompactLevelReport,
  type CameraRecord,
  type LaserCamStat,
  type LaserFeatureRecord,
  type ReprojLevelGap,
  type TargetFeatureRecord,
  type ViewerManifest,
  type ViewerStage,
} from './schema';
import './styles.css';

type ImageMode = 'target' | 'laser';

interface LoadedViewerManifest {
  kind: 'viewer';
  manifest: ViewerManifest;
  manifestUrl: string;
}

interface LoadedBenchRecord {
  kind: 'bench';
  record: BenchRecord;
  recordUrl: string;
}

type LoadedData = LoadedViewerManifest | LoadedBenchRecord;

function defaultDataUrl(): string {
  const params = new URLSearchParams(window.location.search);
  const benchUrl = params.get('bench');
  if (benchUrl) return benchUrl;
  return params.get('manifest') ?? '/viewer-data/puzzle/viewer_manifest.json';
}

function App() {
  const [loaded, setLoaded] = useState<LoadedData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [stageId, setStageId] = useState('stage4');
  const [poseIndex, setPoseIndex] = useState(0);
  const [cameraIndex, setCameraIndex] = useState(0);
  const [imageMode, setImageMode] = useState<ImageMode>('target');
  const [selected, setSelected] = useState<string>('rig');
  const [showPlanes, setShowPlanes] = useState(true);
  const [showBoards, setShowBoards] = useState(true);
  const [showRobot, setShowRobot] = useState(false);

  async function loadDataUrl(url: string) {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      applyLoaded(parseLoadedJson(await response.json(), new URL(url, window.location.href).toString()));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadDataUrl(defaultDataUrl());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function applyLoaded(next: LoadedData) {
    setLoaded(next);
    if (next.kind === 'viewer') {
      setStageId(next.manifest.stages.at(-1)?.id ?? next.manifest.stages[0].id);
      setPoseIndex(next.manifest.poses[0]?.index ?? 0);
      setCameraIndex(0);
    }
  }

  async function loadFromFolder(files: FileList | null) {
    if (!files) return;
    const manifestFile = Array.from(files).find((f) => f.name === 'viewer_manifest.json');
    if (!manifestFile) {
      setError('Selected folder does not contain viewer_manifest.json');
      return;
    }
    setLoading(true);
    try {
      const parsed = parseManifest(JSON.parse(await manifestFile.text()));
      const blobMap = new Map<string, string>();
      for (const file of Array.from(files)) {
        blobMap.set(file.webkitRelativePath || file.name, URL.createObjectURL(file));
      }
      const folderUrl = `folder://${manifestFile.webkitRelativePath || manifestFile.name}`;
      applyLoaded({ kind: 'viewer', manifest: patchFolderUrls(parsed, blobMap), manifestUrl: folderUrl });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function loadFromJsonFile(file: File | null) {
    if (!file) return;
    setLoading(true);
    try {
      applyLoaded(parseLoadedJson(JSON.parse(await file.text()), `file://${file.name}`));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const manifest = loaded?.kind === 'viewer' ? loaded.manifest : undefined;
  const manifestUrl = loaded?.kind === 'viewer' ? loaded.manifestUrl : '';
  const stage = manifest?.stages.find((s) => s.id === stageId) ?? manifest?.stages.at(-1);
  const pose = manifest?.poses.find((p) => p.index === poseIndex) ?? manifest?.poses[0];
  const title = loaded?.kind === 'bench'
    ? loaded.record.ident.dataset_id
    : manifest?.dataset.name ?? 'No report loaded';
  const eyebrow = loaded?.kind === 'bench'
    ? 'Benchmark Quality Dashboard'
    : 'Calibration Geometry Inspector';

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
        </div>
        <div className="load-controls">
          <button onClick={() => loadDataUrl(defaultDataUrl())} disabled={loading}>
            {loading ? <Loader2 className="spin" size={16} /> : <FolderOpen size={16} />}
            Load URL
          </button>
          <label className="file-button">
            <BarChart3 size={16} />
            Open JSON
            <input
              type="file"
              accept="application/json,.json"
              onChange={(event) => void loadFromJsonFile(event.currentTarget.files?.[0] ?? null)}
            />
          </label>
          <label className="file-button">
            <FolderOpen size={16} />
            Open Folder
            <input
              type="file"
              // @ts-expect-error webkitdirectory is intentionally used for local debug tooling.
              webkitdirectory=""
              multiple
              onChange={(event) => loadFromFolder(event.currentTarget.files)}
            />
          </label>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <AlertCircle size={18} />
          {error}
        </div>
      )}

      {loaded?.kind === 'bench' ? (
        <BenchDashboard record={loaded.record} />
      ) : !manifest || !stage || !pose ? (
        <EmptyState loading={loading} onLoad={() => loadDataUrl(defaultDataUrl())} />
      ) : (
        <main className="workbench">
          <aside className="sidebar">
            <SectionTitle icon={<Layers size={16} />} label="Stage" />
            <div className="segmented vertical">
              {manifest.stages.map((s) => (
                <button
                  key={s.id}
                  className={s.id === stage.id ? 'active' : ''}
                  onClick={() => setStageId(s.id)}
                >
                  {s.label}
                  <span>{formatPx(s.mean_reproj_error_px)}</span>
                </button>
              ))}
            </div>

            <SectionTitle icon={<MousePointer2 size={16} />} label="Selection" />
            <label>
              Pose
              <select value={poseIndex} onChange={(e) => setPoseIndex(Number(e.target.value))}>
                {manifest.poses.map((p) => (
                  <option key={p.index} value={p.index}>
                    {p.index} · {p.snap_type}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Camera
              <select value={cameraIndex} onChange={(e) => setCameraIndex(Number(e.target.value))}>
                {stage.geometry.cameras.map((c) => (
                  <option key={c.index} value={c.index}>
                    Camera {c.index}
                  </option>
                ))}
              </select>
            </label>

            <SectionTitle icon={<Box size={16} />} label="3D Layers" />
            <label className="checkline">
              <input type="checkbox" checked={showPlanes} onChange={(e) => setShowPlanes(e.target.checked)} />
              Laser planes
            </label>
            <label className="checkline">
              <input type="checkbox" checked={showBoards} onChange={(e) => setShowBoards(e.target.checked)} />
              Board poses
            </label>
            <label className="checkline">
              <input type="checkbox" checked={showRobot} onChange={(e) => setShowRobot(e.target.checked)} />
              Robot poses
            </label>

            <SelectionDetails selected={selected} stage={stage} cameraIndex={cameraIndex} />
            <PlaneProjection stage={stage} cameraIndex={cameraIndex} />
          </aside>

          <section className="main-grid">
            <div className="viewport-panel">
              <Canvas camera={{ position: [0.35, -0.45, 0.32], fov: 42 }}>
                <color attach="background" args={['#f8faf9']} />
                <ambientLight intensity={0.8} />
                <directionalLight position={[0.5, -0.8, 1.0]} intensity={1.3} />
                <GeometryScene
                  manifest={manifest}
                  stage={stage}
                  activePose={poseIndex}
                  activeCamera={cameraIndex}
                  showPlanes={showPlanes}
                  showBoards={showBoards}
                  showRobot={showRobot}
                  onSelect={setSelected}
                />
                <OrbitControls makeDefault enableDamping />
              </Canvas>
            </div>

            <ImageInspector
              manifest={manifest}
              manifestUrl={manifestUrl}
              stage={stage}
              poseIndex={poseIndex}
              cameraIndex={cameraIndex}
              mode={imageMode}
              onModeChange={setImageMode}
            />
          </section>

          <aside className="rightbar">
            <Diagnostics stage={stage} cameraIndex={cameraIndex} />
            <WorstFeatures stage={stage} onPick={(pose, camera) => {
              setPoseIndex(pose);
              setCameraIndex(camera);
              setImageMode('target');
            }} />
          </aside>
        </main>
      )}
    </div>
  );
}

function parseLoadedJson(value: unknown, url: string): LoadedData {
  if (looksLikeBenchRecord(value)) {
    return { kind: 'bench', record: parseBenchRecord(value), recordUrl: url };
  }
  const manifest = parseManifest(value);
  return { kind: 'viewer', manifest, manifestUrl: url };
}

function patchFolderUrls(manifest: ViewerManifest, blobMap: Map<string, string>): ViewerManifest {
  const resolve = (path: string) => {
    const direct = blobMap.get(path);
    if (direct) return direct;
    const suffix = Array.from(blobMap.entries()).find(([key]) => key.endsWith(path));
    return suffix?.[1] ?? path;
  };
  return {
    ...manifest,
    poses: manifest.poses.map((pose) => ({
      ...pose,
      target_image: resolve(pose.target_image),
      laser_image: pose.laser_image ? resolve(pose.laser_image) : pose.laser_image,
    })),
  };
}

function EmptyState({ loading, onLoad }: { loading: boolean; onLoad: () => void }) {
  return (
    <main className="empty-state">
      <Camera size={32} />
      <h2>Load a calibration viewer manifest</h2>
      <p>Use <code>?manifest=/viewer-data/puzzle/viewer_manifest.json</code> or select a generated folder.</p>
      <button onClick={onLoad} disabled={loading}>
        {loading ? <Loader2 className="spin" size={16} /> : <FolderOpen size={16} />}
        Load default URL
      </button>
    </main>
  );
}

function BenchDashboard({ record }: { record: BenchRecord }) {
  const levels = record.reproj_report?.levels ?? [];
  const finalLevel = levels.at(-1);
  const floorLevel = levels.find((level) => level.level === 'intrinsic');
  const finalGap = record.reproj_report?.gaps.at(-1);
  const headline = record.reproj_report?.headline_px ?? record.fit.reported_mean_reproj_px;
  const detectionCoverage = coveragePct(record.detection?.total_detected, record.detection?.total_expected);
  const status = record.convergence.init_ok
    ? record.convergence.converged ? 'Converged' : 'Optimizer stopped'
    : 'Initialization failed';
  const statusClass = record.convergence.init_ok && record.convergence.converged ? 'good' : 'bad';

  return (
    <main className="bench-dashboard">
      <section className="bench-hero">
        <div>
          <div className="eyebrow">Workspace Data Benchmark</div>
          <h2>{problemLabel(record.ident.problem)}</h2>
          <div className="bench-meta">
            <span>{record.ident.dataset_id}</span>
            <span>Tier {record.ident.tier.toUpperCase()}</span>
            <span>{pipelineLabel(levels)}</span>
            <span>{shortSha(record.ident.git_sha)}</span>
          </div>
        </div>
        <div className={`bench-status ${statusClass}`}>
          <strong>{status}</strong>
          <span>{record.convergence.report.num_iters} iters · cost {formatScalar(record.convergence.report.final_cost)}</span>
        </div>
      </section>

      <section className="bench-kpi-grid">
        <BenchKpi
          icon={<BarChart3 size={18} />}
          label="Headline reprojection"
          value={formatPx(headline)}
          detail={finalLevel ? `${levelLabel(finalLevel.level)} · ${finalLevel.overall.count.toLocaleString()} residuals` : 'fit summary'}
        />
        <BenchKpi
          icon={<Layers size={18} />}
          label="Intrinsic floor"
          value={floorLevel ? formatPx(floorLevel.overall.mean) : 'n/a'}
          detail={floorLevel ? `${formatPx(floorLevel.overall.p95)} p95` : 'no level report'}
        />
        <BenchKpi
          icon={<AlertCircle size={18} />}
          label="Final level gap"
          value={finalGap ? `x${formatScalar(finalGap.ratio_to_intrinsic ?? finalGap.ratio_to_previous ?? 0)}` : 'n/a'}
          detail={finalGap ? `${formatPx(finalGap.mean_delta_px)} over ${levelLabel(finalGap.from)}` : 'no chain gap'}
        />
        <BenchKpi
          icon={<Camera size={18} />}
          label="Detection coverage"
          value={detectionCoverage == null ? 'n/a' : `${detectionCoverage.toFixed(1)}%`}
          detail={record.detection ? `${record.detection.total_detected.toLocaleString()} / ${record.detection.total_expected.toLocaleString()} features` : 'not measured'}
        />
        <BenchKpi
          icon={<Box size={18} />}
          label="Robot correction"
          value={record.robot_corrections ? `${record.robot_corrections.max_trans_mm.toFixed(2)} mm` : 'fixed'}
          detail={record.robot_corrections ? `${record.robot_corrections.max_rot_deg.toFixed(3)} deg max` : 'no per-view deltas'}
        />
        <BenchKpi
          icon={<Loader2 size={18} />}
          label="Runtime"
          value={formatMs(record.timing.total_ms)}
          detail={`${formatMs(record.timing.detection_ms)} detection · ${formatMs(record.timing.optimize_ms)} optimize`}
        />
      </section>

      <section className="bench-layout">
        <div className="bench-main-column">
          <QualityLadder levels={levels} gaps={record.reproj_report?.gaps ?? []} />
          <CameraQualityTable record={record} finalLevel={finalLevel} />
          <WorstViewTable finalLevel={finalLevel} />
          <WorstOutlierTable finalLevel={finalLevel} />
        </div>
        <aside className="bench-side-column">
          <RunTiming timing={record.timing} />
          <RobotCorrectionPanel corrections={record.robot_corrections ?? null} />
          <LaserPanel laser={record.laser ?? null} />
        </aside>
      </section>
    </main>
  );
}

function BenchKpi({
  icon,
  label,
  value,
  detail,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <section className="bench-kpi">
      <div className="bench-kpi-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </section>
  );
}

function QualityLadder({ levels, gaps }: { levels: CompactLevelReport[]; gaps: ReprojLevelGap[] }) {
  if (levels.length === 0) {
    return (
      <section className="bench-panel">
        <SectionTitle icon={<Layers size={16} />} label="Pipeline Quality" />
        <p className="muted">No compact reprojection levels are present in this record.</p>
      </section>
    );
  }
  return (
    <section className="bench-panel">
      <SectionTitle icon={<Layers size={16} />} label="Pipeline Quality" />
      <div className="bench-table-wrap">
        <table className="bench-table">
          <thead>
            <tr>
              <th>Level</th>
              <th>Mean</th>
              <th>RMS</th>
              <th>P95</th>
              <th>Max</th>
              <th>Count</th>
              <th>Gap</th>
            </tr>
          </thead>
          <tbody>
            {levels.map((level) => {
              const gap = gaps.find((candidate) => candidate.to === level.level);
              return (
                <tr key={level.level}>
                  <td>{levelLabel(level.level)}</td>
                  <td>{formatPx(level.overall.mean)}</td>
                  <td>{formatPx(level.overall.rms)}</td>
                  <td>{formatPx(level.overall.p95)}</td>
                  <td>{formatPx(level.overall.max)}</td>
                  <td>{level.overall.count.toLocaleString()}</td>
                  <td>{gap ? `${formatPx(gap.mean_delta_px)} · x${formatScalar(gap.ratio_to_intrinsic ?? gap.ratio_to_previous ?? 0)}` : 'floor'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CameraQualityTable({ record, finalLevel }: { record: BenchRecord; finalLevel?: CompactLevelReport }) {
  const cameraCount = Math.max(
    record.fit.per_camera.length,
    record.detection?.per_camera.length ?? 0,
    finalLevel?.per_camera.length ?? 0,
    record.laser?.per_camera.length ?? 0,
  );
  const rows = Array.from({ length: cameraCount }, (_, index) => {
    const detection = record.detection?.per_camera[index];
    const laser = record.laser?.per_camera[index];
    return {
      id: detection?.camera_id ?? laser?.camera_id ?? `cam${index}`,
      fit: record.fit.per_camera[index],
      final: finalLevel?.per_camera[index],
      detection,
      laser,
    };
  });
  return (
    <section className="bench-panel">
      <SectionTitle icon={<Camera size={16} />} label="Per-Camera Quality" />
      <div className="bench-table-wrap">
        <table className="bench-table">
          <thead>
            <tr>
              <th>Camera</th>
              <th>Fit mean</th>
              <th>Final mean</th>
              <th>Final max</th>
              <th>Coverage</th>
              <th>Laser points</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>{row.id}</td>
                <td>{row.fit ? formatPx(row.fit.mean) : 'n/a'}</td>
                <td>{row.final ? formatPx(row.final.mean) : 'n/a'}</td>
                <td>{row.final ? formatPx(row.final.max) : 'n/a'}</td>
                <td>{row.detection ? `${row.detection.coverage_pct.toFixed(1)}%` : 'n/a'}</td>
                <td>{row.laser ? row.laser.points_extracted.toLocaleString() : 'n/a'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function WorstViewTable({ finalLevel }: { finalLevel?: CompactLevelReport }) {
  const rows = useMemo(
    () => (finalLevel?.per_view ?? [])
      .map((stats, index) => ({ stats, index }))
      .sort((a, b) => b.stats.max - a.stats.max)
      .slice(0, 8),
    [finalLevel],
  );
  return (
    <section className="bench-panel">
      <SectionTitle icon={<BarChart3 size={16} />} label="Worst Views" />
      {rows.length === 0 ? (
        <p className="muted">No per-view statistics are present.</p>
      ) : (
        <div className="bench-table-wrap">
          <table className="bench-table">
            <thead>
              <tr>
                <th>View</th>
                <th>Mean</th>
                <th>P95</th>
                <th>Max</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.index}>
                  <td>{row.index}</td>
                  <td>{formatPx(row.stats.mean)}</td>
                  <td>{formatPx(row.stats.p95)}</td>
                  <td>{formatPx(row.stats.max)}</td>
                  <td>{row.stats.count.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function WorstOutlierTable({ finalLevel }: { finalLevel?: CompactLevelReport }) {
  const outliers = useMemo(
    () => [...(finalLevel?.top_outliers ?? [])]
      .sort((a, b) => (b.error_px ?? 0) - (a.error_px ?? 0))
      .slice(0, 10),
    [finalLevel],
  );
  return (
    <section className="bench-panel">
      <SectionTitle icon={<AlertCircle size={16} />} label="Top Feature Outliers" />
      {outliers.length === 0 ? (
        <p className="muted">No compact outliers are present.</p>
      ) : (
        <div className="bench-outlier-grid">
          {outliers.map((feature) => (
            <div key={`${feature.pose}-${feature.camera}-${feature.feature}`} className="bench-outlier">
              <span>P{feature.pose} · C{feature.camera} · #{feature.feature}</span>
              <strong>{formatPx(feature.error_px ?? 0)}</strong>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function RunTiming({ timing }: { timing: BenchRecord['timing'] }) {
  return (
    <section className="bench-panel">
      <SectionTitle icon={<Loader2 size={16} />} label="Timing" />
      <Metric label="Initialization" value={formatMs(timing.init_ms)} />
      <Metric label="Optimization" value={formatMs(timing.optimize_ms)} />
      <Metric label="Detection" value={formatMs(timing.detection_ms)} />
      <Metric label="Total" value={formatMs(timing.total_ms)} />
    </section>
  );
}

function RobotCorrectionPanel({ corrections }: { corrections: BenchRecord['robot_corrections'] | null }) {
  return (
    <section className="bench-panel">
      <SectionTitle icon={<Box size={16} />} label="Robot Pose Corrections" />
      {!corrections ? (
        <p className="muted">No robot-pose deltas were optimized.</p>
      ) : (
        <>
          <Metric label="Views" value={corrections.count.toLocaleString()} />
          <Metric label="Translation mean" value={`${corrections.mean_trans_mm.toFixed(3)} mm`} />
          <Metric label="Translation max" value={`${corrections.max_trans_mm.toFixed(3)} mm`} />
          <Metric label="Rotation mean" value={`${corrections.mean_rot_deg.toFixed(4)} deg`} />
          <Metric label="Rotation max" value={`${corrections.max_rot_deg.toFixed(4)} deg`} />
        </>
      )}
    </section>
  );
}

function LaserPanel({ laser }: { laser: BenchRecord['laser'] | null }) {
  return (
    <section className="bench-panel">
      <SectionTitle icon={<Layers size={16} />} label="Laser Extraction" />
      {!laser ? (
        <p className="muted">No laser metrics are present.</p>
      ) : (
        <>
          <Metric label="Pixels" value={laser.total_points.toLocaleString()} />
          <Metric label="Images used" value={laser.total_images_used.toLocaleString()} />
          <Metric label="Extraction time" value={formatMs(laser.extract_ms)} />
          <div className="mini-list">
            {laser.per_camera.map((camera) => (
              <LaserCameraLine key={camera.camera_id} camera={camera} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function LaserCameraLine({ camera }: { camera: LaserCamStat }) {
  const residual = camera.line_residual_px?.mean != null
    ? formatPx(camera.line_residual_px.mean)
    : camera.plane_residual_m?.mean != null
      ? formatMm(camera.plane_residual_m.mean)
      : 'n/a';
  return (
    <div className="mini-list-row">
      <span>{camera.camera_id}</span>
      <strong>{camera.points_extracted.toLocaleString()}</strong>
      <small>{residual}</small>
    </div>
  );
}

function coveragePct(detected?: number, expected?: number): number | null {
  if (!expected || expected <= 0 || detected == null) return null;
  return (detected / expected) * 100;
}

function formatMs(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${Math.round(value)} ms`;
}

function formatScalar(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(3);
}

function levelLabel(level: string): string {
  switch (level) {
    case 'intrinsic':
      return 'Intrinsic';
    case 'rig_extrinsic':
      return 'Rig extrinsic';
    case 'hand_eye':
      return 'Hand-eye';
    case 'laser':
      return 'Laser';
    default:
      return level;
  }
}

function problemLabel(problem: string): string {
  return problem
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function pipelineLabel(levels: CompactLevelReport[]): string {
  if (levels.length === 0) return 'fit only';
  return levels.map((level) => levelLabel(level.level)).join(' → ');
}

function shortSha(sha: string): string {
  if (!sha || sha === 'unknown') return 'unknown git';
  return sha.slice(0, 8);
}

function SectionTitle({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="section-title">
      {icon}
      {label}
    </div>
  );
}

function GeometryScene({
  manifest,
  stage,
  activePose,
  activeCamera,
  showPlanes,
  showBoards,
  showRobot,
  onSelect,
}: {
  manifest: ViewerManifest;
  stage: ViewerStage;
  activePose: number;
  activeCamera: number;
  showPlanes: boolean;
  showBoards: boolean;
  showRobot: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <group>
      <gridHelper args={[0.8, 16, '#b8c0bc', '#e1e6e2']} rotation={[Math.PI / 2, 0, 0]} />
      <axesHelper args={[0.08]} />
      <FrameLabel label="Rig" position={[0, 0, 0.04]} />
      {stage.geometry.cameras.map((camera) => (
        <CameraFrustum
          key={camera.index}
          camera={camera}
          active={camera.index === activeCamera}
          onClick={() => onSelect(`camera ${camera.index}`)}
        />
      ))}
      {showPlanes && stage.geometry.cameras.map((camera) => (
        <LaserPlaneMesh
          key={camera.index}
          camera={camera}
          active={camera.index === activeCamera}
          onClick={() => onSelect(`laser plane ${camera.index}`)}
        />
      ))}
      {showBoards && manifest.poses.slice(0, 40).map((pose) => (
        <BoardPose
          key={pose.index}
          manifest={manifest}
          pose={pose}
          stage={stage}
          active={pose.index === activePose}
          onClick={() => onSelect(`pose ${pose.index}`)}
        />
      ))}
      {showRobot && manifest.poses.map((pose) => (
        <RobotPose
          key={pose.index}
          manifest={manifest}
          pose={pose}
          stage={stage}
          active={pose.index === activePose}
        />
      ))}
    </group>
  );
}

function CameraFrustum({ camera, active, onClick }: { camera: CameraRecord; active: boolean; onClick: () => void }) {
  const pose = transformToPose(camera.cam_to_rig);
  const color = active ? '#b43b2e' : '#2f6672';
  return (
    <group position={pose.position} quaternion={pose.quaternion} onClick={(e) => { e.stopPropagation(); onClick(); }}>
      <mesh>
        <boxGeometry args={[0.022, 0.016, 0.012]} />
        <meshStandardMaterial color={color} roughness={0.5} />
      </mesh>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([0, 0, 0, -0.035, -0.026, 0.07, 0, 0, 0, 0.035, -0.026, 0.07, 0, 0, 0, 0.035, 0.026, 0.07, 0, 0, 0, -0.035, 0.026, 0.07]), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color={color} />
      </lineSegments>
      {active && <FrameLabel label={`C${camera.index}`} position={[0, 0.02, 0]} />}
    </group>
  );
}

function LaserPlaneMesh({ camera, active, onClick }: { camera: CameraRecord; active: boolean; onClick: () => void }) {
  const plane = camera.laser_plane_rig;
  const normal = vec3(plane.normal).normalize();
  const center = normal.clone().multiplyScalar(-plane.distance_m);
  const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
  return (
    <mesh position={center} quaternion={quat} onClick={(e) => { e.stopPropagation(); onClick(); }}>
      <planeGeometry args={[0.22, 0.22]} />
      <meshStandardMaterial color={active ? '#d8a12d' : '#78a878'} transparent opacity={active ? 0.42 : 0.22} side={THREE.DoubleSide} />
    </mesh>
  );
}

function BoardPose({
  manifest,
  pose,
  stage,
  active,
  onClick,
}: {
  manifest: ViewerManifest;
  pose: ViewerManifest['poses'][number];
  stage: ViewerStage;
  active: boolean;
  onClick: () => void;
}) {
  const matrix = useMemo(() => {
    if (manifest.handeye_mode === 'EyeToHand') {
      return composeTransforms(stage.geometry.handeye, pose.base_se3_gripper, stage.geometry.target_ref);
    }
    const fallback = new THREE.Matrix4().identity();
    fallback.setPosition(...pose.base_se3_gripper.translation_m);
    return fallback;
  }, [manifest.handeye_mode, pose.base_se3_gripper, stage.geometry.handeye, stage.geometry.target_ref]);
  const boardW = (manifest.dataset.board_cols * manifest.dataset.cell_size_mm) / 1000;
  const boardH = (manifest.dataset.board_rows * manifest.dataset.cell_size_mm) / 1000;
  return (
    <group matrix={matrix} matrixAutoUpdate={false} onClick={(e) => { e.stopPropagation(); onClick(); }}>
      <mesh>
        <planeGeometry args={[boardW, boardH]} />
        <meshStandardMaterial color={active ? '#b43b2e' : '#d6c8a6'} transparent opacity={active ? 0.9 : 0.35} side={THREE.DoubleSide} />
      </mesh>
      {active && <axesHelper args={[0.04]} />}
    </group>
  );
}

function RobotPose({
  manifest,
  pose,
  stage,
  active,
}: {
  manifest: ViewerManifest;
  pose: ViewerManifest['poses'][number];
  stage: ViewerStage;
  active: boolean;
}) {
  const matrix = useMemo(() => {
    if (manifest.handeye_mode === 'EyeToHand') {
      return composeTransforms(stage.geometry.handeye, pose.base_se3_gripper);
    }
    const fallback = new THREE.Matrix4().identity();
    fallback.setPosition(...pose.base_se3_gripper.translation_m);
    return fallback;
  }, [manifest.handeye_mode, pose.base_se3_gripper, stage.geometry.handeye]);
  return (
    <group matrix={matrix} matrixAutoUpdate={false}>
      <mesh>
        <sphereGeometry args={[active ? 0.01 : 0.005, 12, 12]} />
        <meshStandardMaterial color={active ? '#b43b2e' : '#4d6b8a'} />
      </mesh>
      {active && <axesHelper args={[0.035]} />}
    </group>
  );
}

function FrameLabel({ label, position }: { label: string; position: [number, number, number] }) {
  return (
    <Html position={position} distanceFactor={4}>
      <span className="scene-label">{label}</span>
    </Html>
  );
}

function ImageInspector({
  manifest,
  manifestUrl,
  stage,
  poseIndex,
  cameraIndex,
  mode,
  onModeChange,
}: {
  manifest: ViewerManifest;
  manifestUrl: string;
  stage: ViewerStage;
  poseIndex: number;
  cameraIndex: number;
  mode: ImageMode;
  onModeChange: (mode: ImageMode) => void;
}) {
  const [zoom, setZoom] = useState(1);
  const pose = manifest.poses.find((p) => p.index === poseIndex) ?? manifest.poses[0];
  const targetFeatures = useMemo(
    () => stage.target_features.filter((f) => f.pose === poseIndex && f.camera === cameraIndex),
    [stage, poseIndex, cameraIndex],
  );
  const laserFeatures = useMemo(
    () => stage.laser_features.filter((f) => f.pose === poseIndex && f.camera === cameraIndex),
    [stage, poseIndex, cameraIndex],
  );
  const src = mode === 'target'
    ? pose.target_image
    : pose.laser_image ?? pose.target_image;
  const imageUrl = src.startsWith('blob:') ? src : resolveAssetUrl(manifestUrl, src);

  return (
    <div className="image-panel">
      <div className="panel-header">
        <SectionTitle icon={<ImageIcon size={16} />} label={`Pose ${poseIndex} · Camera ${cameraIndex}`} />
        <div className="image-tools">
          <div className="segmented">
            <button className={mode === 'target' ? 'active' : ''} onClick={() => onModeChange('target')}>Target</button>
            <button className={mode === 'laser' ? 'active' : ''} onClick={() => onModeChange('laser')} disabled={!pose.laser_image}>Laser</button>
          </div>
          <button title="Zoom out" aria-label="Zoom out" onClick={() => setZoom((z) => Math.max(1, z / 1.5))}>
            <ZoomOut size={15} />
          </button>
          <button title="Zoom in" aria-label="Zoom in" onClick={() => setZoom((z) => Math.min(12, z * 1.5))}>
            <ZoomIn size={15} />
          </button>
          <button title="Reset zoom" aria-label="Reset zoom" onClick={() => setZoom(1)}>
            <RotateCcw size={15} />
          </button>
          <span className="zoom-readout">{zoom.toFixed(1)}x</span>
        </div>
      </div>
      <div className="image-viewport">
        <div
          className="image-stage"
          style={{
            aspectRatio: `${manifest.dataset.tile_size[0]} / ${manifest.dataset.tile_size[1]}`,
            width: `${zoom * 100}%`,
          }}
        >
          <img
            className="mosaic-image"
            src={imageUrl}
            style={{
              width: `${manifest.dataset.num_cameras * 100}%`,
              transform: `translateX(-${(cameraIndex / manifest.dataset.num_cameras) * 100}%)`,
            }}
          />
          <FeatureOverlay
            tileSize={manifest.dataset.tile_size}
            cameraIndex={cameraIndex}
            targetFeatures={targetFeatures}
            laserFeatures={laserFeatures}
            mode={mode}
          />
        </div>
      </div>
    </div>
  );
}

function FeatureOverlay({
  tileSize,
  cameraIndex,
  targetFeatures,
  laserFeatures,
  mode,
}: {
  tileSize: [number, number];
  cameraIndex: number;
  targetFeatures: TargetFeatureRecord[];
  laserFeatures: LaserFeatureRecord[];
  mode: ImageMode;
}) {
  const [w, h] = tileSize;
  const xOffset = cameraIndex * w;
  return (
    <svg viewBox={`${xOffset} 0 ${w} ${h}`} preserveAspectRatio="none">
      {mode === 'target' && targetFeatures.map((f) => {
        const e = f.error_px ?? 0;
        const color = e <= 1 ? '#20865a' : e <= 2 ? '#d49a1f' : '#b43b2e';
        return (
          <g key={f.feature}>
            <circle cx={f.observed_px[0] + xOffset} cy={f.observed_px[1]} r="2.2" fill="none" stroke={color} strokeWidth="0.8" opacity="0.95">
              <title>{`P${f.pose} C${f.camera} #${f.feature}: ${formatPx(e)}`}</title>
            </circle>
            {f.projected_px && (
              <line
                x1={f.observed_px[0] + xOffset}
                y1={f.observed_px[1]}
                x2={f.projected_px[0] + xOffset}
                y2={f.projected_px[1]}
                stroke={color}
                strokeWidth="0.8"
                opacity="0.65"
              />
            )}
          </g>
        );
      })}
      {mode === 'laser' && laserFeatures.map((f) => {
        const e = Math.abs(f.residual_m ?? 0);
        const color = e <= 0.001 ? '#20865a' : e <= 0.004 ? '#d49a1f' : '#b43b2e';
        return (
          <circle key={f.feature} cx={f.observed_px[0] + xOffset} cy={f.observed_px[1]} r="2" fill="none" stroke={color} strokeWidth="0.8" opacity="0.95">
            <title>{`P${f.pose} C${f.camera} #${f.feature}: ${formatMm(e)}`}</title>
          </circle>
        );
      })}
      {mode === 'laser' && laserFeatures[0]?.projected_line_px && (
        <line
          x1={laserFeatures[0].projected_line_px[0][0] + xOffset}
          y1={laserFeatures[0].projected_line_px[0][1]}
          x2={laserFeatures[0].projected_line_px[1][0] + xOffset}
          y2={laserFeatures[0].projected_line_px[1][1]}
          stroke="#f4f0e8"
          strokeDasharray="6 4"
          strokeWidth="1.2"
          opacity="0.85"
        />
      )}
    </svg>
  );
}

function PlaneProjection({ stage, cameraIndex }: { stage: ViewerStage; cameraIndex: number }) {
  const lines = stage.geometry.cameras.flatMap((camera) => {
    const [nx, ny] = camera.laser_plane_rig.normal;
    const d = camera.laser_plane_rig.distance_m;
    const norm = Math.hypot(nx, ny);
    if (norm < 1e-9) return [];
    const dir = [-ny / norm, nx / norm] as const;
    const p0 = [-d * nx / (norm * norm), -d * ny / (norm * norm)] as const;
    const half = 0.14;
    return [{
      index: camera.index,
      x1: p0[0] - dir[0] * half,
      y1: p0[1] - dir[1] * half,
      x2: p0[0] + dir[0] * half,
      y2: p0[1] + dir[1] * half,
    }];
  });
  const scale = 560;
  const toSvg = (x: number, y: number) => [110 + x * scale, 110 - y * scale] as const;
  return (
    <section className="plane-projection">
      <SectionTitle icon={<Box size={16} />} label="Rig XY Plane Cut" />
      <svg viewBox="0 0 220 220">
        <circle cx="110" cy="110" r="3" />
        <line x1="110" y1="110" x2="190" y2="110" className="axis-x" />
        <line x1="110" y1="110" x2="110" y2="30" className="axis-y" />
        {lines.map((line) => {
          const a = toSvg(line.x1, line.y1);
          const b = toSvg(line.x2, line.y2);
          return (
            <line
              key={line.index}
              x1={a[0]}
              y1={a[1]}
              x2={b[0]}
              y2={b[1]}
              className={line.index === cameraIndex ? 'active-plane-line' : 'plane-line'}
            />
          );
        })}
      </svg>
    </section>
  );
}

function Diagnostics({ stage, cameraIndex }: { stage: ViewerStage; cameraIndex: number }) {
  const stats = stage.per_camera_stats[cameraIndex];
  if (!stats) return null;
  return (
    <section className="diagnostics">
      <SectionTitle icon={<BarChart3 size={16} />} label="Diagnostics" />
      <Metric label="Reproj mean" value={formatPx(stats.mean_reproj_error_px)} />
      <Metric label="Reproj max" value={formatPx(stats.max_reproj_error_px)} />
      <Metric label="Laser RMS" value={formatMm(stats.mean_laser_err_m)} />
      <Metric label="Laser max" value={formatMm(stats.max_laser_err_m)} />
      <Histogram values={stats.reproj_histogram_px} labels={['1', '2', '5', '10', '>']} />
      <Histogram values={stats.laser_histogram_m} labels={['0.1', '1', '10', '100', '>']} />
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Histogram({ values, labels }: { values: number[]; labels: string[] }) {
  const max = Math.max(...values, 1);
  return (
    <div className="histogram">
      {values.map((value, idx) => (
        <div key={idx}>
          <span style={{ height: `${Math.max(5, (value / max) * 64)}px` }} />
          <small>{labels[idx]}</small>
        </div>
      ))}
    </div>
  );
}

function WorstFeatures({ stage, onPick }: { stage: ViewerStage; onPick: (pose: number, camera: number) => void }) {
  const worst = useMemo(
    () => [...stage.target_features]
      .filter((f) => Number.isFinite(f.error_px ?? NaN))
      .sort((a, b) => (b.error_px ?? 0) - (a.error_px ?? 0))
      .slice(0, 9),
    [stage],
  );
  return (
    <section className="worst-table">
      <SectionTitle icon={<AlertCircle size={16} />} label="Worst Target Features" />
      {worst.map((f) => (
        <button key={`${f.pose}-${f.camera}-${f.feature}`} onClick={() => onPick(f.pose, f.camera)}>
          <span>P{f.pose} C{f.camera} #{f.feature}</span>
          <strong>{formatPx(f.error_px ?? 0)}</strong>
        </button>
      ))}
    </section>
  );
}

function SelectionDetails({ selected, stage, cameraIndex }: { selected: string; stage: ViewerStage; cameraIndex: number }) {
  const camera = stage.geometry.cameras[cameraIndex];
  return (
    <section className="selection-details">
      <SectionTitle icon={<MousePointer2 size={16} />} label="Details" />
      <p>{selected}</p>
      {camera && (
        <dl>
          <dt>fx/fy</dt>
          <dd>{camera.intrinsics[0].toFixed(2)} / {camera.intrinsics[1].toFixed(2)}</dd>
          <dt>cx/cy</dt>
          <dd>{camera.intrinsics[2].toFixed(2)} / {camera.intrinsics[3].toFixed(2)}</dd>
          <dt>tau x/y</dt>
          <dd>{camera.scheimpflug[0].toFixed(4)} / {camera.scheimpflug[1].toFixed(4)}</dd>
          <dt>plane d</dt>
          <dd>{formatMm(camera.laser_plane_rig.distance_m)}</dd>
        </dl>
      )}
    </section>
  );
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
