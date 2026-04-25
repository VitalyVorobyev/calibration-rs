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
  resolveAssetUrl,
  type CameraRecord,
  type LaserFeatureRecord,
  type TargetFeatureRecord,
  type ViewerManifest,
  type ViewerStage,
} from './schema';
import './styles.css';

type ImageMode = 'target' | 'laser';

interface LoadedManifest {
  manifest: ViewerManifest;
  manifestUrl: string;
}

function defaultManifestUrl(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get('manifest') ?? '/viewer-data/puzzle/viewer_manifest.json';
}

function App() {
  const [loaded, setLoaded] = useState<LoadedManifest | null>(null);
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

  async function loadManifest(url: string) {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      const parsed = parseManifest(await response.json());
      setLoaded({ manifest: parsed, manifestUrl: new URL(url, window.location.href).toString() });
      setStageId(parsed.stages.at(-1)?.id ?? parsed.stages[0].id);
      setPoseIndex(parsed.poses[0]?.index ?? 0);
      setCameraIndex(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadManifest(defaultManifestUrl());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      setLoaded({ manifest: patchFolderUrls(parsed, blobMap), manifestUrl: folderUrl });
      setStageId(parsed.stages.at(-1)?.id ?? parsed.stages[0].id);
      setPoseIndex(parsed.poses[0]?.index ?? 0);
      setCameraIndex(0);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const manifest = loaded?.manifest;
  const stage = manifest?.stages.find((s) => s.id === stageId) ?? manifest?.stages.at(-1);
  const pose = manifest?.poses.find((p) => p.index === poseIndex) ?? manifest?.poses[0];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Calibration Geometry Inspector</div>
          <h1>{manifest?.dataset.name ?? 'No manifest loaded'}</h1>
        </div>
        <div className="load-controls">
          <button onClick={() => loadManifest(defaultManifestUrl())} disabled={loading}>
            {loading ? <Loader2 className="spin" size={16} /> : <FolderOpen size={16} />}
            Load URL
          </button>
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

      {!manifest || !stage || !pose ? (
        <EmptyState loading={loading} onLoad={() => loadManifest(defaultManifestUrl())} />
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
              manifestUrl={loaded.manifestUrl}
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
