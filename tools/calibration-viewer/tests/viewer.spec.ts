import { expect, test } from '@playwright/test';

test('renders 3D geometry and image overlays from a manifest', async ({ page }) => {
  await page.goto('/?manifest=/tests/fixtures/viewer_manifest.json');
  await expect(page.getByRole('heading', { name: 'synthetic_viewer' })).toBeVisible();
  await expect(page.locator('canvas')).toBeVisible();
  await expect(page.locator('.image-stage svg')).toBeVisible();

  const dataUrlLength = await page.locator('canvas').evaluate((canvas) => {
    const c = canvas as HTMLCanvasElement;
    return c.toDataURL('image/png').length;
  });
  expect(dataUrlLength).toBeGreaterThan(1500);

  const overlayBox = await page.locator('.image-stage svg').boundingBox();
  expect(overlayBox?.width).toBeGreaterThan(100);
  expect(overlayBox?.height).toBeGreaterThan(60);

  await page.getByRole('button', { name: /Target/ }).click();
  await expect(page.getByText('Worst Target Features')).toBeVisible();
});

test('renders compact benchmark records as a quality dashboard', async ({ page }) => {
  await page.route('/bench-record.json', async (route) => {
    await route.fulfill({
      json: {
        ident: {
          dataset_id: 'stereo_rig',
          problem: 'rig_extrinsics',
          tier: 'b',
          git_sha: 'deadbeef',
          timestamp_rfc3339: '2026-05-31T12:00:00Z',
          config_hash: 42,
          bench_schema_version: 3,
          features: ['tier-b'],
        },
        convergence: {
          init_ok: true,
          converged: true,
          report: { final_cost: 0.25, num_iters: 8 },
        },
        fit: {
          overall: { mean: 0.25, rms: 0.3, max: 1.2, count: 120 },
          per_camera: [{ mean: 0.2, rms: 0.25, max: 0.9, count: 60 }],
          per_camera_hist: [],
          reported_mean_reproj_px: 0.25,
          reported_per_cam_px: [0.2],
        },
        generalization: null,
        stability: null,
        detection: {
          total_detected: 118,
          total_expected: 120,
          per_camera: [{
            camera_id: 'cam0',
            images_total: 10,
            images_used: 10,
            features_detected: 118,
            features_expected: 120,
            coverage_pct: 98.333,
            detect_ms: 50,
          }],
        },
        laser: null,
        robot_corrections: {
          count: 10,
          mean_rot_deg: 0.01,
          max_rot_deg: 0.02,
          mean_trans_mm: 0.2,
          max_trans_mm: 0.4,
        },
        artifacts: {
          spatial_unit: 'mm',
          angle_unit: 'deg',
          cameras: [{
            camera_id: 'cam0',
            camera_matrix_px: [[800, 0, 320], [0, 801, 240], [0, 0, 1]],
            intrinsics_px: { fx: 800, fy: 801, cx: 320, cy: 240, skew: 0 },
            distortion_model: 'brown_conrady5',
            distortion: { k1: -0.1, k2: 0.02, k3: 0, p1: 0.001, p2: -0.001 },
            scheimpflug: null,
          }],
          transforms: [{
            name: 'cam0_se3_rig',
            to_frame: 'cam0',
            from_frame: 'rig',
            translation_mm: [1, 2, 3],
            rotation_quat_xyzw: [0, 0, 0, 1],
            rotation_rotvec_deg: [0, 0, 0],
          }],
        },
        delta_to_prior: null,
        timing: { init_ms: 2, optimize_ms: 10, total_ms: 70, detection_ms: 50 },
        reproj_report: {
          headline_px: 0.25,
          levels: [{
            level: 'intrinsic',
            overall: { mean: 0.18, median: 0.16, rms: 0.2, p95: 0.4, max: 0.8, count: 120 },
            per_camera: [{ mean: 0.18, median: 0.16, rms: 0.2, p95: 0.4, max: 0.8, count: 120 }],
            per_view: [{ mean: 0.18, median: 0.16, rms: 0.2, p95: 0.4, max: 0.8, count: 12 }],
            residual_count: 120,
            top_outliers: [],
          }, {
            level: 'rig_extrinsic',
            overall: { mean: 0.25, median: 0.22, rms: 0.3, p95: 0.6, max: 1.2, count: 120 },
            per_camera: [{ mean: 0.25, median: 0.22, rms: 0.3, p95: 0.6, max: 1.2, count: 120 }],
            per_view: [{ mean: 0.25, median: 0.22, rms: 0.3, p95: 0.6, max: 1.2, count: 12 }],
            residual_count: 120,
            top_outliers: [{
              pose: 0,
              camera: 0,
              feature: 1,
              target_xyz_m: [0, 0, 0],
              observed_px: [10, 10],
              projected_px: [10.2, 10.1],
              error_px: 0.22,
            }],
          }],
          gaps: [{
            from: 'intrinsic',
            to: 'rig_extrinsic',
            mean_delta_px: 0.07,
            ratio_to_previous: 1.389,
            ratio_to_intrinsic: 1.389,
          }],
        },
      },
    });
  });

  await page.goto('/?bench=/bench-record.json');
  await expect(page.getByRole('heading', { name: 'stereo_rig' })).toBeVisible();
  await expect(page.getByText('Pipeline Quality')).toBeVisible();
  await expect(page.getByText('Robot Pose Corrections')).toBeVisible();
  await expect(page.getByText('Calibration Artifacts')).toBeVisible();
  await expect(page.getByText('cam0_se3_rig')).toBeVisible();
  await expect(page.getByText('Headline reprojection')).toBeVisible();
});
