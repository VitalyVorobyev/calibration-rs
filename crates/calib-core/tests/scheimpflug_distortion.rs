//! Integration tests for Scheimpflug sensor with distortion models.
//!
//! Validates that the Scheimpflug sensor model correctly combines with
//! Brown-Conrady distortion for forward and inverse projection.

use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, Pinhole, Pt3, ScheimpflugParams, Vec2};

#[test]
fn scheimpflug_with_brown_conrady_roundtrip() {
    // Create camera with Scheimpflug sensor and Brown-Conrady distortion
    let intrinsics = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };

    let sensor_params = ScheimpflugParams {
        tilt_x: 0.02,
        tilt_y: -0.01,
    };
    let sensor = sensor_params.compile();

    let distortion = BrownConrady5 {
        k1: -0.3,
        k2: 0.1,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 5,
    };

    let camera = Camera::new(Pinhole, distortion, sensor, intrinsics);

    // Test multiple 3D points
    let test_points = vec![
        Pt3::new(0.0, 0.0, 2.0),
        Pt3::new(0.2, 0.1, 2.5),
        Pt3::new(-0.15, -0.08, 1.8),
        Pt3::new(0.3, -0.2, 3.0),
        Pt3::new(-0.1, 0.15, 2.2),
    ];

    for point in test_points {
        // Forward projection
        let pixel = camera
            .project_point(&point)
            .expect("point should project successfully");

        // Backward projection (backproject to ray, then scale to original depth)
        let ray = camera.backproject_pixel(&pixel).point;

        // Scale ray to match original depth
        let reconstructed = ray * (point.z / ray.z);

        // Verify roundtrip accuracy
        let dx = (reconstructed.x - point.x).abs();
        let dy = (reconstructed.y - point.y).abs();
        let dz = (reconstructed.z - point.z).abs();

        assert!(
            dx < 1e-6,
            "X coordinate error too large: {} for point {:?}",
            dx,
            point
        );
        assert!(
            dy < 1e-6,
            "Y coordinate error too large: {} for point {:?}",
            dy,
            point
        );
        assert!(
            dz < 1e-6,
            "Z coordinate error too large: {} for point {:?}",
            dz,
            point
        );
    }
}

#[test]
fn scheimpflug_distortion_affects_projection() {
    let intrinsics = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };

    let sensor_params = ScheimpflugParams {
        tilt_x: 0.02,
        tilt_y: -0.01,
    };
    let sensor = sensor_params.compile();

    let distortion = BrownConrady5 {
        k1: -0.3,
        k2: 0.1,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 5,
    };

    let no_distortion = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 5,
    };

    let camera_with_dist = Camera::new(Pinhole, distortion, sensor.clone(), intrinsics);
    let camera_no_dist = Camera::new(Pinhole, no_distortion, sensor, intrinsics);

    let point = Pt3::new(0.2, 0.15, 2.0);

    let pixel_with_dist = camera_with_dist
        .project_point(&point)
        .expect("point should project");
    let pixel_no_dist = camera_no_dist
        .project_point(&point)
        .expect("point should project");

    // Projections should differ when distortion is present
    let diff = ((pixel_with_dist.x - pixel_no_dist.x).powi(2)
        + (pixel_with_dist.y - pixel_no_dist.y).powi(2))
    .sqrt();

    assert!(
        diff > 1e-3,
        "Distortion should visibly affect projection, but diff is only {}",
        diff
    );
}

#[test]
fn scheimpflug_tilt_affects_projection() {
    let intrinsics = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };

    let tilted_params = ScheimpflugParams {
        tilt_x: 0.02,
        tilt_y: -0.01,
    };
    let tilted_sensor = tilted_params.compile();

    let identity_params = ScheimpflugParams {
        tilt_x: 0.0,
        tilt_y: 0.0,
    };
    let identity_sensor = identity_params.compile();

    let distortion = BrownConrady5 {
        k1: -0.3,
        k2: 0.1,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 5,
    };

    let camera_tilted = Camera::new(Pinhole, distortion, tilted_sensor, intrinsics);
    let camera_identity = Camera::new(Pinhole, distortion, identity_sensor, intrinsics);

    let point = Pt3::new(0.2, 0.15, 2.0);

    let pixel_tilted = camera_tilted
        .project_point(&point)
        .expect("point should project");
    let pixel_identity = camera_identity
        .project_point(&point)
        .expect("point should project");

    // Projections should differ when Scheimpflug tilt is present
    let diff = ((pixel_tilted.x - pixel_identity.x).powi(2)
        + (pixel_tilted.y - pixel_identity.y).powi(2))
    .sqrt();

    assert!(
        diff > 1e-3,
        "Scheimpflug tilt should visibly affect projection, but diff is only {}",
        diff
    );
}

#[test]
fn combined_scheimpflug_distortion_unproject() {
    let intrinsics = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };

    let sensor_params = ScheimpflugParams {
        tilt_x: 0.02,
        tilt_y: -0.01,
    };
    let sensor = sensor_params.compile();

    let distortion = BrownConrady5 {
        k1: -0.3,
        k2: 0.1,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 5,
    };

    let camera = Camera::new(Pinhole, distortion, sensor, intrinsics);

    // Test various pixel locations
    let test_pixels = vec![
        Vec2::new(640.0, 360.0), // Principal point
        Vec2::new(400.0, 200.0), // Upper left
        Vec2::new(880.0, 520.0), // Lower right
        Vec2::new(640.0, 200.0), // Top center
        Vec2::new(400.0, 360.0), // Left center
    ];

    for pixel in test_pixels {
        let ray = camera.backproject_pixel(&pixel).point;

        // Ray point is on z=1 plane, normalize to unit direction
        let ray_norm = (ray.x * ray.x + ray.y * ray.y + ray.z * ray.z).sqrt();
        let ray_dir = ray / ray_norm;

        // Forward-backward consistency: project ray back
        let point_on_ray = ray_dir * 2.0; // Arbitrary depth along ray
        let reprojected = camera
            .project_point(&Pt3::from(point_on_ray))
            .expect("ray point should project");

        let reproject_diff =
            ((reprojected.x - pixel.x).powi(2) + (reprojected.y - pixel.y).powi(2)).sqrt();

        assert!(
            reproject_diff < 1e-3,
            "Reprojection error too large: {} for pixel {:?}",
            reproject_diff,
            pixel
        );
    }
}
