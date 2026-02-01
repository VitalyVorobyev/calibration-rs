# Multi-Camera Rig Initialization

A multi-camera rig is a set of cameras rigidly mounted on a common frame. The extrinsics of the rig describe the relative pose of each camera with respect to a reference camera (or a rig frame). This chapter covers the linear initialization of rig extrinsics from per-camera pose estimates.

## Problem Statement

**Given**: Per-camera, per-view pose estimates $T_{C_k, T}^{(v)}$ (camera $k$, view $v$, target to camera).

**Find**: Camera-to-rig transforms $\{T_{R, C_k}\}$ and rig-to-target poses $\{T_{R, T}^{(v)}\}$.

**Assumptions**:
- All cameras observe the same calibration target simultaneously
- Per-camera poses have been estimated (e.g., via PnP or homography decomposition)
- The rig is rigid (camera-to-rig transforms are constant across views)

## Algorithm

### Reference Camera

One camera is designated as the **reference camera** (default: camera 0). Its camera-to-rig transform is identity:

$$T_{R, C_0} = I$$

This defines the rig frame to coincide with the reference camera frame.

### Rig-to-Target from Reference

For each view $v$, the rig-to-target pose is the reference camera's pose:

$$T_{R, T}^{(v)} = T_{C_0, T}^{(v)}$$

### Camera-to-Rig via Averaging

For each non-reference camera $k$, collect the camera-to-rig estimate from each view where both cameras have observations:

$$\hat{T}_{R, C_k}^{(v)} = T_{R, T}^{(v)} \cdot \left(T_{C_k, T}^{(v)}\right)^{-1} = T_{C_0, T}^{(v)} \cdot \left(T_{C_k, T}^{(v)}\right)^{-1}$$

Then average across views:

- **Rotation**: Quaternion averaging with hemisphere correction. Before averaging, flip quaternions to the same hemisphere (since $\mathbf{q}$ and $-\mathbf{q}$ represent the same rotation). Then compute the mean quaternion and normalize.
- **Translation**: Arithmetic mean of translation vectors.

$$T_{R, C_k} = \text{average}\left(\left\{ \hat{T}_{R, C_k}^{(v)} \right\}_v\right)$$

### Quaternion Hemisphere Correction

Quaternions have a double-cover of SO(3): both $\mathbf{q}$ and $-\mathbf{q}$ represent the same rotation. Before averaging, all quaternions are flipped to the same hemisphere as the first:

$$\mathbf{q}_i \leftarrow \begin{cases} \mathbf{q}_i & \text{if } \mathbf{q}_i \cdot \mathbf{q}_1 \geq 0 \\ -\mathbf{q}_i & \text{if } \mathbf{q}_i \cdot \mathbf{q}_1 < 0 \end{cases}$$

This prevents averaging artifacts where opposite quaternions cancel out.

## Accuracy

The initialization accuracy depends on:

- **Per-camera pose accuracy**: Noisy single-camera poses (from homography decomposition or PnP) propagate to the rig extrinsics
- **Number of views**: More views improve the averaging
- **View diversity**: Diverse viewpoints reduce systematic errors

Typical accuracy: 1-5° rotation, 5-15% translation. This is refined in the rig bundle adjustment (see [Multi-Camera Rig Extrinsics](rig_extrinsics.md)).

## API

```rust
let extrinsic_poses = estimate_extrinsics_from_cam_target_poses(
    &cam_se3_target,    // Vec<Vec<Option<Iso3>>>: per-view, per-camera poses
    ref_cam_idx,        // usize: reference camera index
)?;

// extrinsic_poses.cam_to_rig: Vec<Iso3>  (T_{R,C} for each camera)
// extrinsic_poses.rig_from_target: Vec<Iso3>  (T_{R,T} for each view)
```

## Usage in Calibration Pipeline

In the rig extrinsics calibration pipeline:

1. **Per-camera intrinsics** are estimated and optimized independently
2. **Per-camera poses** are recovered from the calibration board
3. **Rig extrinsics** are initialized via averaging (this chapter)
4. **Rig bundle adjustment** jointly optimizes all parameters
