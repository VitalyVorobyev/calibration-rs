//! View-related structures and functions.

use crate::CorrespondenceView;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Single-camera observation view with attached metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct View<Meta> {
    /// 2D-3D correspondences observed in this view.
    pub obs: CorrespondenceView,
    /// Arbitrary metadata payload associated with the view.
    pub meta: Meta,
}

impl<Meta> View<Meta> {
    /// Number of observations in this view.
    pub fn num_observations(&self) -> usize {
        self.obs.points_3d.len()
    }

    /// Create a view from correspondences and metadata.
    pub fn new(obs: CorrespondenceView, meta: Meta) -> Self {
        Self { obs, meta }
    }
}

impl View<NoMeta> {
    /// Create a view without metadata.
    pub fn without_meta(obs: CorrespondenceView) -> Self {
        Self { obs, meta: NoMeta }
    }
}

/// Multi-camera observations for one rig view/frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigViewObs {
    /// Per-camera observation; None if camera didn't observe in this view.
    pub cameras: Vec<Option<CorrespondenceView>>,
}

/// One time-synchronized rig frame containing per-camera observations and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigView<Meta> {
    /// Per-camera observations for this rig frame.
    pub obs: RigViewObs,
    /// Arbitrary metadata payload associated with the view.
    pub meta: Meta,
}

/// Multi-view dataset for a camera rig.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigDataset<Meta> {
    /// Number of cameras in the rig.
    pub num_cameras: usize,
    /// Sequence of rig views.
    pub views: Vec<RigView<Meta>>,
}

impl<Meta> RigDataset<Meta> {
    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }

    /// Construct a rig dataset and validate per-view camera counts.
    pub fn new(views: Vec<RigView<Meta>>, num_cameras: usize) -> Result<Self> {
        ensure!(!views.is_empty(), "need at least one view");
        for (idx, view) in views.iter().enumerate() {
            ensure!(
                view.obs.cameras.len() == num_cameras,
                "view {} has {} cameras, expected {}",
                idx,
                view.obs.cameras.len(),
                num_cameras
            );
        }
        Ok(Self { num_cameras, views })
    }
}

/// Empty metadata marker for views that do not need extra metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoMeta;

/// A planar dataset consisting of multiple views.
///
/// Each view observes a planar calibration target in pixel coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarDataset {
    /// Sequence of planar calibration views.
    pub views: Vec<View<NoMeta>>,
}

impl PlanarDataset {
    /// Construct a planar dataset and validate basic cardinality constraints.
    pub fn new(views: Vec<View<NoMeta>>) -> Result<Self> {
        ensure!(!views.is_empty(), "need at least one view for calibration");
        for (i, view) in views.iter().enumerate() {
            ensure!(
                view.obs.len() >= 4,
                "view {} has too few points (need >=4)",
                i
            );
        }
        Ok(Self { views })
    }

    /// Number of planar views.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }
}
