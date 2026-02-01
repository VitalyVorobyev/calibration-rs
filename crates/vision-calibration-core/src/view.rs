//! View-related structures and functions.
//!

use crate::CorrespondenceView;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct View<Meta> {
    pub obs: CorrespondenceView,
    pub meta: Meta,
}

impl<Meta> View<Meta> {
    /// Number of observations in this view.
    pub fn num_observations(&self) -> usize {
        self.obs.points_3d.len()
    }

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigView<Meta> {
    pub obs: RigViewObs,
    pub meta: Meta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigDataset<Meta> {
    pub num_cameras: usize,
    pub views: Vec<RigView<Meta>>,
}

impl<Meta> RigDataset<Meta> {
    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }

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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoMeta;

/// A planar dataset consisting of multiple views.
///
/// Each view observes a planar calibration target in pixel coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarDataset {
    pub views: Vec<View<NoMeta>>,
}

impl PlanarDataset {
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

    pub fn num_views(&self) -> usize {
        self.views.len()
    }
}
