//! View-related structures and functions.
//!

use crate::CorrespondenceView;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct View<Meta> {
    pub obs: CorrespondenceView,
    pub meta: Meta,
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
