//! Intermediate state for joint rig hand-eye laserline calibration.

use serde::{Deserialize, Serialize};

/// Joint calibration has no public intermediate step state today; the session
/// state exists to satisfy the generic `ProblemType` plumbing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct RigHandeyeLaserlineState;
