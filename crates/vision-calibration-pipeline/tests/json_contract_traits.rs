#![allow(missing_docs)]

use serde::{Serialize, de::DeserializeOwned};
use vision_calibration_pipeline::{
    common::config::CameraConfig,
    laserline_device::{
        DeviceOptimizeConfig, LaserlineDeviceConfig, LaserlineDeviceEstimate,
        LaserlineDeviceExport, LaserlineDeviceInput, LaserlineDeviceProblem,
    },
    planar_intrinsics::{
        IntrinsicsManualInit, PlanarIntrinsicsConfig, PlanarIntrinsicsEstimate,
        PlanarIntrinsicsExport, PlanarIntrinsicsProblem,
    },
    rig_extrinsics::{RigExtrinsicsConfig, RigExtrinsicsExport, RigExtrinsicsInput},
    rig_handeye::{RigHandeyeConfig, RigHandeyeExport, RigHandeyeInput},
    session::{CalibrationSession, ExportRecord, LogEntry, SessionMetadata},
    single_cam_handeye::{
        HandeyeMeta, SingleCamHandeyeConfig, SingleCamHandeyeExport, SingleCamHandeyeInput,
    },
};

fn assert_json_contract<T>()
where
    T: Serialize + DeserializeOwned,
{
}

#[test]
fn session_contract_types_are_serde() {
    assert_json_contract::<SessionMetadata>();
    assert_json_contract::<LogEntry>();
    assert_json_contract::<ExportRecord<PlanarIntrinsicsExport>>();
    assert_json_contract::<CalibrationSession<PlanarIntrinsicsProblem>>();
    assert_json_contract::<CalibrationSession<LaserlineDeviceProblem>>();
}

#[test]
fn planar_intrinsics_contract_types_are_serde() {
    assert_json_contract::<PlanarIntrinsicsConfig>();
    assert_json_contract::<CameraConfig>();
    assert_json_contract::<IntrinsicsManualInit>();
    assert_json_contract::<PlanarIntrinsicsEstimate>();
    assert_json_contract::<PlanarIntrinsicsExport>();
}

#[test]
fn rig_extrinsics_contract_types_are_serde() {
    assert_json_contract::<RigExtrinsicsInput>();
    assert_json_contract::<RigExtrinsicsConfig>();
    assert_json_contract::<RigExtrinsicsExport>();
}

#[test]
fn single_cam_handeye_contract_types_are_serde() {
    assert_json_contract::<HandeyeMeta>();
    assert_json_contract::<SingleCamHandeyeInput>();
    assert_json_contract::<SingleCamHandeyeConfig>();
    assert_json_contract::<SingleCamHandeyeExport>();
}

#[test]
fn rig_handeye_contract_types_are_serde() {
    assert_json_contract::<RigHandeyeInput>();
    assert_json_contract::<RigHandeyeConfig>();
    assert_json_contract::<RigHandeyeExport>();
}

#[test]
fn laserline_contract_types_are_serde() {
    assert_json_contract::<LaserlineDeviceInput>();
    assert_json_contract::<LaserlineDeviceConfig>();
    assert_json_contract::<DeviceOptimizeConfig>();
    assert_json_contract::<LaserlineDeviceEstimate>();
    assert_json_contract::<LaserlineDeviceExport>();
}
