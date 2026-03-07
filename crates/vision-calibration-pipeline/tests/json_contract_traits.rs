use serde::{Serialize, de::DeserializeOwned};
use vision_calibration_pipeline::{
    laserline_device::{
        LaserlineDeviceConfig, LaserlineDeviceExport, LaserlineDeviceInitConfig,
        LaserlineDeviceInput, LaserlineDeviceOptimizeConfig, LaserlineDeviceOutput,
        LaserlineDeviceProblem, LaserlineDeviceSolverConfig,
    },
    planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsExport, PlanarIntrinsicsProblem},
    rig_extrinsics::{RigExtrinsicsConfig, RigExtrinsicsExport, RigExtrinsicsInput},
    rig_handeye::{
        RigHandeyeBaConfig, RigHandeyeConfig, RigHandeyeExport, RigHandeyeInitConfig,
        RigHandeyeInput, RigHandeyeIntrinsicsConfig, RigHandeyeRigConfig, RigHandeyeSolverConfig,
    },
    scheimpflug_intrinsics::{
        ScheimpflugFixMask, ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsExport,
        ScheimpflugIntrinsicsInput, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsProblem,
        ScheimpflugIntrinsicsResult,
    },
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
    assert_json_contract::<CalibrationSession<ScheimpflugIntrinsicsProblem>>();
    assert_json_contract::<CalibrationSession<LaserlineDeviceProblem>>();
}

#[test]
fn planar_intrinsics_contract_types_are_serde() {
    assert_json_contract::<PlanarIntrinsicsConfig>();
    assert_json_contract::<PlanarIntrinsicsExport>();
}

#[test]
fn scheimpflug_intrinsics_contract_types_are_serde() {
    assert_json_contract::<ScheimpflugFixMask>();
    assert_json_contract::<ScheimpflugIntrinsicsInput>();
    assert_json_contract::<ScheimpflugIntrinsicsConfig>();
    assert_json_contract::<ScheimpflugIntrinsicsParams>();
    assert_json_contract::<ScheimpflugIntrinsicsResult>();
    assert_json_contract::<ScheimpflugIntrinsicsExport>();
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
    assert_json_contract::<RigHandeyeIntrinsicsConfig>();
    assert_json_contract::<RigHandeyeRigConfig>();
    assert_json_contract::<RigHandeyeInitConfig>();
    assert_json_contract::<RigHandeyeSolverConfig>();
    assert_json_contract::<RigHandeyeBaConfig>();
    assert_json_contract::<RigHandeyeExport>();
}

#[test]
fn laserline_contract_types_are_serde() {
    assert_json_contract::<LaserlineDeviceInput>();
    assert_json_contract::<LaserlineDeviceConfig>();
    assert_json_contract::<LaserlineDeviceInitConfig>();
    assert_json_contract::<LaserlineDeviceSolverConfig>();
    assert_json_contract::<LaserlineDeviceOptimizeConfig>();
    assert_json_contract::<LaserlineDeviceOutput>();
    assert_json_contract::<LaserlineDeviceExport>();
}
