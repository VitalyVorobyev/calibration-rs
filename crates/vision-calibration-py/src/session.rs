//! Calibration sessions: one Python class per problem type, each wrapping a
//! `CalibrationSession<P>`.
//!
//! Everything a session does is written once, generic over the problem type
//! (the `session_*` functions below); the `session_class!` macro only stamps
//! out the per-problem `#[pyclass]` and its step methods, each a one-line call
//! into the facade's step function. Payloads cross as their serde shapes; the
//! Python layer wraps them in dataclasses.

use pyo3::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use vision_calibration::session::{CalibrationSession, ProblemType};

use crate::errors::calib_err;
use crate::validation::{reject_non_finite, value_err};
use crate::{parse_optional_payload, parse_payload, runtime_err};

/// Serialize `value` to its Python payload.
pub(crate) fn to_py<T: Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    Ok(pythonize::pythonize(py, value)
        .map_err(|err| runtime_err(format!("failed to convert result to Python: {err}")))?
        .unbind())
}

/// Parse a payload after rejecting non-finite floats in it.
pub(crate) fn parse_checked<T: DeserializeOwned>(
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<T> {
    reject_non_finite(name, value)?;
    parse_payload(value, name)
}

/// Parse an optional payload (`None` → `None`) after rejecting non-finite
/// floats in it.
pub(crate) fn parse_optional_checked<T: DeserializeOwned>(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<T>> {
    if let Some(v) = value {
        reject_non_finite(name, v)?;
    }
    parse_optional_payload(value, name)
}

fn session_new<P: ProblemType>(description: Option<String>) -> CalibrationSession<P> {
    match description {
        Some(text) => CalibrationSession::with_description(text),
        None => CalibrationSession::new(),
    }
}

fn session_set_input<P: ProblemType>(
    session: &mut CalibrationSession<P>,
    input: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let input: P::Input = parse_checked(input, "input")?;
    session
        .set_input(input)
        .map_err(|err| value_err(format!("invalid input: {err}")))
}

fn session_set_config<P: ProblemType>(
    session: &mut CalibrationSession<P>,
    config: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let config: P::Config = parse_checked(config, "config")?;
    session
        .set_config(config)
        .map_err(|err| value_err(format!("invalid config: {err}")))
}

fn session_export<P: ProblemType>(
    py: Python<'_>,
    session: &mut CalibrationSession<P>,
    notes: Option<String>,
) -> PyResult<Py<PyAny>> {
    let export = match notes {
        Some(notes) => session.export_with_notes(notes),
        None => session.export(),
    }
    .map_err(|err| calib_err("export failed", err))?;
    to_py(py, &export)
}

fn session_to_json<P: ProblemType>(session: &CalibrationSession<P>) -> PyResult<String> {
    session
        .to_json()
        .map_err(|err| runtime_err(format!("failed to serialize the session: {err}")))
}

fn session_from_json<P: ProblemType>(json: &str) -> PyResult<CalibrationSession<P>> {
    CalibrationSession::from_json(json)
        .map_err(|err| value_err(format!("invalid session JSON: {err}")))
}

/// Stamp out the session class of one problem type.
///
/// `steps` take optional step options; `seeded` steps also take the manual
/// init seed. Every name resolves in the problem's facade module, which the
/// invocation glob-imports.
macro_rules! session_class {
    (
        $(#[$doc:meta])*
        $class:ident = $name:literal, $problem:ty;
        steps { $( $step:ident ( $opts:ty ) -> $res:ty ; )* }
        seeded { $( $sstep:ident ( $seed:ty , $sopts:ty ) -> $sres:ty ; )* }
    ) => {
        $(#[$doc])*
        #[pyclass(module = "vision_calibration._vision_calibration", name = $name)]
        pub(crate) struct $class {
            inner: CalibrationSession<$problem>,
        }

        #[pymethods]
        impl $class {
            #[new]
            #[pyo3(signature = (description=None))]
            fn new(description: Option<String>) -> Self {
                Self { inner: session_new(description) }
            }

            fn set_input(&mut self, input: &Bound<'_, PyAny>) -> PyResult<()> {
                session_set_input(&mut self.inner, input)
            }

            fn input(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
                self.inner.input().map(|input| to_py(py, input)).transpose()
            }

            fn has_input(&self) -> bool {
                self.inner.has_input()
            }

            fn set_config(&mut self, config: &Bound<'_, PyAny>) -> PyResult<()> {
                session_set_config(&mut self.inner, config)
            }

            fn config(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                to_py(py, &self.inner.config)
            }

            fn has_output(&self) -> bool {
                self.inner.has_output()
            }

            fn output(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
                self.inner.output().map(|output| to_py(py, output)).transpose()
            }

            #[pyo3(signature = (notes=None))]
            fn export(&mut self, py: Python<'_>, notes: Option<String>) -> PyResult<Py<PyAny>> {
                session_export(py, &mut self.inner, notes)
            }

            fn exports(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                to_py(py, &self.inner.exports)
            }

            fn log(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                to_py(py, &self.inner.log())
            }

            fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                to_py(py, self.inner.metadata())
            }

            fn validate(&self) -> PyResult<()> {
                self.inner.validate().map_err(|err| calib_err("invalid session", err))
            }

            fn reset(&mut self) {
                self.inner.reset();
            }

            fn to_json(&self) -> PyResult<String> {
                session_to_json(&self.inner)
            }

            #[staticmethod]
            fn from_json(json: &str) -> PyResult<Self> {
                Ok(Self { inner: session_from_json(json)? })
            }

            fn run_calibration(&mut self, py: Python<'_>) -> PyResult<()> {
                let inner = &mut self.inner;
                py.detach(|| run_calibration(inner))
                    .map_err(|err| calib_err("run_calibration failed", err))
            }

            $(
                #[pyo3(signature = (options=None))]
                fn $step(
                    &mut self,
                    py: Python<'_>,
                    options: Option<&Bound<'_, PyAny>>,
                ) -> PyResult<Py<PyAny>> {
                    let opts: Option<$opts> = parse_optional_checked(options, "options")?;
                    let inner = &mut self.inner;
                    let result: $res = py
                        .detach(|| $step(inner, opts))
                        .map_err(|err| calib_err(concat!(stringify!($step), " failed"), err))?;
                    to_py(py, &result)
                }
            )*

            $(
                #[pyo3(signature = (seed, options=None))]
                fn $sstep(
                    &mut self,
                    py: Python<'_>,
                    seed: &Bound<'_, PyAny>,
                    options: Option<&Bound<'_, PyAny>>,
                ) -> PyResult<Py<PyAny>> {
                    let seed: $seed = parse_checked(seed, "seed")?;
                    let opts: Option<$sopts> = parse_optional_checked(options, "options")?;
                    let inner = &mut self.inner;
                    let result: $sres = py
                        .detach(|| $sstep(inner, seed, opts))
                        .map_err(|err| calib_err(concat!(stringify!($sstep), " failed"), err))?;
                    to_py(py, &result)
                }
            )*
        }
    };
}

pub(crate) mod planar {
    use super::*;
    use vision_calibration::planar_intrinsics::*;

    session_class! {
        /// `CalibrationSession<PlanarIntrinsicsProblem>`.
        PlanarIntrinsicsSession = "PlanarIntrinsicsSession", PlanarIntrinsicsProblem;
        steps {
            step_init(IntrinsicsInitOptions) -> IntrinsicsInitResult;
            step_optimize(IntrinsicsOptimizeOptions) -> IntrinsicsOptimizeResult;
            step_filter(IntrinsicsFilterOptions) -> ();
            run_calibration_with_filtering(IntrinsicsFilterOptions) -> ();
        }
        seeded {
            step_init_with_seed(IntrinsicsManualInit, IntrinsicsInitOptions) -> IntrinsicsInitResult;
        }
    }
}

pub(crate) mod single_cam_handeye {
    use super::*;
    use vision_calibration::single_cam_handeye::*;

    session_class! {
        /// `CalibrationSession<SingleCamHandeyeProblem>`.
        SingleCamHandeyeSession = "SingleCamHandeyeSession", SingleCamHandeyeProblem;
        steps {
            step_intrinsics_init(IntrinsicsInitOptions) -> IntrinsicsInitResult;
            step_intrinsics_optimize(IntrinsicsOptimizeOptions) -> IntrinsicsOptimizeResult;
            step_handeye_init(HandeyeInitOptions) -> HandeyeInitResult;
            step_handeye_optimize(HandeyeOptimizeOptions) -> HandeyeOptimizeResult;
        }
        seeded {
            step_intrinsics_init_with_seed(IntrinsicsManualInit, IntrinsicsInitOptions)
                -> IntrinsicsInitResult;
            step_handeye_init_with_seed(HandeyeManualInit, HandeyeInitOptions) -> HandeyeInitResult;
        }
    }
}

pub(crate) mod laserline_device {
    use super::*;
    use vision_calibration::laserline_device::*;

    session_class! {
        /// `CalibrationSession<LaserlineDeviceProblem>`.
        LaserlineDeviceSession = "LaserlineDeviceSession", LaserlineDeviceProblem;
        steps {
            step_init(DeviceInitOptions) -> DeviceInitResult;
            step_optimize(DeviceOptimizeOptions) -> DeviceOptimizeResult;
        }
        seeded {
            step_init_with_seed(DeviceManualInit, DeviceInitOptions) -> DeviceInitResult;
        }
    }
}

pub(crate) mod rig_extrinsics {
    use super::*;
    use vision_calibration::rig_extrinsics::*;

    session_class! {
        /// `CalibrationSession<RigExtrinsicsProblem>`.
        RigExtrinsicsSession = "RigExtrinsicsSession", RigExtrinsicsProblem;
        steps {
            step_intrinsics_init_all(IntrinsicsInitOptions) -> IntrinsicsInitResult;
            step_intrinsics_optimize_all(IntrinsicsOptimizeOptions) -> IntrinsicsOptimizeResult;
            step_rig_init(RigInitOptions) -> RigInitResult;
            step_rig_optimize(RigOptimizeOptions) -> RigOptimizeResult;
        }
        seeded {
            step_intrinsics_init_all_with_seed(IntrinsicsManualInit, IntrinsicsInitOptions)
                -> IntrinsicsInitResult;
            step_rig_init_with_seed(RigManualInit, RigInitOptions) -> RigInitResult;
        }
    }
}

pub(crate) mod rig_handeye {
    use super::*;
    use vision_calibration::rig_handeye::*;

    session_class! {
        /// `CalibrationSession<RigHandeyeProblem>`.
        RigHandeyeSession = "RigHandeyeSession", RigHandeyeProblem;
        steps {
            step_intrinsics_init_all(IntrinsicsInitOptions) -> IntrinsicsInitResult;
            step_intrinsics_optimize_all(IntrinsicsOptimizeOptions) -> IntrinsicsOptimizeResult;
            step_rig_init(RigInitOptions) -> RigInitResult;
            step_rig_optimize(RigOptimizeOptions) -> RigOptimizeResult;
            step_handeye_init(HandeyeInitOptions) -> HandeyeInitResult;
            step_handeye_optimize(HandeyeOptimizeOptions) -> HandeyeOptimizeResult;
        }
        seeded {
            step_intrinsics_init_all_with_seed(IntrinsicsManualInit, IntrinsicsInitOptions)
                -> IntrinsicsInitResult;
            step_rig_init_with_seed(RigManualInit, RigInitOptions) -> RigInitResult;
            step_handeye_init_with_seed(HandeyeManualInit, HandeyeInitOptions) -> HandeyeInitResult;
        }
    }
}

pub(crate) mod rig_laserline_device {
    use super::*;
    use vision_calibration::rig_laserline_device::*;

    session_class! {
        /// `CalibrationSession<RigLaserlineDeviceProblem>`.
        RigLaserlineDeviceSession = "RigLaserlineDeviceSession", RigLaserlineDeviceProblem;
        steps {
            step_init(PlaneInitOptions) -> ();
            step_optimize(PlaneOptimizeOptions) -> ();
        }
        seeded {
            step_init_with_seed(PlaneManualInit, PlaneInitOptions) -> ();
        }
    }
}

pub(crate) mod rig_handeye_laserline {
    use super::*;
    use vision_calibration::rig_handeye_laserline::*;

    session_class! {
        /// `CalibrationSession<RigHandeyeLaserlineProblem>`.
        RigHandeyeLaserlineSession = "RigHandeyeLaserlineSession", RigHandeyeLaserlineProblem;
        steps {}
        seeded {}
    }
}

/// Register the session classes on the extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<planar::PlanarIntrinsicsSession>()?;
    m.add_class::<single_cam_handeye::SingleCamHandeyeSession>()?;
    m.add_class::<laserline_device::LaserlineDeviceSession>()?;
    m.add_class::<rig_extrinsics::RigExtrinsicsSession>()?;
    m.add_class::<rig_handeye::RigHandeyeSession>()?;
    m.add_class::<rig_laserline_device::RigLaserlineDeviceSession>()?;
    m.add_class::<rig_handeye_laserline::RigHandeyeLaserlineSession>()?;
    Ok(())
}
