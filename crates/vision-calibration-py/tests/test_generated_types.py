"""The payloads the package exchanges with the extension have the shapes of
the generated ``TypedDict``\\ s.

``_generated_types.py`` is rendered from the JSON Schemas of the Rust types.
These tests record the raw payloads of real runs — what the dataclasses'
``to_payload`` sends and what the extension returns — and check each against
the ``TypedDict`` of its Rust type, so a dataclass that drifts from the Rust
shape, or a generated type that misdescribes it, fails here.
"""

from __future__ import annotations

import types
import typing
import unittest
from typing import Any
from unittest import mock

import vision_calibration as vc
import vision_calibration._api as vc_api
import vision_calibration._generated_types as gen
import vision_calibration.types as vc_types
from _fixtures import planar_calibration_dataset
from test_distortion_sensor_consumption import _rig_extrinsics_dataset
from test_rig_handeye_laserline import _eye_to_hand_config, _synthetic_dataset


def shape_errors(value: Any, tp: Any, path: str = "$") -> list[str]:
    """Where ``value`` departs from the type ``tp``; empty when it conforms."""
    if tp is Any:
        return []
    if tp is None or tp is type(None):
        return [] if value is None else [f"{path}: expected None, got {value!r:.60}"]
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        member_errors = [shape_errors(value, member, path) for member in args]
        if any(not errors for errors in member_errors):
            return []
        return [f"{path}: matches no member of {tp}"] + min(member_errors, key=len)
    if origin is typing.Literal:
        return [] if value in args else [f"{path}: {value!r} is not one of {args}"]
    if origin is list:
        if not isinstance(value, list):
            return [f"{path}: expected a list, got {type(value).__name__}"]
        return [e for i, item in enumerate(value) for e in shape_errors(item, args[0], f"{path}[{i}]")]
    if origin is tuple:
        if not isinstance(value, tuple) or len(value) != len(args):
            return [f"{path}: expected a {len(args)}-tuple, got {value!r:.60}"]
        return [e for i, (item, arg) in enumerate(zip(value, args)) for e in shape_errors(item, arg, f"{path}[{i}]")]
    if origin is dict:
        if not isinstance(value, dict):
            return [f"{path}: expected a dict, got {type(value).__name__}"]
        return [e for k, item in value.items() for e in shape_errors(item, args[1], f"{path}.{k}")]
    if typing.is_typeddict(tp):
        if not isinstance(value, dict):
            return [f"{path}: expected {tp.__name__}, got {type(value).__name__}"]
        hints = typing.get_type_hints(tp)
        errors = [f"{path}: missing required key {k!r} of {tp.__name__}" for k in sorted(tp.__required_keys__ - value.keys())]
        errors += [f"{path}: key {k!r} is not in {tp.__name__}" for k in sorted(value.keys() - hints.keys())]
        for key in sorted(value.keys() & hints.keys()):
            errors += shape_errors(value[key], hints[key], f"{path}.{key}")
        return errors
    if tp is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif tp in (str, bool):
        ok = isinstance(value, tp)
    else:
        raise TypeError(f"{path}: the checker does not handle {tp!r}")
    return [] if ok else [f"{path}: expected {tp.__name__}, got {value!r:.60}"]


class _Recorder:
    """Stands in for the extension module, recording each call's payloads."""

    def __init__(self, native: Any) -> None:
        self._native = native
        self.calls: list[tuple[str, tuple[Any, ...], Any]] = []

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._native, name)

        def call(*args: Any) -> Any:
            result = target(*args)
            self.calls.append((name, args, result))
            return result

        return call


class GeneratedShapesTest(unittest.TestCase):
    def assertConforms(self, value: Any, tp: Any) -> None:
        errors = shape_errors(value, tp)
        self.assertEqual(errors, [], f"payload does not match {tp}:\n" + "\n".join(errors[:20]))

    def test_default_config_payloads_match_the_schema(self) -> None:
        for name in [
            "PlanarIntrinsicsConfig",
            "SingleCamHandeyeConfig",
            "RigExtrinsicsConfig",
            "RigHandeyeConfig",
            "LaserlineDeviceConfig",
            "RigLaserlineDeviceConfig",
            "RigHandeyeLaserlineConfig",
            "QualityThresholds",
        ]:
            with self.subTest(name):
                self.assertConforms(getattr(vc, name)().to_payload(), getattr(gen, name))

    def test_workflow_payloads_match_the_schema(self) -> None:
        recorder = _Recorder(vc_api._native)
        with mock.patch.object(vc_api, "_native", recorder):
            planar = vc.run_planar_intrinsics(planar_calibration_dataset(), vc.PlanarIntrinsicsConfig())
            vc.assess(planar, vc.QualityThresholds(), [(1280, 720)])
            vc.projection_uncertainty(planar, (1280, 720))
            vc.run_rig_extrinsics(_rig_extrinsics_dataset(), vc.RigExtrinsicsConfig())
            vc.run_rig_handeye_laserline(_synthetic_dataset(), _eye_to_hand_config())

        expected = {
            "run_planar_intrinsics": "PlanarIntrinsics",
            "run_rig_extrinsics": "RigExtrinsics",
            "run_rig_handeye_laserline": "RigHandeyeLaserline",
        }
        seen = set()
        for name, args, result in recorder.calls:
            seen.add(name)
            with self.subTest(name):
                if name in expected:
                    problem = expected[name]
                    self.assertConforms(args[0], getattr(gen, f"{problem}Input"))
                    self.assertConforms(args[1], getattr(gen, f"{problem}Config"))
                    self.assertConforms(result, getattr(gen, f"{problem}Export"))
                elif name == "assess":
                    self.assertConforms(args[0], gen.PlanarIntrinsicsExport)
                    self.assertConforms(args[1], gen.QualityThresholds)
                    self.assertConforms(result, gen.QualityReport)
                elif name == "projection_uncertainty":
                    self.assertConforms(result, gen.ProjectionUncertainty)
        self.assertEqual(seen, {*expected, "assess", "projection_uncertainty"})

    def test_a_partial_config_is_not_a_config(self) -> None:
        # The library reads a config whole: a key the schema requires cannot
        # be left out, and the generated type says so.
        partial = {"solver": {"max_iters": 5}}
        self.assertNotEqual(shape_errors(partial, gen.PlanarIntrinsicsConfig), [])
        with self.assertRaises(TypeError):
            vc_api._native.run_planar_intrinsics(planar_calibration_dataset().to_payload(), partial)

    def test_types_module_reexports_the_generated_types(self) -> None:
        for name in gen.__all__:
            self.assertIs(getattr(vc_types, name), getattr(gen, name), name)
        self.assertIs(vc_types.Transform, gen.Iso3)
        self.assertIs(vc_types.DistortionModel, gen.DistortionKind)
        self.assertIs(vc_types.RobustLossHuber, gen.RobustLoss_Huber)


if __name__ == "__main__":
    unittest.main()
