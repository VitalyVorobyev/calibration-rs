# Python parity audit (D3)

Audit of the PyO3 binding surface (`vision-calibration-py`, module
`_vision_calibration`) against the Rust facade (`vision-calibration`), as of
2026-06-21. Goal: identify what the Python package can and cannot reach, so the
remaining binding work toward the v1.0 "binding parity" gate is concrete.

## Binding style

Bindings are **JSON/serde-based**, not native `pyclass` wrappers. The
`run_problem` helper (`src/lib.rs`) uses `pythonize`/`depythonize` to convert
Python dicts ⇄ Rust types via `serde`, so a function is bindable in this style
**iff** its inputs are `DeserializeOwned` and its outputs are `Serialize`. The
Python side ships dataclass wrappers (`python/vision_calibration/models.py`,
`types.py`) plus type stubs (`__init__.pyi`).

## Bound today — parity exists

**Seven of the eight** facade calibration workflows and their helpers are bound
(the eighth, `rig_handeye_laserline`, is missing — see [G0](#g0--rig-hand-eye--laserline-workflow-unbound-cheap-highest-priority)):

| Rust facade | Python |
|---|---|
| `run_calibration` for `PlanarIntrinsics` | `run_planar_intrinsics` |
| `ScheimpflugIntrinsics` | `run_scheimpflug_intrinsics` |
| `SingleCamHandeye` | `run_single_cam_handeye` |
| `RigExtrinsics` | `run_rig_extrinsics` |
| `RigHandeye` | `run_rig_handeye` |
| `LaserlineDevice` | `run_laserline_device` |
| `RigLaserlineDevice` | `run_rig_laserline_device` |
| `RobustLoss` constructors | `robust_{none,huber,cauchy,arctan}` |
| `pixel_to_gripper_point` | `pixel_to_gripper_point` |
| workspace version | `library_version` |

The core calibration value proposition — run a calibration from Python with
JSON-friendly config/input/export — is met for seven of the eight workflows.

## Gaps — Rust-only

### G0 — rig hand-eye + laserline workflow unbound [cheap, highest priority]

The facade exports an **eighth** calibration workflow,
`rig_handeye_laserline::run_calibration` with `RigHandeyeLaserlineProblem`
(`vision-calibration/src/lib.rs:446`), but the PyO3 module registers only the
seven runners (`vision-calibration-py/src/lib.rs:357–367`) — there is **no**
`run_rig_handeye_laserline`. Python users cannot run the joint rig hand-eye +
laserline calibration. **Effort: low** — it is the same `run_problem::<P>`
JSON/serde pattern as the other seven (the `RigHandeyeLaserlineProblem`
`Input`/`Config`/`Export` are serde types like the rest); add one
`#[pyfunction]`, register it, and ship the Python config/result dataclass +
`.pyi` + a smoke test. This should be the **first** fill item — it completes the
calibration-workflow surface, the binding's core value.

### G1 — MVG surface (`geometry` + `mvg` modules) [biggest gap]

Entirely unbound. None of the following are reachable from Python:

- `geometry`: `epipolar` (fundamental/essential/decompose), `homography`
  (DLT + RANSAC), `triangulation`, `camera_matrix`.
- `mvg`: `pose_recovery::recover_relative_pose`,
  `triangulation::triangulate_nview`, `rectification::rectify_stereo_pair`,
  `robust::*` (RANSAC), `cheirality`, `degeneracy`, and
  `bundle_adjust::bundle_adjust` (behind the facade `refine` feature).

**Effort: medium.** The MVG API takes/returns **raw nalgebra types**
(`Pt2`/`Pt3`/`Mat3`/`Iso3`/`Correspondence2D`), not JSON DTOs, so binding in the
existing `pythonize` style needs serde-friendly payload structs (or numpy
interop) per entry point. `bundle_adjust` is `refine`-gated → conditional
binding + the `tiny-solver` build. This is the load-bearing item for the v1.0
"binding parity" goal and is best done as its own slice (DTOs → bind
triangulation + rectification + pose recovery → robust → BA → `.pyi` + parity
tests).

### G2 — M-WIRE distortion-model selection [cheap]

`PlanarIntrinsicsConfig.distortion_model: DistortionKind` (added in the M-WIRE
slice, PR #79) is **not** surfaced in the Python `PlanarIntrinsicsConfig`
wrapper (`models.py`/`types.py` carry `fix_distortion` but no
`distortion_model`). Expected — M-WIRE was scoped Rust-core-only — but now
actionable. **Effort: low** (add the field to the config dataclass + the stub;
a string/enum maps to the serde `snake_case` `DistortionKind` tag).

### G3 — low-level / by-design-unbound modules

`linear`, `optim` (backend internals), `synthetic`, `analysis`, and the raw
`session` API are not bound. Most are internal or low-level; the `session` flow
is already covered transitively by the `run_*` functions. Parity here is
**probably not required** for v1.0 — revisit only if a Python consumer needs a
specific low-level entry point.

## Parity-test gap

There is no test asserting that every facade `run_*` workflow has a Python
binding, nor round-trip tests for the (future) MVG bindings. A small
binding-coverage test (introspect the module's `__all__` against the known
workflow list) would catch a future workflow being added in Rust but forgotten
in Python.

## Recommended sequencing toward the v1.0 binding-parity gate

1. **G0** (`run_rig_handeye_laserline`) — cheap, completes the
   calibration-workflow surface (the binding's core value). Do first.
2. **G2** (distortion-model field) — cheap, completes M-WIRE for Python.
3. **G1 MVG bindings**, in value order: serde DTOs → `triangulate_nview` +
   `rectify_stereo_pair` + `recover_relative_pose` → `robust::*` →
   `bundle_adjust` (`refine`-gated). Ship `.pyi` stubs + per-binding
   round-trip parity tests alongside.
4. **Binding-coverage parity test** (workflow list ⇄ module surface) — this is
   the guard that would have caught G0; add it so a future Rust workflow can't
   silently miss its Python binding.

G3 is deferred pending a concrete consumer.
