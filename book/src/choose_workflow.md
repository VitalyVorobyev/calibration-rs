# Choose Your Workflow

Three questions settle how to calibrate a system: how you want to run it,
which workflow matches your hardware, and which camera model matches your
lenses. This chapter answers them in that order and ends with what to
capture.

## 1. How you run it

| Route | When | Start at |
|---|---|---|
| **Desktop app** | You have a folder of images and want a result and its diagnostics without writing code | [The Desktop App](app.md) |
| **Python** | You script your lab work or analysis in Python | [Getting Started](getting_started.md), the Python half |
| **Rust** | You embed calibration in a Rust program or need the session steps | [Getting Started](getting_started.md), the Rust half |

All three run the same solvers and produce the same JSON export, so a result
from one opens in the others.

## 2. Which workflow

The workflow follows from the hardware: how many cameras there are, whether
a robot moves the camera or the target, and whether a laser projects a line
for triangulation.

| Your system | Workflow | Manifest `topology` | Python runner (`vc.<module>.run_calibration`) |
|---|---|---|---|
| One camera; you want its intrinsics and distortion | [Planar intrinsics](planar_intrinsics.md) | `planar_intrinsics` | `planar_intrinsics.run_calibration` |
| One camera on a robot, or a fixed camera watching a target the robot holds | [Single-camera hand-eye](handeye_workflow.md) | `single_cam_handeye` | `single_cam_handeye.run_calibration` |
| Two or more rigidly mounted cameras: a stereo pair or a multi-camera rig | [Rig extrinsics](rig_extrinsics.md) | `rig_extrinsics` | `rig_extrinsics.run_calibration` |
| A camera rig on a robot, or a fixed rig watching the robot's target | [Rig hand-eye](rig_handeye.md) | `rig_handeye` | `rig_handeye.run_calibration` |
| One camera and a laser line projector mounted together | [Laserline device](laserline.md) | `laserline_device` | `laserline_device.run_calibration` |
| A rig on a robot with one laser per camera, rig already calibrated | [Rig laserline device](scheimpflug_laser_rigs.md) | `rig_laserline_device` | `rig_laserline_device.run_calibration` |
| The same rig with the rig, hand-eye and laser planes in one solve | [Rig hand-eye laserline](scheimpflug_laser_rigs.md) | `rig_handeye_laserline` | `rig_handeye_laserline.run_calibration` |
| One camera and its laser on a robot, with the camera, plane and hand-eye in one solve | [Rig hand-eye laserline](scheimpflug_laser_rigs.md#one-camera) (a rig of one) | `rig_handeye_laserline` | `rig_handeye_laserline.run_calibration` |

Each workflow also calibrates the intrinsics of every camera it uses: a rig
does not need a planar-intrinsics run per camera first. The hand-eye
workflows cover both mountings, the camera on the gripper (eye-in-hand, the
default) and the camera fixed in the scene (eye-to-hand); the configuration
says which.

The two rig laser workflows differ in staging. `rig_laserline_device` fits
the laser planes against a rig hand-eye result you computed earlier and
keep fixed; `rig_handeye_laserline` estimates everything at once.

A single laser head on a robot can be calibrated in two workflows, the
laserline device and then the single-camera hand-eye. If you do this, hold the
first workflow's camera fixed in the second. The laser plane is in that
camera's frame, and a hand-eye found with a re-estimated camera does not
compose with it. `rig_handeye_laserline` with one camera avoids the question.

## 3. Which camera model

The camera model is a setting of every workflow, not a workflow of its own.
A fisheye stereo pair is a `rig_extrinsics` calibration with a fisheye
camera model; a tilted-sensor laser profiler is a `laserline_device`
calibration with a Scheimpflug sensor. The model has three parts:

| Part | Choices | Pick |
|---|---|---|
| Projection | pinhole; equidistant (fisheye) | pinhole for ordinary lenses; equidistant for fisheye lenses, and always for rays 90° or more off the optical axis, which a pinhole cannot represent |
| Distortion | Brown-Conrady (5 terms, the default), rational (8), thin prism (9), division (1), Kannala-Brandt (4, for fisheye) | the default for ordinary lenses; Kannala-Brandt with the equidistant projection; the richer kinds only when the residuals show a pattern the default leaves |
| Sensor | straight (default); Scheimpflug (tilted) | Scheimpflug only when the sensor is deliberately tilted, as in laser profilers |

The details are in [Choosing a Distortion Model](distortion_selection.md),
[Fisheye Projection](fisheye.md) and
[Sensor Models and Scheimpflug Tilt](sensor.md). A rig may mix models: each
camera can override the shared one.

## 4. What to capture

Every workflow observes a planar calibration target: a chessboard, a
ChArUco board, a PuzzleBoard or a ring grid
([Calibration Targets and Detection](detection.md)). Measure the square or
ring pitch of the printed board; the calibration's scale comes from it.

- **Views.** At least 3 per camera for any workflow to run; 15 to 30 for a
  result worth keeping. Tilt the board 15–30° about both of its axes, make
  it a third to a half of the image wide, and move it into the corners of
  the image, where distortion is largest. [Capturing Calibration
  Images](capture.md) measures what each of these buys.
- **Rigs.** Every camera needs its own 3 or more views of the board, and the
  rig's cameras need views in which they see the board at the same time, so
  the board links them.
- **Robots.** One robot pose per view, from the controller, with the frame
  convention and units written down: the manifest asks for them and does
  not guess. Rotate the gripper about at least two axes between views; a
  hand-eye calibration from translations alone is undetermined.
- **Lasers.** Two frames per view: one with the board lit and the laser off,
  one with the laser on ([Laser Datasets](manifest_laser.md)).

Every result carries its parameter uncertainty, which shows how well the
views constrained it ([Parameter Uncertainty](uncertainty.md)). For a single
camera, the library also ranks candidate board poses by how much each would
help as the next view ([Planning the Next View](uncertainty_next_view.md)).

## Next

Describe your image folder in a [dataset manifest](manifest.md). The folder
sniffer writes a first draft from the file and folder names and leaves what
only you know, such as the target and the robot's pose convention, marked
for you to fill in; the desktop app edits the manifest in a form. Then follow the chapter for your workflow from the table
above, and judge the result with [Judging a Result](quality.md).
