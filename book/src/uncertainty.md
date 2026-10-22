# Parameter Uncertainty

A calibration result says not only what the parameters are but how well the
data determines them. Every result carries an `uncertainty`: the covariance
of the estimated parameters at the solution, the noise level the residuals
imply, and, when no covariance can be given, the reason. The laserline
device result keeps it with its estimate, as `estimate.uncertainty` (in
Python also `result.uncertainty`).

Reprojection error alone does not answer this. A camera calibrated from a
handful of similar views can reproject its own corners to 0.1 px and still have
a focal length that is off by a percent, with the principal point
traded off against it. The covariance shows that directly: a large `σ` of `fx`,
and a strong `fx`–`cx` correlation. On the real images of `data/kuka_1`, the `planar_real` example reports `fx = 2057.8 ± 0.3 px`.

## What a σ means

The covariance is first order at the solution: it describes how the estimate
would scatter if the calibration were repeated with new noise of the same
kind, for the same views. The noise level is taken from the residuals
themselves, so nothing has to be known in advance. Two consequences follow:

- `σ` measures the **noise-driven** error. A systematic error, such as a wrong
  board cell size, a board that is not flat, or the wrong distortion model,
  leaves `σ` small and the estimate biased. Watch the per-feature residuals for
  structure.
- `σ` shrinks with more data roughly as `1/√views` and grows quickly when the
  views are similar. The most effective way to reduce `σ` of `fx` and `cx` is
  board tilt: views at varied angles separate focal length from distance.

With a robust loss (`Huber`, `Cauchy`, `Arctan`), the covariance is that of
the robust estimate actually computed, including the bounded pull of the
residuals the loss down-weights. It does not ignore the outliers, and it does
not treat them as Gaussian noise either. Without a loss it is the classical
`σ̂² (JᵀJ)⁻¹`.

These σ were checked against Monte-Carlo scatter on synthetic scenes (single
cameras, a stereo rig, an eye-in-hand camera, a laserline device with corner
and laser residuals together), a few hundred noise realizations each. The
predicted σ matched the empirical scatter within 10 % on clean data, and
within 20 % with 5 % of the corners displaced by 10–30 px under Huber or
Cauchy. The projection σ ([Projection Uncertainty](uncertainty_projection.md)) was checked the same way.

## In This Part

- [Reading the Covariance](uncertainty_reading.md): the labels, the views, the fixed parameters and the noise estimate, from Rust, Python and JSON.
- [Projection Uncertainty](uncertainty_projection.md): what the parameter covariance implies for where a ray lands in the image.
- [Limits and Unavailable Results](uncertainty_limits.md): when no covariance can be given, what it is conditional on, and what it does not cover.
- [Judging a Result](quality.md): a pass, warn or fail verdict on a result, built from its covariance, coverage and residuals.
