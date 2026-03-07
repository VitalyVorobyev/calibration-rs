# M7-T06: `#[doc(hidden)]` Leak Audit

Date: 2026-03-07
Commit: pending

## Scope

- Audited re-export surfaces for internal implementation-detail leakage.
- Verified key internal type-erasure details are already hidden in docs where applicable:
  - `AnyProjection`
  - `AnyDistortion`
  - `AnySensor`
  - `AnyIntrinsics`
- Reviewed facade re-export boundaries and confirmed no additional internal symbols required new `#[doc(hidden)]` annotations for this milestone.

## Files changed

- `docs/backlog.md`
- `docs/report/2026-03-07-M7-T06-doc-hidden-audit.md`

## Validation run

- Structural audit via symbol/re-export inspection (`rg` over public items and existing `#[doc(hidden)]` use)
- `cargo rustc -p vision-calibration --lib -- -W missing-docs` -> pass
- `cargo doc -p vision-calibration --no-deps` -> pass

## Follow-ups / risks

- If facade/optim re-export policy changes in future (for example broader backend internals), rerun this audit and add `#[doc(hidden)]` as needed.
