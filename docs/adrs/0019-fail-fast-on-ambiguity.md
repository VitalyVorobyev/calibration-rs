# ADR 0019: Fail Fast on Ambiguity (`AskUser` runtime contract)

- Status: Accepted
- Date: 2026-05-02

## Context

The "complete calibration workflow in the app" goal pulls in two
sources of runtime ambiguity:

1. **AI-generated dataset manifests** that may leave fields under-
   specified — e.g. an AI inspecting a foreign folder can list image
   paths from filename patterns but cannot infer `pose_convention.transform`
   from quaternion data alone.
2. **User-authored manifests** with structurally-valid but
   semantically-ambiguous content — e.g. two cameras whose filenames
   produce the same SharedFilenameToken but differ in count.

The user explicitly requested ("In any not clear situation stop and ask
user") that the runtime never silently guess. Wrong frame conventions
silently produce plausible-but-wrong calibrations 6 weeks downstream;
ambiguous pose pairings silently drop frames; "we picked a default for
you" is the worst possible UX in this domain.

## Decision

### 1. Two-tier field model in `DatasetSpec`

Every manifest field is implicitly tagged either:

- `infer_from_data` — the AI generator is expected to populate this
  from filenames / file extensions / sample data. Examples: image
  globs, camera count, target type from a README.
- `human_or_doc_required` — the AI is _forbidden_ from guessing.
  It must read documentation, find the value in metadata, or leave it
  `null` and add the field path to `_unresolved`. Examples: every
  `pose_convention.*` field, target physical dimensions when not in a
  README.

Tier metadata is encoded as a `x-calib-tier` schema extension so both
Rust and TypeScript form generators can render the right UX (e.g. a
red badge on `human_or_doc_required` fields that are still null).

### 2. `_unresolved` field at manifest root

`DatasetSpec._unresolved: Vec<String>` lists the dotted paths of
fields the AI couldn't determine. Validation refuses to dispatch a
manifest with non-empty `_unresolved`; the Run workspace renders the
list as red badges and blocks the Run button until each entry is
resolved.

### 3. Structured `AskUser` error variant

Runtime ambiguity that survives validation surfaces as a typed error:

```rust
pub enum RunError {
    AskUser {
        field: String,
        prompt: String,
        suggestions: Vec<String>,
    },
    …
}
```

The Run workspace catches `AskUser` and shows a blocking modal with
the prompt + suggestions; the user response feeds back into the
runner via an event channel and the conversion resumes. The CLI
(generate-manifest binary) catches `AskUser` too, prints the prompt
+ suggestions, and exits with status 2 — surface parity makes the
generator scriptable.

This is a single error variant, not a separate ambiguity-handling
machinery, deliberately: "ambiguity" is just a structured error like
any other, and it composes naturally through `?`.

### 4. No silent defaults anywhere

This applies to:

- Missing manifest fields → `Validation(Unresolved(...))`.
- Conflicting detector params → detector implementations must reject,
  not "auto-correct".
- Ambiguous image-to-pose pairings → pairing impl must report the
  conflict via `AskUser`.
- Empty glob matches → `EmptyImageMatch { camera, pattern }` with
  the offending base path.
- Insufficient features per view (< 4 corners) → `InsufficientUsableViews`
  with usable / total counts; no "let's see how the solver feels
  about it".

## Consequences

- The runtime is verbose by design when given ambiguous input. We pay
  this cost in dev-time UX in exchange for never producing a silent-
  failure calibration.
- Adding a new ambiguity source is a new `RunError` variant, not a new
  branch in some catch-all. Compile-time enforcement of "did you
  handle this?" through pattern matching.
- The `AskUser` modal is the second-most-important UI affordance in
  the Run workspace (after the Run button itself). PR 3 invests in
  its UX accordingly.
- The CLI parity means the AI-manifest generator can run unattended
  against a folder and produce either a complete manifest or a
  human-readable list of "you need to tell me about: …".

## Status of work

- ✅ `_unresolved` field in `DatasetSpec` with validator enforcement.
- ✅ Closed-enum frame conventions with no defaults (ADR 0016).
- ✅ `AskUser` variant in `pipeline::dataset_runner::RunError`.
- ✅ `EmptyImageMatch`, `InsufficientUsableViews`,
  `UnsupportedTopology`, `UnsupportedTarget`, and `MissingPoseConvention`
  variants for the structurally-rejectable ambiguities.
- ⏳ React `AskUserModal` component (PR 1 task #10).
- ⏳ Tauri event channel for resuming a paused conversion after the
  user answers (PR 3, alongside the AI-manifest UI).
