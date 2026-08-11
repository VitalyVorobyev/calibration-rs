# Release Runbook

A checklist to follow under pressure when cutting a `calibration-rs`
release. For the *why* behind each rule see `.claude/CLAUDE.md`'s
"Releasing — version-source lockstep" section — that is the source of
truth this runbook distills; if the two disagree, fix this file.

We are pre-1.0: breaking changes are expected and are batched into a
single minor (`0.x.0`) bump rather than dribbled across patch releases.

## 1. Version sources — must move together

Four files, updated in the **same commit**:

1. `Cargo.toml` → `[workspace.package] version` (~line 21).
2. `Cargo.toml` → `[workspace.dependencies]` path-dep `version = "…"`
   pins for every publishable workspace crate (~lines 33–45). As of
   0.7.0 that is **nine** crates: `vision-calibration-core`,
   `vision-geometry`, `vision-calibration-dataset`,
   `vision-calibration-detect`, `vision-calibration-linear`,
   `vision-calibration-optim`, `vision-mvg`,
   `vision-calibration-pipeline`, `vision-calibration`.
3. `crates/vision-calibration-py/pyproject.toml` → `[project] version`.
   **Not** wired into `[workspace.package]` — `release-pypi.yml`'s
   `Verify tag/version sync` job reads this file directly and will
   skip the wheel/sdist build + PyPI upload if it drifts.
4. `crates/vision-calibration-examples-private/Cargo.toml` → the
   package `version` **plus every path-dep `version = "…"` pin inside
   it**. This crate is outside the crates.io publish set but is still
   compiled in CI, so a stale pin here breaks the examples build, not
   the release. **Count the pins before assuming there are four** — the
   pin count tracks how many published crates the private examples
   depend on and has grown over time (6 as of 0.7.0: `vision-calibration`,
   `vision-calibration-core`, `vision-calibration-optim`,
   `vision-calibration-pipeline`, `vision-calibration-detect`,
   `vision-mvg`). Grep the file, don't trust a remembered number.

After editing all four, refresh the lockfile once:

```bash
cargo build --workspace   # only workspace-crate version strings change in Cargo.lock
```

### The app crate is a fifth place to check, not a fifth lockstep source

`app/src-tauri/Cargo.toml` has its own product `version` (currently
`0.1.0`, independent of the workspace — do not bump it as part of a
library release) but pulls in `vision-calibration` via a **path** dep.
Check whether that path dep (or any other workspace-crate path dep in
this file) carries a `version = "…"` constraint:

- If it does not (current state), nothing to edit in the `.toml`, but
  its **committed** `Cargo.lock` still resolves the workspace crate's
  version string and must be refreshed:
  ```bash
  cargo build --manifest-path app/src-tauri/Cargo.toml
  ```
- If a future change adds a version constraint there, bump it in step
  1 above and rebuild the same way.

## 2. Crates.io publish DAG (`release.yml`)

Publish order follows the dependency graph, leaf crates first, facade
last — `vision-calibration-py` is **not** in this job, it ships to PyPI
via `release-pypi.yml`:

```
vision-calibration-core
  → vision-geometry
    → vision-calibration-dataset
    → vision-calibration-detect
      → vision-calibration-linear
      → vision-calibration-optim
        → vision-mvg
          → vision-calibration-pipeline
            → vision-calibration
```

`dataset` and `detect` have no internal deps but are non-optional deps
of `pipeline`, so they must land before it. `vision-geometry` and
`vision-mvg` are a standalone MVG island (no calibration-chain crate
depends on them) but ship in DAG order anyway: geometry after core, mvg
after geometry.

The job is idempotent (skips a crate/version already on the index) and
`sleep 20`s between publishes for index propagation, so a mid-release
failure can be fixed and the tag-triggered workflow re-run safely.

## 3. Per-crate Trusted Publishing gotcha (2026-07-08 incident)

crates.io Trusted Publishing (OIDC, no long-lived token) is configured
**per crate**, not per repo. Every crate in the DAG above needs its own
Trusted Publisher entry on crates.io:

- Repository: `VitalyVorobyev/calibration-rs`
- Workflow: `release.yml`
- Environment: `crates-io`

**A crate that has never been published cannot have Trusted Publishing
configured before its first upload** — crates.io only offers the
"add a trusted publisher" UI on a crate that already exists. `release.yml`
handles this gracefully for a first-ever publish (404 on the sparse
index → warn and skip rather than hard-fail), but that means a newly
promoted crate can silently sit **unpublished** across a whole release
if nobody follows up with `cargo publish` by hand + the crates.io UI
config. This is exactly what happened to `vision-mvg` in the 0.6.0
release (`vision-geometry`/`vision-mvg` joined the publish set that
release and `vision-mvg` never got a Trusted Publisher entry, so it
quietly never reached crates.io until the gap was noticed and fixed
out of band).

**When a crate joins the publish set for the first time:**

1. Publish it manually once: `cargo publish -p <crate>` with a real
   token (or ask someone with crates.io ownership to do it).
2. Immediately add the Trusted Publisher entry (repo / workflow /
   environment above) on the new crate's crates.io settings page.
3. Only then does the next tagged release pick it up automatically.

**Pre-tag check:** for every crate in the DAG, confirm on crates.io
that (a) the crate exists and (b) it has a Trusted Publisher entry for
`release.yml` / `crates-io`. Don't assume "it published before, so it's
fine" — go look, especially for crates newer than the last release you
personally drove.

### Earlier incident: PyPI pyproject drift (0.4.0 / 0.5.0)

`0.4.0` and `0.5.0` shipped to crates.io but never reached PyPI:
`crates/vision-calibration-py/pyproject.toml`'s `project.version` was
not bumped alongside the workspace, which tripped `release-pypi.yml`'s
`Verify tag/version sync` job and skipped the wheel/sdist build and
upload for both tags. `0.5.1` was cut specifically to repair this.
Lesson: the pyproject version is not part of `[workspace.package]` and
is easy to forget — it's source #3 in §1 above precisely because of
this incident.

## 4. "Adding public API to a published crate is a release event"

`0.6.0` exists because PR #67 added the public
`vision_calibration_core::linalg` module to the **already-published**
`core@0.5.1` without a version bump. Local `core@0.5.1` then diverged
from the immutable registry `core@0.5.1`. Publishing `vision-geometry`
(which re-exports `core::linalg`) failed `--dry-run` with `E0432`,
because `cargo publish` strips path deps and resolves the registry's
*old* `core@0.5.1`, which has no `linalg`.

**Rule:** once a crate has a version on crates.io, any change to its
public surface — new module, new item, changed signature — must ride a
workspace-wide version bump. Never land public-API changes at the same
version a crate was already published at, even mid-development on
`main`, if there's any chance a dependent crate publishes before the
next bump.

## 5. Pre-tag checklist

Run all of these locally before pushing the tag. All must pass.

```bash
# 1. All four version sources print the same vX.Y.Z (see §1 for what to grep —
#    remember examples-private has more than one path-dep pin per crate).
grep -RHn 'version = "' Cargo.toml \
    crates/vision-calibration-py/pyproject.toml \
    crates/vision-calibration-examples-private/Cargo.toml \
  | grep -v 'edition\|rust-version'

# 2. Full workspace gates.
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features

# 3. Doc build with the same strictness publish-docs.yml uses (that workflow
#    only runs on push-to-main, not on PRs — catch broken intra-doc-links here).
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

# 4. PyO3 build + runtime tests (release-pypi.yml's verify job repeats this).
maturin develop -m crates/vision-calibration-py/Cargo.toml
python -m unittest discover -s crates/vision-calibration-py/tests -p "test_*.py"

# 5. Binding-surface sanity (cheap, catches drift before CI does).
python3 scripts/check_pyi_coverage.py --check
python3 scripts/check_binding_parity.py --check

# 6. Full local acceptance — public AND private registries. The laser
#    datasets (rtv3d family) hard-fail without the `laser` feature; plain
#    `tier-b` is only enough for the public kuka subset that CI runs.
cargo run --release -p vision-calibration-bench --features "tier-b laser" \
  --bin calib-bench -- accept \
  --registry crates/vision-calibration-bench/registry/public.json
cargo run --release -p vision-calibration-bench --features "tier-b laser" \
  --bin calib-bench -- accept \
  --registry crates/vision-calibration-bench/registry/private.json

# 7. Publish dry-run in DAG order (§2) — catches E0432-style path-dep
#    resolution failures (§4) before they wedge the real publish job.
#    Read §5.1 first: on a lockstep bump most of these are EXPECTED to fail.
for c in vision-calibration-core vision-geometry vision-calibration-dataset \
         vision-calibration-detect vision-calibration-linear \
         vision-calibration-optim vision-mvg vision-calibration-pipeline \
         vision-calibration; do
  echo "=== $c ==="; cargo publish -p "$c" --dry-run --locked
done

# 8. Trusted Publishing coverage (§3) — manual crates.io UI check, one page
#    per crate in the DAG. No CLI shortcut; just go look.
```

### 5.1 How to read the step-7 dry-run

`cargo publish --dry-run` strips path dependencies and resolves every
workspace dep against the crates.io index. On a **lockstep bump** the new
version is not on the index yet, so every crate that depends on another
workspace crate necessarily fails with:

```
candidate versions found which didn't match: 0.7.0, 0.6.0, …
location searched: crates.io index
```

That is the expected result, not a defect — those crates only become
verifiable once their dependencies are actually published, which is exactly
what `release.yml` does by publishing in DAG order. For `0.8.0` the
leaf-only crates (`vision-calibration-core`, `-dataset`, `-detect`) passed
and the other six reported the message above.

What step 7 *does* catch is the §4 failure: a crate whose dependency is
already on the index at the version being requested, but whose registry copy
lacks an item the local source uses. That shows up as a compile error
(`E0432` and friends), not as "candidate versions found". **Triage rule:**
"candidate versions" on a lockstep bump is fine; any *compile* error is a
hard stop.

Do not "fix" this by loosening the `[workspace.dependencies]` pins to accept
the previous version — that would let a crate publish against a stale
dependency and reintroduce the `0.6.0` incident.

## 6. Tag and what fires

Tag format `vX.Y.Z` (matches `[workspace.package] version` and the
pyproject version exactly — `release-pypi.yml`'s verify job hard-fails
otherwise). Do not sign or amend after pushing; if a workflow fails
mid-release, fix forward (the publish jobs are idempotent, see §2).

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Three workflows key off `push: tags: 'v*'`:

| Workflow | Triggers | Produces |
|---|---|---|
| `release.yml` | tag push | gates (fmt/clippy/test) → publishes the nine crates to crates.io in DAG order (§2) → GitHub release with auto-generated notes |
| `release-pypi.yml` | tag push | verifies tag/version sync → builds wheels (Linux/macOS/Windows, abi3-py310) + sdist → publishes to PyPI |
| `app-bundle.yml` | tag push (also `workflow_dispatch`) | unsigned macOS (`.app`/`.dmg`) and Linux (AppImage/`.deb`) desktop bundles, uploaded as **workflow-run artifacts only** — not attached to the GitHub release, not signed/notarized |

None of these push back to `main`; the version-bump PR that prepared
the release is the only commit that touches version strings.
