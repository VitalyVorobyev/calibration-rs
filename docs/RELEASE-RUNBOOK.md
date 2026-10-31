# Release Runbook

The procedure for cutting a `calibration-rs` release. Pre-1.0, breaking
changes are batched into a minor (`0.x.0`) bump rather than spread across
patch releases.

## 1. Procedure

### 1.1 Bump the version sources (one commit)

All four move together — a missed one ships a mismatched PyPI wheel or breaks
the private examples:

1. `Cargo.toml`: `[workspace.package] version`.
2. `Cargo.toml`: every `version = "…"` pin in `[workspace.dependencies]` that
   has a `path = "./crates/…"` (grep for it).
3. `crates/vision-calibration-py/pyproject.toml`: `[project] version`. It is
   not wired to the workspace version; `release-pypi.yml` fails its
   `Verify tag/version sync` job (tag, workspace, py crate and pyproject must
   all match) and no wheel is built if it drifts.
4. `crates/vision-calibration-examples-private/Cargo.toml`: the package
   `version` and every path-dep `version = "…"` pin in it. The crate is
   outside the workspace and CI, so a stale pin breaks only the private
   examples.

Then refresh the lockfiles:

```bash
cargo build --workspace
cargo build --manifest-path app/src-tauri/Cargo.toml   # committed app Cargo.lock
```

The app has its own product version (`app/src-tauri/Cargo.toml`); leave it
alone in a library release. Its `vision-calibration` path dep carries no
`version` constraint; if one is ever added, bump it in step 2 too.

### 1.2 Update the CHANGELOG

Rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD` and open a fresh empty
`[Unreleased]` above it. Releases older than the previous minor family live
verbatim in `docs/changelog/<version>.md`, one file per release, indexed under
"Older releases" at the bottom of `CHANGELOG.md`; archive a family there when
it leaves the active window.

### 1.3 Run the checks locally

All must pass before tagging. First the full gate list of AGENTS.md §3,
every line: its doc build is the one `publish-docs.yml` runs after merge,
so a broken intra-doc link shows up here rather than on the live site.
Then:

```bash
# Version sources agree.
grep -RHn 'version = "' Cargo.toml \
    crates/vision-calibration-py/pyproject.toml \
    crates/vision-calibration-examples-private/Cargo.toml \
  | grep -v 'edition\|rust-version'

# Python extension build and tests (release-pypi.yml repeats them).
maturin develop -m crates/vision-calibration-py/Cargo.toml
python -m unittest discover -s crates/vision-calibration-py/tests -p "test_*.py"
python3 scripts/check_pyi_coverage.py --check
python3 scripts/check_binding_parity.py --check

# Acceptance on the private registry (CI checks the public datasets through
# the regression suite). The laser datasets hard-fail without `laser`.
cargo run --release -p vision-calibration-bench --features "tier-b laser" \
  --bin calib-bench -- accept

# Publish dry-run in the §2 order. Read §3 for how to interpret it.
for c in vision-calibration-core vision-geometry vision-calibration-dataset \
         vision-calibration-detect vision-calibration-linear \
         vision-calibration-optim vision-mvg vision-calibration-pipeline \
         vision-calibration; do
  echo "=== $c ==="; cargo publish -p "$c" --dry-run --locked
done
```

Also confirm on crates.io, once per crate in §2, that the crate exists and has
a Trusted Publisher entry (§4). There is no CLI for this.

### 1.4 Tag

The tag is `vX.Y.Z` and must equal the workspace and pyproject versions.

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Do not sign or amend after pushing. If a workflow fails mid-release, fix
forward and re-run it: the crates.io job skips versions already published.

Workflows fired by the tag push:

| Workflow | Produces |
|---|---|
| `release.yml` | fmt / clippy / test gates, then publishes the §2 crates to crates.io in order, then a GitHub release with generated notes |
| `release-pypi.yml` | tag/version check, Python runtime tests, wheels (Linux, macOS, Windows; abi3-py310) and sdist, then PyPI upload |
| `app-bundle.yml` | unsigned macOS (`.app`/`.dmg`) and Linux (AppImage/`.deb`) desktop bundles as workflow-run artifacts only (not attached to the release) |
| `publish-docs.yml` | rebuilds and deploys the docs site |

None of them push back to `main`.

## 2. Publish order

`release.yml` publishes in this order (leaf crates first, facade last):

1. `vision-calibration-core`
2. `vision-geometry` (core)
3. `vision-calibration-dataset`, `vision-calibration-detect` (no workspace deps)
4. `vision-calibration-linear` (core, geometry)
5. `vision-calibration-optim` (core)
6. `vision-mvg` (core, geometry)
7. `vision-calibration-pipeline` (core, linear, optim, geometry, dataset, detect)
8. `vision-calibration` (facade)

`vision-calibration-py` ships to PyPI through `release-pypi.yml`; the bench and
examples-private crates are not published. Dev-dependencies do not constrain
the order (`cargo publish` strips them). The job waits 20 s between publishes
for index propagation.

## 3. Reading the dry-run

`cargo publish --dry-run` strips path dependencies and resolves workspace deps
against the crates.io index. On a lockstep bump the new version is not on the
index, so every crate with a workspace dependency fails with
`candidate versions found which didn't match: X.Y.Z, …`. That is expected;
only crates without workspace dependencies (`core`, `dataset`, `detect`) pass.

Any other failure, in particular a compile error such as `E0432`, is a hard
stop: it means a crate uses an item that the registry copy of its dependency
lacks. Do not loosen the `[workspace.dependencies]` pins to the previous
version to silence it.

Adding a public item to a crate at an already-published version causes exactly
this failure for its dependents. Once a crate has a version on crates.io, any
public-surface change must ride a workspace-wide version bump.

## 4. Trusted Publishing (per crate)

crates.io Trusted Publishing (OIDC, no long-lived token) is configured per
crate. Each crate in §2 needs an entry with:

- Repository: `VitalyVorobyev/calibration-rs`
- Workflow: `release.yml`
- Environment: `crates-io`

When a crate joins the publish set:

1. Publish it manually once (`cargo publish -p <crate>` with a real token, or
   ask a crates.io owner). The Trusted Publisher UI exists only for crates
   that already exist.
2. Add the Trusted Publisher entry on its crates.io settings page.
3. The next tagged release then publishes it automatically.

Until then `release.yml` warns and skips a crate whose index lookup returns
404, so the crate sits unpublished across the release without failing it. Add
the new crate to the `crates` array in `release.yml` and to §2 and the dry-run
loop above.
