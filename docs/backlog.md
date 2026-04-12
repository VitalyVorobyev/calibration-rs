# Backlog and Milestones

Planning model:

- Architecture decisions live in `docs/adrs/`.
- Execution tracking lives in this backlog.
- Automated workflow: `/orchestrate`, `/architect`, `/implement`, `/review`, `/gate-check`.

Execution workflow:

- Each implemented task must produce:
  1. A concise report in `docs/report/`
  2. A status update in this backlog
  3. A dedicated git commit
- Task IDs use `M<milestone>-T<nn>` (example: `M1-T03`).

---

## Active Milestones

---

## Standard Gate Checklist (Per Milestone Completion)

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --workspace --all-features`
- [ ] `cargo doc --workspace --no-deps` (no warnings)
- [ ] `python3 -m compileall crates/vision-calibration-py/python/vision_calibration`
