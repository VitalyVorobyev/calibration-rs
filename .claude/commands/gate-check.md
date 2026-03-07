# Gate Check: Run All Quality Gates

Run the full quality gate checklist and report results.

## Gates

Run each command and report pass/fail:

1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
3. `cargo test --workspace --all-features`
4. `cargo doc --workspace --no-deps 2>&1 | grep -E "warning|error" | head -30`
5. `python3 -m compileall crates/vision-calibration-py/python/vision_calibration 2>&1`

## Output Format

```
Gate Results:
  fmt:     PASS/FAIL
  clippy:  PASS/FAIL
  tests:   PASS/FAIL (N passed, M failed)
  doc:     PASS/FAIL (N warnings)
  python:  PASS/FAIL
```

If any gate fails, show the first 10 lines of error output.
