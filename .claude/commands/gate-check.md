# Gate Check: Run All Quality Gates

Run the full quality gate checklist and report results.

## Gates

Run every command in the canonical gate list (AGENTS.md §3; add §12 when
`app/` changed) and report pass/fail for each.

## Output Format

```
Gate Results:
  fmt:     PASS/FAIL
  clippy:  PASS/FAIL
  tests:   PASS/FAIL (N passed, M failed)
  doc:     PASS/FAIL (N warnings)
  schemas: PASS/FAIL
  python:  PASS/FAIL
```

If any gate fails, show the first 10 lines of error output.
