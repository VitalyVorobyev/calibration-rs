# M9 Rebaseline: Python Dictless Typed API Migration

Date: 2026-03-07
Commit: pending

## Assessment summary

`docs/python-bindings-dictless-todo.md` and current package state are misaligned with previous `M9` backlog scope.

Current Python package still exposes dict-centric behavior in the public high-level surface:

- `_api.py` high-level runners accept `Mapping[...]` inputs/configs and explicitly document raw mapping support.
- `models.py` high-level result dataclasses still expose dict payload fields (`camera`, `cameras`, `estimate`, `stats`) and `raw` snapshots.
- `__init__.pyi` propagates mapping unions in public signatures.
- `types.py` remains a large public TypedDict surface and is exported from `__init__.py`.
- README/examples/tests still include dict-style result access and payload patterns.

This means the previous `M9` wording (parity + one step-by-step API) is not the right next release target.

## Backlog change

Re-scoped `M9` from "Python API Parity" to "Python Dictless Typed API Migration" with tasks aligned to the TODO plan:

- typed camera/result payload models
- typed-only high-level signatures (raw compatibility moved to low-level helpers)
- `types.py` compatibility-only positioning
- migration/deprecation policy
- typed tests/examples across all runners

## Files changed

- `docs/backlog.md`
- `docs/report/2026-03-07-M9-rebaseline-dictless-migration.md`
