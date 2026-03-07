# ADR 0004: Planning Process (ADR + Backlog)

- Status: Accepted
- Date: 2026-03-07

## Context

Planning has been tracked in ad-hoc top-level files (for example `IMPLEMENTATION_PLAN.md`),
which makes decisions, execution items, and release milestones harder to maintain consistently.

## Decision

Use a two-layer planning model:

1. ADRs (`docs/adrs/`) define architecture decisions and constraints.
2. `docs/backlog.md` tracks execution tasks, milestones, and acceptance criteria.

`IMPLEMENTATION_PLAN.md` is removed and must not be reintroduced.

## Consequences

Positive:

- Clear separation between long-lived decisions and short-lived task tracking.
- Better traceability from task to architecture rationale.
- Easier release planning and milestone management.

Negative:

- Requires discipline to keep ADR/backlog updated.

## Maintenance Rules

- New architectural changes must add/update an ADR first.
- Backlog items must reference relevant ADR IDs.
- Release readiness is evaluated against backlog milestone acceptance criteria.
