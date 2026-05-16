# Development Harness

This harness keeps Claude/GLM development aligned with the V1 design document.

It is not the game runtime. It is the control layer for:

- reading the right context;
- selecting one active task;
- preventing rule drift;
- tracking progress;
- reviewing output against the design;
- protecting private information boundaries.

## How To Use

1. Start with `CLAUDE.md`.
2. Read `PROGRESS.md`.
3. Read the relevant phase plan in `harness/plans/`.
4. Implement only the current active task.
5. Run the relevant checklist in `harness/checklists/`.
6. Update `PROGRESS.md`.

## File Map

- `context/`: stable project rules and architecture boundaries.
- `plans/`: phase-specific implementation plans.
- `checklists/`: acceptance checks before claiming completion.
- `templates/`: task and review report templates.
