# Command: Review Against Design

Use this before claiming a task is complete.

Check:

1. Does the implementation match `docs/design/werewolf-agent-v1-design.md`?
2. Does it preserve RuleEngine authority?
3. Does it preserve visibility boundaries?
4. Are critical V1 rules covered by tests?
5. Is `PROGRESS.md` updated?
6. Are skipped checks explicitly documented?

Write findings using `harness/templates/review-report.md`.
