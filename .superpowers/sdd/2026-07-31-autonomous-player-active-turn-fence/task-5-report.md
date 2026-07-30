# Task 5 PostgreSQL Active-Turn Fence Report

## Delivered scope

- Added nullable `active_turn_fence_json JSONB` to fresh PostgreSQL dispatch
  schema plus the idempotent legacy `ALTER TABLE` upgrade.
- Round-trip serialized dispatch fences through one shared SELECT-column list
  and shared insert helper, retaining NULL compatibility for historical rows.
- Unified PostgreSQL mutation boundaries: advisory game lock first, then the
  `games` row, followed by schedule, managed-turn, and dispatch rows.
- Implemented atomic fenced reservation and fenced terminalization with
  expected CAS/recovery errors re-raised directly and unexpected failures
  rolled back and sanitized as `ActiveTurnFenceTransactionError`.
- Restricted 23505 mapping to the exact dispatch primary-key and
  executor/provider-key constraints. Unknown constraints are sanitized after
  rollback.
- Added stateful fake-connection tests covering lock order, reservation/finish
  commit and rollback, unresolved completion blocking, context validation
  before idempotency checks, and exact unique-constraint behavior.

## TDD evidence

| Group | RED evidence | GREEN evidence |
| --- | --- | --- |
| Schema and serialization | 2 failures: missing JSONB fence column and dropped row fence | 2 passed |
| Advisory lock order | Missing `pg_advisory_xact_lock` assertion failure | 1 passed |
| Reservation transaction | `create_active_turn_dispatch` missing | 1 passed |
| Fenced terminal transaction | 2 failures: `finish_active_turn_fenced` missing | 2 passed |
| Exact 23505 mapping | Unknown unique constraint incorrectly mapped to idempotency conflict | 1 passed |
| Plain create fence rejection | Caller-supplied fence reached a PostgreSQL query | 1 passed |
| Context-before-idempotency order | Context drift returned dispatch-id conflict | 1 passed |

## Verification

```text
conda run -n wofkill python -m pytest \
  tests/storage/test_postgres_autonomous_commit.py \
  tests/storage/test_active_turn_fence.py -k postgres -v
56 passed

conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/storage/postgres_store.py \
  tests/storage/test_postgres_autonomous_commit.py \
  tests/storage/test_active_turn_fence.py
All checks passed!

conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/storage/postgres_store.py
Success: no issues found in 1 source file
```

`git diff --check` also exits successfully. These are PostgreSQL schema and
fake-connection contract tests only; no real PostgreSQL service integration
was run or claimed.

## Self-review

- All new fence paths use the game advisory lock before the games row and
  before schedule/turn/dispatch row locking. Dispatch-ID transition discovery
  stays unlocked only long enough to discover `game_id`, then reloads `FOR
  UPDATE` under that boundary.
- Fenced reservation validates schedule/turn context before idempotency and
  recovery checks, inserts the generated fence, and CAS-updates the managed
  turn in one transaction.
- Terminalization locks exact-turn attempts in deterministic order and rolls
  back cancellation, turn, and schedule changes as one unit.
- The diff is limited to PostgreSQL storage, focused fake-connection tests,
  the active-fence plain-create regression, and this Task 5 report. It does
  not modify Memory, SQLite, HostRuntime, legacy PlayerAgent, live paths, or
  ToolResult Markdown.

## Fix round 1: transition lock order

- Review identified that `transition_active_turn()` locked the managed-turn
  row before its schedule after the advisory/game boundary, which violated the
  shared `game -> schedule -> managed turn -> dispatch attempts` order.
- The method still performs its original unlocked identity discovery only to
  obtain `schedule_id` and `game_id`. After advisory lock plus `games FOR
  UPDATE`, it now reloads and locks the authoritative schedule first, then
  reloads and locks the managed turn before the unchanged version CAS and
  transition preparation.
- Added `test_postgres_transition_locks_schedule_before_managed_turn`, which
  ignores the harmless discovery SELECT. RED showed the prior locked-row order
  as managed turn index 3 before schedule index 4; GREEN passed after the
  reorder.
