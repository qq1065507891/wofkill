# Task 4 report: SQLite active-turn fence

## Scope

Implemented the SQLite-only active-turn fence capability, nullable schema
upgrade, canonical fence JSON persistence, combined reservation and terminal
transactions, and focused SQLite tests. PostgreSQL, `MigrationManager`, host
runtime, live paths, and legacy `PlayerAgent` code were not changed.

## Changes

- `werewolf_agent/storage/sqlite_store.py`
  - Adds nullable `active_turn_fence_json` to new durable-dispatch DDL and an
    idempotent startup `PRAGMA table_info`/`ALTER TABLE` upgrade for historical
    databases.
  - Serializes the optional fence as canonical compact JSON and loads `NULL`
    legacy rows as unfenced `DispatchAttempt` records.
  - Rejects caller-provided fences on plain `create_dispatch()`.
  - Adds `supports_active_turn_fence()`,
    `create_active_turn_dispatch()`, and `finish_active_turn_fenced()`.
    Both operations own one `BEGIN IMMEDIATE` transaction, use schedule/turn
    CAS updates, and convert unexpected backend errors to the sanitized
    `ActiveTurnFenceTransactionError` after rollback.
- `tests/storage/test_sqlite_migrations.py`
  - Covers fresh nullable DDL and a complete pre-fence legacy dispatch row.
- `tests/storage/test_active_turn_fence.py`
  - Covers reservation round-trip, plain-create rejection, conflict/recovery
    rollback, trigger-injected rollback at both write boundaries, atomic
    cancellation, and real two-connection create-vs-complete/cancel/transition
    races.

## TDD evidence

### Schema and round-trip RED

Before the SQLite implementation:

```bash
conda run -n wofkill python -m pytest tests/storage/test_sqlite_migrations.py -k "active_turn_fence or nullable_fence" -v
```

Result: `2 failed`; the fresh fence column was absent and
`supports_active_turn_fence()` did not exist.

The first SQLite behavior tests also failed before the implementation:

```bash
conda run -n wofkill python -m pytest tests/storage/test_active_turn_fence.py -k sqlite -v
```

Result: `2 failed`; `create_active_turn_dispatch()` was absent and plain
`create_dispatch()` accepted a caller-supplied fence.

### Transaction RED

After writing the trigger/rollback cases and before adding fenced terminal
support:

```bash
conda run -n wofkill python -m pytest tests/storage/test_active_turn_fence.py -k "sqlite and (rolls_back or conflicts)" -v
```

Result: `4 passed, 1 failed`; the terminal rollback case failed because
`finish_active_turn_fenced()` was absent. The reservation rollback and conflict
checks already passed once reservation was implemented.

### Mutation RED

The reservation round-trip test was run against a temporary deliberate CAS
mutation (`expected_turn_version + 1`). It failed with the expected
`TurnStateConflict`, proving the test catches loss of the managed-turn CAS.
The correct expected version was immediately restored before final checks.

### GREEN

```bash
conda run -n wofkill python -m pytest tests/storage/test_active_turn_fence.py -k sqlite -v
conda run -n wofkill python -m pytest tests/storage/test_sqlite_migrations.py tests/storage/test_autonomous_turns.py tests/storage/test_autonomous_commit.py -k "sqlite or dispatch or turn" -v
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/storage/sqlite_store.py tests/storage/test_active_turn_fence.py tests/storage/test_sqlite_migrations.py
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/storage/sqlite_store.py
git diff --check
```

Results: `11 passed`; `134 passed`; Ruff reported `All checks passed!`; mypy
reported `Success: no issues found in 1 source file`; and `git diff --check`
was clean.

## Self-review

- Every `DispatchAttempt` materializing SQLite SELECT now includes the nullable
  fence column; a full pre-existing dispatch/turn regression run caught and
  verified the two list-query additions.
- Legacy `NULL` fence JSON remains valid and no `MigrationManager` migration
  was added.
- Reservation inserts the fenced attempt before the managed-turn CAS update;
  terminalization updates changed attempts, then the managed turn and schedule,
  all within one transaction. Trigger failures prove rollback of each order.
- The race tests use two distinct `SqliteGameRepository` objects sharing one
  database file, rather than one process-local lock.

## Fix round 1

### Review findings addressed

- Reservation now calls `prepare_active_turn_dispatch()` immediately after the
  schedule and managed-turn CAS checks. A malformed cross-game attempt is
  therefore rejected before dispatch uniqueness or recovery queries can expose
  another game's state.
- The legacy migration fixture now creates an explicit pre-fence dispatch table
  instead of reusing current durable-dispatch DDL. It verifies the fence column
  is absent before repository initialization and nullable afterwards.
- `_cancel()` now accepts the shared `ActiveTurnFenceRepository` protocol.

### RED/GREEN evidence

The new priority regression initially failed with `DispatchRecoveryBlocked:
game-2` when a game-2 `DISPATCHING` row existed:

```bash
conda run -n wofkill python -m pytest tests/storage/test_active_turn_fence.py -k "sqlite_fenced_create_rejects_context_before_cross_game_recovery_barrier" -v
```

After moving the pure validation, this command passed. The upgraded legacy
fixture also passed its focused migration assertion. Final verification:

```bash
conda run -n wofkill python -m pytest tests/storage/test_active_turn_fence.py -k sqlite -v
conda run -n wofkill python -m pytest tests/storage/test_sqlite_migrations.py tests/storage/test_autonomous_turns.py tests/storage/test_autonomous_commit.py -k "sqlite or dispatch or turn" -v
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/storage/sqlite_store.py tests/storage/test_active_turn_fence.py tests/storage/test_sqlite_migrations.py
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/storage/sqlite_store.py
git diff --check
```

Results: `12 passed`; `134 passed`; Ruff and mypy passed; and the diff check
was clean.
