# Autonomous Player Durable Active-Turn Fence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a durable, cross-backend active-turn fence so dispatch reservation and schedule/turn terminalization compete atomically on one persisted CAS identity.

**Architecture:** Add a strict fence payload to `DispatchAttempt`, a combined `ActiveTurnFenceRepository` capability, and pure preparation helpers that reserve a dispatch by incrementing `ManagedAgentTurn.state_version`. Memory, SQLite, and PostgreSQL implement the same create/finish transactions; HostRuntime becomes the only production-facing coordinator and requires one physical repository for turn, dispatch, and fence state.

**Tech Stack:** Python 3.12, Pydantic v2, protocols, in-memory `RLock`, SQLite WAL/`BEGIN IMMEDIATE`, PostgreSQL JSONB/advisory locks/row locks, pytest, Ruff, mypy; every Python command uses `conda run -n wofkill`.

## Global Constraints

- `docs/superpowers/specs/2026-07-28-autonomous-player-agent-runtime-design.md` remains the sole authoritative architecture.
- Use `docs/superpowers/specs/2026-07-31-autonomous-player-active-turn-fence-design.md` for this stage's exact contracts and race semantics.
- Do not connect `GameRunner`, a live game, old `PlayerAgent`, old prompts, `ModelRouter`, or `_dispatch_agent`.
- Do not implement ToolResult Markdown projection.
- Do not change `GameRevision`; dispatch persistence is not game truth.
- Keep historical unfenced durable-dispatch rows readable, but never accept them as production active-turn authorization.
- Use the lock order `game -> schedule -> managed turn -> dispatch attempts`.
- Keep Memory, SQLite, and PostgreSQL behavior equivalent.
- Any validation, CAS, uniqueness, or backend failure must leave no partial attempt, managed-turn version bump, cancellation, schedule update, or turn update.
- New or substantially edited Python files use the project header order and concise Chinese comments; set `修改日期: 2026-07-31` on non-trivially changed modules.
- Run tests, Ruff, and mypy only through `conda run -n wofkill`.

---

## File Map

- Modify: `werewolf_agent/player_agents/contracts/dispatch.py` — add `ActiveTurnDispatchFence` and the optional persisted fence on attempts.
- Modify: `werewolf_agent/player_agents/contracts/__init__.py` — export the new strict contract.
- Create: `werewolf_agent/storage/active_turn_fence.py` — combined capability, stable errors, guard, and pure reservation/terminal preparation.
- Modify: `werewolf_agent/storage/memory_store.py` — atomic in-memory reservation and terminal publication.
- Modify: `werewolf_agent/storage/sqlite_store.py` — nullable fence JSON upgrade and `BEGIN IMMEDIATE` transactions.
- Modify: `werewolf_agent/storage/postgres_store.py` — nullable fence JSONB upgrade, unified advisory lock, and combined transactions.
- Modify: `werewolf_agent/player_agents/runtime/host.py` — production-facing fenced creation and fenced terminal methods.
- Create: `tests/storage/test_active_turn_fence.py` — strict contract, pure helpers, shared Memory/SQLite behavior, rollback, and real SQLite races.
- Modify: `tests/player_agents/test_dispatch_contracts.py` — attempt round-trip with and without the fence.
- Modify: `tests/player_agents/test_host_runtime.py` — combined capability, fenced creation, completion, cancellation, expiry, and interleaving tests.
- Modify: `tests/storage/test_autonomous_commit.py` — plain durable create rejects caller-supplied fence and existing dispatch semantics remain unchanged.
- Modify: `tests/storage/test_sqlite_migrations.py` — fresh/legacy nullable-column initialization.
- Modify: `tests/storage/test_postgres_autonomous_commit.py` — PostgreSQL DDL, lock order, stateful transactions, rollback, and exact conflict mapping.
- Modify: `handoff.md` — record completed fence semantics, evidence, and the next unique milestone.

### Task 1: Define the Fence Contract, Capability, and Pure Transactions

**Files:**
- Modify: `werewolf_agent/player_agents/contracts/dispatch.py:1-136`
- Modify: `werewolf_agent/player_agents/contracts/__init__.py`
- Create: `werewolf_agent/storage/active_turn_fence.py`
- Create: `tests/storage/test_active_turn_fence.py`
- Modify: `tests/player_agents/test_dispatch_contracts.py`

**Interfaces:**
- Consumes: `DispatchAttempt`, `DispatchStatus`, `ManagedAgentTurn`, `SerialPublicSchedule`, `AgentTurnStatus`, `TerminalDisposition`, and `prepare_active_finish`.
- Produces: `ActiveTurnDispatchFence`, `ActiveTurnFenceRepository`, `ActiveTurnFenceUnsupported`, `ActiveTurnFenceRejected`, `ActiveTurnFenceTransactionError`, `prepare_active_turn_dispatch`, `prepare_fenced_active_finish`, and `require_active_turn_fence_repository`.

- [ ] **Step 1: Write failing strict-contract tests**

Add literal fixtures and tests that name the break: removing any persisted
identity field must fail validation, and a JSON round trip must preserve the
fence exactly.

```python
def _fence(**updates: object) -> ActiveTurnDispatchFence:
    payload: dict[str, object] = {
        "schedule_id": "schedule-1",
        "schedule_state_version": 1,
        "turn_state_version": 4,
        "window_id": "speech-d1",
        "window_version": 1,
        "base_game_revision": 4,
    }
    payload.update(updates)
    return ActiveTurnDispatchFence.model_validate(payload)


def test_active_turn_fence_is_strict_frozen_and_json_round_trips() -> None:
    fence = _fence()
    assert ActiveTurnDispatchFence.model_validate_json(
        fence.model_dump_json(),
    ) == fence
    with pytest.raises(ValidationError):
        ActiveTurnDispatchFence.model_validate({
            **fence.model_dump(),
            "unexpected": True,
        })
    with pytest.raises((ValidationError, TypeError)):
        fence.turn_state_version = 5  # type: ignore[misc]


def test_dispatch_attempt_round_trips_optional_active_turn_fence() -> None:
    fenced = _attempt(active_turn_fence=_fence())
    restored = DispatchAttempt.model_validate_json(fenced.model_dump_json())
    assert restored == fenced
    assert restored.active_turn_fence == _fence()
    assert _attempt().active_turn_fence is None
```

- [ ] **Step 2: Run contract tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_dispatch_contracts.py \
  tests/storage/test_active_turn_fence.py -k "fence or round_trips_optional" -v
```

Expected: collection fails because `ActiveTurnDispatchFence` and
`werewolf_agent.storage.active_turn_fence` do not exist.

- [ ] **Step 3: Implement the strict persisted fence**

Add this model before `DispatchAttempt`, add its optional field, export the
name, and update the module description/date without changing existing enum
values:

```python
class ActiveTurnDispatchFence(StrictFrozenModel):
    """把一次生产 dispatch 绑定到持久化活动回合身份。"""

    schedule_id: NonEmptyId
    schedule_state_version: int = Field(ge=0)
    turn_state_version: int = Field(ge=1)
    window_id: NonEmptyId
    window_version: int = Field(ge=1)
    base_game_revision: int = Field(ge=0)


class DispatchAttempt(StrictFrozenModel):
    # existing fields stay unchanged
    active_turn_fence: ActiveTurnDispatchFence | None = None
```

Use `turn_state_version >= 1` because a fenced attempt always records the
post-reservation managed-turn version.

- [ ] **Step 4: Write failing pure reservation tests**

Build an open schedule and active managed turn at versions 1 and 3. Assert the
successful literal outcome and table-drive every mismatch:

```python
def test_prepare_active_turn_dispatch_reserves_turn_and_builds_fence() -> None:
    schedule, managed = _active_turn(turn_version=3)
    updated, attempt = prepare_active_turn_dispatch(
        schedule,
        managed,
        _attempt_for(managed),
        NOW,
    )
    assert updated.state_version == 4
    assert updated.turn == managed.turn
    assert updated.updated_at == NOW
    assert attempt.active_turn_fence == ActiveTurnDispatchFence(
        schedule_id=schedule.schedule_id,
        schedule_state_version=schedule.state_version,
        turn_state_version=4,
        window_id=managed.turn.window.window_id,
        window_version=managed.turn.window.version,
        base_game_revision=managed.turn.revision.base_revision,
    )


@pytest.mark.parametrize(
    "attempt_update",
    [
        {"game_id": "other-game"},
        {"turn_id": "other-turn"},
        {"actor_id": "p02"},
        {"lease_hash": "b" * 64},
        {"view_fingerprint": "b" * 64},
        {"deadline": DEADLINE + timedelta(seconds=1)},
    ],
)
def test_prepare_active_turn_dispatch_rejects_context_drift(
    attempt_update: dict[str, object],
) -> None:
    schedule, managed = _active_turn()
    with pytest.raises(ActiveTurnFenceRejected):
        prepare_active_turn_dispatch(
            schedule,
            managed,
            _attempt_for(managed, **attempt_update),
            NOW,
        )
```

Add separate tests for closed/inactive schedules, terminal managed turns,
caller-supplied fences, non-PENDING/version-nonzero attempts, naive
`observed_at`, `observed_at >= attempt.deadline`, and
`observed_at >= window.deadline`.

- [ ] **Step 5: Write failing pure fenced-terminal tests**

Use literal attempts in every durable status:

```python
def test_prepare_fenced_cancel_cancels_only_cancellable_attempts() -> None:
    schedule, managed = _active_turn(status=AgentTurnStatus.THINKING)
    attempts = (
        _attempt_for(managed, dispatch_id="pending"),
        _attempt_for(
            managed,
            dispatch_id="dispatching",
            status=DispatchStatus.DISPATCHING,
            state_version=1,
        ),
        _attempt_for(
            managed,
            dispatch_id="dispatched",
            status=DispatchStatus.DISPATCHED,
            state_version=2,
        ),
    )
    updated_schedule, updated_turn, updated_attempts = prepare_fenced_active_finish(
        schedule,
        managed,
        attempts,
        AgentTurnStatus.CANCELLED,
        TerminalDisposition.REPLACE,
        reason_code="operator_cancelled",
        now=NOW,
    )
    assert updated_schedule.active_turn_id is None
    assert updated_turn.turn.status is AgentTurnStatus.CANCELLED
    assert tuple(item.status for item in updated_attempts) == (
        DispatchStatus.CANCELLED,
        DispatchStatus.CANCELLED,
        DispatchStatus.DISPATCHED,
    )


@pytest.mark.parametrize(
    "status",
    [DispatchStatus.PENDING, DispatchStatus.DISPATCHING, DispatchStatus.DISPATCHED],
)
def test_prepare_fenced_commit_rejects_unresolved_attempt(status: DispatchStatus) -> None:
    schedule, managed = _active_turn(status=AgentTurnStatus.VALIDATING)
    with pytest.raises(DispatchRecoveryBlocked):
        prepare_fenced_active_finish(
            schedule,
            managed,
            (_attempt_for(managed, status=status),),
            AgentTurnStatus.COMMITTED,
            TerminalDisposition.ADVANCE,
            reason_code=None,
            now=NOW,
        )
```

Also assert `RESULT_RECORDED`, `UNKNOWN_OUTCOME`, and `CANCELLED` do not block
completion, and attempts for another turn are rejected rather than mutated.

- [ ] **Step 6: Implement stable errors, pure helpers, and capability guard**

Create the module with the required Chinese header and these exact signatures:

```python
class ActiveTurnFenceError(RuntimeError):
    code: ClassVar[str] = "active_turn_fence_error"


class ActiveTurnFenceUnsupported(ActiveTurnFenceError):
    code = "active_turn_fence_unsupported"


class ActiveTurnFenceRejected(ActiveTurnFenceError):
    code = "active_turn_fence_rejected"


class ActiveTurnFenceTransactionError(ActiveTurnFenceError):
    code = "active_turn_fence_transaction_error"


class ActiveTurnFenceRepository(Protocol):
    def supports_active_turn_fence(self) -> bool: ...
    def create_active_turn_dispatch(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        attempt: DispatchAttempt,
        observed_at: datetime,
    ) -> DispatchAttempt: ...
    def finish_active_turn_fenced(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        terminal_status: AgentTurnStatus,
        disposition: TerminalDisposition,
        reason_code: str | None,
    ) -> SerialPublicSchedule: ...
```

`require_active_turn_fence_repository(turn_repository, dispatch_repository)`
must require object identity, call all three explicit capability methods, map
any missing/false/raising capability to `ActiveTurnFenceUnsupported`, and
return `cast(ActiveTurnFenceRepository, turn_repository)`.

Implement `prepare_active_turn_dispatch()` exactly from design section 6.
Implement `prepare_fenced_active_finish()` by validating all attempts belong to
the exact game/turn, blocking unresolved completion, preparing cancellable
attempt copies with version increments, then delegating schedule/turn changes
to `prepare_active_finish()`.

- [ ] **Step 7: Verify GREEN and commit the contract layer**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_dispatch_contracts.py \
  tests/storage/test_active_turn_fence.py -k "contract or prepare or capability" -v
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents/contracts/dispatch.py \
  werewolf_agent/player_agents/contracts/__init__.py \
  werewolf_agent/storage/active_turn_fence.py \
  tests/player_agents/test_dispatch_contracts.py \
  tests/storage/test_active_turn_fence.py
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/storage/active_turn_fence.py
git add \
  werewolf_agent/player_agents/contracts/dispatch.py \
  werewolf_agent/player_agents/contracts/__init__.py \
  werewolf_agent/storage/active_turn_fence.py \
  tests/player_agents/test_dispatch_contracts.py \
  tests/storage/test_active_turn_fence.py
git commit -m "feat: define durable active-turn fence"
```

Expected: selected tests pass; Ruff and mypy exit 0.

### Task 2: Implement the In-Memory Atomic Fence

**Files:**
- Modify: `werewolf_agent/storage/memory_store.py:75-825`
- Modify: `tests/storage/test_active_turn_fence.py`
- Modify: `tests/storage/test_autonomous_commit.py`

**Interfaces:**
- Consumes: Task 1's capability and pure helper signatures.
- Produces: `InMemoryGameRepository.supports_active_turn_fence`, `create_active_turn_dispatch`, and `finish_active_turn_fenced`.

- [ ] **Step 1: Write failing in-memory reservation tests**

```python
def test_memory_fenced_create_persists_attempt_and_turn_version() -> None:
    repository, schedule, managed = _memory_active_turn()
    stored = repository.create_active_turn_dispatch(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        _attempt_for(managed),
        NOW,
    )
    current = repository.load_managed_turn(managed.turn.turn_id)
    assert current is not None
    assert current.state_version == managed.state_version + 1
    assert stored.active_turn_fence is not None
    assert stored.active_turn_fence.turn_state_version == current.state_version
    assert repository.load_dispatch(stored.dispatch_id) == stored


def test_memory_plain_create_rejects_caller_supplied_fence() -> None:
    repository, _, managed = _memory_active_turn()
    with pytest.raises(DispatchInvalidTransition):
        repository.create_dispatch(
            _attempt_for(managed, active_turn_fence=_fence()),
        )
```

Add table-driven tests for schedule/turn CAS, recovery barrier, dispatch ID,
provider key, context mismatch, and expired deadline. After every exception,
assert the original managed turn and all dispatch indexes are unchanged.

- [ ] **Step 2: Run Memory tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/storage/test_active_turn_fence.py -k memory -v
```

Expected: `InMemoryGameRepository` lacks the fence capability methods.

- [ ] **Step 3: Implement prepare-then-publish reservation**

Under the existing `RLock`:

```python
def create_active_turn_dispatch(
    self,
    schedule_id: str,
    expected_schedule_version: int,
    turn_id: str,
    expected_turn_version: int,
    attempt: DispatchAttempt,
    observed_at: datetime,
) -> DispatchAttempt:
    with self._lock:
        schedule = self._serial_public_schedules.get(schedule_id)
        if schedule is None:
            raise ScheduleNotFound("schedule not found")
        if schedule.state_version != expected_schedule_version:
            raise ScheduleStateConflict("schedule state version conflict")
        managed = self._managed_agent_turns.get(turn_id)
        if managed is None:
            raise ManagedTurnNotFound("managed turn not found")
        if managed.state_version != expected_turn_version:
            raise TurnStateConflict("managed turn state version conflict")
        self._assert_dispatch_allowed_unlocked(attempt.game_id)
        key = (attempt.executor_id, attempt.provider_idempotency_key)
        if attempt.dispatch_id in self._dispatch_attempts or key in self._dispatch_key_index:
            raise DispatchIdempotencyConflict(attempt.dispatch_id)
        updated_managed, fenced_attempt = prepare_active_turn_dispatch(
            schedule,
            managed,
            attempt,
            observed_at,
        )
        self._managed_agent_turns[turn_id] = updated_managed.model_copy(deep=True)
        self._dispatch_attempts[fenced_attempt.dispatch_id] = fenced_attempt.model_copy(
            deep=True,
        )
        self._dispatch_key_index[key] = fenced_attempt.dispatch_id
        return fenced_attempt.model_copy(deep=True)
```

Do not mutate containers until every check and model construction succeeds.
Add the explicit capability method and reject `active_turn_fence is not None`
at the start of plain `create_dispatch()`.

- [ ] **Step 4: Write failing in-memory fenced-terminal and race tests**

```python
def test_memory_fenced_cancel_rolls_attempts_and_turn_forward_together() -> None:
    repository, schedule, managed = _memory_active_turn()
    pending = _reserve(repository, schedule, managed, dispatch_id="pending")
    managed = repository.load_managed_turn(managed.turn.turn_id)
    assert managed is not None
    updated = repository.finish_active_turn_fenced(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        AgentTurnStatus.CANCELLED,
        TerminalDisposition.REPLACE,
        "operator_cancelled",
    )
    assert updated.active_turn_id is None
    assert repository.load_dispatch(pending.dispatch_id).status is DispatchStatus.CANCELLED  # type: ignore[union-attr]


def test_memory_create_and_complete_have_one_cas_winner() -> None:
    repository, schedule, managed = _memory_validating_turn()
    barrier = threading.Barrier(2)
    results = _race(
        lambda: _reserve_after_barrier(repository, schedule, managed, barrier),
        lambda: _finish_after_barrier(repository, schedule, managed, barrier),
    )
    assert sum(result.succeeded for result in results) == 1
    assert {result.error_type for result in results if not result.succeeded} <= {
        TurnStateConflict,
        ActiveTurnFenceRejected,
        DispatchRecoveryBlocked,
    }
```

Add create-vs-cancel, create-vs-expire, and create-vs-nonterminal-transition
races. For cancellation/expiry, retry after reloading and assert no terminal
turn owns a `PENDING` or `DISPATCHING` attempt.

- [ ] **Step 5: Implement atomic fenced finish**

Load exact schedule, managed turn, and deterministic exact-turn attempts under
the same `RLock`; validate both CAS versions; call
`prepare_fenced_active_finish`; then publish all updated attempts, the managed
turn, schedule, and active-schedule index together. Build all copies first and
restore snapshots if an injected container assignment raises.

- [ ] **Step 6: Verify Memory behavior and commit**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/storage/test_active_turn_fence.py -k memory -v
conda run -n wofkill python -m pytest \
  tests/storage/test_autonomous_commit.py -k dispatch -v
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/storage/memory_store.py \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_autonomous_commit.py
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/storage/memory_store.py
git add \
  werewolf_agent/storage/memory_store.py \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_autonomous_commit.py
git commit -m "feat: add in-memory active-turn fence"
```

Expected: Memory fence and existing durable-dispatch tests pass; checks exit 0.

### Task 3: Route HostRuntime Through the Combined Fence

**Files:**
- Modify: `werewolf_agent/player_agents/runtime/host.py:1-289`
- Modify: `tests/player_agents/test_host_runtime.py`

**Interfaces:**
- Consumes: Task 2's in-memory combined repository and Task 1's guard.
- Produces: `HostRuntime.create_active_turn_dispatch(schedule_id, attempt)` and fenced terminal routing for complete/cancel/expire.

- [ ] **Step 1: Write failing Host capability and creation tests**

```python
def test_host_requires_one_physical_active_turn_fence_repository() -> None:
    turn_repository = _repository(_schedule())
    dispatch_repository = _repository(_schedule(game_id="other-game"))
    reconciler = DispatchReconciler(dispatch_repository, PendingResolver())
    with pytest.raises(ActiveTurnFenceUnsupported):
        HostRuntime(turn_repository, dispatch_repository, reconciler, clock=lambda: NOW)


def test_host_creates_repository_generated_fenced_attempt() -> None:
    repository, host, managed = _host_with_active_turn()
    attempt = host.create_active_turn_dispatch(
        managed.schedule_id,
        _attempt(managed),
    )
    current = repository.load_managed_turn(managed.turn.turn_id)
    assert current is not None
    assert attempt.active_turn_fence is not None
    assert attempt.active_turn_fence.turn_state_version == current.state_version
```

Add tests for unrecovered/blocked games, expired deadlines, caller-supplied
fences, and a replacement interleaving between Host read and repository call.

- [ ] **Step 2: Run Host creation tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_host_runtime.py -k "physical or fenced_attempt or active_turn_dispatch" -v
```

Expected: HostRuntime does not require the combined capability and has no
`create_active_turn_dispatch` method.

- [ ] **Step 3: Implement combined construction and fenced creation**

In `HostRuntime.__init__`, retain the public parameters but replace separate
capability acceptance with:

```python
self._fence_repository = require_active_turn_fence_repository(
    turn_repository,
    dispatch_repository,
)
self._scheduler = SerialPublicScheduler(turn_repository)
self._dispatch_repository = require_durable_dispatch_repository(
    dispatch_repository,
)
```

Add:

```python
def create_active_turn_dispatch(
    self,
    schedule_id: str,
    attempt: DispatchAttempt,
) -> DispatchAttempt:
    schedule, managed = self._require_active_schedule_turn(schedule_id)
    self._require_recovered(schedule.game_id)
    return self._fence_repository.create_active_turn_dispatch(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        attempt,
        self._aware_clock_now(),
    )
```

Do not add provider callbacks or expose the repository.

- [ ] **Step 4: Write failing fenced terminal-routing tests**

Replace direct setup calls with `_reserve()` where production authorization is
under test. Add a spy repository that raises if the old terminal method or
`list_dispatches_for_turn()` is called:

```python
def test_host_cancel_uses_one_fenced_terminal_call_without_prescan() -> None:
    repository, host, managed = _host_with_active_turn(repository_type=FenceSpyRepository)
    host.create_active_turn_dispatch(managed.schedule_id, _attempt(managed))
    host.cancel_active_turn(
        managed.schedule_id,
        "operator_cancelled",
        TerminalDisposition.ADVANCE,
    )
    assert repository.fenced_finish_calls == 1
    assert repository.dispatch_scan_calls == 0
    assert repository.unfenced_finish_calls == 0
```

Cover completion, each cancel disposition, deterministic expiry, an already
`DISPATCHED` attempt, and existing captured-identity replacement tests.

- [ ] **Step 5: Route every terminal method through the fence**

Delete `_cancel_turn_dispatches`. `complete_active_turn`,
`cancel_active_turn`, and `expire_due_turns` call
`finish_active_turn_fenced()` with the first captured schedule and managed-turn
versions. Keep the recovery rules and deterministic expiry ordering unchanged.
`SerialPublicScheduler.finish_active_turn()` stays available only as the
low-level scheduling facade and is no longer called by HostRuntime.

- [ ] **Step 6: Verify Host behavior and commit**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_host_runtime.py \
  tests/player_agents/test_runtime_import_boundary.py -v
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents/runtime/host.py \
  tests/player_agents/test_host_runtime.py
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/player_agents/runtime
git add \
  werewolf_agent/player_agents/runtime/host.py \
  tests/player_agents/test_host_runtime.py
git commit -m "feat: route host through active-turn fence"
```

Expected: Host and import-boundary tests pass; checks exit 0.

### Task 4: Implement SQLite Persistence, Migration, Rollback, and Races

**Files:**
- Modify: `werewolf_agent/storage/sqlite_store.py:160-1578`
- Modify: `tests/storage/test_active_turn_fence.py`
- Modify: `tests/storage/test_sqlite_migrations.py`

**Interfaces:**
- Consumes: Task 1's fence JSON contract/helpers and Task 3's Host API.
- Produces: SQLite fence capability, schema upgrade, full attempt round-trip, and atomic create/finish transactions.

- [ ] **Step 1: Write failing fresh and legacy schema tests**

```python
def test_sqlite_fresh_schema_has_nullable_active_turn_fence_column(tmp_path) -> None:
    repository = SqliteGameRepository(str(tmp_path / "fresh.db"))
    columns = {
        row[1]: row
        for row in repository._conn.execute(
            "PRAGMA table_info(autonomous_dispatch_attempts)",
        ).fetchall()
    }
    assert "active_turn_fence_json" in columns
    assert columns["active_turn_fence_json"][3] == 0


def test_sqlite_legacy_dispatch_schema_adds_nullable_fence_column(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    _create_pre_fence_database(path)
    repository = SqliteGameRepository(str(path))
    assert repository.supports_active_turn_fence() is True
    assert repository.load_dispatch("legacy-dispatch").active_turn_fence is None  # type: ignore[union-attr]
```

The legacy fixture creates the old dispatch table and one complete unfenced
row. It must not edit `MigrationManager` migrations.

- [ ] **Step 2: Run schema tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/storage/test_sqlite_migrations.py -k "active_turn_fence or nullable_fence" -v
```

Expected: the column and capability are absent.

- [ ] **Step 3: Add the idempotent nullable-column upgrade and round-trip**

Include `active_turn_fence_json TEXT` in fresh DDL. Add a startup helper:

```python
def _ensure_active_turn_fence_schema(conn: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(autonomous_dispatch_attempts)",
        ).fetchall()
    }
    if "active_turn_fence_json" not in columns:
        conn.execute(
            "ALTER TABLE autonomous_dispatch_attempts "
            "ADD COLUMN active_turn_fence_json TEXT",
        )
```

Call it before setting schema-ready flags. Extend every dispatch SELECT/INSERT
and `_dispatch_from_row` with the nullable field, encoding canonical compact
JSON from `ActiveTurnDispatchFence.model_dump(mode="json")`.

- [ ] **Step 4: Write failing SQLite transaction and rollback tests**

Reuse the shared behavior assertions through `_sqlite_active_turn(tmp_path)`.
Add triggers at both write boundaries:

```python
def test_sqlite_fenced_create_rolls_back_attempt_when_turn_update_fails(tmp_path) -> None:
    repository, schedule, managed = _sqlite_active_turn(tmp_path)
    repository._conn.execute("""
        CREATE TRIGGER fail_fence_turn_update
        BEFORE UPDATE ON autonomous_managed_turns
        BEGIN SELECT RAISE(ABORT, 'forced turn update failure'); END;
    """)
    with pytest.raises(ActiveTurnFenceTransactionError):
        _reserve(repository, schedule, managed)
    assert repository.load_dispatch("dispatch-1") is None
    assert repository.load_managed_turn(managed.turn.turn_id) == managed


def test_sqlite_fenced_finish_rolls_back_cancel_when_schedule_update_fails(tmp_path) -> None:
    repository, schedule, managed = _sqlite_active_turn(tmp_path)
    attempt = _reserve(repository, schedule, managed)
    managed = repository.load_managed_turn(managed.turn.turn_id)
    _install_schedule_update_failure(repository)
    with pytest.raises(ActiveTurnFenceTransactionError):
        _cancel(repository, schedule, managed)
    assert repository.load_dispatch(attempt.dispatch_id).status is DispatchStatus.PENDING  # type: ignore[union-attr]
    assert repository.load_managed_turn(managed.turn.turn_id) == managed
```

Also cover ID/key conflicts and recovery barrier rollback.

- [ ] **Step 5: Implement SQLite combined transactions**

Both methods use one `BEGIN IMMEDIATE`, load schedule/turn inside it, compare
CAS versions, run the pure helper, and use existing row-count CAS update
helpers. Reservation inserts the attempt and updates the managed turn before
one commit. Terminalization loads exact-turn attempts in deterministic order,
updates changed attempts, updates managed turn and schedule, then commits.

Catch and re-raise stable expected errors. Roll back and wrap all other
exceptions in `ActiveTurnFenceTransactionError`.

Reject caller-supplied fences in plain `create_dispatch()`.

- [ ] **Step 6: Write and run real two-connection SQLite races**

Create two `SqliteGameRepository` instances for one database path and use a
barrier to start operations together:

```python
def test_sqlite_two_connections_create_and_complete_have_one_cas_winner(tmp_path) -> None:
    first, second, schedule, managed = _sqlite_two_connection_validating_turn(tmp_path)
    results = _race_two_repositories(
        lambda: _reserve(first, schedule, managed),
        lambda: _complete(second, schedule, managed),
    )
    assert sum(result.succeeded for result in results) == 1
    observed_schedule = first.load_serial_public_schedule(schedule.schedule_id)
    observed_turn = first.load_managed_turn(managed.turn.turn_id)
    observed_attempt = first.load_dispatch("dispatch-1")
    assert _is_valid_single_winner_state(
        observed_schedule,
        observed_turn,
        observed_attempt,
    )
```

Add create-vs-cancel and create-vs-transition variants. Retry cancellation
after a create winner and assert the attempt is cancelled atomically.

- [ ] **Step 7: Verify SQLite behavior and commit**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/storage/test_active_turn_fence.py -k sqlite -v
conda run -n wofkill python -m pytest \
  tests/storage/test_sqlite_migrations.py \
  tests/storage/test_autonomous_turns.py \
  tests/storage/test_autonomous_commit.py -k "sqlite or dispatch or turn" -v
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/storage/sqlite_store.py \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_sqlite_migrations.py
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/storage/sqlite_store.py
git add \
  werewolf_agent/storage/sqlite_store.py \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_sqlite_migrations.py
git commit -m "feat: add SQLite active-turn fence"
```

Expected: SQLite tests including two-connection races pass; checks exit 0.

### Task 5: Implement PostgreSQL Locking, Schema, and Transactions

**Files:**
- Modify: `werewolf_agent/storage/postgres_store.py:200-2058`
- Modify: `tests/storage/test_postgres_autonomous_commit.py`
- Modify: `tests/storage/test_active_turn_fence.py`

**Interfaces:**
- Consumes: Task 1's helpers and Task 4's persisted nullable-field semantics.
- Produces: PostgreSQL fence capability with advisory-lock-first ordering and exact rollback/conflict behavior.

- [ ] **Step 1: Write failing PostgreSQL schema and lock-order tests**

```python
def test_postgres_schema_adds_nullable_active_turn_fence_jsonb() -> None:
    repository, connection = _schema_repository()
    repository._ensure_autonomous_schema(connection)
    sql = " ".join(connection.executed_sql).lower()
    assert "active_turn_fence_json jsonb" in sql
    assert (
        "alter table autonomous_dispatch_attempts "
        "add column if not exists active_turn_fence_json jsonb"
    ) in sql


def test_postgres_fenced_create_locks_game_before_schedule_turn_and_dispatch() -> None:
    repository, connection, schedule, managed = _postgres_active_turn()
    repository.create_active_turn_dispatch(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        _attempt_for(managed),
        NOW,
    )
    statements = connection.normalized_statements
    assert _index_of(statements, "pg_advisory_xact_lock") < _index_of(
        statements,
        "autonomous_serial_public_schedules",
    )
    assert _index_of(statements, "autonomous_serial_public_schedules") < _index_of(
        statements,
        "autonomous_managed_turns",
    )
    assert _index_of(statements, "autonomous_managed_turns") < _index_of(
        statements,
        "autonomous_dispatch_attempts",
    )
```

Also assert `_transition_dispatch()` acquires the advisory game lock before
locking/updating a fenced attempt.

- [ ] **Step 2: Run PostgreSQL tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/storage/test_postgres_autonomous_commit.py \
  tests/storage/test_active_turn_fence.py -k postgres -v
```

Expected: schema, capability, methods, and lock order are absent.

- [ ] **Step 3: Extend schema and attempt serialization**

Add the nullable JSONB column to fresh table DDL and execute the idempotent
`ALTER TABLE` immediately after table creation. Extend
`_dispatch_select_columns`, `_dispatch_from_row`, and every insert with the
nullable canonical JSONB value. Preserve all historical field offsets through
one shared select-column helper rather than duplicating lists.

- [ ] **Step 4: Unify the PostgreSQL game boundary**

Make `_lock_dispatch_game` an instance method that first calls
`_lock_game_transaction(conn, game_id)` and then verifies/locks the `games`
row. Existing plain dispatch create and state transitions may keep their
behavior, but they now participate in the same advisory boundary as schedule
and fenced terminal transactions.

Do not acquire a dispatch row before the advisory lock. Where the dispatch ID
is the only input, perform an unlocked lookup only to discover `game_id`, then
take the advisory lock and reload the dispatch `FOR UPDATE`, preserving the
existing recheck.

- [ ] **Step 5: Write failing stateful transaction and rollback tests**

Extend the stateful fake connection to hold schedules, turns, attempts, and a
transaction snapshot. Add:

```python
def test_postgres_fenced_create_commits_attempt_and_turn_version_once() -> None:
    repository, connection, schedule, managed = _postgres_active_turn()
    attempt = _postgres_reserve(repository, schedule, managed)
    assert connection.committed == 1
    assert connection.rolled_back == 0
    assert connection.turns[managed.turn.turn_id]["state_version"] == 1
    assert connection.attempts[attempt.dispatch_id]["active_turn_fence"] == {
        "schedule_id": schedule.schedule_id,
        "schedule_state_version": schedule.state_version,
        "turn_state_version": 1,
        "window_id": managed.turn.window.window_id,
        "window_version": managed.turn.window.version,
        "base_game_revision": managed.turn.revision.base_revision,
    }


def test_postgres_fenced_finish_rolls_back_cancel_and_turn_on_schedule_failure() -> None:
    repository, connection, schedule, managed, attempt = _postgres_reserved_turn(
        fail_on_schedule_update=True,
    )
    with pytest.raises(ActiveTurnFenceTransactionError):
        _postgres_cancel(repository, schedule, managed)
    assert connection.rolled_back == 1
    assert connection.attempts[attempt.dispatch_id]["status"] == "pending"
    assert connection.turns[managed.turn.turn_id]["turn"]["status"] != "cancelled"
```

Cover schedule/turn CAS misses, completion blocked by unresolved work, and a
failure after attempt insert but before managed-turn update.

- [ ] **Step 6: Implement PostgreSQL combined transactions**

Reservation:

1. resolve game from schedule, acquire advisory lock, then verify game;
2. lock schedule and managed turn in order;
3. compare CAS versions and prepare the update;
4. check exact uniqueness and game recovery barrier under the lock;
5. insert the fenced attempt and CAS-update the managed turn; and
6. commit once.

Terminalization:

1. acquire advisory lock from schedule;
2. lock schedule, turn, then exact-turn attempts in deterministic order;
3. call `prepare_fenced_active_finish`;
4. update changed attempts, managed turn, and schedule by CAS; and
5. commit once.

Map only known expected exceptions directly. For `23505`, inspect the exact
constraint name before mapping to `DispatchIdempotencyConflict`; wrap all
other failures in `ActiveTurnFenceTransactionError` after rollback.
Reject `active_turn_fence is not None` at the start of PostgreSQL plain
`create_dispatch()` so callers cannot bypass validation with manufactured
fence metadata.

- [ ] **Step 7: Verify PostgreSQL behavior and commit**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/storage/test_postgres_autonomous_commit.py \
  tests/storage/test_active_turn_fence.py -k postgres -v
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/storage/postgres_store.py \
  tests/storage/test_postgres_autonomous_commit.py \
  tests/storage/test_active_turn_fence.py
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/storage/postgres_store.py
git add \
  werewolf_agent/storage/postgres_store.py \
  tests/storage/test_postgres_autonomous_commit.py \
  tests/storage/test_active_turn_fence.py
git commit -m "feat: add PostgreSQL active-turn fence"
```

Expected: PostgreSQL contract tests pass; Ruff and mypy exit 0. Do not claim a
real PostgreSQL service integration was run.

### Task 6: Cross-Backend Hardening, Documentation, and Final Verification

**Files:**
- Modify: `tests/storage/test_active_turn_fence.py`
- Modify: `tests/player_agents/test_host_runtime.py`
- Modify: `handoff.md`

**Interfaces:**
- Consumes: all prior task interfaces.
- Produces: one documented, regression-verified fence milestone with no live-path integration.

- [ ] **Step 1: Add the shared backend conformance matrix**

Parameterize Memory and SQLite repositories over the same literal scenarios:

```python
@pytest.mark.parametrize("repository_factory", [_memory_factory, _sqlite_factory])
def test_fence_backend_conformance(repository_factory, tmp_path) -> None:
    repository, schedule, managed = repository_factory(tmp_path)
    attempt = _reserve(repository, schedule, managed)
    managed = repository.load_managed_turn(managed.turn.turn_id)
    assert managed is not None
    assert attempt.active_turn_fence is not None
    assert attempt.active_turn_fence.turn_state_version == managed.state_version
    terminal = repository.finish_active_turn_fenced(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        AgentTurnStatus.CANCELLED,
        TerminalDisposition.ADVANCE,
        "operator_cancelled",
    )
    assert terminal.active_turn_id is None
    assert repository.load_dispatch(attempt.dispatch_id).status is DispatchStatus.CANCELLED  # type: ignore[union-attr]
```

Include mismatch/error-code equivalence and defensive-copy assertions.

- [ ] **Step 2: Run the full new-runtime focused suite**

Run:

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents \
  tests/storage/test_autonomous_commit.py \
  tests/storage/test_autonomous_turns.py \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_durable_dispatch_protocol.py \
  tests/storage/test_postgres_autonomous_commit.py -q
```

Expected: exit 0 with no failures. If a regression appears, write the narrowest
failing test that names the broken contract before modifying production code.

- [ ] **Step 3: Run scoped static checks and repository diff checks**

Run:

```bash
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents \
  werewolf_agent/storage/active_turn_fence.py \
  werewolf_agent/storage/memory_store.py \
  werewolf_agent/storage/sqlite_store.py \
  werewolf_agent/storage/postgres_store.py \
  tests/player_agents \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_autonomous_commit.py \
  tests/storage/test_autonomous_turns.py \
  tests/storage/test_durable_dispatch_protocol.py \
  tests/storage/test_postgres_autonomous_commit.py
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/player_agents \
  werewolf_agent/storage/active_turn_fence.py \
  werewolf_agent/storage/autonomous_turns.py \
  werewolf_agent/storage/durable_dispatch.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 4: Run the full repository suite**

Run:

```bash
conda run -n wofkill python -m pytest -q
```

Expected: exit 0 with only the repository's documented skips and third-party
warnings. Record the exact pass/skip/warning counts in `handoff.md` from this
fresh run.

- [ ] **Step 5: Update the handoff milestone**

Update `handoff.md` with:

- active-turn fence listed under implemented infrastructure;
- exact persisted identity and managed-turn version reservation semantics;
- atomic cancel/expire/complete behavior;
- Memory/SQLite/PostgreSQL evidence and the lack of real PostgreSQL service
  validation;
- exact fresh focused/full test counts and check commands;
- unchanged prohibitions on legacy/live paths and ToolResult Markdown; and
- the next unique milestone: isolated player document projections,
  `ObservationFrame`, context-budget accounting, structured compaction, and
  checkpoint rehydration, without claiming the first playable vertical slice
  is complete.

- [ ] **Step 6: Review the final diff against the design invariants**

Run:

```bash
git status --short --branch
git diff --stat HEAD~5..HEAD
git diff HEAD~5..HEAD -- \
  werewolf_agent/player_agents \
  werewolf_agent/storage \
  tests/player_agents \
  tests/storage \
  handoff.md
rg -n "PlayerAgent|ModelRouter|_dispatch_agent|ToolResult.*Markdown" \
  werewolf_agent/player_agents \
  werewolf_agent/storage/active_turn_fence.py \
  tests/storage/test_active_turn_fence.py
```

Expected: only the deliberate boundary-test literals may mention forbidden
legacy names; no production import or live-path call exists. Compare every
design acceptance criterion to a test or static check before completion.

- [ ] **Step 7: Commit verification documentation**

Run:

```bash
git add handoff.md tests/storage/test_active_turn_fence.py tests/player_agents/test_host_runtime.py
git commit -m "docs: record active-turn fence completion"
git status --short --branch
```

Expected: commit succeeds and the worktree is clean.
