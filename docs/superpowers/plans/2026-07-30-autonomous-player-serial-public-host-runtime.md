# Autonomous Player Serial Public Host Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement durable `serial_public` scheduling and a host-owned autonomous-player turn lifecycle without connecting any live game, model, tool, workspace, or legacy player path.

**Architecture:** Add strict scheduling contracts and a separate `AutonomousTurnRepository` capability backed by memory, SQLite, and PostgreSQL. Shared pure state-preparation helpers define admission, lifecycle CAS, and atomic terminal advancement; `HostRuntime` coordinates those operations with the existing durable-dispatch recovery barrier and per-turn cancellation query.

**Tech Stack:** Python 3.12, Pydantic v2, dataclasses/protocols, in-memory RLock, SQLite transactions, PostgreSQL JSONB/row locks, pytest, Ruff, mypy; all Python commands use `conda run -n wofkill`.

**Progress:** Planned steps complete (`52/52`); final-review fixes in progress
(`3/3`).

**Design:** `docs/superpowers/specs/2026-07-30-serial-public-scheduler-host-runtime-design.md`

---

## File Map

- Create: `werewolf_agent/player_agents/contracts/scheduling.py` - strict schedule, slot, admission, managed-turn, status, and terminal-disposition contracts.
- Modify: `werewolf_agent/player_agents/contracts/__init__.py` - export the stable scheduling contract surface.
- Create: `werewolf_agent/storage/autonomous_turns.py` - capability protocol, stable errors, and shared pure lifecycle preparation helpers.
- Modify: `werewolf_agent/storage/durable_dispatch.py` - add deterministic per-turn dispatch lookup to the explicit capability.
- Modify: `werewolf_agent/storage/memory_store.py` - memory schedule/turn CAS plus per-turn dispatch lookup.
- Modify: `werewolf_agent/storage/sqlite_store.py` - scheduling schema, canonical payload persistence, transactions, and per-turn dispatch lookup.
- Modify: `werewolf_agent/storage/postgres_store.py` - JSONB scheduling schema, row-locked transactions, and per-turn dispatch lookup.
- Create: `werewolf_agent/player_agents/runtime/__init__.py` - isolated runtime namespace and public exports.
- Create: `werewolf_agent/player_agents/runtime/serial_public.py` - narrow scheduler facade over the turn repository.
- Create: `werewolf_agent/player_agents/runtime/host.py` - recovery, admission, lifecycle, cancellation, expiry, and completion coordinator.
- Create: `tests/player_agents/test_scheduling_contracts.py` - strict contract and binding tests.
- Create: `tests/storage/test_autonomous_turns.py` - shared memory/SQLite lifecycle and atomicity matrix.
- Modify: `tests/storage/test_durable_dispatch_protocol.py` - capability fixture and per-turn lookup contract.
- Modify: `tests/storage/test_sqlite_migrations.py` - fresh/legacy scheduling schema isolation checks.
- Modify: `tests/storage/test_postgres_autonomous_commit.py` - PostgreSQL scheduling DDL and transaction mocks.
- Create: `tests/player_agents/test_host_runtime.py` - HostRuntime recovery, cancellation, expiry, and completion tests.
- Create: `tests/player_agents/test_runtime_import_boundary.py` - AST boundary check for the new runtime package.

### Task 1: Define Strict Scheduling Contracts

**Files:**
- Create: `werewolf_agent/player_agents/contracts/scheduling.py`
- Modify: `werewolf_agent/player_agents/contracts/__init__.py`
- Test: `tests/player_agents/test_scheduling_contracts.py`

- [x] **Step 1: Write failing schedule and admission contract tests**

Create fixed UTC fixtures and cover the public invariants with these tests:

```python
def test_schedule_requires_serial_public_ordered_unique_slots() -> None:
    schedule = _schedule()
    assert tuple(slot.ordinal for slot in schedule.slots) == (0, 1)
    assert schedule.current_slot.player_id == "p01"

    payload = schedule.model_dump()
    payload["slots"] = (
        {"ordinal": 0, "player_id": "p01"},
        {"ordinal": 2, "player_id": "p02"},
    )
    with pytest.raises(ValidationError, match="contiguous"):
        SerialPublicSchedule.model_validate(payload)


def test_schedule_rejects_non_serial_window_and_participant_drift() -> None:
    schedule = _schedule()
    private_window = schedule.window.model_copy(
        update={"conflict_class": ConflictClass.SERIAL_PRIVATE},
    )
    with pytest.raises(ValidationError, match="serial_public"):
        SerialPublicSchedule.model_validate({
            **schedule.model_dump(),
            "window": private_window,
        })

    with pytest.raises(ValidationError, match="window participants"):
        SerialPublicSchedule.model_validate({
            **schedule.model_dump(),
            "slots": ({"ordinal": 0, "player_id": "p01"},),
        })


def test_managed_turn_and_admission_are_strict_frozen_models() -> None:
    admission = _admission()
    with pytest.raises(ValidationError):
        TurnAdmission.model_validate({
            **admission.model_dump(),
            "unexpected": True,
        })
    with pytest.raises((ValidationError, TypeError)):
        admission.player_id = "p09"  # type: ignore[misc]
```

The fixture uses a `LegalActionWindow` with participants `("p01", "p02")`,
`ConflictClass.SERIAL_PUBLIC`, revision 4, and an aware deadline. It creates an
open schedule at state version 0, no active turn, and `next_slot_ordinal=0`.

- [x] **Step 2: Run the contract tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_scheduling_contracts.py -v
```

Expected: collection fails because `contracts.scheduling` does not exist.

- [x] **Step 3: Implement the closed enums and strict models**

Create the module with the required project header and these exact public fields:

```python
class SerialPublicScheduleStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class TerminalDisposition(StrEnum):
    ADVANCE = "advance"
    REPLACE = "replace"
    CLOSE = "close"


class SerialPublicSlot(StrictFrozenModel):
    ordinal: int = Field(ge=0)
    player_id: NonEmptyId


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class SerialPublicSchedule(StrictFrozenModel):
    schedule_id: NonEmptyId
    game_id: NonEmptyId
    window: LegalActionWindow
    slots: tuple[SerialPublicSlot, ...] = Field(min_length=1)
    next_slot_ordinal: int = Field(ge=0)
    active_turn_id: NonEmptyId | None = None
    status: SerialPublicScheduleStatus
    state_version: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime

    @property
    def current_slot(self) -> SerialPublicSlot:
        if self.status is not SerialPublicScheduleStatus.OPEN:
            raise ValueError("serial public schedule is not open")
        return self.slots[self.next_slot_ordinal]

    @model_validator(mode="after")
    def _consistent_schedule(self) -> Self:
        if self.window.conflict_class is not ConflictClass.SERIAL_PUBLIC:
            raise ValueError("schedule window must be serial_public")
        if self.game_id != self.window.game_id:
            raise ValueError("schedule game_id must match window game_id")
        ordinals = tuple(slot.ordinal for slot in self.slots)
        if ordinals != tuple(range(len(self.slots))):
            raise ValueError("slot ordinals must be contiguous from zero")
        player_ids = tuple(slot.player_id for slot in self.slots)
        require_unique(player_ids, field_name="slot player IDs")
        if set(player_ids) != set(self.window.participant_ids):
            raise ValueError("slot players must match window participants")
        if self.next_slot_ordinal > len(self.slots):
            raise ValueError("next_slot_ordinal exceeds slot count")
        if self.status is SerialPublicScheduleStatus.OPEN:
            if self.next_slot_ordinal >= len(self.slots):
                raise ValueError("open schedule must have a current slot")
        elif self.active_turn_id is not None:
            raise ValueError("terminal schedule cannot have an active turn")
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class TurnAdmission(StrictFrozenModel):
    turn_id: NonEmptyId
    player_id: NonEmptyId
    role_id: NonEmptyId
    phase: NonEmptyId
    revision: RevisionContext
    read_set: tuple[ReadReference, ...] = ()
    model_lease_hash: ContentHash
    budget: TurnBudget
    idempotency_key: NonEmptyId

    @model_validator(mode="after")
    def _unique_reads(self) -> Self:
        require_unique(
            (item.record_id for item in self.read_set),
            field_name="read_set record IDs",
        )
        return self


class ManagedAgentTurn(StrictFrozenModel):
    schedule_id: NonEmptyId
    turn: AgentTurn
    state_version: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    terminal_reason: NonEmptyId | None = None

    @model_validator(mode="after")
    def _consistent_timestamps(self) -> Self:
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self
```

Add an aware-datetime validator for `ManagedAgentTurn` and keep
`AgentTurn.status` as its only status field.

- [x] **Step 4: Add contract cross-field rejection tests**

Add tests that reject an open schedule at the end of its slots, a terminal
schedule with an active turn, a naive timestamp, duplicate read references,
and an admission whose player does not match the current slot:

```python
@pytest.mark.parametrize(
    "updates",
    [
        {"next_slot_ordinal": 2},
        {
            "status": SerialPublicScheduleStatus.CLOSED,
            "active_turn_id": "turn-1",
        },
        {"updated_at": datetime(2026, 7, 30, 10)},  # noqa: DTZ001
    ],
)
def test_schedule_rejects_inconsistent_state(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SerialPublicSchedule.model_validate({
            **_schedule().model_dump(),
            **updates,
        })
```

- [x] **Step 5: Export contracts and verify GREEN**

Export all six public contract names from `contracts/__init__.py`, update its
`修改日期` to `2026-07-30`, then run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_scheduling_contracts.py -v
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/player_agents/contracts tests/player_agents/test_scheduling_contracts.py
```

Expected: all tests and Ruff checks pass.

- [x] **Step 6: Commit the scheduling contract layer**

```bash
git add werewolf_agent/player_agents/contracts tests/player_agents/test_scheduling_contracts.py
git commit -m "feat: define serial public scheduling contracts"
```

### Task 2: Add the Turn Repository Protocol and Pure State Preparation

**Files:**
- Create: `werewolf_agent/storage/autonomous_turns.py`
- Test: `tests/storage/test_autonomous_turns.py`

- [x] **Step 1: Write failing capability and state-preparation tests**

Use the fixtures from Task 1 and assert explicit capability rejection plus the
three pure transitions:

```python
def test_turn_capability_guard_rejects_legacy_repository() -> None:
    with pytest.raises(AutonomousTurnsUnsupported):
        require_autonomous_turn_repository(object())


def test_prepare_admission_binds_fresh_turn_to_current_slot() -> None:
    schedule, managed = prepare_serial_public_admission(
        _schedule(),
        _admission(),
        NOW,
    )
    assert schedule.active_turn_id == "turn-1"
    assert schedule.state_version == 1
    assert managed.turn.status is AgentTurnStatus.OPEN
    assert managed.turn.window == schedule.window
    assert managed.state_version == 0


def test_prepare_finish_advance_consumes_slot_atomically() -> None:
    schedule, managed = _admitted_validating()
    updated_schedule, updated_turn = prepare_active_finish(
        schedule,
        managed,
        AgentTurnStatus.COMMITTED,
        TerminalDisposition.ADVANCE,
        reason_code=None,
        now=NOW,
    )
    assert updated_turn.turn.status is AgentTurnStatus.COMMITTED
    assert updated_schedule.active_turn_id is None
    assert updated_schedule.next_slot_ordinal == 1
```

Also assert that `REPLACE` retains the current ordinal and `CLOSE` produces a
cancelled schedule.

- [x] **Step 2: Run the helper tests and verify RED**

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_turns.py -k "capability or prepare" -v
```

Expected: import error for `storage.autonomous_turns`.

- [x] **Step 3: Implement stable errors and the explicit protocol**

Define these stable errors and signatures:

```python
class AutonomousTurnError(RuntimeError):
    code: ClassVar[str] = "autonomous_turn_error"


class AutonomousTurnsUnsupported(AutonomousTurnError):
    code = "autonomous_turns_unsupported"


class ScheduleNotFound(AutonomousTurnError):
    code = "schedule_not_found"


class ManagedTurnNotFound(AutonomousTurnError):
    code = "managed_turn_not_found"


class ScheduleStateConflict(AutonomousTurnError):
    code = "schedule_state_conflict"


class TurnStateConflict(AutonomousTurnError):
    code = "turn_state_conflict"


class InvalidScheduleTransition(AutonomousTurnError):
    code = "invalid_schedule_transition"


class InvalidTurnAdmission(AutonomousTurnError):
    code = "invalid_turn_admission"


class AutonomousTurnTransactionError(AutonomousTurnError):
    code = "autonomous_turn_transaction_error"


class AutonomousTurnRepository(Protocol):
    def supports_autonomous_turns(self) -> bool: ...
    def create_serial_public_schedule(
        self, schedule: SerialPublicSchedule,
    ) -> SerialPublicSchedule: ...
    def load_serial_public_schedule(
        self, schedule_id: str,
    ) -> SerialPublicSchedule | None: ...
    def load_active_serial_public_schedule(
        self, game_id: str,
    ) -> SerialPublicSchedule | None: ...
    def list_open_serial_public_schedules(
        self,
    ) -> tuple[SerialPublicSchedule, ...]: ...
    def admit_serial_public_turn(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        admission: TurnAdmission,
    ) -> ManagedAgentTurn: ...
    def transition_active_turn(
        self,
        turn_id: str,
        expected_turn_version: int,
        next_status: AgentTurnStatus,
    ) -> ManagedAgentTurn: ...
    def finish_active_turn(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        terminal_status: AgentTurnStatus,
        disposition: TerminalDisposition,
        reason_code: str | None,
    ) -> SerialPublicSchedule: ...
    def load_managed_turn(self, turn_id: str) -> ManagedAgentTurn | None: ...
```

`require_autonomous_turn_repository` accepts only a callable capability method
that returns true, mirroring the completed commit and dispatch guards.

- [x] **Step 4: Implement shared pure preparation helpers**

Implement `prepare_serial_public_admission`,
`prepare_active_transition`, and `prepare_active_finish`. The admission
helper constructs the existing `AgentTurn` from schedule-owned game, task,
window, and deadline fields:

```python
turn = AgentTurn(
    turn_id=admission.turn_id,
    game_id=schedule.game_id,
    player_id=admission.player_id,
    role_id=admission.role_id,
    phase=admission.phase,
    task_type=schedule.window.task_type,
    revision=admission.revision,
    window=schedule.window,
    read_set=admission.read_set,
    model_lease_hash=admission.model_lease_hash,
    budget=admission.budget,
    status=AgentTurnStatus.OPEN,
    idempotency_key=admission.idempotency_key,
)
```

Before constructing it, reject a non-open schedule, an existing active turn, a
wrong player, a stale window ID/version, or a base revision below
`window.opened_revision`. Return a schedule copy with `active_turn_id` and
incremented `state_version`, plus a version-zero `ManagedAgentTurn`.

`prepare_active_transition` verifies the schedule active ID, rejects terminal
targets, calls `transition_turn`, and increments only the managed-turn version.

`prepare_active_finish` verifies both active identities, calls
`transition_turn` for `COMMITTED`, `CANCELLED`, or `EXPIRED`, increments
the managed-turn version, clears `active_turn_id`, and applies:

```python
if disposition is TerminalDisposition.ADVANCE:
    next_ordinal = schedule.next_slot_ordinal + 1
    next_status = (
        SerialPublicScheduleStatus.CLOSED
        if next_ordinal == len(schedule.slots)
        else SerialPublicScheduleStatus.OPEN
    )
elif disposition is TerminalDisposition.REPLACE:
    next_ordinal = schedule.next_slot_ordinal
    next_status = SerialPublicScheduleStatus.OPEN
else:
    next_ordinal = schedule.next_slot_ordinal
    next_status = SerialPublicScheduleStatus.CANCELLED
```

Translate Pydantic or transition failures to `InvalidTurnAdmission` or
`InvalidScheduleTransition` without leaking payload data.

- [x] **Step 5: Verify helper edge cases and static checks**

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_turns.py -k "capability or prepare" -v
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/storage/autonomous_turns.py tests/storage/test_autonomous_turns.py
conda run -n wofkill python -m mypy werewolf_agent/storage/autonomous_turns.py
```

Expected: focused tests pass and both static checks report no errors.

- [x] **Step 6: Commit protocol and shared lifecycle logic**

```bash
git add werewolf_agent/storage/autonomous_turns.py tests/storage/test_autonomous_turns.py
git commit -m "feat: add autonomous turn repository protocol"
```

### Task 3: Add Deterministic Per-Turn Dispatch Lookup

**Files:**
- Modify: `werewolf_agent/storage/durable_dispatch.py`
- Modify: `werewolf_agent/storage/memory_store.py`
- Modify: `werewolf_agent/storage/sqlite_store.py`
- Modify: `werewolf_agent/storage/postgres_store.py`
- Modify: `tests/storage/test_durable_dispatch_protocol.py`
- Modify: `tests/storage/test_autonomous_commit.py`
- Modify: `tests/storage/test_postgres_autonomous_commit.py`

- [x] **Step 1: Write failing shared lookup tests**

For memory and SQLite, create attempts for two turns and multiple statuses, then
assert exact filtering and deterministic order:

```python
def test_dispatches_for_turn_are_filtered_and_ordered(dispatch_repository) -> None:
    first = _dispatch_attempt(dispatch_id="d2", turn_id="turn-1")
    second = _dispatch_attempt(
        dispatch_id="d1",
        turn_id="turn-1",
        created_at=DISPATCH_NOW - timedelta(seconds=1),
        updated_at=DISPATCH_NOW - timedelta(seconds=1),
    )
    other = _dispatch_attempt(
        dispatch_id="d3",
        turn_id="turn-2",
        provider_idempotency_key="provider-key-3",
    )
    for attempt in (first, second, other):
        dispatch_repository.create_dispatch(attempt)

    assert [
        attempt.dispatch_id
        for attempt in dispatch_repository.list_dispatches_for_turn(
            "g1", "turn-1",
        )
    ] == ["d1", "d2"]
```

Add a protocol-fixture method to `InMemoryDispatchFixture` and a PostgreSQL SQL
assertion that both `game_id` and `turn_id` are parameters.

- [x] **Step 2: Run lookup tests and verify RED**

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_commit.py tests/storage/test_durable_dispatch_protocol.py tests/storage/test_postgres_autonomous_commit.py -k "dispatches_for_turn" -v
```

Expected: repositories and protocol lack `list_dispatches_for_turn`.

- [x] **Step 3: Extend the capability and all three backends**

Add to `DurableDispatchRepository`:

```python
def list_dispatches_for_turn(
    self,
    game_id: str,
    turn_id: str,
) -> list[DispatchAttempt]: ...
```

The memory implementation filters exact game/turn identity, sorts by
`(created_at, dispatch_id)`, and returns defensive copies. SQLite and
PostgreSQL select all statuses with:

```sql
WHERE game_id = ? AND turn_id = ?
ORDER BY created_at, dispatch_id
```

and:

```sql
WHERE game_id = %s AND turn_id = %s
ORDER BY created_at, dispatch_id
```

Add `idx_dispatch_game_turn_created` /
`idx_autonomous_dispatch_game_turn_created` indexes to the respective schema
blocks. Update modified Python module descriptions and `修改日期` only where the
current description no longer covers turn-scoped lookup.

- [x] **Step 4: Verify lookup behavior and dispatch regressions**

```bash
conda run -n wofkill python -m pytest tests/storage/test_durable_dispatch_protocol.py tests/storage/test_autonomous_commit.py tests/storage/test_postgres_autonomous_commit.py -k "dispatch" -v
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/storage/durable_dispatch.py werewolf_agent/storage/memory_store.py werewolf_agent/storage/sqlite_store.py werewolf_agent/storage/postgres_store.py tests/storage/test_durable_dispatch_protocol.py tests/storage/test_autonomous_commit.py tests/storage/test_postgres_autonomous_commit.py
```

Expected: all dispatch tests and Ruff checks pass.

- [x] **Step 5: Commit the turn-scoped dispatch query**

```bash
git add werewolf_agent/storage/durable_dispatch.py werewolf_agent/storage/memory_store.py werewolf_agent/storage/sqlite_store.py werewolf_agent/storage/postgres_store.py tests/storage
git commit -m "feat: query durable dispatches by turn"
```

### Task 4: Implement the In-Memory Turn Repository

**Files:**
- Modify: `werewolf_agent/storage/memory_store.py`
- Modify: `tests/storage/test_autonomous_turns.py`

- [x] **Step 1: Write failing in-memory lifecycle and CAS tests**

Add a repository fixture with `GameState(game_id="game-1")` and cover creation,
admission, transition, finish, replace, close, missing game, duplicate active
schedule, stale schedule version, stale turn version, and defensive reads:

```python
def test_memory_schedule_admits_only_current_slot() -> None:
    repository = _memory_repository()
    created = repository.create_serial_public_schedule(_schedule())
    managed = repository.admit_serial_public_turn(
        created.schedule_id,
        expected_schedule_version=0,
        admission=_admission(),
    )
    assert managed.turn.player_id == "p01"
    assert repository.load_serial_public_schedule(
        created.schedule_id,
    ).active_turn_id == managed.turn.turn_id


def test_memory_stale_admission_does_not_publish_partial_turn() -> None:
    repository = _memory_repository()
    repository.create_serial_public_schedule(_schedule())
    repository.admit_serial_public_turn("schedule-1", 0, _admission())

    with pytest.raises(ScheduleStateConflict):
        repository.admit_serial_public_turn(
            "schedule-1", 0, _admission(turn_id="turn-stale"),
        )
    assert repository.load_managed_turn("turn-stale") is None
```

- [x] **Step 2: Run memory tests and verify RED**

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_turns.py -k "memory" -v
```

Expected: `InMemoryGameRepository` lacks the autonomous-turn capability.

- [x] **Step 3: Add in-memory state and capability methods**

Initialize these maps under the existing `RLock`:

```python
self._serial_public_schedules: dict[str, SerialPublicSchedule] = {}
self._managed_agent_turns: dict[str, ManagedAgentTurn] = {}
self._active_schedule_by_game: dict[str, str] = {}
```

Implement `supports_autonomous_turns`, create/load/list methods, and use the
Task 2 pure helpers for admission and transition. Creation rejects a missing
game, a duplicate schedule ID, or an already-open schedule for the same game.

- [x] **Step 4: Implement atomic finish with prepare-then-publish**

Inside the lock, load both records and verify expected versions before calling
`prepare_active_finish`. Publish the updated turn and schedule only after all
checks pass:

```python
updated_schedule, updated_turn = prepare_active_finish(
    schedule,
    managed,
    terminal_status,
    disposition,
    reason_code=reason_code,
    now=datetime.now(timezone.utc),
)
self._managed_agent_turns[turn_id] = updated_turn
self._serial_public_schedules[schedule_id] = updated_schedule
if updated_schedule.status is SerialPublicScheduleStatus.OPEN:
    self._active_schedule_by_game[updated_schedule.game_id] = schedule_id
else:
    self._active_schedule_by_game.pop(updated_schedule.game_id, None)
return updated_schedule.model_copy(deep=True)
```

Update `delete_game` to remove the game's schedules and their managed turns.
Do not alter legacy save/load/event behavior.

- [x] **Step 5: Verify memory concurrency and rollback semantics**

Add a 20-worker duplicate-admission test: exactly one admission succeeds and
all failures are stable CAS or invalid-schedule errors; exactly one managed turn
exists. Run:

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_turns.py -k "memory" -v
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/storage/memory_store.py werewolf_agent/storage/autonomous_turns.py
```

Expected: memory tests pass and mypy reports no errors.

- [x] **Step 6: Commit the memory implementation**

```bash
git add werewolf_agent/storage/memory_store.py tests/storage/test_autonomous_turns.py
git commit -m "feat: persist serial public turns in memory"
```

### Task 5: Implement SQLite Turn Persistence

**Files:**
- Modify: `werewolf_agent/storage/sqlite_store.py`
- Modify: `tests/storage/test_autonomous_turns.py`
- Modify: `tests/storage/test_sqlite_migrations.py`

- [x] **Step 1: Write failing SQLite schema, lifecycle, and rollback tests**

Extend the shared repository fixture to SQLite. Assert fresh repository
initialization creates the two scheduling tables and indexes while
`MigrationManager` does not. Add forced turn-ID conflict rollback:

```python
def test_sqlite_admission_conflict_rolls_back_schedule_pointer(tmp_path) -> None:
    repository = _sqlite_repository(tmp_path)
    repository.create_serial_public_schedule(_schedule())
    repository._conn.execute(
        "INSERT INTO autonomous_managed_turns "
        "(turn_id, schedule_id, game_id, player_id, status, state_version, "
        "turn_json, terminal_reason, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "turn-1", "schedule-1", "game-1", "p09", "open", 0,
            _managed_json_for_conflict(), None, NOW.isoformat(), NOW.isoformat(),
        ),
    )
    repository._conn.commit()

    with pytest.raises(AutonomousTurnTransactionError):
        repository.admit_serial_public_turn("schedule-1", 0, _admission())

    stored = repository.load_serial_public_schedule("schedule-1")
    assert stored is not None
    assert stored.active_turn_id is None
    assert stored.state_version == 0
```

- [x] **Step 2: Run SQLite tests and verify RED**

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_turns.py tests/storage/test_sqlite_migrations.py -k "sqlite or scheduling" -v
```

Expected: scheduling tables and SQLite capability are missing.

- [x] **Step 3: Add isolated SQLite scheduling schema**

Create `_AUTONOMOUS_SCHEDULING_SCHEMA` and execute it in repository
initialization after the dispatch schema:

```sql
CREATE TABLE IF NOT EXISTS autonomous_serial_public_schedules (
    schedule_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    window_id TEXT NOT NULL,
    status TEXT NOT NULL,
    next_slot_ordinal INTEGER NOT NULL,
    active_turn_id TEXT,
    state_version INTEGER NOT NULL,
    schedule_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS autonomous_managed_turns (
    turn_id TEXT PRIMARY KEY,
    schedule_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    status TEXT NOT NULL,
    state_version INTEGER NOT NULL,
    turn_json TEXT NOT NULL,
    terminal_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (schedule_id)
        REFERENCES autonomous_serial_public_schedules(schedule_id)
        ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_open_serial_public_schedule
    ON autonomous_serial_public_schedules (game_id)
    WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_managed_turn_schedule_status
    ON autonomous_managed_turns (schedule_id, status);
```

Keep this schema out of `MigrationManager` and its legacy version numbering.

- [x] **Step 4: Implement SQLite serializers and read operations**

Serialize full strict models as canonical UTF-8 JSON with sorted keys and
compact separators. Rebuild through `model_validate_json`. Implement
capability, create, load by ID, load open by game, list open ordered by
`(created_at, schedule_id)`, and load managed turn. Capability returns true
only after schema initialization succeeds.

```python
@staticmethod
def _canonical_contract_json(value: StrictFrozenModel) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

def load_serial_public_schedule(
    self,
    schedule_id: str,
) -> SerialPublicSchedule | None:
    with self._lock:
        row = self._conn.execute(
            "SELECT schedule_json FROM autonomous_serial_public_schedules "
            "WHERE schedule_id = ?",
            (schedule_id,),
        ).fetchone()
        return (
            None
            if row is None
            else SerialPublicSchedule.model_validate_json(row[0])
        )
```

- [x] **Step 5: Implement SQLite CAS admission, transition, and finish**

Use `BEGIN IMMEDIATE` under `self._lock`. Load strict objects, validate
expected versions, call the Task 2 preparation helper, then update payload and
indexed columns with `WHERE state_version = ?`. Admission inserts the managed
turn before updating the schedule; finish updates both records before commit.
Any exception rolls back. Preserve stable not-found, state-conflict, admission,
and transition exceptions; wrap unexpected database failures in
`AutonomousTurnTransactionError`.

```python
with self._lock:
    try:
        self._conn.execute("BEGIN IMMEDIATE")
        schedule = self._load_schedule_unlocked(schedule_id)
        managed = self._load_managed_turn_unlocked(turn_id)
        if schedule is None:
            raise ScheduleNotFound(schedule_id)
        if managed is None:
            raise ManagedTurnNotFound(turn_id)
        if schedule.state_version != expected_schedule_version:
            raise ScheduleStateConflict(schedule_id)
        if managed.state_version != expected_turn_version:
            raise TurnStateConflict(turn_id)
        updated_schedule, updated_turn = prepare_active_finish(
            schedule, managed, terminal_status, disposition,
            reason_code=reason_code, now=datetime.now(timezone.utc),
        )
        self._update_managed_turn_unlocked(updated_turn, expected_turn_version)
        self._update_schedule_unlocked(updated_schedule, expected_schedule_version)
        self._conn.commit()
        return updated_schedule
    except (
        ScheduleNotFound,
        ManagedTurnNotFound,
        ScheduleStateConflict,
        TurnStateConflict,
        InvalidScheduleTransition,
        InvalidTurnAdmission,
    ):
        self._conn.rollback()
        raise
    except Exception as exc:
        self._conn.rollback()
        raise AutonomousTurnTransactionError(
            "autonomous turn finish transaction failed",
        ) from exc
```

- [x] **Step 6: Run SQLite lifecycle, concurrency, migration, and static checks**

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_turns.py tests/storage/test_sqlite_migrations.py -k "sqlite or scheduling" -v
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/storage/sqlite_store.py tests/storage/test_autonomous_turns.py tests/storage/test_sqlite_migrations.py
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/storage/sqlite_store.py
```

Expected: SQLite and migration tests pass; static checks report no errors.

- [x] **Step 7: Commit SQLite persistence**

```bash
git add werewolf_agent/storage/sqlite_store.py tests/storage/test_autonomous_turns.py tests/storage/test_sqlite_migrations.py
git commit -m "feat: persist serial public turns in sqlite"
```

### Task 6: Implement PostgreSQL Turn Persistence

**Files:**
- Modify: `werewolf_agent/storage/postgres_store.py`
- Modify: `tests/storage/test_postgres_autonomous_commit.py`

- [x] **Step 1: Write failing PostgreSQL DDL and capability tests**

Add assertions for both tables, JSONB, TIMESTAMPTZ, the partial unique open
schedule index, the managed-turn index, and unsupported capability without an
initialized connection:

```python
def test_postgres_schema_contains_autonomous_turn_tables() -> None:
    repository = _repository_without_connection()
    connection = _clean_schema_connection()
    repository._ensure_schema_transaction(connection)
    sql = " ".join(
        call.args[0].lower()
        for call in connection.execute.call_args_list
    )
    assert "autonomous_serial_public_schedules" in sql
    assert "autonomous_managed_turns" in sql
    assert "jsonb" in sql
    assert "where status = 'open'" in sql


def test_uninitialized_postgres_reports_autonomous_turns_unsupported() -> None:
    repository = _repository_without_connection()
    assert repository.supports_autonomous_turns() is False
```

- [x] **Step 2: Run PostgreSQL tests and verify RED**

```bash
conda run -n wofkill python -m pytest tests/storage/test_postgres_autonomous_commit.py -k "autonomous_turn or serial_public" -v
```

Expected: DDL and capability are absent.

- [x] **Step 3: Add PostgreSQL JSONB scheduling schema**

Add equivalent tables in `_ensure_schema_transaction`, using `BIGINT`,
`JSONB`, `TIMESTAMPTZ`, game/schedule foreign keys, and:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS uq_open_serial_public_schedule
ON autonomous_serial_public_schedules (game_id)
WHERE status = 'open'
```

Set no separate readiness flag; reuse the existing successful autonomous schema
initialization state.

- [x] **Step 4: Implement row decoders and read/capability methods**

Accept both psycopg dictionaries and JSON strings for JSONB payloads. Load
strict `SerialPublicSchedule` and `ManagedAgentTurn` objects, return defensive
validated copies, and order open schedules deterministically.

```python
@staticmethod
def _schedule_from_jsonb(payload: object) -> SerialPublicSchedule:
    if isinstance(payload, str):
        return SerialPublicSchedule.model_validate_json(payload)
    return SerialPublicSchedule.model_validate(payload)

def supports_autonomous_turns(self) -> bool:
    return self._conn is not None and bool(
        getattr(self, "_autonomous_schema_ready", False),
    )
```

- [x] **Step 5: Implement PostgreSQL CAS transactions**

For every mutation, acquire the existing game advisory transaction lock, select
the schedule and active turn `FOR UPDATE`, validate expected versions, run the
Task 2 pure helper, and update both JSONB payloads plus indexed columns. Verify
`rowcount == 1` for every CAS update. Roll back all failures and preserve stable
lifecycle exceptions; wrap unexpected driver failures.

```python
game_row = conn.execute(
    "SELECT game_id FROM autonomous_serial_public_schedules "
    "WHERE schedule_id = %s",
    (schedule_id,),
).fetchone()
if game_row is None:
    raise ScheduleNotFound(schedule_id)
conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (game_row[0],))
schedule_row = conn.execute(
    "SELECT schedule_json FROM autonomous_serial_public_schedules "
    "WHERE schedule_id = %s FOR UPDATE",
    (schedule_id,),
).fetchone()
turn_row = conn.execute(
    "SELECT turn_json FROM autonomous_managed_turns "
    "WHERE turn_id = %s FOR UPDATE",
    (turn_id,),
).fetchone()
updated_schedule, updated_turn = prepare_active_finish(
    self._schedule_from_jsonb(schedule_row[0]),
    self._managed_turn_from_jsonb(turn_row[0]),
    terminal_status,
    disposition,
    reason_code=reason_code,
    now=datetime.now(timezone.utc),
)
```

- [x] **Step 6: Add fake-connection atomicity tests**

Use a stateful fake connection that records schedule and managed-turn payloads.
Force the second write of `finish_active_turn` to fail and assert rollback
leaves both records unchanged. Also assert SQL contains `FOR UPDATE`, the
advisory lock, `state_version`, and `%s` placeholders.

- [x] **Step 7: Run PostgreSQL and adjacent storage checks**

```bash
conda run -n wofkill python -m pytest tests/storage/test_postgres_autonomous_commit.py tests/storage/test_postgres_store.py -q
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/storage/postgres_store.py tests/storage/test_postgres_autonomous_commit.py
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/storage/postgres_store.py
```

Expected: PostgreSQL tests pass without a live server; static checks pass.

- [x] **Step 8: Commit PostgreSQL persistence**

```bash
git add werewolf_agent/storage/postgres_store.py tests/storage/test_postgres_autonomous_commit.py
git commit -m "feat: persist serial public turns in postgres"
```

### Task 7: Implement SerialPublicScheduler and HostRuntime

**Files:**
- Create: `werewolf_agent/player_agents/runtime/__init__.py`
- Create: `werewolf_agent/player_agents/runtime/serial_public.py`
- Create: `werewolf_agent/player_agents/runtime/host.py`
- Create: `tests/player_agents/test_host_runtime.py`
- Create: `tests/player_agents/test_runtime_import_boundary.py`

- [x] **Step 1: Write failing isolated-runtime boundary test**

Scan every Python file below `werewolf_agent/player_agents/runtime` and reject
imports starting with:

```python
FORBIDDEN_PREFIXES = (
    "werewolf_agent.agents",
    "werewolf_agent.model_gateway",
    "werewolf_agent.runtime.agent_action_pipeline",
    "werewolf_agent.runtime.agent_adapter",
    "werewolf_agent.runtime.nodes",
    "werewolf_agent.runtime.strategy",
)
```

Assert the runtime package exists and has no violation.

- [x] **Step 2: Write failing HostRuntime lifecycle tests**

Cover fresh admission, restarted-runtime recovery requirement, pending recovery
block, unsafe-to-unknown recovery success, lifecycle transitions, committed
advance, cancellation, replacement, close, and deterministic expiry:

```python
def test_restarted_host_requires_recovery_before_admission() -> None:
    repository = _repository_with_schedule()
    restarted = _host(repository, resolver=PendingResolver())

    with pytest.raises(HostRecoveryRequired):
        restarted.admit_next_turn("schedule-1", _admission())


def test_complete_active_turn_advances_after_validating() -> None:
    repository, host, managed = _host_with_active_turn()
    managed = host.transition_active_turn(
        managed.turn.turn_id, managed.state_version, AgentTurnStatus.OBSERVING,
    )
    managed = host.transition_active_turn(
        managed.turn.turn_id, managed.state_version, AgentTurnStatus.THINKING,
    )
    managed = host.transition_active_turn(
        managed.turn.turn_id, managed.state_version, AgentTurnStatus.SUBMITTED,
    )
    managed = host.transition_active_turn(
        managed.turn.turn_id, managed.state_version, AgentTurnStatus.VALIDATING,
    )

    schedule = host.complete_active_turn("schedule-1")
    assert schedule.next_slot_ordinal == 1
    assert repository.load_managed_turn(
        managed.turn.turn_id,
    ).turn.status is AgentTurnStatus.COMMITTED
```

- [x] **Step 3: Run runtime tests and verify RED**

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_runtime_import_boundary.py tests/player_agents/test_host_runtime.py -v
```

Expected: runtime modules and `HostRuntime` do not exist.

- [x] **Step 4: Implement the narrow scheduler facade**

`SerialPublicScheduler` accepts only
`require_autonomous_turn_repository(repository)`. Its methods delegate
create/load/admit/transition/finish while keeping repository access private.
It loads repository state before admission and non-terminal transitions and
raises stable not-found or inactive-turn errors. Terminal operations instead
accept the schedule and turn IDs plus expected versions captured together by
`HostRuntime`; the facade must not reload a newer active turn and accidentally
apply an older completion or cancellation decision to it.

```python
class SerialPublicScheduler:
    def __init__(self, repository: AutonomousTurnRepository | object) -> None:
        self._repository = require_autonomous_turn_repository(repository)

    def admit_next_turn(
        self,
        schedule_id: str,
        admission: TurnAdmission,
    ) -> ManagedAgentTurn:
        schedule = self.require_schedule(schedule_id)
        return self._repository.admit_serial_public_turn(
            schedule_id,
            schedule.state_version,
            admission,
        )
```

- [x] **Step 5: Implement HostRuntime recovery and admission gates**

Define `HostRuntimeError`, `HostRecoveryRequired`, and
`HostRecoveryBlocked` with stable `code` values. The constructor validates
both capabilities and stores a set of recovered game IDs plus the latest
recovery report per game.

`create_schedule` persists through the scheduler and marks only that newly
created game as admitted in the current process. `recover_game` calls the
injected `DispatchReconciler`, then `assert_dispatch_allowed`; it adds the
game ID only when `report.barrier_open` is true and the assertion passes.

Before admit, transition, complete, cancel, or expire, require the game ID in
the recovered set and call `assert_dispatch_allowed` when the operation starts
new work.

```python
def recover_game(self, game_id: str) -> RecoveryReport:
    report = self._reconciler.reconcile_game(game_id)
    self._recovery_reports[game_id] = report
    if report.barrier_open:
        try:
            self._dispatch_repository.assert_dispatch_allowed(game_id)
        except DispatchRecoveryBlocked:
            self._recovered_games.discard(game_id)
        else:
            self._recovered_games.add(game_id)
    else:
        self._recovered_games.discard(game_id)
    return report

def _require_recovered(self, game_id: str) -> None:
    if game_id in self._recovered_games:
        return
    report = self._recovery_reports.get(game_id)
    if report is not None and not report.barrier_open:
        raise HostRecoveryBlocked(game_id)
    raise HostRecoveryRequired(game_id)
```

- [x] **Step 6: Implement completion, cancellation, and expiry**

`complete_active_turn` loads the schedule and managed turn, then calls finish
with `COMMITTED`, `ADVANCE`, and no reason.

Cancellation queries
`list_dispatches_for_turn(game_id, turn_id)`, CAS-cancels only `PENDING` and
`DISPATCHING` attempts, leaves later statuses durable, then finishes with
`CANCELLED` and the requested disposition.

`expire_due_turns(now)` requires an aware `now`, scans
`list_open_serial_public_schedules()`, skips schedules with no active turn,
and for recovered games whose active deadline is at or before `now`, runs the
same dispatch cancellation followed by `EXPIRED` and `ADVANCE`. Return
changed schedules in deterministic game/schedule order.

```python
def _cancel_turn_dispatches(
    self,
    managed: ManagedAgentTurn,
    reason_code: str,
) -> None:
    attempts = self._dispatch_repository.list_dispatches_for_turn(
        managed.turn.game_id,
        managed.turn.turn_id,
    )
    for attempt in attempts:
        if attempt.status in {DispatchStatus.PENDING, DispatchStatus.DISPATCHING}:
            self._dispatch_repository.cancel_dispatch(
                attempt.dispatch_id,
                attempt.state_version,
                reason_code,
            )
```

- [x] **Step 7: Verify recovery and late-result behavior**

Create a dispatched attempt, cancel the active turn, assert the attempt remains
recoverable and a new admission is blocked. Reconcile it to
`UNKNOWN_OUTCOME`, then assert the barrier opens and later admission may
continue without reusing the cancelled turn ID or idempotency key.

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_host_runtime.py tests/player_agents/test_runtime_import_boundary.py tests/storage/test_durable_dispatch_protocol.py -v
```

Expected: all runtime and recovery tests pass.

- [x] **Step 8: Run runtime static checks and commit**

```bash
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/player_agents/runtime tests/player_agents/test_host_runtime.py tests/player_agents/test_runtime_import_boundary.py
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/player_agents/runtime
git add werewolf_agent/player_agents/runtime tests/player_agents/test_host_runtime.py tests/player_agents/test_runtime_import_boundary.py
git commit -m "feat: add serial public host runtime"
```

Expected: Ruff and the scoped runtime mypy check pass, followed by a successful
commit. Repository-wide dependency checking remains part of Task 8.

### Task 8: Run Cross-Backend and Repository-Wide Verification

**Files:**
- Modify: `tests/storage/test_autonomous_turns.py`
- Modify: `docs/superpowers/plans/2026-07-30-autonomous-player-serial-public-host-runtime.md`

- [x] **Step 1: Add the final shared atomicity matrix**

Parameterize memory and SQLite for: schedule creation, current-slot admission,
stale CAS, every legal non-terminal edge used by HostRuntime, advance, replace,
close, final-slot closure, 20 concurrent duplicate admissions, deletion
cleanup, and forced rollback. Assert schedule and managed-turn objects are
byte-equivalent after canonical JSON serialization where backend timestamps are
fixed by fixtures.

```python
@pytest.mark.parametrize("repository_kind", ("memory", "sqlite"))
def test_atomic_finish_matrix(repository_kind, tmp_path) -> None:
    repository = _repository(repository_kind, tmp_path)
    schedule, managed = _admit_to_validating(repository)
    updated = repository.finish_active_turn(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        AgentTurnStatus.COMMITTED,
        TerminalDisposition.ADVANCE,
        None,
    )
    stored = repository.load_managed_turn(managed.turn.turn_id)
    assert stored is not None
    assert stored.turn.status is AgentTurnStatus.COMMITTED
    assert updated.active_turn_id is None
    assert updated.next_slot_ordinal == 1
```

- [x] **Step 2: Run focused player-agent and storage suites**

```bash
conda run -n wofkill python -m pytest tests/player_agents tests/storage/test_autonomous_turns.py tests/storage/test_durable_dispatch_protocol.py tests/storage/test_autonomous_commit.py tests/storage/test_sqlite_migrations.py tests/storage/test_postgres_autonomous_commit.py -v
```

Expected: all focused and adjacent tests pass.

- [x] **Step 3: Run scoped Ruff and mypy**

```bash
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/player_agents werewolf_agent/storage/autonomous_turns.py werewolf_agent/storage/durable_dispatch.py werewolf_agent/storage/memory_store.py werewolf_agent/storage/sqlite_store.py werewolf_agent/storage/postgres_store.py tests/player_agents tests/storage/test_autonomous_turns.py tests/storage/test_durable_dispatch_protocol.py tests/storage/test_postgres_autonomous_commit.py tests/storage/test_sqlite_migrations.py
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/player_agents werewolf_agent/storage/autonomous_turns.py werewolf_agent/storage/durable_dispatch.py werewolf_agent/storage/memory_store.py werewolf_agent/storage/sqlite_store.py werewolf_agent/storage/postgres_store.py
```

Expected: both commands report no errors.

- [x] **Step 4: Verify the live-runtime boundary**

Run:

```bash
rg -n "HostRuntime|SerialPublicScheduler|AutonomousTurnRepository" werewolf_agent/runtime werewolf_agent/agents werewolf_agent/model_gateway
```

Expected: no output. This stage must not be wired into the old runtime.

- [x] **Step 5: Run the full test suite and diff checks**

```bash
conda run -n wofkill python -m pytest -q
git diff --check
git status --short
```

Expected: the full suite passes with only the repository's existing warnings;
`git diff --check` is silent; status lists only this plan's intended files.

- [x] **Step 6: Mark progress complete and commit the integrated stage**

Change this plan's progress to `52/52` and every completed checkbox to
`[x]`, then run:

```bash
git add werewolf_agent/player_agents werewolf_agent/storage tests/player_agents tests/storage docs/superpowers/plans/2026-07-30-autonomous-player-serial-public-host-runtime.md
git commit -m "feat: add autonomous player serial public host runtime"
```

### Post-Review Fixes

- [x] **Fix 1: Preserve recovery qualification when new dispatch work is blocked**

  Keep successful recovery/current-process qualification separate from the
  transient `assert_dispatch_allowed` result. A blocked transition or admission
  must not make later cancellation or expiry behave as though recovery never
  completed; a genuinely unrecovered restarted game must remain blocked.

- [x] **Fix 2: Require a fresh initial schedule at creation**

  All three backends must reject schedule creation unless it starts `open`, at
  slot zero, with no active turn and state version zero. Final-slot coverage
  must reach the last slot through normal admission/advance rather than creating
  a pre-advanced schedule.

- [x] **Fix 3: Enforce fresh replacement idempotency keys durably**

  A replacement turn must not reuse an idempotency key already stored for the
  same schedule. Memory, SQLite, and PostgreSQL must enforce equivalent atomic
  behavior and expose the same stable admission error under duplicates.

## Completion Criteria

- Strict schedules persist speaker slots but create each `AgentTurn` only when
  its slot becomes current.
- Memory, SQLite, and PostgreSQL expose explicit and equivalent autonomous-turn
  capabilities.
- Every mutation uses CAS; terminal turn and schedule advancement are atomic.
- Restarted persisted schedules cannot perform work before dispatch
  reconciliation opens the barrier.
- Cancellation and expiry cancel cancellable dispatch attempts and never reuse
  a terminal turn's ID, observation, or idempotency key.
- `UNKNOWN_OUTCOME` remains terminal and budget-consuming without permanently
  blocking recovery.
- No model, tool, workspace, RuleEngine, legacy player, feature gate, or live
  game path is connected by this plan.
- Focused tests, scoped Ruff/mypy, the full pytest suite, and import-boundary
  scans all pass.
