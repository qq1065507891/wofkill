# Autonomous Player Durable Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不接入旧 runtime 的前提下，为自主玩家 model/tool 请求提供可恢复、可幂等、带 CAS 状态迁移的 durable dispatch 能力。

**Architecture:** 新增严格的 `DispatchAttempt`/`DispatchResultRecord` contract 和独立 `DurableDispatchRepository` capability。内存、SQLite、PostgreSQL 实现相同的状态机；`DispatchReconciler` 只依赖 capability 与 fake/provider resolver，在重启后处理未决请求和 per-game recovery barrier。整个阶段不修改 `GameRepository`、`ModelRouter`、`_dispatch_agent`、scheduler 或 `HostRuntime`。

**Tech Stack:** Python 3.12, Pydantic v2, dataclasses, SQLite WAL/transactions, PostgreSQL JSONB/row locks, pytest, ruff, mypy; all Python commands use `conda run -n wofkill`.

---

## File Map

- Create: `werewolf_agent/player_agents/contracts/dispatch.py` — strict dispatch attempt/result models, enums, JSON payload freezing, result dispositions.
- Modify: `werewolf_agent/player_agents/contracts/_base.py` — move the existing deep JSON freeze/thaw helpers to a shared internal location.
- Modify: `werewolf_agent/player_agents/contracts/transactions.py` — reuse shared JSON helpers without changing public transaction behavior.
- Modify: `werewolf_agent/player_agents/contracts/__init__.py` — export dispatch contract names.
- Create: `werewolf_agent/storage/durable_dispatch.py` — capability protocol, stable storage errors, resolver protocol, reconciliation report, and reconciler.
- Modify: `werewolf_agent/storage/memory_store.py` — in-memory dispatch tables and atomic CAS transitions.
- Modify: `werewolf_agent/storage/sqlite_store.py` — independent dispatch schema initialization and SQLite transactions.
- Modify: `werewolf_agent/storage/postgres_store.py` — PostgreSQL dispatch schema and row-locked transitions.
- Create: `tests/player_agents/test_dispatch_contracts.py` — strict model and JSON round-trip tests.
- Create: `tests/storage/test_durable_dispatch_protocol.py` — capability guard, resolver, reconciliation, and stable error tests.
- Modify: `tests/storage/test_autonomous_commit.py` — shared in-memory/SQLite dispatch behavior and recovery tests.
- Modify: `tests/storage/test_sqlite_migrations.py` — assert legacy migrations remain unchanged while repository initialization adds dispatch tables.
- Modify: `tests/storage/test_postgres_autonomous_commit.py` — PostgreSQL DDL and transition mocks.

## Task 1: Add Strict Dispatch Contracts

**Files:**
- Create: `werewolf_agent/player_agents/contracts/dispatch.py`
- Modify: `werewolf_agent/player_agents/contracts/_base.py`
- Modify: `werewolf_agent/player_agents/contracts/transactions.py`
- Modify: `werewolf_agent/player_agents/contracts/__init__.py`
- Test: `tests/player_agents/test_dispatch_contracts.py`

- [ ] **Step 1: Write failing contract tests**

Create a fixture using timezone-aware datetimes and 64-character hashes:

```python
from datetime import datetime, timezone

from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.dispatch import (
    DispatchAttempt,
    DispatchOperationKind,
    DispatchRecoveryPolicy,
    DispatchResultRecord,
    DispatchResultOutcome,
    DispatchStatus,
)

HASH = "a" * 64


def _attempt(**updates: object) -> DispatchAttempt:
    data: dict[str, object] = {
        "dispatch_id": "dispatch-1",
        "game_id": "game-1",
        "turn_id": "turn-1",
        "actor_id": "p01",
        "operation_kind": DispatchOperationKind.MODEL,
        "executor_id": "mock-provider",
        "provider_idempotency_key": "provider-key-1",
        "recovery_policy": DispatchRecoveryPolicy.IDEMPOTENT_LOOKUP_OR_REISSUE,
        "request_hash": HASH,
        "lease_hash": HASH,
        "view_fingerprint": HASH,
        "deadline": datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
        "created_at": datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        "status": DispatchStatus.PENDING,
        "state_version": 0,
    }
    data.update(updates)
    return DispatchAttempt.model_validate(data)


def _result(**updates: object) -> DispatchResultRecord:
    data: dict[str, object] = {
        "result_id": "result-1",
        "dispatch_id": "dispatch-1",
        "request_hash": HASH,
        "lease_hash": HASH,
        "result_hash": HASH,
        "result_kind": "model_response",
        "outcome": DispatchResultOutcome.SUCCESS,
        "payload": {"accepted": True},
        "recorded_at": datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    }
    data.update(updates)
    return DispatchResultRecord.model_validate(data)


def test_dispatch_attempt_is_strict_frozen_and_json_round_trips() -> None:
    attempt = _attempt()
    restored = DispatchAttempt.model_validate_json(attempt.model_dump_json())
    assert restored == attempt
    with pytest.raises(ValidationError):
        DispatchAttempt.model_validate({**attempt.model_dump(), "state_version": "0"})
    with pytest.raises(ValidationError):
        attempt.state_version = 1


def test_dispatch_attempt_rejects_naive_deadline_and_invalid_hash() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _attempt(deadline=datetime(2026, 7, 29, 12))
    with pytest.raises(ValidationError):
        _attempt(request_hash="short")


def test_dispatch_result_payload_is_deeply_immutable() -> None:
    result = DispatchResultRecord(
        result_id="result-1",
        dispatch_id="dispatch-1",
        request_hash=HASH,
        lease_hash=HASH,
        result_hash=HASH,
        result_kind="model_response",
        outcome=DispatchResultOutcome.SUCCESS,
        payload={"nested": [{"safe": True}]},
        recorded_at=datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    )
    with pytest.raises(TypeError):
        result.payload["nested"] = []  # type: ignore[index]
    assert result.payload["nested"] == ({"safe": True},)

## Task 2: Add the Capability Protocol and Reconciler

**Files:**
- Create: werewolf_agent/storage/durable_dispatch.py
- Test: tests/storage/test_durable_dispatch_protocol.py

- [ ] **Step 1: Write failing protocol and recovery tests**

Add a minimal fake repository and resolver in the test module. Verify that a
plain object is rejected by the capability guard, that a resolver can return a
typed found result, and that the reconciler produces a barrier report:

~~~python
def test_capability_guard_requires_explicit_support() -> None:
    with pytest.raises(DurableDispatchUnsupported):
        require_durable_dispatch_repository(object())


def test_reconciler_marks_non_idempotent_attempt_unknown() -> None:
    store = InMemoryDispatchFixture([_attempt(
        recovery_policy=DispatchRecoveryPolicy.AT_MOST_ONCE_UNKNOWN,
        status=DispatchStatus.DISPATCHED,
        state_version=2,
    )])
    report = DispatchReconciler(store, resolver=NeverCalledResolver()).reconcile_game("game-1")
    assert report.unknown == 1
    assert store.load_dispatch("dispatch-1").status is DispatchStatus.UNKNOWN_OUTCOME
    assert report.barrier_open is True


def test_reconciler_records_found_idempotent_result_without_new_dispatch_id() -> None:
    store = InMemoryDispatchFixture([_attempt(
        status=DispatchStatus.DISPATCHED,
        state_version=2,
    )])
    resolver = FoundResolver(_result())
    report = DispatchReconciler(store, resolver=resolver).reconcile_game("game-1")
    assert report.resolved == 1
    assert store.load_dispatch("dispatch-1").status is DispatchStatus.RESULT_RECORDED
    assert resolver.seen_keys == ["provider-key-1"]


def test_reconciler_leaves_pending_provider_and_keeps_barrier_closed() -> None:
    store = InMemoryDispatchFixture([_attempt(
        status=DispatchStatus.DISPATCHING,
        state_version=1,
    )])
    report = DispatchReconciler(store, resolver=PendingResolver()).reconcile_game("game-1")
    assert report.pending == 1
    assert report.barrier_open is False
    with pytest.raises(DispatchRecoveryBlocked):
        store.assert_dispatch_allowed("game-1")
~~~

The fixture may implement only the protocol methods required by the tests; it
must preserve state_version and return defensive model copies.

- [ ] **Step 2: Run the protocol tests and verify RED**

Run:

~~~bash
conda run -n wofkill python -m pytest -q tests/storage/test_durable_dispatch_protocol.py
~~~

Expected: collection fails because the capability protocol, errors, resolver,
and reconciler do not exist.

- [ ] **Step 3: Implement stable errors and capability guard**

Define DurableDispatchUnsupported, DispatchNotFound,
DispatchStateConflict, DispatchInvalidTransition,
DispatchIdempotencyConflict, DispatchLeaseMismatch,
DispatchResultConflict, DispatchRecoveryBlocked, and DispatchTransactionError.
Each error exposes a stable code string. Add:

~~~python
def require_durable_dispatch_repository(repository: object) -> DurableDispatchRepository:
    supports = getattr(repository, "supports_durable_dispatch", None)
    if not callable(supports) or not supports():
        raise DurableDispatchUnsupported("repository does not support durable dispatch")
    return cast(DurableDispatchRepository, repository)
~~~

The guard must never infer support from the presence of create_dispatch or
commit_turn.

- [ ] **Step 4: Implement resolver types and DispatchReconciler**

Use immutable dataclasses for resolver input/output:

~~~python
class RecoveryResolutionKind(StrEnum):
    FOUND = "found"
    REISSUED = "reissued"
    PENDING = "pending"
    UNAVAILABLE = "unavailable"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class RecoveryResolution:
    kind: RecoveryResolutionKind
    result: DispatchResultRecord | None = None
    reason_code: str = ""


class DispatchResolver(Protocol):
    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
        pass
~~~

DispatchReconciler.reconcile_game must sort attempts by
(created_at, dispatch_id), mark non-idempotent attempts unknown without
calling the resolver, record FOUND results with the current state version,
leave PENDING/UNAVAILABLE unresolved, and mark UNSAFE unknown. It must return
a frozen RecoveryReport with resolved, unknown, pending, errors,
budget_consumption_required, and barrier_open counts. A resolver cannot mutate
an attempt or create a dispatch ID.

- [ ] **Step 5: Run focused protocol tests and commit**

~~~bash
conda run -n wofkill python -m pytest -q tests/storage/test_durable_dispatch_protocol.py
conda run -n wofkill python -m ruff check --ignore UP009 tests/storage/test_durable_dispatch_protocol.py werewolf_agent/storage/durable_dispatch.py
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/storage/durable_dispatch.py
git add werewolf_agent/storage/durable_dispatch.py tests/storage/test_durable_dispatch_protocol.py
git commit -m "feat: add durable dispatch capability protocol"
~~~

## Task 3: Implement In-Memory Dispatch Storage

**Files:**
- Modify: werewolf_agent/storage/memory_store.py
- Modify: tests/storage/test_autonomous_commit.py

- [ ] **Step 1: Add failing in-memory transition tests**

Add tests that create a game, insert a PENDING attempt, and assert every valid
transition increments state_version. Add these negative cases:

~~~python
with pytest.raises(DispatchInvalidTransition):
    repository.mark_dispatched("dispatch-1", expected_version=0)
with pytest.raises(DispatchStateConflict):
    repository.mark_dispatching("dispatch-1", expected_version=99)
with pytest.raises(DispatchIdempotencyConflict):
    repository.create_dispatch(_attempt(provider_idempotency_key="provider-key-1"))
~~~

Also test first result, same-result replay, different-result conflict,
cancel_dispatch, mark_unknown_outcome, and DISCARDED_LATE after each terminal
status. Test assert_dispatch_allowed rejects a game containing a DISPATCHING
or DISPATCHED attempt.

- [ ] **Step 2: Run the in-memory tests and verify RED**

~~~bash
conda run -n wofkill python -m pytest -q tests/storage/test_autonomous_commit.py -k dispatch
~~~

Expected: InMemoryGameRepository has no durable dispatch methods.

- [ ] **Step 3: Add isolated in-memory state**

Initialize these dictionaries beside the existing autonomous commit state:

~~~python
self._dispatch_attempts: dict[str, DispatchAttempt] = {}
self._dispatch_results: dict[str, DispatchResultRecord] = {}
self._dispatch_key_index: dict[tuple[str, str], str] = {}
~~~

Keep all durable dispatch methods under the existing RLock. Include dispatch
state in delete_game cleanup and return model_copy(deep=True) from every read
method.

- [ ] **Step 4: Implement atomic transition helpers**

Implement a private _transition_dispatch that:

1. loads the ID or raises DispatchNotFound;
2. verifies state_version == expected_version;
3. checks the exact allowed source status;
4. updates status, updated_at, reason_code, and increments version;
5. stores the new frozen model and returns a defensive copy.

Wrap multi-container writes in a full snapshot/restore block like the existing
commit_turn rollback path. record_result must first validate dispatch ID,
request hash, lease hash, and result dispatch ID; it then writes result and
attempt together. Repeating an identical result returns REPLAYED without a
version increment; a different result raises DispatchResultConflict. Before
creating a new attempt, reject the game with DispatchRecoveryBlocked when any
DISPATCHING or DISPATCHED attempt is present.

- [ ] **Step 5: Verify in-memory behavior and commit**

~~~bash
conda run -n wofkill python -m pytest -q tests/storage/test_autonomous_commit.py -k dispatch
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/storage/memory_store.py tests/storage/test_autonomous_commit.py
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/storage/memory_store.py
git add werewolf_agent/storage/memory_store.py tests/storage/test_autonomous_commit.py
git commit -m "feat: implement in-memory durable dispatch"
~~~

## Task 4: Add SQLite Dispatch Schema and Transactions

**Files:**
- Modify: werewolf_agent/storage/sqlite_store.py
- Modify: tests/storage/test_sqlite_migrations.py
- Modify: tests/storage/test_autonomous_commit.py

- [ ] **Step 1: Add failing schema/restart tests**

Add a fresh-schema test asserting both tables and the (game_id, status,
created_at) index. Create a database with only legacy games/events, open it
through SqliteGameRepository, insert an attempt, close it, reopen it, and
assert list_recoverable_dispatches("game-1") returns the same attempt. Assert
the existing MigrationManager still creates no autonomous dispatch tables.

- [ ] **Step 2: Run the SQLite tests and verify RED**

~~~bash
conda run -n wofkill python -m pytest -q tests/storage/test_sqlite_migrations.py tests/storage/test_autonomous_commit.py -k dispatch
~~~

Expected: dispatch tables and repository methods are missing.

- [ ] **Step 3: Add idempotent SQLite schema outside MigrationManager**

Define _AUTONOMOUS_DISPATCH_SCHEMA with CREATE TABLE IF NOT EXISTS for:

~~~sql
CREATE TABLE IF NOT EXISTS autonomous_dispatch_attempts (
    dispatch_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    operation_kind TEXT NOT NULL,
    executor_id TEXT NOT NULL,
    provider_idempotency_key TEXT NOT NULL,
    recovery_policy TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    lease_hash TEXT NOT NULL,
    view_fingerprint TEXT NOT NULL,
    deadline TEXT NOT NULL,
    status TEXT NOT NULL,
    state_version INTEGER NOT NULL,
    reason_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS autonomous_dispatch_results (
    result_id TEXT PRIMARY KEY,
    dispatch_id TEXT NOT NULL UNIQUE,
    request_hash TEXT NOT NULL,
    lease_hash TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    result_kind TEXT NOT NULL,
    outcome TEXT NOT NULL,
    result_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (dispatch_id)
        REFERENCES autonomous_dispatch_attempts(dispatch_id)
        ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dispatch_executor_key
    ON autonomous_dispatch_attempts (executor_id, provider_idempotency_key);
CREATE INDEX IF NOT EXISTS idx_dispatch_game_status_created
    ON autonomous_dispatch_attempts (game_id, status, created_at);
~~~

Use TEXT for JSON and timestamps, INTEGER for state_version, and foreign keys
to games(game_id) with ON DELETE CASCADE. Run this schema after the existing
event schema upgrade and before the existing sequence integrity check; do not
alter MIGRATIONS or its version numbers.

- [ ] **Step 4: Implement SQLite CAS transitions**

Use the existing repository lock and BEGIN IMMEDIATE. Every update must
include the expected state version:

~~~sql
UPDATE autonomous_dispatch_attempts
SET status = ?, state_version = state_version + 1,
    reason_code = ?, updated_at = ?
WHERE dispatch_id = ? AND state_version = ? AND status = ?
~~~

Check rowcount == 1; otherwise reload the row and raise either
DispatchStateConflict or DispatchInvalidTransition. Insert result and
transition the attempt in one transaction. On any exception rollback and wrap
unexpected SQLite errors as DispatchTransactionError while preserving stable
dispatch errors. The create operation must query for existing
DISPATCHING/DISPATCHED rows for the same game and raise
DispatchRecoveryBlocked before inserting a new attempt.

- [ ] **Step 5: Verify SQLite behavior and commit**

~~~bash
conda run -n wofkill python -m pytest -q tests/storage/test_sqlite_migrations.py tests/storage/test_autonomous_commit.py -k dispatch
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/storage/sqlite_store.py tests/storage/test_sqlite_migrations.py tests/storage/test_autonomous_commit.py
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/storage/sqlite_store.py
git add werewolf_agent/storage/sqlite_store.py tests/storage/test_sqlite_migrations.py tests/storage/test_autonomous_commit.py
git commit -m "feat: implement sqlite durable dispatch"
~~~

## Task 5: Implement PostgreSQL Dispatch Storage

**Files:**
- Modify: werewolf_agent/storage/postgres_store.py
- Modify: tests/storage/test_postgres_autonomous_commit.py

- [ ] **Step 1: Add failing PostgreSQL contract tests**

Extend the existing mock connection to recognize dispatch DDL and row-lock
queries. Assert supports_durable_dispatch() is false before schema setup and
that _ensure_schema_transaction contains both dispatch tables, the executor
key unique index, and the game/status/created index.

- [ ] **Step 2: Run the PostgreSQL tests and verify RED**

~~~bash
conda run -n wofkill python -m pytest -q tests/storage/test_postgres_autonomous_commit.py -k dispatch
~~~

Expected: the repository has no dispatch capability or dispatch schema.

- [ ] **Step 3: Add PostgreSQL DDL to the existing schema transaction**

Create the same two tables with JSONB result payloads, TIMESTAMPTZ timestamps,
BIGINT state versions, foreign keys to games, and the two indexes. Keep
_autonomous_schema_ready false if any DDL fails and preserve the existing
rollback/retry behavior.

- [ ] **Step 4: Implement row-locked state transitions**

For every operation, acquire the game and dispatch rows with SELECT FOR UPDATE,
validate state and hashes in Python, update with expected state version, and
commit only after a result insert and attempt transition have both succeeded.
Use the same stable errors and DispatchResultDisposition as the memory and
SQLite implementations. A missing psycopg connection must return the explicit
unsupported capability rather than opening a connection during a capability
check. The create operation must lock the game row, query for existing
DISPATCHING/DISPATCHED rows, and raise DispatchRecoveryBlocked before inserting
a new attempt.

- [ ] **Step 5: Verify PostgreSQL mocks and commit**

~~~bash
conda run -n wofkill python -m pytest -q tests/storage/test_postgres_autonomous_commit.py -k dispatch
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/storage/postgres_store.py tests/storage/test_postgres_autonomous_commit.py
conda run -n wofkill python -m mypy --follow-imports=skip werewolf_agent/storage/postgres_store.py
git add werewolf_agent/storage/postgres_store.py tests/storage/test_postgres_autonomous_commit.py
git commit -m "feat: implement postgres durable dispatch"
~~~

## Task 6: Cross-Backend Recovery and Fault Injection

**Files:**
- Modify: tests/storage/test_autonomous_commit.py
- Modify: tests/storage/test_postgres_autonomous_commit.py
- Modify: tests/player_agents/test_dispatch_contracts.py

- [ ] **Step 1: Add the shared backend matrix**

Parameterize the memory and SQLite repositories with the same _attempt and
_result fixtures. For each backend, verify:

~~~python
repository.create_dispatch(attempt)
dispatching = repository.mark_dispatching("dispatch-1", expected_version=0)
dispatched = repository.mark_dispatched("dispatch-1", expected_version=1)
assert dispatched.state_version == 2
assert repository.record_result("dispatch-1", 2, result) is DispatchResultDisposition.RECORDED
assert repository.record_result("dispatch-1", 3, result) is DispatchResultDisposition.REPLAYED
~~~

Use a fresh SQLite repository after close/reopen to simulate a process restart;
use the memory repository's retained dictionaries to simulate an in-process
crash boundary.

- [ ] **Step 2: Add recovery fault injection**

Cover these exact cases for memory and SQLite:

1. attempt left in DISPATCHING is discovered after restart;
2. FOUND records a result and does not create another attempt;
3. REISSUED reuses the original provider key and remains DISPATCHED;
4. PENDING and UNAVAILABLE leave the barrier closed;
5. AT_MOST_ONCE_UNKNOWN never calls the resolver and becomes unknown;
6. late result after cancel and late result after unknown both return
   DISCARDED_LATE;
7. a mismatched lease or request hash raises the stable binding error and does
   not mutate the attempt or result table.

- [ ] **Step 3: Add concurrency and rollback injection**

Use ThreadPoolExecutor(max_workers=10) to race two mark_dispatching calls with
the same expected version and assert exactly one succeeds. Replace the memory
result dictionary with a failing mapping and inject a SQLite insert failure;
assert no partial result and no state-version increment remain.

- [ ] **Step 4: Run focused cross-backend checks and commit**

~~~bash
conda run -n wofkill python -m pytest -q tests/player_agents/test_dispatch_contracts.py tests/storage/test_durable_dispatch_protocol.py tests/storage/test_autonomous_commit.py tests/storage/test_postgres_autonomous_commit.py
conda run -n wofkill python -m ruff check --ignore UP009 tests/player_agents/test_dispatch_contracts.py tests/storage/test_durable_dispatch_protocol.py tests/storage/test_autonomous_commit.py tests/storage/test_postgres_autonomous_commit.py
git add tests/player_agents/test_dispatch_contracts.py tests/storage/test_durable_dispatch_protocol.py tests/storage/test_autonomous_commit.py tests/storage/test_postgres_autonomous_commit.py
git commit -m "test: verify durable dispatch recovery semantics"
~~~

## Task 7: Final Verification and Review Gate

**Files:**
- No production files beyond Tasks 1-6.
- Review all changed files against docs/superpowers/specs/2026-07-29-autonomous-player-durable-dispatch-design.md.

- [ ] **Step 1: Run the full test suite**

~~~bash
conda run -n wofkill python -m pytest -q
~~~

Expected: exit code 0 with only the repository's known warnings/skips.

- [ ] **Step 2: Run scoped Ruff and mypy**

~~~bash
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents/contracts/_base.py \
  werewolf_agent/player_agents/contracts/dispatch.py \
  werewolf_agent/player_agents/contracts/transactions.py \
  werewolf_agent/storage/durable_dispatch.py \
  werewolf_agent/storage/memory_store.py \
  werewolf_agent/storage/sqlite_store.py \
  werewolf_agent/storage/postgres_store.py \
  tests/player_agents/test_dispatch_contracts.py \
  tests/storage/test_durable_dispatch_protocol.py \
  tests/storage/test_autonomous_commit.py \
  tests/storage/test_sqlite_migrations.py \
  tests/storage/test_postgres_autonomous_commit.py

conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/player_agents/contracts/_base.py \
  werewolf_agent/player_agents/contracts/dispatch.py \
  werewolf_agent/player_agents/contracts/transactions.py \
  werewolf_agent/storage/durable_dispatch.py \
  werewolf_agent/storage/memory_store.py \
  werewolf_agent/storage/sqlite_store.py \
  werewolf_agent/storage/postgres_store.py
~~~

Expected: Ruff and isolated mypy pass for all changed implementation and test
files. Existing unrelated full-package mypy errors must be reported rather
than broadened into this stage.

- [ ] **Step 3: Verify the runtime boundary**

Run:

~~~bash
rg -n "DurableDispatch|DispatchReconciler|DispatchAttempt" werewolf_agent/runtime werewolf_agent/agents
~~~

Expected: no new live runtime import or call site. Only the new contracts,
storage modules, and focused tests may reference the capability.

- [ ] **Step 4: Request code review and inspect the worktree**

~~~bash
git diff --check
git status --short
git log --oneline --decorate -8
~~~

The feature worktree must be clean, all task commits must be present, and the
review must confirm no scheduler, HostRuntime, ModelRouter, or feature-gate
integration slipped into this stage.

- [ ] **Step 5: Commit only documentation/checklist updates if needed**

~~~bash
git add docs/superpowers/plans/2026-07-29-autonomous-player-durable-dispatch.md
git commit -m "docs: track durable dispatch implementation plan"
~~~

No runtime code is considered complete until the full test, scoped static
checks, boundary scan, and independent review all pass.
