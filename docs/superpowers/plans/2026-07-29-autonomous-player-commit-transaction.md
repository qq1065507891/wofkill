# Autonomous Player Commit Transaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变旧运行时的前提下，为内存、SQLite、PostgreSQL 仓储提供一致的自主玩家 `CommitTurn` 原子提交能力。

**Architecture:** 新增严格的事务契约和 capability protocol，仓储实现各自的原子写入；旧 `GameRepository` 方法保留不变。SQLite 和 PostgreSQL 共享相同的 JSON 记录结构，内存仓储使用同一套哈希、revision、幂等和回滚辅助逻辑。

**Tech Stack:** Python 3.12, Pydantic v2, dataclasses, SQLite WAL transactions, PostgreSQL JSONB/advisory locks, pytest, ruff, mypy; all Python commands use `conda run -n wofkill`.

---

## File Map

- Create: `werewolf_agent/player_agents/contracts/transactions.py` — strict request/result, event candidate, audit and outbox records.
- Modify: `werewolf_agent/player_agents/contracts/__init__.py` — export transaction contracts.
- Create: `werewolf_agent/storage/autonomous_commit.py` — capability protocol, stable errors, canonical request hashing and event/result helpers.
- Modify: `werewolf_agent/storage/memory_store.py` — in-memory autonomous commit implementation.
- Modify: `werewolf_agent/storage/sqlite_store.py` — autonomous tables, initialization and SQLite transaction implementation.
- Modify: `werewolf_agent/storage/postgres_store.py` — PostgreSQL schema and transaction implementation.
- Create: `tests/player_agents/test_transaction_contracts.py` — contract strictness and hash tests.
- Create: `tests/storage/test_autonomous_commit.py` — shared in-memory/SQLite behavior and rollback suite.
- Create: `tests/storage/test_postgres_autonomous_commit.py` — PostgreSQL capability and SQL/schema contract checks without requiring a live server.
- Modify: `tests/storage/test_sqlite_migrations.py` — assert autonomous tables are present in fresh and legacy-initialized schemas.

### Task 1: Add Strict Transaction Contracts

**Files:**
- Create: `werewolf_agent/player_agents/contracts/transactions.py`
- Modify: `werewolf_agent/player_agents/contracts/__init__.py`
- Test: `tests/player_agents/test_transaction_contracts.py`

- [ ] **Step 1: Write failing contract tests**

Add tests that construct a valid request from the existing speech envelope fixture and assert:

```python
def test_commit_request_binds_turn_and_rejects_extra_fields() -> None:
    request = _request()
    assert request.proposal.turn_id == request.turn_id
    with pytest.raises(ValidationError):
        CommitTurnRequest.model_validate({**request.model_dump(), "unexpected": True})


def test_commit_request_rejects_duplicate_audit_and_outbox_ids() -> None:
    with pytest.raises(ValidationError, match="audit IDs must not contain duplicates"):
        _request(audit_ids=("audit-1", "audit-1"))
    with pytest.raises(ValidationError, match="outbox IDs must not contain duplicates"):
        _request(outbox_ids=("outbox-1", "outbox-1"))


def test_request_hash_is_order_independent_for_json_object_keys() -> None:
    left = _request(rule_result={"accepted": True, "reason": "ok"})
    right = _request(rule_result={"reason": "ok", "accepted": True})
    assert request_hash(left) == request_hash(right)
```

The helper must create a two-move `SpeechProposalEnvelope`, one `EventCandidate`,
one `CriticalAuditRecord`, and one `ProjectionOutboxRecord`; it must not import
legacy `SpeechAct`, `PlayerAction`, or strategy modules.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_transaction_contracts.py -v
```

Expected: collection fails because `transactions.py` and `request_hash` do not exist.

- [ ] **Step 3: Implement the immutable transaction models**

Implement these exact public models:

```python
class EventCandidate(StrictFrozenModel):
    type: NonEmptyId
    payload: dict[str, Any] = Field(default_factory=dict)
    visibility: EventVisibility | None = None


class CriticalAuditRecord(StrictFrozenModel):
    audit_id: NonEmptyId
    kind: NonEmptyId
    payload: dict[str, Any] = Field(default_factory=dict)


class ProjectionOutboxRecord(StrictFrozenModel):
    outbox_id: NonEmptyId
    kind: NonEmptyId
    payload: dict[str, Any] = Field(default_factory=dict)


class CommitTurnRequest(StrictFrozenModel):
    game_id: NonEmptyId
    turn_id: NonEmptyId
    idempotency_key: NonEmptyId
    base_game_revision: int = Field(ge=0)
    read_set: tuple[ReadReference, ...] = ()
    proposal: SpeechProposalEnvelope
    rule_result: dict[str, Any] = Field(default_factory=dict)
    event: EventCandidate
    public_record: PublicSpeechRecord | None = None
    critical_audit_records: tuple[CriticalAuditRecord, ...] = ()
    projection_outbox_records: tuple[ProjectionOutboxRecord, ...] = ()


class CommitResult(StrictFrozenModel):
    game_id: NonEmptyId
    turn_id: NonEmptyId
    idempotency_key: NonEmptyId
    committed_revision: int = Field(ge=1)
    event_id: NonEmptyId
    public_record_id: NonEmptyId | None = None
    audit_ids: tuple[NonEmptyId, ...] = ()
    outbox_ids: tuple[NonEmptyId, ...] = ()
    request_hash: ContentHash
    replayed: bool = False
```

Use `model_validator` to bind `proposal.turn_id`, public record game/turn IDs,
and enforce unique read, audit, and outbox IDs. Export all five classes from
`contracts/__init__.py`.

- [ ] **Step 4: Run focused contract checks**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_transaction_contracts.py -v
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/player_agents tests/player_agents/test_transaction_contracts.py
```

Expected: all focused tests and Ruff checks pass.

- [ ] **Step 5: Commit the contract layer**

```bash
git add werewolf_agent/player_agents/contracts tests/player_agents/test_transaction_contracts.py
git commit -m "feat: add autonomous commit transaction contracts"
```

### Task 2: Add Capability Protocol and Shared Commit Helpers

**Files:**
- Create: `werewolf_agent/storage/autonomous_commit.py`
- Test: `tests/storage/test_autonomous_commit.py`

- [ ] **Step 1: Write failing protocol and helper tests**

Cover stable exceptions, canonical hashes, event identity, and capability rejection:

```python
def test_request_hash_is_sha256_hex() -> None:
    digest = request_hash(_request())
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_capability_guard_rejects_legacy_repository() -> None:
    with pytest.raises(AutonomousCommitUnsupported):
        require_autonomous_commit_repository(object())


def test_build_event_assigns_authoritative_identity() -> None:
    event = build_committed_event("g1", _request().event, 3)
    assert event.event_id == "g1:e000003"
    assert event.sequence_number == 3
    assert event.game_id == "g1"
    assert event.schema_version == "2"
```

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_commit.py -k "hash or capability or identity" -v
```

Expected: import errors for the new helpers.

- [ ] **Step 3: Implement protocol, exceptions, hashing, and identity helpers**

Implement:

```python
class AutonomousCommitUnsupported(RuntimeError): ...
class StaleCommitError(RuntimeError): ...
class IdempotencyConflictError(RuntimeError): ...
class CommitTransactionError(RuntimeError): ...

class AutonomousCommitRepository(Protocol):
    def supports_autonomous_commit(self) -> bool: ...
    def commit_turn(self, request: CommitTurnRequest) -> CommitResult: ...
    def load_game_revision(self, game_id: str) -> int: ...
    def load_outbox(self, game_id: str) -> list[ProjectionOutboxRecord]: ...

def request_hash(request: CommitTurnRequest) -> str: ...
def build_committed_event(game_id: str, candidate: EventCandidate, revision: int) -> GameEvent: ...
def bind_public_record(record: PublicSpeechRecord | None, revision: int) -> PublicSpeechRecord | None: ...
def build_commit_result(request: CommitTurnRequest, digest: str, revision: int, event: GameEvent, record: PublicSpeechRecord | None) -> CommitResult: ...
def require_autonomous_commit_repository(repository: object) -> AutonomousCommitRepository: ...
```

`request_hash` uses `json.dumps(..., ensure_ascii=False, sort_keys=True,
separators=(",", ":"))` and SHA-256. `build_committed_event` uses UTC-aware
`datetime.now(timezone.utc)`, event visibility inference, and the canonical
`{game_id}:e{revision:06d}` identity. The guard requires a callable capability
method returning `True`; it never treats the presence of `commit_turn` alone as
support.

- [ ] **Step 4: Run helper checks and commit**

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_commit.py -k "hash or capability or identity" -v
git add werewolf_agent/storage/autonomous_commit.py tests/storage/test_autonomous_commit.py
git commit -m "feat: add autonomous commit repository protocol"
```

### Task 3: Implement In-Memory Atomic Commit

**Files:**
- Modify: `werewolf_agent/storage/memory_store.py`
- Test: `tests/storage/test_autonomous_commit.py`

- [ ] **Step 1: Add failing in-memory behavior tests**

Parameterize the tests with `InMemoryGameRepository` and a temporary
`SqliteGameRepository` placeholder. Cover first commit, stale CAS, replay,
idempotency conflict, outbox listing, and duplicate failure:

```python
def test_first_commit_advances_revision_and_writes_all_records(repo) -> None:
    repo.save_game(GameState(game_id="g1"))
    result = repo.commit_turn(_request(game_id="g1", base_revision=0))
    assert result.committed_revision == 1
    assert repo.load_game_revision("g1") == 1
    assert len(repo.load_events("g1")) == 1
    assert repo.load_outbox("g1")[0].outbox_id == "outbox-1"


def test_duplicate_submit_replays_without_second_event(repo) -> None:
    repo.save_game(GameState(game_id="g1"))
    request = _request(game_id="g1", base_revision=0)
    first = repo.commit_turn(request)
    replay = repo.commit_turn(request)
    assert replay.replayed is True
    assert replay.committed_revision == first.committed_revision
    assert len(repo.load_events("g1")) == 1


def test_stale_submit_and_idempotency_conflict_are_distinct(repo) -> None:
    repo.save_game(GameState(game_id="g1"))
    request = _request(game_id="g1", base_revision=0)
    repo.commit_turn(request)
    with pytest.raises(StaleCommitError):
        repo.commit_turn(_request(game_id="g1", base_revision=0, turn_id="turn-2"))
    with pytest.raises(IdempotencyConflictError):
        repo.commit_turn(_request(game_id="g1", base_revision=0, event_type="other"))
```

- [ ] **Step 2: Run the behavior tests and verify RED**

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_commit.py -v
```

Expected: the in-memory repository lacks capability and commit methods.

- [ ] **Step 3: Implement in-memory commit with prepare-then-publish semantics**

Add autonomous state dictionaries initialized in `__init__` and implement:

```python
def supports_autonomous_commit(self) -> bool:
    return True

def commit_turn(self, request: CommitTurnRequest) -> CommitResult:
    with self._lock:
        digest = request_hash(request)
        existing = self._autonomous_commits.get((request.game_id, request.turn_id, request.idempotency_key))
        if existing is not None:
            if existing.request_hash != digest:
                raise IdempotencyConflictError("idempotency key conflicts with an existing proposal")
            return existing.model_copy(update={"replayed": True})
        current = self._autonomous_revision(request.game_id)
        if request.base_game_revision != current:
            raise StaleCommitError(f"expected revision {current}, got {request.base_game_revision}")
        next_revision = current + 1
        event = build_committed_event(request.game_id, request.event, next_revision)
        record = bind_public_record(request.public_record, next_revision)
        result = build_commit_result(request, digest, next_revision, event, record)
        self._check_new_ids(request)
        # 以上全部校验成功后才更新事件、stream、audit、commit 和 outbox。
        self._events.setdefault(request.game_id, []).append(event)
        self._autonomous_revision_by_game[request.game_id] = next_revision
        self._publish_autonomous_records(request, result, record)
        return result
```

Keep the helper calls private and make all copies defensive when returning
outbox data. Existing `save_game` and `append_events` behavior remains unchanged.

- [ ] **Step 4: Run in-memory tests and commit**

```bash
conda run -n wofkill python -m pytest tests/storage/test_autonomous_commit.py -k "not sqlite and not postgres" -v
git add werewolf_agent/storage/memory_store.py tests/storage/test_autonomous_commit.py
git commit -m "feat: implement in-memory autonomous commit"
```

### Task 4: Add SQLite Schema and Atomic Commit

**Files:**
- Modify: `werewolf_agent/storage/sqlite_store.py`
- Modify: `tests/storage/test_sqlite_migrations.py`
- Test: `tests/storage/test_autonomous_commit.py`

- [ ] **Step 1: Add failing SQLite schema and rollback tests**

Assert all autonomous tables exist for a fresh database and a database created
with only legacy `games/events` tables. Inject a conflicting existing outbox ID
and verify no revision, event, audit, public record, commit, or outbox row is
left behind.

- [ ] **Step 2: Run the SQLite tests and verify RED**

```bash
conda run -n wofkill python -m pytest tests/storage/test_sqlite_migrations.py tests/storage/test_autonomous_commit.py -k sqlite -v
```

Expected: missing autonomous tables and `commit_turn` implementation.

- [ ] **Step 3: Add idempotent SQLite tables**

Extend `_SCHEMA` with the four stream tables plus `autonomous_public_records`,
and call a focused `_ensure_autonomous_schema` after `_ensure_event_schema_v2`.
The tables must use `IF NOT EXISTS`, foreign keys to `games` where applicable,
primary keys for audit/outbox/record IDs, a unique `(game_id, turn_id,
idempotency_key)` constraint, and an index on `(game_id, committed_revision)`.
Do not change `MigrationManager`'s existing version numbering.

- [ ] **Step 4: Implement SQLite `commit_turn`**

Use `BEGIN IMMEDIATE` inside `self._lock`. In order: validate game existence,
compute the request hash, return an exact stored replay, create or lock the
stream row using `MAX(events.seq)` as the initial revision, compare the base
revision, build the authoritative event, bind the public record revision,
insert event/public/audit/outbox/idempotency rows, update the stream revision,
and commit. Catch `StaleCommitError` and `IdempotencyConflictError` unchanged;
rollback and wrap every other failure in `CommitTransactionError`.

- [ ] **Step 5: Run SQLite behavior and migration checks**

```bash
conda run -n wofkill python -m pytest tests/storage/test_sqlite_migrations.py tests/storage/test_autonomous_commit.py -k sqlite -v
```

Expected: all SQLite transaction, replay, stale, conflict, concurrency, and
rollback tests pass.

- [ ] **Step 6: Commit SQLite support**

```bash
git add werewolf_agent/storage/sqlite_store.py tests/storage/test_sqlite_migrations.py tests/storage/test_autonomous_commit.py
git commit -m "feat: add sqlite autonomous commit transaction"
```

### Task 5: Implement PostgreSQL Schema and Transaction

**Files:**
- Modify: `werewolf_agent/storage/postgres_store.py`
- Create: `tests/storage/test_postgres_autonomous_commit.py`

- [ ] **Step 1: Add failing PostgreSQL capability/schema tests**

Construct the repository with `initialize=False` and assert it reports
unsupported until a connection exists. Inspect `_ensure_schema_transaction`
source for every autonomous table and test the SQL placeholder style uses `%s`
and JSONB casts. Use the existing fake connection/cursor fixtures for a commit
smoke test; no live PostgreSQL service is required.

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run -n wofkill python -m pytest tests/storage/test_postgres_autonomous_commit.py -v
```

Expected: missing capability method, tables, and transaction implementation.

- [ ] **Step 3: Add PostgreSQL tables to `_ensure_schema_transaction`**

Create the same stream, commit, public-record, audit, and outbox tables using
`JSONB`, `TIMESTAMP`, `BIGINT`, `PRIMARY KEY`, `UNIQUE`, and the game/revision
index. Keep the operation inside the existing schema transaction and do not
change the legacy event migration behavior.

- [ ] **Step 4: Implement PostgreSQL `commit_turn`**

Acquire the existing advisory transaction lock for the game, lock/create the
stream row, query idempotency before CAS, and perform the same inserts and
result serialization as SQLite with PostgreSQL placeholders. Roll back on
every failure and preserve stable exception types. `supports_autonomous_commit`
returns `True` only when the repository has an initialized connection and the
autonomous schema initialization completed.

- [ ] **Step 5: Run PostgreSQL tests and commit**

```bash
conda run -n wofkill python -m pytest tests/storage/test_postgres_autonomous_commit.py tests/storage/test_postgres_store.py -q
git add werewolf_agent/storage/postgres_store.py tests/storage/test_postgres_autonomous_commit.py
git commit -m "feat: add postgres autonomous commit transaction"
```

### Task 6: Cross-Backend Fault Injection and Regression Gates

**Files:**
- Modify: `tests/storage/test_autonomous_commit.py`
- Modify: `tests/storage/test_repository.py`

- [ ] **Step 1: Add the shared atomicity matrix**

Run the same cases for memory and SQLite: successful commit, replay, stale
revision, request-hash conflict, missing game, duplicate existing outbox ID,
duplicate audit ID, and 50 concurrent identical submissions. Assert exactly one
event, one revision, one idempotency result, one public record, all requested
audit records, and all requested outbox records.

- [ ] **Step 2: Add explicit capability assertions**

Assert memory and SQLite return `True`, an uninitialized PostgreSQL repository
returns `False`, and a plain legacy stub is rejected by
`require_autonomous_commit_repository`.

- [ ] **Step 3: Run the complete verification set**

```bash
conda run -n wofkill python -m pytest tests/player_agents tests/storage/test_autonomous_commit.py tests/storage/test_sqlite_migrations.py tests/storage/test_postgres_autonomous_commit.py -v
conda run -n wofkill python -m pytest tests/agents/test_schemas.py tests/agents/test_speech_intent_parser.py tests/runtime/test_event_metadata_v2.py tests/storage/test_repository.py -v
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/player_agents tests/player_agents werewolf_agent/storage tests/storage/test_autonomous_commit.py tests/storage/test_postgres_autonomous_commit.py
conda run -n wofkill python -m mypy werewolf_agent/player_agents werewolf_agent/storage
```

Expected: all focused and adjacent tests pass; the scoped Ruff and mypy checks
report no errors. The existing full-repository lint baseline is not part of
this plan because it contains unrelated historical findings.

- [ ] **Step 4: Run the full test suite and commit the integrated slice**

```bash
conda run -n wofkill python -m pytest -q
git add werewolf_agent/player_agents/contracts werewolf_agent/storage tests/player_agents tests/storage
git commit -m "feat: add autonomous commit transaction capability"
```

## Completion Criteria

- `CommitTurnRequest` and `CommitResult` are strict, immutable, hashable by
  canonical JSON, and exported from the player-agent contracts package.
- Capability checks are explicit and legacy repositories remain unsupported.
- Memory, SQLite, and initialized PostgreSQL implement the same CAS,
  idempotency, revision, event, public-record, audit, outbox, and rollback rules.
- Duplicate identical submissions replay one result; conflicting reuse and stale
  revisions never mutate truth.
- A failed transaction leaves no stream, event, public-record, audit, commit, or
  outbox partial write.
- No HostRuntime, scheduler, model dispatch, renderer, RuleEngine, or feature
  gate behavior is changed.
