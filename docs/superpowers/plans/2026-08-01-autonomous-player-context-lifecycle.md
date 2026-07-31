# Autonomous Player Context Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for every behavior change and `superpowers:verification-before-completion` before any completion claim. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Host-owned context accounting, durable compaction checkpoints, deterministic rehydration, and restart recovery for autonomous player turns across Memory, SQLite, and PostgreSQL, without connecting a model, ToolGateway, `HostRuntime`, `CommitTurn`, or the live game path.

**Architecture:** Add a framework-neutral `player_agents.context` package for strict contracts and pure lifecycle rules, plus an independent `ContextCheckpointRepository` capability implemented by the same physical repository that owns autonomous turns. Checkpoint insertion and `COMPACTING -> THINKING` CAS are one backend transaction; `ContextLifecycleService` is the narrow Host-facing coordinator and recovery remains an explicit observation API.

**Tech Stack:** Python 3.12, Pydantic v2 strict frozen models, SHA-256 canonical JSON, protocols, `RLock`, SQLite WAL/`BEGIN IMMEDIATE`, PostgreSQL JSONB/advisory locks/row locks, pytest, Ruff, mypy. Every Python command runs through `conda run -n wofkill`.

## Global Constraints

- The approved specification is `docs/superpowers/specs/2026-08-01-autonomous-player-context-lifecycle-design.md`; do not broaden it during implementation.
- Before implementation, use `superpowers:using-git-worktrees` to create an isolated `codex/` branch/worktree unless the user explicitly selects the current checkout.
- Preserve `CommitTurn`, the game engine, the existing live runtime, and ordinary observation semantics.
- Do not import or call Deep Agents, legacy `PlayerAgent`, old tools schemas, provider/tokenizer SDKs, `ModelRouter`, RuleEngine, `CommitTurn`, `GameRunner`, AgentLoop, or ToolGateway.
- The checkpoint is the only resumable authority. `CompactionHandoff` is optional `UNTRUSTED_DATA` and is never parsed into facts, evidence, permissions, memory, tools, or proposals.
- Admission is strictly below 55%; automatic compaction is inclusive at 80%; every rehydration plan is at most 55%; maximum compactions default to two.
- `ContextTokenAccounting` derives and verifies its margin, total, and ratio. Threshold decisions use integer cross-multiplication, never rounded display ratios.
- The same repository object must satisfy both `AutonomousTurnRepository` and `ContextCheckpointRepository`; split repositories are rejected.
- Lock order is `game -> schedule -> managed turn -> latest checkpoint lineage row`. Memory uses its existing `RLock`; SQLite uses `BEGIN IMMEDIATE`; PostgreSQL uses its existing game advisory transaction lock before row locks.
- Checkpoint insert and `COMPACTING -> THINKING` CAS are all-or-nothing. Any authority, deadline, lineage, CAS, uniqueness, or backend failure publishes neither side.
- New and substantially edited Python files follow `AGENTS.md`: encoding header, accurate Chinese module docstring, author, and `2026-08-01` creation/modification date; comments are concise Chinese.
- Do not claim a playable vertical slice. Completion still excludes model/tool dispatch, proposal validation, RuleEngine/`CommitTurn` orchestration, and live integration.

## File Map

- Create: `werewolf_agent/player_agents/context/__init__.py`
- Create: `werewolf_agent/player_agents/context/contracts.py`
- Create: `werewolf_agent/player_agents/context/errors.py`
- Create: `werewolf_agent/player_agents/context/accounting.py`
- Create: `werewolf_agent/player_agents/context/lifecycle.py`
- Create: `werewolf_agent/player_agents/context/service.py`
- Create: `werewolf_agent/storage/context_checkpoints.py`
- Modify: `werewolf_agent/player_agents/observation/service.py`
- Modify: `werewolf_agent/player_agents/observation/__init__.py`
- Modify: `werewolf_agent/storage/memory_store.py`
- Modify: `werewolf_agent/storage/sqlite_store.py`
- Modify: `werewolf_agent/storage/postgres_store.py`
- Create: `tests/player_agents/test_context_contracts.py`
- Create: `tests/player_agents/test_context_lifecycle.py`
- Create: `tests/player_agents/test_context_checkpoint_conformance.py`
- Create: `tests/player_agents/test_context_import_boundary.py`
- Modify: `tests/player_agents/test_observation_service.py`
- Create: `tests/storage/test_context_checkpoints.py`
- Create: `tests/storage/test_postgres_context_checkpoints.py`
- Create: `tests/storage/test_postgres_context_checkpoints_live.py`
- Modify: `tests/storage/test_sqlite_migrations.py`
- Modify: `handoff.md`

---

### Task 1: Define Strict Context Contracts, Errors, and Accounting

**Files:**
- Create: `werewolf_agent/player_agents/context/contracts.py`
- Create: `werewolf_agent/player_agents/context/errors.py`
- Create: `werewolf_agent/player_agents/context/accounting.py`
- Create: `werewolf_agent/player_agents/context/__init__.py`
- Create: `tests/player_agents/test_context_contracts.py`

**Public interfaces:**

```python
class TokenizerMode(StrEnum):
    PROVIDER = "provider"
    CONSERVATIVE = "conservative"


class HandoffTrustClass(StrEnum):
    UNTRUSTED_DATA = "UNTRUSTED_DATA"


class ContextBudgetPolicy(StrictFrozenModel):
    model_context_limit_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(ge=0)
    reserved_tool_schema_tokens: int = Field(ge=0)
    estimator_version: NonEmptyId
    tokenizer_mode: TokenizerMode
    conservative_safety_margin: Decimal = Decimal("0.10")
    auto_compact_threshold: Decimal = Decimal("0.80")
    post_compact_target: Decimal = Decimal("0.55")
    max_compactions_per_turn: int = Field(default=2, ge=1)


class ContextTokenAccounting(StrictFrozenModel):
    immutable_prefix_tokens: int = Field(ge=0)
    active_history_tokens: int = Field(ge=0)
    exposed_tool_schema_tokens: int = Field(ge=0)
    reserved_output_tokens: int = Field(ge=0)
    safety_margin_tokens: int = Field(ge=0)
    total_active_tokens: int = Field(ge=0)
    model_context_limit_tokens: int = Field(gt=0)
    occupancy_ratio: Decimal
    estimator_version: NonEmptyId
    tokenizer_mode: TokenizerMode
```

`ContextTokenAccounting` uses a six-decimal `ROUND_HALF_EVEN` ratio for serialization only. `build_context_accounting()` computes conservative margin as `ceil((immutable_prefix_tokens + active_history_tokens + exposed_tool_schema_tokens) * 0.10)`; provider mode adds zero. Both modes include output reserve in total.

Define these stable errors exactly:

```python
class ContextLifecycleError(RuntimeError):
    code: ClassVar[str] = "context_lifecycle_error"


class ContextLifecycleUnsupported(ContextLifecycleError):
    code = "context_lifecycle_unsupported"


class ContextAdmissionRejected(ContextLifecycleError):
    code = "context_admission_rejected"


class ContextCompactionRequired(ContextLifecycleError):
    code = "context_compaction_required"


class CheckpointIntegrityFailed(ContextLifecycleError):
    code = "checkpoint_integrity_failed"


class CheckpointAuthorityConflict(ContextLifecycleError):
    code = "checkpoint_authority_conflict"


class CheckpointLineageConflict(ContextLifecycleError):
    code = "checkpoint_lineage_conflict"


class CheckpointDeadlineExpired(ContextLifecycleError):
    code = "checkpoint_deadline_expired"


class CheckpointTransactionFailed(ContextLifecycleError):
    code = "checkpoint_transaction_failed"


class RehydrationTargetExceeded(ContextLifecycleError):
    code = "rehydration_target_exceeded"


class CheckpointRecoveryRejected(ContextLifecycleError):
    code = "checkpoint_recovery_rejected"
```

- [ ] **Step 1: Write RED tests for strictness and derived values**

```python
def test_default_policy_and_derived_accounting_are_exact() -> None:
    policy = _policy(tokenizer_mode=TokenizerMode.CONSERVATIVE)
    accounting = build_context_accounting(
        policy,
        immutable_prefix_tokens=100,
        active_history_tokens=200,
    )
    assert policy.auto_compact_threshold == Decimal("0.80")
    assert policy.post_compact_target == Decimal("0.55")
    assert policy.max_compactions_per_turn == 2
    assert accounting.exposed_tool_schema_tokens == 50
    assert accounting.reserved_output_tokens == 100
    assert accounting.safety_margin_tokens == 35
    assert accounting.total_active_tokens == 485


def test_accounting_rejects_tampered_derived_values() -> None:
    valid = build_context_accounting(_policy(), 100, 200)
    with pytest.raises(ValidationError):
        ContextTokenAccounting.model_validate({
            **valid.model_dump(),
            "total_active_tokens": valid.total_active_tokens + 1,
        })
```

Also test strict/frozen/extra-forbid behavior, list-to-tuple defensive freezing, invalid policy ordering/reserves, provider mode zero margin, and all stable codes.

- [ ] **Step 2: Run RED**

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_context_contracts.py \
  -o addopts='' -k "policy or accounting or error" -v
```

Expected: collection fails because `werewolf_agent.player_agents.context` does not exist.

- [ ] **Step 3: Implement policy validation and pure accounting**

Expose these exact functions:

```python
def build_context_accounting(
    policy: ContextBudgetPolicy,
    immutable_prefix_tokens: int,
    active_history_tokens: int,
) -> ContextTokenAccounting: ...


def require_context_admission(
    policy: ContextBudgetPolicy,
    immutable_prefix_tokens: int,
) -> ContextTokenAccounting: ...


def require_model_request_ready(
    accounting: ContextTokenAccounting,
    policy: ContextBudgetPolicy,
) -> ContextTokenAccounting: ...


def require_rehydration_target(
    accounting: ContextTokenAccounting,
    policy: ContextBudgetPolicy,
) -> ContextTokenAccounting: ...
```

Use exact integer comparisons:

```python
def _meets_ratio(total: int, limit: int, ratio: Decimal) -> bool:
    numerator, denominator = ratio.as_integer_ratio()
    return total * denominator >= limit * numerator
```

Admission rejects equality at 55%; model request rejects equality at 80%; rehydration accepts equality at 55%.

- [ ] **Step 4: Run GREEN and commit**

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_context_contracts.py -o addopts='' -q
git add werewolf_agent/player_agents/context tests/player_agents/test_context_contracts.py
git commit -m "feat: define autonomous context accounting contracts"
```

---

### Task 2: Define Checkpoint, Handoff, and Rehydration Contracts

**Files:**
- Modify: `werewolf_agent/player_agents/context/contracts.py`
- Modify: `werewolf_agent/player_agents/context/__init__.py`
- Modify: `tests/player_agents/test_context_contracts.py`

**Exact contracts:**

```python
BoundedContextText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]


class IntendedOperation(StrictFrozenModel):
    operation_kind: NonEmptyId
    objective: BoundedContextText
    candidate_action: NonEmptyId | None = None
    candidate_target_ids: tuple[NonEmptyId, ...] = ()


class CompactionHandoff(StrictFrozenModel):
    trust_class: Literal[HandoffTrustClass.UNTRUSTED_DATA]
    current_progress: tuple[BoundedContextText, ...] = Field(max_length=8)
    decisions_made: tuple[BoundedContextText, ...] = Field(max_length=8)
    important_constraints: tuple[BoundedContextText, ...] = Field(max_length=8)
    remaining_steps: tuple[BoundedContextText, ...] = Field(max_length=12)
    critical_reference_ids: tuple[NonEmptyId, ...] = Field(max_length=64)
    estimated_tokens: int = Field(ge=0, le=2000)


class CheckpointRehydrationReceipt(StrictFrozenModel):
    ordered_authority_component_hashes: tuple[ContentHash, ...] = Field(min_length=5)
    accounting: ContextTokenAccounting
    final_context_hash: ContentHash


class RehydrationPlan(StrictFrozenModel):
    checkpoint_id: NonEmptyId
    checkpoint_hash: ContentHash
    ordered_component_hashes: tuple[ContentHash, ...] = Field(min_length=5)
    accounting: ContextTokenAccounting
    handoff_included: bool
    final_context_hash: ContentHash
```

`CompactionCheckpoint` has these exact fields:

```python
class CompactionCheckpoint(StrictFrozenModel):
    schema_version: Literal[1]
    checkpoint_id: NonEmptyId
    ordinal: int = Field(ge=1)
    parent_checkpoint_id: NonEmptyId | None
    parent_checkpoint_hash: ContentHash | None
    game_id: NonEmptyId
    player_id: NonEmptyId
    schedule_id: NonEmptyId
    schedule_state_version: int = Field(ge=0)
    turn_id: NonEmptyId
    observation_turn_state_version: int = Field(ge=0)
    compacting_turn_state_version: int = Field(ge=1)
    resumed_thinking_turn_state_version: int = Field(ge=2)
    window_id: NonEmptyId
    window_version: int = Field(ge=1)
    task_type: NonEmptyId
    base_game_revision: int = Field(ge=0)
    view_fingerprint: ContentHash
    model_lease_hash: ContentHash
    original_workspace_revision: ContentHash
    original_workspace_hash: ContentHash
    observation_frame_hash: ContentHash
    source_history_hash: ContentHash
    event_cursor: int = Field(ge=0)
    legal_action_snapshot: tuple[NonEmptyId, ...]
    legal_target_snapshot: tuple[NonEmptyId, ...]
    remaining_budget: TurnBudget
    remaining_compactions: int = Field(ge=0)
    deadline: datetime
    confirmed_read_references: tuple[ReadReference, ...]
    projection_source_references: tuple[ProjectionSourceReference, ...]
    public_reference_ids: tuple[NonEmptyId, ...]
    intended_operation: IntendedOperation
    pre_compaction_accounting: ContextTokenAccounting
    checkpoint_rehydration_receipt: CheckpointRehydrationReceipt
    created_at: datetime
    persisted_at: datetime
    checkpoint_hash: ContentHash
```

The checkpoint model validates parent pair/ordinal consistency, consecutive CAS versions, aware ordered timestamps before deadline, unique reference identities, and its self-hash. `checkpoint_authority_hash()` hashes the canonical checkpoint payload excluding `checkpoint_rehydration_receipt`, `persisted_at`, and `checkpoint_hash`, preventing circular receipt construction. `checkpoint_self_hash()` hashes every field except `checkpoint_hash`.

- [ ] **Step 1: Write RED contract tests**

Cover self-hash tampering, ordinal-one parent absence, ordinal-N parent requirement, CAS version sequence, handoff list freezing, 2,000-token cap, duplicate reference IDs, and JSON round trips.

```python
def test_checkpoint_self_hash_detects_any_mutation() -> None:
    checkpoint = _checkpoint()
    assert checkpoint_self_hash(checkpoint) == checkpoint.checkpoint_hash
    with pytest.raises(ValidationError):
        CompactionCheckpoint.model_validate({
            **checkpoint.model_dump(),
            "event_cursor": checkpoint.event_cursor + 1,
        })
```

- [ ] **Step 2: Run RED, implement, and run GREEN**

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_context_contracts.py \
  -o addopts='' -k "checkpoint or handoff or rehydration" -v
conda run -n wofkill python -m pytest \
  tests/player_agents/test_context_contracts.py -o addopts='' -q
```

- [ ] **Step 3: Commit**

```bash
git add werewolf_agent/player_agents/context tests/player_agents/test_context_contracts.py
git commit -m "feat: define compaction checkpoint authority"
```

---

### Task 3: Implement Pure Checkpoint Build, Handoff Filtering, and Recovery Validation

**Files:**
- Create: `werewolf_agent/player_agents/context/lifecycle.py`
- Create: `tests/player_agents/test_context_lifecycle.py`

**Interfaces:**

```python
def build_compaction_checkpoint(
    *,
    checkpoint_id: str,
    schedule: SerialPublicSchedule,
    compacting_turn: ManagedAgentTurn,
    observation: ObservationBundle,
    policy: ContextBudgetPolicy,
    immutable_system_contract_hash: str,
    immutable_prefix_tokens: int,
    checkpoint_authority_tokens: int,
    source_history_hash: str,
    event_cursor: int,
    public_reference_ids: tuple[str, ...],
    intended_operation: IntendedOperation,
    pre_compaction_accounting: ContextTokenAccounting,
    parent: CompactionCheckpoint | None,
    created_at: datetime,
    persisted_at: datetime,
) -> CompactionCheckpoint: ...


def validate_handoff(
    checkpoint: CompactionCheckpoint,
    handoff: CompactionHandoff | None,
) -> CompactionHandoff | None: ...


def build_rehydration_plan(
    checkpoint: CompactionCheckpoint,
    policy: ContextBudgetPolicy,
    *,
    handoff: CompactionHandoff | None = None,
) -> RehydrationPlan: ...


def validate_checkpoint_lineage(
    checkpoints: tuple[CompactionCheckpoint, ...],
) -> tuple[CompactionCheckpoint, ...]: ...


def recover_rehydration_plan(
    *,
    checkpoints: tuple[CompactionCheckpoint, ...],
    schedule: SerialPublicSchedule,
    thinking_turn: ManagedAgentTurn,
    rebuilt_observation: ObservationBundle,
    policy: ContextBudgetPolicy,
) -> RehydrationPlan: ...
```

Source references are collected deterministically from the observation frame's manifest plus `recent_commitment_references`; duplicate identical identities collapse once, conflicting hashes fail. Handoff critical IDs must be a subset of confirmed read IDs, projection source record IDs, and public reference IDs. Invalid/missing/oversized handoff returns `None`; checkpoint-only overflow raises `RehydrationTargetExceeded`.

- [ ] **Step 1: Write RED lifecycle tests**

```python
def test_invalid_handoff_is_dropped_without_blocking_checkpoint_only_plan() -> None:
    checkpoint = _checkpoint_at_target(Decimal("0.50"))
    invalid = _handoff(critical_reference_ids=("not-authorized",))
    plan = build_rehydration_plan(checkpoint, _policy(), handoff=invalid)
    assert plan.handoff_included is False
    assert len(plan.ordered_component_hashes) == 5


def test_checkpoint_only_over_target_fails_closed() -> None:
    checkpoint = _checkpoint_at_target(Decimal("0.550001"))
    with pytest.raises(RehydrationTargetExceeded):
        build_rehydration_plan(checkpoint, _policy())
```

Table-drive stale game/player/schedule/turn/window/revision/view/lease/legal/source/deadline, budget increase, remaining-compaction increase, parent fork, skipped ordinal, bad self-hash, and recovery side-effect absence.

- [ ] **Step 2: Run RED**

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_context_lifecycle.py -o addopts='' -v
```

- [ ] **Step 3: Implement pure functions and GREEN**

Use `canonical_json_hash()` from observation contracts for all component/final hashes. Recovery compares stable authority lineage and intentionally does not require rebuilt `ProjectionIdentity.turn_state_version`, workspace revision, or workspace hash to equal the original values.

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_context_contracts.py \
  tests/player_agents/test_context_lifecycle.py -o addopts='' -q
git add werewolf_agent/player_agents/context tests/player_agents/test_context_lifecycle.py
git commit -m "feat: build and validate context checkpoints"
```

---

### Task 4: Add Repository Capability and Atomic Memory Backend

**Files:**
- Create: `werewolf_agent/storage/context_checkpoints.py`
- Modify: `werewolf_agent/storage/memory_store.py:89-110,260-283,927-990`
- Create: `tests/storage/test_context_checkpoints.py`

**Repository interface:**

```python
class ContextCheckpointRepository(Protocol):
    def supports_context_checkpoints(self) -> bool: ...

    def load_latest_compaction_checkpoint(
        self,
        turn_id: str,
    ) -> CompactionCheckpoint | None: ...

    def load_compaction_checkpoint_lineage(
        self,
        turn_id: str,
    ) -> tuple[CompactionCheckpoint, ...]: ...

    def commit_compaction_checkpoint(
        self,
        checkpoint: CompactionCheckpoint,
    ) -> ManagedAgentTurn: ...


def require_context_checkpoint_repository(
    repository: object,
) -> ContextCheckpointRepository: ...


def prepare_checkpoint_commit(
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
    latest: CompactionCheckpoint | None,
    checkpoint: CompactionCheckpoint,
) -> ManagedAgentTurn: ...
```

`prepare_checkpoint_commit()` validates the entire active authority and lineage, then calls
`prepare_active_transition(schedule, managed, AgentTurnStatus.THINKING)` and requires
its resulting version to equal `resumed_thinking_turn_state_version`.

- [ ] **Step 1: Write RED protocol and pure transaction tests**

Test capability rejection, sanitized errors, every identity/CAS/deadline mismatch, first/next lineage, and the exact `COMPACTING -> THINKING` version.

- [ ] **Step 2: Implement the protocol and pure preparation**

Only `CheckpointAuthorityConflict`, `CheckpointLineageConflict`, `CheckpointDeadlineExpired`, and `CheckpointIntegrityFailed` may escape validation; unexpected errors are chained under `CheckpointTransactionFailed` at backend boundaries.

- [ ] **Step 3: Write RED Memory tests, then implement under the existing `RLock`**

Add:

```python
self._compaction_checkpoints: dict[str, CompactionCheckpoint] = {}
self._checkpoint_lineage_by_turn: dict[str, list[str]] = {}
```

Within one `with self._lock`, load schedule/turn/latest, prepare, reject duplicate ID and ordinal, insert a defensive checkpoint copy, replace the managed turn, and return a deep copy. Extend `delete_game()` to remove matching checkpoint IDs and lineage entries.

```python
def test_memory_checkpoint_commit_is_atomic_and_defensive() -> None:
    repository, checkpoint = _memory_compacting_turn()
    resumed = repository.commit_compaction_checkpoint(checkpoint)
    assert resumed.turn.status is AgentTurnStatus.THINKING
    assert repository.load_latest_compaction_checkpoint("turn-1") == checkpoint
    assert repository.load_compaction_checkpoint_lineage("turn-1") == (checkpoint,)
```

- [ ] **Step 4: Run GREEN and commit**

```bash
conda run -n wofkill python -m pytest \
  tests/storage/test_context_checkpoints.py \
  -o addopts='' -k "prepare or capability or memory" -q
git add werewolf_agent/storage/context_checkpoints.py \
  werewolf_agent/storage/memory_store.py \
  tests/storage/test_context_checkpoints.py
git commit -m "feat: persist context checkpoints in memory"
```

---

### Task 5: Implement SQLite Schema, Atomic Commit, and Reopen Recovery

**Files:**
- Modify: `werewolf_agent/storage/sqlite_store.py:222-265,439-465,568-882`
- Modify: `tests/storage/test_context_checkpoints.py`
- Modify: `tests/storage/test_sqlite_migrations.py`

**Schema:**

```sql
CREATE TABLE IF NOT EXISTS autonomous_compaction_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    schedule_id TEXT NOT NULL REFERENCES autonomous_serial_public_schedules(schedule_id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL REFERENCES autonomous_managed_turns(turn_id) ON DELETE CASCADE,
    player_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
    parent_checkpoint_id TEXT REFERENCES autonomous_compaction_checkpoints(checkpoint_id),
    checkpoint_hash TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    persisted_at TEXT NOT NULL,
    UNIQUE (turn_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_compaction_checkpoint_turn_ordinal
ON autonomous_compaction_checkpoints(turn_id, ordinal DESC);
```

Initialization calls `_ensure_compaction_checkpoint_integrity()` before enabling `_context_checkpoint_schema_ready`; it validates historical duplicate `(turn_id, ordinal)`, parent ownership/hash, and contiguous ordinals before creating/enabling constraints. Fresh schemas and legacy-schema rejection both require tests.

- [ ] **Step 1: Write RED SQLite tests**

Cover successful atomic commit, stale schedule/turn CAS, non-COMPACTING/terminal turn, duplicate ID, duplicate ordinal, fork/orphan/skip, forced insert failure, forced turn-update failure, defensive reads, and close/reopen full-lineage recovery.

- [ ] **Step 2: Implement `BEGIN IMMEDIATE` transaction**

```python
def commit_compaction_checkpoint(
    self,
    checkpoint: CompactionCheckpoint,
) -> ManagedAgentTurn:
    with self._lock:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            schedule = self._load_schedule_unlocked(checkpoint.schedule_id)
            managed = self._load_managed_turn_unlocked(checkpoint.turn_id)
            latest = self._load_latest_compaction_checkpoint_unlocked(
                checkpoint.turn_id,
            )
            if schedule is None or managed is None:
                raise CheckpointAuthorityConflict(
                    "checkpoint authority is unavailable",
                )
            resumed = prepare_checkpoint_commit(
                schedule,
                managed,
                latest,
                checkpoint,
            )
            self._conn.execute(
                "INSERT INTO autonomous_compaction_checkpoints "
                "(checkpoint_id, game_id, schedule_id, turn_id, player_id, "
                "ordinal, parent_checkpoint_id, checkpoint_hash, checkpoint_json, "
                "created_at, persisted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._checkpoint_row(checkpoint),
            )
            self._update_managed_turn_unlocked(
                resumed,
                checkpoint.compacting_turn_state_version,
            )
            self._conn.commit()
            return resumed.model_copy(deep=True)
        except ContextLifecycleError:
            self._conn.rollback()
            raise
        except Exception as exc:
            self._conn.rollback()
            raise CheckpointTransactionFailed(
                "context checkpoint transaction failed",
            ) from exc
```

- [ ] **Step 3: Run GREEN and commit**

```bash
conda run -n wofkill python -m pytest \
  tests/storage/test_context_checkpoints.py \
  tests/storage/test_sqlite_migrations.py \
  -o addopts='' -k "context_checkpoint or sqlite" -q
git add werewolf_agent/storage/sqlite_store.py \
  tests/storage/test_context_checkpoints.py \
  tests/storage/test_sqlite_migrations.py
git commit -m "feat: persist context checkpoints in sqlite"
```

---

### Task 6: Implement PostgreSQL DDL, Lock Order, CAS, and Rollback

**Files:**
- Modify: `werewolf_agent/storage/postgres_store.py:107-119,255-418,619-662,2038-2240`
- Create: `tests/storage/test_postgres_context_checkpoints.py`

- [ ] **Step 1: Build a stateful fake connection and write RED tests**

The fake must record normalized SQL, stage checkpoint/turn writes in a transaction snapshot, restore both on rollback, expose `rowcount`, and emulate JSONB dict returns. Assert this exact ordering:

```python
def test_postgres_checkpoint_commit_uses_fixed_lock_order() -> None:
    repository, connection, checkpoint = _repository_with_compacting_turn()
    repository.commit_compaction_checkpoint(checkpoint)
    sql = [statement for statement, _ in connection.executed]
    assert _index(sql, "pg_advisory_xact_lock") < _index(sql, "schedule_json")
    assert _index(sql, "schedule_json") < _index(sql, "turn_json")
    assert _index(sql, "turn_id = %s ORDER BY ordinal DESC") < _index(
        sql, "INSERT INTO autonomous_compaction_checkpoints"
    )
    assert _index(sql, "INSERT INTO autonomous_compaction_checkpoints") < _index(
        sql, "UPDATE autonomous_managed_turns"
    )
```

Cover DDL columns/constraints/index, capability readiness, JSONB round trip, stale schedule/turn, duplicate mappings, lineage rollback, checkpoint-insert failure, turn-CAS failure, commit failure, and sanitized database errors.

- [ ] **Step 2: Add PostgreSQL schema and helpers**

Use the same columns as SQLite with PostgreSQL types `TEXT`, `BIGINT`, `JSONB`, and `TIMESTAMPTZ`. Foreign keys reference games/schedules/turns; unique `(turn_id, ordinal)` and descending lineage index are mandatory. Schema readiness becomes true only after the checkpoint table and integrity scan succeed.

- [ ] **Step 3: Implement atomic commit**

Within the existing process lock:

1. locate game from checkpoint turn/schedule identity;
2. call `_lock_game_transaction(conn, checkpoint.game_id)`;
3. `SELECT schedule_json FROM autonomous_serial_public_schedules WHERE schedule_id = %s FOR UPDATE`;
4. `SELECT turn_json FROM autonomous_managed_turns WHERE turn_id = %s FOR UPDATE`;
5. `SELECT checkpoint_json FROM autonomous_compaction_checkpoints WHERE turn_id = %s ORDER BY ordinal DESC LIMIT 1 FOR UPDATE`;
6. call `prepare_checkpoint_commit()`;
7. insert checkpoint JSONB;
8. CAS-update the managed turn;
9. commit once.

Map SQLSTATE `23505` and `23503` to stable lineage/authority errors only after inspecting the named constraint; all other database errors become chained `CheckpointTransactionFailed` without SQL or payload text in the public message.

- [ ] **Step 4: Run GREEN and commit**

```bash
conda run -n wofkill python -m pytest \
  tests/storage/test_postgres_context_checkpoints.py \
  -o addopts='' -q
git add werewolf_agent/storage/postgres_store.py \
  tests/storage/test_postgres_context_checkpoints.py
git commit -m "feat: persist context checkpoints in postgres"
```

---

### Task 7: Add Explicit Recovery Observation and ContextLifecycleService

**Files:**
- Modify: `werewolf_agent/player_agents/observation/service.py:54-57,73-111,292-372`
- Modify: `werewolf_agent/player_agents/observation/__init__.py`
- Modify: `tests/player_agents/test_observation_service.py`
- Create: `werewolf_agent/player_agents/context/service.py`
- Modify: `tests/player_agents/test_context_lifecycle.py`

**Observation API:**

```python
def build_serial_public_recovery_observation(
    self,
    schedule_id: str,
    turn_id: str,
    observed_at: datetime,
) -> ObservationBundle: ...
```

Refactor the private builder to accept an explicit allowed-status set. The public ordinary method continues to pass `{OPEN, OBSERVING}`. The recovery method passes only `{THINKING}`, requires the supplied `turn_id` to equal `schedule.active_turn_id`, and performs the same two-phase schedule/turn rereads, authority verification, workspace projection, deadline checks, error sanitation, and final unchanged check. Do not add `THINKING` to `_ALLOWED_OBSERVATION_STATUSES`.

**Service API:**

```python
class ContextLifecycleService:
    def __init__(
        self,
        repository: object,
        observation_service: ObservationProjectionService,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None: ...

    def require_admission(
        self,
        policy: ContextBudgetPolicy,
        immutable_prefix_tokens: int,
    ) -> ContextTokenAccounting: ...

    def require_request_ready(
        self,
        policy: ContextBudgetPolicy,
        immutable_prefix_tokens: int,
        active_history_tokens: int,
    ) -> ContextTokenAccounting: ...

    def commit_compaction(
        self,
        *,
        checkpoint_id: str,
        schedule_id: str,
        observation: ObservationBundle,
        policy: ContextBudgetPolicy,
        immutable_system_contract_hash: str,
        immutable_prefix_tokens: int,
        checkpoint_authority_tokens: int,
        source_history_hash: str,
        event_cursor: int,
        public_reference_ids: tuple[str, ...],
        intended_operation: IntendedOperation,
        pre_compaction_accounting: ContextTokenAccounting,
        handoff: CompactionHandoff | None = None,
    ) -> RehydrationPlan: ...

    def recover(
        self,
        *,
        schedule_id: str,
        turn_id: str,
        observed_at: datetime,
        policy: ContextBudgetPolicy,
    ) -> RehydrationPlan: ...
```

Constructor invokes both capability guards on the same object. `commit_compaction()` requires the current turn to be `THINKING`, checks `>=80%`, transitions it to `COMPACTING` with existing turn CAS, loads latest parent, builds checkpoint-only receipt/plan, optionally adds valid handoff, then calls atomic repository commit. A failure after entering `COMPACTING` stays fail-closed and does not authorize dispatch. `recover()` rebuilds a THINKING observation through the explicit API, loads full lineage, and returns only the deterministic plan; it never invokes model/tool callbacks.

- [ ] **Step 1: Write RED observation tests**

```python
def test_ordinary_observation_still_rejects_thinking_turn() -> None:
    service = _service(turn_status=AgentTurnStatus.THINKING)
    with pytest.raises(ActiveObservationConflict):
        service.build_serial_public_observation("schedule-1", NOW)


def test_explicit_recovery_observation_accepts_only_bound_thinking_turn() -> None:
    service = _service(turn_status=AgentTurnStatus.THINKING)
    bundle = service.build_serial_public_recovery_observation(
        "schedule-1", "turn-1", NOW,
    )
    assert bundle.frame.identity.turn_id == "turn-1"
```

- [ ] **Step 2: Write RED service tests**

Test split/missing capability rejection, exact state sequence, 79.999% no compaction, 80% inclusive compaction, optional handoff drop, checkpoint-only overflow, atomic commit result, restart recovery, stale lineage, and zero external dispatch calls.

- [ ] **Step 3: Implement and run GREEN**

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_observation_service.py \
  tests/player_agents/test_context_lifecycle.py \
  -o addopts='' -q
git add werewolf_agent/player_agents/observation \
  werewolf_agent/player_agents/context/service.py \
  tests/player_agents/test_observation_service.py \
  tests/player_agents/test_context_lifecycle.py
git commit -m "feat: coordinate context compaction and recovery"
```

---

### Task 8: Add Shared Backend Conformance and Real PostgreSQL Service Tests

**Files:**
- Create: `tests/player_agents/test_context_checkpoint_conformance.py`
- Create: `tests/storage/test_postgres_context_checkpoints_live.py`

- [ ] **Step 1: Add shared Memory/SQLite canonical trace**

Run the same command sequence against both factories and compare serialized schedules, turns, checkpoint lineages, stable error codes, and recovery plan hashes. Include successful ordinals 1 and 2, stale CAS zero-partial-write, duplicate/fork/orphan/skip, defensive reads, and SQLite reopen.

```python
@pytest.mark.parametrize("repository_factory", [_memory_factory, _sqlite_factory])
def test_checkpoint_repository_canonical_trace(repository_factory) -> None:
    trace = _canonical_checkpoint_trace(repository_factory())
    assert trace == EXPECTED_CANONICAL_TRACE
```

- [ ] **Step 2: Add isolated live PostgreSQL fixture**

The live test reads `WOFKILL_CONTEXT_TEST_PG_DSN`. If absent it skips unless `WOFKILL_REQUIRE_REAL_POSTGRES=1`, in which case it fails. Connect to the service database with autocommit, create a unique `wofkill_context_<uuid>` database using `psycopg.sql.Identifier`, yield its DSN, close repositories/connections, terminate remaining sessions for that exact generated database, and drop only that generated database.

Live tests must verify:

- schema columns, FKs, unique constraints, and descending index;
- successful checkpoint insert plus turn CAS;
- a forced stale lineage leaves no row and no turn change;
- two repository instances racing the same ordinal produce exactly one success;
- advisory locking serializes the competing transactions;
- `close()` followed by a new repository instance reloads latest/full lineage and recovers the identical plan.

- [ ] **Step 3: Start the approved real service and run the mandatory gate**

```bash
docker compose up -d postgres
docker compose ps postgres
WOFKILL_CONTEXT_TEST_PG_DSN=postgresql://wofkill:wofkill-dev@localhost:5432/wofkill \
WOFKILL_REQUIRE_REAL_POSTGRES=1 \
conda run -n wofkill python -m pytest \
  tests/storage/test_postgres_context_checkpoints_live.py \
  -o addopts='' -vv -rs
```

Expected: all live tests pass, zero skips. Do not stop or delete the shared PostgreSQL volume unless the user separately asks; the fixture deletes only its UUID-named test database.

- [ ] **Step 4: Run all context/backend tests and commit**

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_context_contracts.py \
  tests/player_agents/test_context_lifecycle.py \
  tests/player_agents/test_context_checkpoint_conformance.py \
  tests/storage/test_context_checkpoints.py \
  tests/storage/test_postgres_context_checkpoints.py \
  tests/storage/test_postgres_context_checkpoints_live.py \
  -o addopts='' -q
git add tests/player_agents/test_context_checkpoint_conformance.py \
  tests/storage/test_postgres_context_checkpoints_live.py
git commit -m "test: verify context checkpoints across backends"
```

---

### Task 9: Enforce Import Boundaries, Run Regression Gates, Review, and Update Handoff

**Files:**
- Create: `tests/player_agents/test_context_import_boundary.py`
- Modify: `handoff.md`
- Modify only files implicated by verified failures from the gates below.

- [ ] **Step 1: Write and pass import-boundary tests**

Parse the AST of `werewolf_agent/player_agents/context/*.py` and
`werewolf_agent/storage/context_checkpoints.py`. Reject imports rooted at:

```python
FORBIDDEN_ROOTS = {
    "deepagents",
    "langgraph",
    "werewolf_agent.agents",
    "werewolf_agent.tools",
    "werewolf_agent.model_gateway",
    "werewolf_agent.rules",
    "werewolf_agent.runtime",
}
```

Also reject text references to `CommitTurn`, `GameRunner`, `PlayerAgent`,
`ToolGateway`, and `_dispatch_agent` outside docstrings used to state non-goals.

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_context_import_boundary.py \
  -o addopts='' -q
```

- [ ] **Step 2: Run focused tests and static checks**

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_context_contracts.py \
  tests/player_agents/test_context_lifecycle.py \
  tests/player_agents/test_context_checkpoint_conformance.py \
  tests/player_agents/test_context_import_boundary.py \
  tests/player_agents/test_observation_contracts.py \
  tests/player_agents/test_observation_service.py \
  tests/player_agents/test_observation_conformance.py \
  tests/storage/test_context_checkpoints.py \
  tests/storage/test_postgres_context_checkpoints.py \
  tests/storage/test_postgres_context_checkpoints_live.py \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_autonomous_turns.py \
  -o addopts='' -q
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents/context \
  werewolf_agent/player_agents/observation/service.py \
  werewolf_agent/storage/context_checkpoints.py \
  werewolf_agent/storage/memory_store.py \
  werewolf_agent/storage/sqlite_store.py \
  werewolf_agent/storage/postgres_store.py \
  tests/player_agents/test_context_contracts.py \
  tests/player_agents/test_context_lifecycle.py \
  tests/player_agents/test_context_checkpoint_conformance.py \
  tests/player_agents/test_context_import_boundary.py \
  tests/storage/test_context_checkpoints.py \
  tests/storage/test_postgres_context_checkpoints.py \
  tests/storage/test_postgres_context_checkpoints_live.py
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/player_agents/context \
  werewolf_agent/player_agents/observation/service.py \
  werewolf_agent/storage/context_checkpoints.py
git diff --check
```

- [ ] **Step 3: Run the full regression suite**

```bash
conda run -n wofkill python -m pytest -q
```

Record pass/fail/skip counts and duration. A green suite does not replace the explicit zero-skip live PostgreSQL command from Task 8.

- [ ] **Step 4: Invoke review and address only verified findings**

Use `superpowers:requesting-code-review`. Review against the approved design, focusing on transaction atomicity, lock order, exact 80%/55% boundaries, self-hash/lineage validation, recovery authority, sanitized errors, and scope exclusions. Process feedback with `superpowers:receiving-code-review`; rerun the narrow RED/GREEN test for every accepted correction.

- [ ] **Step 5: Update the canonical handoff**

Replace the current milestone with evidence-backed status:

- what context lifecycle capabilities now exist;
- exact focused, live PostgreSQL, static, and full-suite results;
- intentional non-goals versus blockers;
- reading order and exact verification commands;
- red lines: checkpoint-only authority, no Markdown parsing, no split repository, no process-local database fence, no ToolGateway/provider/AgentLoop/live path;
- one unique next milestone taken from the parent autonomous-runtime design.

- [ ] **Step 6: Verify completion evidence and commit**

Invoke `superpowers:verification-before-completion`, rerun any evidence it requires, then:

```bash
git add werewolf_agent/player_agents/context \
  werewolf_agent/player_agents/observation \
  werewolf_agent/storage/context_checkpoints.py \
  werewolf_agent/storage/memory_store.py \
  werewolf_agent/storage/sqlite_store.py \
  werewolf_agent/storage/postgres_store.py \
  tests/player_agents/test_context_contracts.py \
  tests/player_agents/test_context_lifecycle.py \
  tests/player_agents/test_context_checkpoint_conformance.py \
  tests/player_agents/test_context_import_boundary.py \
  tests/player_agents/test_observation_service.py \
  tests/storage/test_context_checkpoints.py \
  tests/storage/test_postgres_context_checkpoints.py \
  tests/storage/test_postgres_context_checkpoints_live.py \
  tests/storage/test_sqlite_migrations.py \
  handoff.md
git commit -m "docs: hand off autonomous context lifecycle"
```

## Completion Criteria

The milestone is complete only when all of the following are evidenced in the current checkout:

1. Strict contracts and stable errors pass their focused tests.
2. 55% admission, 80% inclusive trigger, and 55% inclusive rehydration use exact integer comparisons.
3. Checkpoint self-hash, parent lineage, authority, deadlines, budgets, and source hashes are validated.
4. Invalid or oversized handoff is dropped without weakening checkpoint-only continuation.
5. Memory, SQLite, and PostgreSQL atomically insert the checkpoint and CAS `COMPACTING -> THINKING`.
6. SQLite reopen and PostgreSQL reconnect recover identical full lineage and deterministic plans.
7. Real PostgreSQL tests pass with zero skips, including two-connection contention.
8. Ordinary observation still rejects `THINKING`; only the explicit recovery API accepts the bound THINKING turn.
9. Import-boundary, Ruff, mypy, focused regression, `git diff --check`, and full pytest gates pass.
10. `handoff.md` states that this is a context-lifecycle foundation, not a model/tool/live playable slice.
