# Serial Public Scheduler and Host Runtime Design

Date: 2026-07-30
Status: Draft for review; design direction approved
Owner: Codex development session
Parent: `docs/superpowers/specs/2026-07-28-autonomous-player-agent-runtime-design.md`

## 1. Purpose and Scope

This document defines the next bounded stage of the autonomous-player rewrite:
durable `serial_public` scheduling and the host-owned player-turn lifecycle.
It builds on the completed contract, `CommitTurn`, and durable-dispatch stages.

The stage delivers a new isolated runtime path that can:

1. persist a serial public speaking schedule and its currently active turn;
2. admit one fresh `AgentTurn` at a time in speaker order;
3. advance, cancel, expire, or replace that turn through compare-and-swap
   transitions;
4. recover safely after restart by reconciling durable dispatch before any new
   turn work begins; and
5. use identical lifecycle semantics in memory, SQLite, and PostgreSQL.

The following remain explicitly out of scope:

- `ModelRouter`, provider invocation, tool invocation, and an `AgentLoop`;
- player workspace, observations, context budgets beyond the existing turn
  contract, compaction, reflection, RAG, and skills;
- `RuleEngine`, legacy `GameRunner`, `_dispatch_agent`, old player code, and
  feature-gate integration;
- proposal validation, `CommitTurn` execution, rendering, and public-game
  routing.

The new modules must not import legacy player-decision modules. They are a
tested orchestration capability, not a compatibility adapter.

## 2. Design Decisions

### 2.1 Repository-backed orchestration

`HostRuntime` uses a new explicit `AutonomousTurnRepository` capability. An
in-memory-only scheduler is rejected because it loses active turns on restart.
Reconstructing the scheduler from committed events is also rejected: an
unsubmitted, cancelled, or expired turn has no canonical game event from which
to recover its lifecycle state.

The capability is implemented by the existing in-memory, SQLite, and
PostgreSQL repositories. A repository that does not explicitly support it
cannot enable this runtime.

### 2.2 Slots are durable; turns are admitted just in time

A `SerialPublicSchedule` stores an ordered immutable list of speaker slots,
but it does not pre-create an `AgentTurn` for every speaker. The host admits a
turn only when its slot becomes current. The caller supplies a fresh
`TurnAdmission` containing the current revision, read set, view fingerprint,
lease hash, budget, deadline, turn ID, and idempotency key.

This distinction is required because every committed public utterance changes
the next speaker's observable state. Pre-creating later turns would pin stale
revisions and observations before the prior speaker finishes.

### 2.3 Host owns mechanics, never strategy

`HostRuntime` checks schedule order, lifecycle status, deadlines, repository
capabilities, recovery barriers, and dispatch cancellation. It does not select
a player action, score evidence, interpret a result, or call a model. A later
AgentLoop stage supplies those capabilities through explicit host APIs.

### 2.4 Recovery is a hard admission barrier

For a persisted game with an unfinished schedule, a newly created
`HostRuntime` must call `recover_game(game_id)` before it admits or transitions
an active turn. Recovery delegates to the existing `DispatchReconciler`.
The barrier opens only when its `RecoveryReport.barrier_open` is true and the
durable-dispatch repository accepts new work. A pending, malformed, or errored
reconciliation report keeps the game blocked. `UNKNOWN_OUTCOME` is
already terminal in the durable-dispatch state machine: it remains auditable
and reports budget consumption, but does not itself keep the barrier closed.

Freshly created schedules have no prior runtime process to recover and may be
admitted immediately, subject to `assert_dispatch_allowed(game_id)`.

## 3. Contracts and State Model

### 3.1 New strict contracts

Create `werewolf_agent.player_agents.contracts.scheduling` with the following
strict, immutable models and enums:

| Name | Required responsibility |
| --- | --- |
| `SerialPublicSlot` | Immutable ordinal and participant identity for one speaker position. |
| `SerialPublicScheduleStatus` | Closed enum: `open`, `closed`, `cancelled`. |
| `SerialPublicSchedule` | Window binding, ordered slots, next-slot ordinal, active turn ID, CAS version, timestamps, and schedule status. |
| `TurnAdmission` | All fresh values required to construct one `AgentTurn` for the current slot. |
| `ManagedAgentTurn` | Persisted `AgentTurn` plus schedule ID, state version, timestamps, and optional terminal reason. |
| `TerminalDisposition` | Closed enum: `advance`, `replace`, `close`. |

`SerialPublicSchedule` validates all of the following:

- `window.conflict_class is ConflictClass.SERIAL_PUBLIC`;
- schedule and window game IDs match;
- slot ordinals are contiguous from zero and slot participants are unique;
- `next_slot_ordinal` is within the slot range while the schedule is open;
- `active_turn_id` is absent when no turn is admitted, and is present only for
  an open schedule with an unconsumed current slot;
- every timestamp is timezone-aware.

`TurnAdmission` is bound to one schedule and one current slot. It is converted
to the existing `AgentTurn` only after checking game ID, participant ID, task
type, window ID/version, and deadline against the schedule's
`LegalActionWindow`. The supplied revision remains authoritative; the
scheduler never invents a revision or observation fingerprint.

`ManagedAgentTurn` is the persistence envelope. Its independent
`state_version` is the CAS counter. Its embedded `AgentTurn.status` remains
the single lifecycle status; the envelope must not duplicate a second status
field.

### 3.2 Lifecycle and scheduling invariants

The existing `transition_turn` graph remains authoritative. The new repository
enforces these additional invariants:

1. a schedule admits at most one active turn at a time;
2. only the schedule's `active_turn_id` may make a non-terminal transition;
3. a terminal transition is atomic with updating the schedule pointer and
   slot cursor;
4. `advance` terminalizes the active turn and consumes its slot;
5. `replace` terminalizes the active turn without consuming the slot; a later
   admission creates a new turn with a new ID and idempotency key;
6. `close` terminalizes the active turn and closes the schedule without
   admitting any later slot; and
7. a closed or cancelled schedule cannot admit another turn.

The scheduler is poll-driven. `HostRuntime.expire_due_turns(now)` finds an
active turn whose deadline is not later than `now` and finishes it with
`AgentTurnStatus.EXPIRED`. It does not start a background thread, which keeps
tests and restart behavior deterministic.

### 3.3 Persistent operations

`AutonomousTurnRepository` provides these narrow operations:

```text
supports_autonomous_turns() -> bool
create_serial_public_schedule(schedule) -> SerialPublicSchedule
load_serial_public_schedule(schedule_id) -> SerialPublicSchedule | None
load_active_serial_public_schedule(game_id) -> SerialPublicSchedule | None
admit_serial_public_turn(schedule_id, expected_schedule_version, admission) -> ManagedAgentTurn
transition_active_turn(turn_id, expected_turn_version, next_status) -> ManagedAgentTurn
finish_active_turn(
    schedule_id,
    expected_schedule_version,
    turn_id,
    expected_turn_version,
    terminal_status,
    disposition,
    reason_code,
) -> SerialPublicSchedule
load_managed_turn(turn_id) -> ManagedAgentTurn | None
```

All mutating operations use CAS and fail with stable lifecycle errors instead
of silently overwriting a newer schedule or turn. `finish_active_turn` validates
that the passed turn is active and that the supplied terminal status is one of
`CANCELLED`, `EXPIRED`, or `COMMITTED` before atomically changing both
records.

SQLite performs these operations in one `BEGIN IMMEDIATE` transaction. The
PostgreSQL implementation locks the schedule row in its existing schema
transaction. The in-memory implementation uses the repository lock and
prepare-then-publish copies. All three store canonical JSON payloads with
separate indexed identity, status, and version columns.

### 3.4 Dispatch cancellation extension

Extend `DurableDispatchRepository` with a read operation that returns attempts
for exactly one `(game_id, turn_id)` in deterministic creation order. All three
backends implement it. `HostRuntime` uses it only while cancelling or expiring
the active turn:

1. load that turn's `PENDING` and `DISPATCHING` attempts;
2. CAS-cancel each attempt with a stable reason code;
3. finish the active turn using the requested terminal disposition.

Already dispatched or recorded attempts are retained for audit. They cannot be
used by a later runtime operation because the enclosing turn is terminal. A
result racing with cancellation may be persisted, but it cannot trigger a game
action in this stage and must be ignored by later consumers unless the active
turn, lease, window, and revision still match.

## 4. Host Runtime API

Create `werewolf_agent.player_agents.runtime.host` and
`werewolf_agent.player_agents.runtime.serial_public`. The public coordinator
is constructed with an `AutonomousTurnRepository`, a
`DurableDispatchRepository`, a `DispatchReconciler`, and an injected UTC
clock for deterministic tests.

Its public methods are:

```text
create_schedule(schedule) -> SerialPublicSchedule
recover_game(game_id) -> RecoveryReport
admit_next_turn(schedule_id, admission) -> ManagedAgentTurn
transition_active_turn(turn_id, expected_turn_version, next_status) -> ManagedAgentTurn
cancel_active_turn(schedule_id, reason_code, disposition) -> SerialPublicSchedule
expire_due_turns(now) -> tuple[SerialPublicSchedule, ...]
load_active_turn(game_id) -> ManagedAgentTurn | None
```

`admit_next_turn` and `transition_active_turn` reject a persisted game until
`recover_game` has opened its barrier. They also call
`assert_dispatch_allowed(game_id)` before authorizing work. The runtime tracks
successful recovery only in process memory; durable dispatch remains the source
of truth after another restart.

The host exposes no generic repository object to callers and never accepts a
model callback. This keeps future model/tool dispatch integration explicit and
prevents a new adapter around the legacy runtime.

## 5. Data Flow

```text
create schedule
  -> persist ordered slots, no AgentTurn
  -> admit current slot with fresh TurnAdmission
  -> persist ManagedAgentTurn and active_turn_id by CAS
  -> non-terminal lifecycle transitions by CAS
  -> cancel/expire: cancel unresolved dispatches, then finish active turn
  -> atomically consume, retain, or close the current slot
  -> later admission receives a new revision and observation

restart with persisted schedule
  -> HostRuntime.recover_game
  -> DispatchReconciler reconciles earlier external I/O
  -> barrier open: inspect/admit only the valid active or next slot
  -> barrier closed: reject all new turn work
```

No operation in this stage submits a proposal or changes canonical game truth.
`CommitTurn` remains the later authority that will terminally commit a
validated speech proposal.

## 6. Errors and Observability

Create stable host/runtime errors for unsupported turn storage, stale schedule
or turn versions, invalid slot admission, inactive-turn transition, recovery
required, recovery blocked, and closed schedule. Error messages contain only
opaque IDs and safe reason codes; they never embed player-private state or
dispatch payloads.

The stage records no new broad telemetry pipeline. It returns immutable
schedule, turn, and `RecoveryReport` values so the following stage can attach
critical audit records without changing lifecycle semantics.

## 7. Test Strategy and Acceptance Criteria

Tests must be written before each production change. They cover:

1. strict contract validation for schedules, slots, admissions, and immutable
   managed turns;
2. ordered admission and the guarantee that a later slot cannot run first;
3. CAS conflicts for duplicate admission, non-terminal transition, and finish;
4. all terminal dispositions, including replacement using a fresh turn ID and
   idempotency key;
5. expiry through an injected clock with no background worker;
6. cancellation of pending and dispatching attempts while recorded attempts
   stay readable but unusable;
7. recovery barriers for pending and resolver-error outcomes, plus the
   terminal, budget-consuming `UNKNOWN_OUTCOME` path for unsafe recovery;
8. identical in-memory and SQLite semantics, SQLite rollback on a forced
   write failure, and PostgreSQL DDL/lock/capability tests without a live
   server; and
9. an AST import-boundary check proving the new runtime imports neither legacy
   player-decision modules nor `ModelRouter` or `_dispatch_agent`.

Completion requires focused tests, scoped Ruff and mypy checks, `git diff
--check`, and the full pytest suite. No live game path changes in this stage.

## 8. Follow-up Boundary

The next stage may add observation projection, player workspace, context
budgeting, and the first AgentLoop. It must consume this host API rather than
modifying scheduler ordering or bypassing recovery/cancellation checks.
Real model and tool invocation will create durable dispatch attempts only
through a later explicit dispatcher; it is deliberately absent here.
