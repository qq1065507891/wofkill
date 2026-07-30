# Autonomous Player Durable Active-Turn Fence Design

Date: 2026-07-31
Status: Draft for written review; design direction approved
Owner: Codex development session
Parent: `docs/superpowers/specs/2026-07-28-autonomous-player-agent-runtime-design.md`

## 1. Purpose

This design closes the durable race between external dispatch creation and
`serial_public` turn terminalization. A future production dispatcher must not
create an externally executable request after another process has cancelled,
expired, replaced, advanced, or committed the active turn.

The stage adds one repository-owned atomic competition boundary that binds a
new dispatch attempt to the exact persisted schedule and managed turn. It also
makes HostRuntime cancellation, expiry, replacement, advancement, and
completion use that same boundary.

This stage does not:

- call a provider or external tool;
- add an AgentLoop, observation projection, workspace, context compaction,
  proposal validation, RuleEngine call, or `CommitTurn` caller;
- connect the new runtime to `GameRunner`, a live game, old `PlayerAgent`, old
  prompts, `ModelRouter`, or `_dispatch_agent`;
- implement ToolResult Markdown projection; or
- change the durable dispatch recovery state machine.

## 2. Existing Race

The current repositories persist schedules, managed turns, and dispatch
attempts, but production-safe attempt creation is absent. The existing
`create_dispatch()` checks only game existence, dispatch idempotency, and the
game recovery barrier. It does not verify that:

- the schedule remains open;
- the same turn remains active;
- schedule and turn CAS versions still match;
- actor, window, revision, view, lease, and deadline match the managed turn; or
- the turn is still non-terminal.

HostRuntime cancellation and expiry currently scan and cancel dispatches in
separate transactions before calling `finish_active_turn()`. Another process
can insert an attempt between that scan and terminalization. A process-local
lock cannot close this gap across repository instances.

PostgreSQL also uses different lock families on the two paths today:
schedule/turn writes use the per-game advisory transaction lock, while plain
dispatch writes lock the `games` row. These are not a shared competition
boundary.

## 3. Approaches Considered

### 3.1 Selected: explicit combined repository capability

Add an `ActiveTurnFenceRepository` capability implemented by the same concrete
repository object that owns schedules, managed turns, and dispatch attempts.
Its creation and terminal operations lock and mutate all relevant records in
one backend transaction.

Advantages:

- one durable source of truth and one lock order;
- exact CAS behavior in Memory, SQLite, and PostgreSQL;
- legacy durable dispatch recovery remains readable;
- no second fence token store or process-local coordination; and
- future production dispatcher code receives a narrow API that cannot confuse
  plain durable intent storage with active-turn authorization.

### 3.2 Rejected: make plain `create_dispatch()` schedule-aware

Changing the existing method globally would break the independently tested
durable-dispatch protocol and historical rows that predate scheduler state.
Recovery and low-level state-machine tests need to load and reason about those
unfenced attempts. A production-safe path must be explicit instead of silently
changing the meaning of the existing method.

### 3.3 Rejected: separate fence-token table or in-process lock

A separate epoch table would create another authority that must remain
consistent with schedule and turn JSON. An in-process lock does not coordinate
multiple hosts. Both approaches add complexity without improving the atomic
guarantee provided by the repository transaction itself.

## 4. Contract Changes

### 4.1 Persistent fence identity

Add the strict frozen contract:

```text
ActiveTurnDispatchFence
  schedule_id
  schedule_state_version
  turn_state_version
  window_id
  window_version
  base_game_revision
```

`DispatchAttempt` gains:

```text
active_turn_fence: ActiveTurnDispatchFence | None = None
```

The existing attempt fields complete the binding:

```text
game_id
turn_id
actor_id
lease_hash
view_fingerprint
deadline
```

The nested fence is optional only for historical and low-level unfenced
durable-dispatch rows. `create_active_turn_dispatch()` requires it to be absent
on input and persists a repository-generated value. Plain `create_dispatch()`
rejects attempts that already carry a fence, so a caller cannot manufacture a
fenced-looking row while bypassing active-turn validation.

`turn_state_version` is the managed-turn version after the repository reserves
the dispatch. This post-reservation value lets a result consumer later prove
that no lifecycle or newer-dispatch mutation occurred after authorization.

### 4.2 Managed-turn CAS meaning

`ManagedAgentTurn.state_version` remains the sole persisted CAS counter for the
managed-turn envelope. Its meaning expands from lifecycle-only mutation to any
authorized managed-turn mutation:

- non-terminal lifecycle transition;
- active-turn dispatch reservation; or
- terminal lifecycle transition.

Creating a fenced attempt increments `state_version` and `updated_at` without
changing `AgentTurn.status`. This is required for a true single-winner race:

- if dispatch creation wins, a terminal caller holding the old turn version
  receives `TurnStateConflict`;
- if terminalization wins, dispatch creation finds a terminal or inactive turn
  and is rejected; and
- neither operation can publish a partial result.

No second dispatch generation counter is introduced.

## 5. Repository Capability

Create `werewolf_agent/storage/active_turn_fence.py` with a combined explicit
capability:

```python
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

The capability guard accepts the repository only when:

1. `supports_autonomous_turns()` is explicitly true;
2. `supports_durable_dispatch()` is explicitly true;
3. `supports_active_turn_fence()` is explicitly true; and
4. the turn, dispatch, and fence arguments are the same repository object.

The guard never infers atomicity from method names. HostRuntime retains its
current constructor shape for compatibility, but construction fails when two
different repository objects are supplied. Cross-database transactions are
not emulated.

Stable fence errors are:

| Error | Code | Meaning |
| --- | --- | --- |
| `ActiveTurnFenceUnsupported` | `active_turn_fence_unsupported` | Repository does not explicitly provide the combined capability. |
| `ActiveTurnFenceRejected` | `active_turn_fence_rejected` | Captured identity, context, status, or deadline is no longer valid. |
| `ActiveTurnFenceTransactionError` | `active_turn_fence_transaction_error` | Backend failure rolled back the combined transaction. |

Existing exact CAS, idempotency, and recovery errors remain authoritative:
`ScheduleStateConflict`, `TurnStateConflict`,
`DispatchIdempotencyConflict`, and `DispatchRecoveryBlocked`.

Safe error messages never include player-private state, request payloads,
provider output, or hidden identities beyond caller-supplied opaque IDs.

## 6. Dispatch Reservation Transaction

`create_active_turn_dispatch()` performs these steps in one transaction:

1. require a new `PENDING`, state-version-zero attempt with no reason code and
   no caller-supplied active-turn fence;
2. acquire the per-game mutation boundary;
3. load and lock the exact schedule and managed turn;
4. compare both caller-supplied CAS versions;
5. require an open schedule whose `active_turn_id` equals `turn_id`;
6. require a non-terminal managed turn owned by the schedule;
7. compare game, turn, actor, current slot, window ID/version, base game
   revision, view fingerprint, and model lease hash exactly;
8. require timezone-aware `observed_at` and require both the attempt deadline
   and window deadline to be later than it;
9. require `attempt.deadline <= turn.window.deadline`;
10. check dispatch ID and executor/provider idempotency uniqueness;
11. check the game recovery barrier for `DISPATCHING` or `DISPATCHED` attempts;
12. increment the managed-turn state version and updated timestamp;
13. construct the fence from locked authoritative records, using the new
    managed-turn version;
14. insert the fenced attempt and update the managed turn; and
15. commit both writes together.

Any validation, CAS, uniqueness, or backend failure leaves neither the attempt
nor the managed-turn version change visible.

The deadline comparison is intentionally not equality. A request may use a
shorter operation deadline, but it may never outlive the legal action window.

## 7. Fenced Terminal Transaction

`finish_active_turn_fenced()` uses the same game-level lock and the same
schedule/turn CAS identity as dispatch reservation.

For `CANCELLED` and `EXPIRED`:

1. load and validate the active schedule and managed turn;
2. atomically change every exact-turn `PENDING` or `DISPATCHING` attempt to
   `CANCELLED`, incrementing each dispatch state version and recording the
   supplied safe reason code;
3. leave `DISPATCHED`, `RESULT_RECORDED`, `UNKNOWN_OUTCOME`, and already
   `CANCELLED` attempts immutable; and
4. terminalize the managed turn and apply `advance`, `replace`, or `close` in
   the same transaction.

For `COMMITTED`:

- the turn must follow the existing lifecycle into `VALIDATING`;
- any exact-turn `PENDING`, `DISPATCHING`, or `DISPATCHED` attempt rejects the
  operation with `DispatchRecoveryBlocked`; and
- only after all dispatched work is terminal may the method commit the
  managed-turn and schedule updates.

This preserves the existing rule that a dispatched unknown request remains
durable and blocks replacement admission until recovery. Completion cannot
hide unfinished work.

The old repository `finish_active_turn()` remains a low-level scheduling
contract for existing tests and non-production migration code. HostRuntime no
longer calls it. All production-facing terminal methods use the fenced
operation.

## 8. HostRuntime Surface

HostRuntime adds:

```text
create_active_turn_dispatch(schedule_id, attempt) -> DispatchAttempt
```

It performs only process-level coordination:

1. load and capture the schedule and active managed turn;
2. require successful game recovery;
3. pass the captured CAS versions, attempt, and injected aware clock value to
   the combined repository method; and
4. return the repository-generated fenced attempt.

The repository repeats every authoritative check under its transaction. The
Host read is not the fence.

`complete_active_turn()`, `cancel_active_turn()`, and `expire_due_turns()` pass
their originally captured schedule/turn versions to
`finish_active_turn_fenced()`. The current pre-terminal dispatch scan is
removed. `replace`, `advance`, and `close` therefore share the same durable
competition boundary.

HostRuntime does not expose the repository and does not accept provider/model
callbacks. Network I/O remains outside this stage.

## 9. Backend Semantics and Lock Order

All backends use this logical order:

```text
game -> schedule -> managed turn -> dispatch attempts
```

### 9.1 Memory

Use the existing repository `RLock`. Prepare defensive copies of the managed
turn, attempt, dispatch indexes, and terminal updates before publishing. Fault
injection after either prepared write must restore every container and index.

### 9.2 SQLite

Use one `BEGIN IMMEDIATE` transaction. Load schedule and turn inside that
transaction, execute CAS updates with row-count checks, and insert or cancel
attempts before commit. Any exception rolls back the entire unit.

Add nullable `active_turn_fence_json TEXT` to
`autonomous_dispatch_attempts`. Fresh schema includes the column. Repository
startup checks `PRAGMA table_info` and performs an idempotent `ALTER TABLE` for
older databases before declaring the capability ready.

Historical NULL rows remain valid unfenced durable-dispatch records. They are
never accepted as production active-turn authorizations.

### 9.3 PostgreSQL

Acquire `pg_advisory_xact_lock(hashtextextended(game_id, 0))` before row
locks. Then lock the schedule, managed turn, and relevant dispatch rows with
`FOR UPDATE`. Dispatch reservation, transition, cancellation, and terminal
paths that participate in the fence all use this advisory boundary.

Add nullable `active_turn_fence_json JSONB` with
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. Existing rows remain NULL.

Unique conflicts are mapped only when SQLSTATE/PGCODE is `23505` and the exact
known constraint identifies the dispatch primary key or executor/provider-key
index. Other failures become `ActiveTurnFenceTransactionError` after rollback.

## 10. Recovery and Result-Consumption Boundary

The existing `DispatchReconciler` behavior does not change. Unfenced legacy
attempts and fenced attempts use the same durable state machine and provider
idempotency key rules.

A later result consumer must revalidate before using a recorded result:

- schedule is still open and has the same active turn;
- schedule version equals the attempt fence;
- managed-turn version equals `turn_state_version` from the fence;
- window ID/version and base revision still match;
- lease hash and view fingerprint still match; and
- the deadline is still valid.

This stage persists the information required for that check but does not add a
provider result consumer or AgentLoop.

## 11. Failure and Concurrency Semantics

The required outcomes are:

| Race | Legal outcome |
| --- | --- |
| Dispatch creation vs completion | Exactly one initial CAS operation succeeds. A pre-existing unresolved attempt blocks completion. |
| Dispatch creation vs cancel/expire | Exactly one initial CAS operation succeeds. Retried terminalization may atomically cancel the winner before finishing. |
| Dispatch creation vs replace/advance/close | The losing captured identity receives a stable schedule or turn conflict; no replacement is accidentally targeted. |
| Dispatch creation vs non-terminal transition | Exactly one captured turn version wins; caller reloads before retrying. |
| Terminalization vs dispatch state transition | Shared game boundary serializes them; cancellation either wins first or observes and cancels the latest cancellable state. |
| Backend failure after attempt insert | Transaction rollback removes the attempt and managed-turn version bump. |
| Backend failure after dispatch cancellation | Transaction rollback restores attempts, schedule, and managed turn. |

Retries always reload authoritative state. They never silently reuse a stale
fence or create a new dispatch ID to hide an earlier unknown outcome.

## 12. Test Strategy

### 12.1 Contracts and pure preparation

- strict/frozen `ActiveTurnDispatchFence` JSON round-trip;
- attempt round-trip with and without a fence;
- plain create rejects a caller-supplied fence;
- every identity mismatch, terminal state, late deadline, and deadline beyond
  the window is rejected;
- successful preparation increments only the managed envelope version and
  persists the post-reservation version in the fence.

### 12.2 Shared Memory and SQLite behavior

- successful fenced creation and restart round-trip;
- recovery barrier failure leaves no partial attempt or turn-version bump;
- dispatch ID and provider-key conflicts roll back the turn-version bump;
- completion rejects all unresolved exact-turn statuses;
- cancel/expire atomically cancel `PENDING` and `DISPATCHING` attempts;
- `DISPATCHED` survives terminalization and blocks replacement admission;
- forced insert, turn-update, schedule-update, and cancellation failures roll
  back every record; and
- real thread races assert the single-winner CAS outcomes described above.

SQLite concurrency tests use two repository instances connected to the same
database file, not only threads sharing one Python lock.

### 12.3 PostgreSQL contract tests

- schema and idempotent nullable-column upgrade SQL;
- advisory game lock precedes schedule/turn/dispatch `FOR UPDATE` statements;
- exact binding queries and CAS update parameters;
- insert and terminal rollback with stateful fake connections;
- exact `23505` constraint mapping; and
- capability remains false before complete schema initialization.

A real PostgreSQL service integration remains a pre-production repository gate
and is not fabricated by mocks in this stage.

### 12.4 Runtime and boundary tests

- HostRuntime rejects distinct turn and dispatch repositories;
- HostRuntime creates only repository-generated fenced attempts;
- completion, cancel, expiry, replace, advance, and close use the fenced API;
- restarted games still require recovery;
- AST import-boundary tests continue to reject legacy player decision modules,
  `ModelRouter`, and `_dispatch_agent`; and
- no live game path, ToolResult Markdown projection, or old player path changes.

## 13. Acceptance Criteria

The stage is complete only when:

1. Memory, SQLite, and PostgreSQL expose equivalent explicit fence capability;
2. every production-facing dispatch attempt contains the persisted exact fence;
3. attempt reservation increments the managed-turn CAS in the same transaction;
4. HostRuntime terminal operations use the same combined repository boundary;
5. no tested race leaves a terminal turn with a newly executable attempt;
6. no fault-injection point leaves a partial attempt, partial cancellation, or
   partial schedule/turn mutation;
7. focused runtime/storage tests, Ruff, mypy, and `git diff --check` pass;
8. the full pytest suite passes before merge; and
9. the diff contains no legacy PlayerAgent/live-game connection and no
   ToolResult Markdown projection implementation.

## 14. Final Invariants

1. The repository transaction, not a Host pre-check, is the active-turn fence.
2. A fenced attempt is authorized against one exact schedule and managed-turn
   CAS identity.
3. Schedule, active turn, window, revision, actor, view, lease, and deadline
   are checked before any attempt becomes durable.
4. Dispatch reservation and turn terminalization compete by the same managed
   turn CAS and game-level lock.
5. Cancellation and expiry never use a second dispatch scan outside the
   terminal transaction.
6. Completion never hides unresolved external work.
7. Historical unfenced attempts remain recoverable but cannot authorize new
   production work.
8. Result recording does not grant authority to consume a result after its
   active-turn fence becomes stale.
9. No process-local coordination or cross-repository composition is presented
   as durable atomicity.
10. This stage remains isolated from legacy player decisions, live games, and
    ToolResult Markdown presentation.
