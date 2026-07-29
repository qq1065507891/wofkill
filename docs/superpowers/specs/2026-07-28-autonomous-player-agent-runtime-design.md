# Autonomous Player Agent Runtime Design

Date: 2026-07-28
Last revised: 2026-07-29
Status: Approved design; execution contracts completed; implementation plan not yet written
Owner: Codex development session

## 1. Decision Summary

This document is the sole authoritative design for the player-agent rewrite.
It replaces the previous structured-speech foundation design and plan.

The approved direction is a clean rewrite of the player decision subsystem:

- preserve `RuleEngine`, `GameState`, `GameEvent`, event replay, ruleset truth,
  and the host's visibility and persistence boundaries;
- replace `PlayerAgent`, its prompt pipeline, directives, retry/fallback logic,
  text-to-action recovery, role strategy handlers, and player-facing tool flow;
- run one bounded autonomous agent loop for each legal action opportunity;
- retain each player's identity, documents, beliefs, commitments, and approved
  lessons across action turns;
- give the agent phase- and role-scoped tools that it chooses to call;
- make reflection a player-callable, host-validated candidate-memory write;
- make every game-changing output a strict structured proposal;
- retain a strongly characterized judge presenter for public narration, but
  never give it rules authority or hidden-state inference authority;
- treat audit as a first-class tamper-evident subsystem, not ordinary logs.

The existing `werewolf_agent/agents/speech_act_schemas.py` implementation and
its tests are rejected artifacts. No new module may import them, adapt them, or
use them as a compatibility contract. They remain only behind the legacy
runtime entry point until the replacement vertical slice has passed replay and
cutover gates; they are deleted in the final migration task, not before a
working replacement exists.

## 2. Public Design Basis

This design borrows public, documented agent patterns. It does not assume
Claude Code's private prompts, internal reasoning, or undocumented compaction
implementation.

Relevant Claude Code patterns are:

- concise checked-in Markdown carries durable instructions;
- a small memory index can point to detailed topic files loaded only when
  needed;
- context is progressively disclosed rather than always injected in full;
- the model may choose tools, inspect their results, revise its approach, and
  continue until it invokes a terminal action;
- Markdown guidance is context, while permissions and mandatory behavior are
  enforced by the host;
- compaction preserves task state while removing redundant conversational
  history;
- strict tool use and structured outputs constrain model-generated arguments.

Relevant public Codex patterns are:

- automatic compaction starts at a configured token threshold, counted against
  either the whole active context or the body after a carried prefix;
- manual and automatic compaction share an explicit lifecycle and observable
  `contextCompaction` event;
- earlier history is replaced by a concise continuation summary while canonical
  initial context is reinjected rather than trusted to the summary;
- the public compaction prompt asks for progress, decisions, constraints,
  remaining work, and critical references needed by a successor model;
- repeated compaction is treated as lossy and monitored rather than assumed to
  preserve unlimited conversation quality.

References:

- <https://code.claude.com/docs/en/memory>
- <https://code.claude.com/docs/en/context-window>
- <https://code.claude.com/docs/en/skills>
- <https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use>
- <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>
- <https://developers.openai.com/codex/codex-manual.md#keep-transcripts-lean-with-compact>
- <https://developers.openai.com/codex/config-reference/>
- <https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs>
- <https://github.com/openai/codex/blob/main/codex-rs/prompts/templates/compact/prompt.md>

The project applies those patterns to a hidden-information game. Markdown is a
bounded player-facing projection; the host owns permissions and side effects;
the RuleEngine owns game mechanics; strict schemas do not replace semantic,
visibility, revision, or legality validation.

## 3. Problem

The current player implementation is primarily a constrained one-shot action
generator. Runtime code assembles a large prompt containing visible state,
role directives, strategy instructions, memory, RAG, skills, quality rules,
and output requirements. The model receives preselected advice rather than
owning a durable observe-tool-reflect-submit loop.

This creates five structural problems:

1. player intelligence is split across prompt code, directives, regexes,
   strategy handlers, retry branches, and model output;
2. the prompt grows with game history and competes with current evidence;
3. heuristics frequently decide what a smart player should do instead of
   enforcing only legality and visibility;
4. memory, RAG, reflection, and world-model components exist but are not owned
   by a coherent player agent lifecycle;
5. free-form language and text recovery blur public facts, hypotheses,
   quotations, commitments, and private reasoning.

The rewrite must make each player an autonomous, bounded agent without making
game rules or hidden information probabilistic.

## 4. Goals

1. Give every player a persistent logical identity and isolated workspace.
2. Run one bounded autonomous loop for every legal speech, vote, night action,
   or role-ability opportunity.
3. Let the player decide which authorized tools and memories to consult.
4. Keep default context size bounded independently of total game length.
5. Separate facts, observations, hypotheses, commitments, and lessons.
6. Let the player reflect during an action without allowing it to rewrite
   facts or durable memory directly.
7. Preserve and measure useful RAG, reflection, and world-model behavior.
8. Make every state-changing proposal strict, visible, legal, revision-pinned,
   and auditable.
9. Preserve an expressive judge personality without giving the judge hidden
   knowledge or rules authority.
10. Support replay, pause/resume, cancellation, crash recovery, and concurrent
    player turns.
11. Remove strategy hard-coding while keeping rules, security, and visibility
    deterministic.
12. Prove value through ablation and human-quality evaluation, not through
    tool-call counts or prompt length.

### 4.1 Delivery Boundaries

The rewrite is deliberately staged. A later-stage capability is not enabled
merely because its package or schema exists.

1. **Foundation and daytime speech:** versioned serial commits, durable turn
   state, strict daytime `SpeechProposal`, public semantic records, deterministic
   rendering, critical audit/outbox, and restart recovery. This is the first
   playable vertical slice.
2. **Independent and team action windows:** votes, ordinary private role
   actions, explicitly commutative multi-player windows, and the wolf-team
   coordination protocol in section 14.
3. **Cognition extensions:** world model, in-turn reflection beyond structural
   validation, RAG, skills, and cross-game semantic memory, each behind its own
   feature gate and ablation gate.
4. **Cutover:** migrate the runtime entry point per game, retain historical
   replay readers, then delete legacy player-decision code only after all
   cutover gates pass.

The first stage intentionally schedules public speech serially. It does not
claim concurrent public-speaking turns before the commit protocol and declared
dependency sets can prove that such turns are safe.

## 5. Non-Goals

1. Do not let a model adjudicate legal actions, role abilities, phase flow, or
   victory.
2. Do not preserve the old `PlayerAgent` API, prompt contracts, directive
   behavior, retry semantics, fallback output, or agent-focused tests.
3. Do not reuse the rejected SpeechAct implementation.
4. Do not make Markdown the source of truth or a direct write channel.
5. Do not expose arbitrary filesystem, shell, moderator, model-routing,
   retry-control, budget-control, or event-append tools to a player.
6. Do not store or reconstruct hidden chain-of-thought.
7. Do not treat RAG, memory, skill advice, or possible worlds as game facts.
8. Do not require every player to use every tool during every action.
9. Do not create multiple planner, critic, and actor model personas in the
   first implementation.
10. Do not keep an unbounded model conversation alive for an entire game.
11. Do not infer live game semantics from regular expressions over rendered
    speech.
12. Do not optimize only for win rate at the cost of diversity, calibration,
    or human-like mistakes and adaptation.

## 6. Authority and Terminology

The following terms have distinct authority:

| Term | Responsibility | May change game truth |
| --- | --- | --- |
| `PlayerAgent` | Observe, call tools, reflect, form hypotheses, submit proposals | No |
| `HostRuntime` | Create turns, enforce ACLs and budgets, execute tools, validate and commit | Only through RuleEngine results |
| `RuleEngine` | Legal windows, action legality, role mechanics, phase flow, victory | Yes |
| `EventStore` | Persist committed objective game facts | Stores truth; does not decide it |
| `JudgePresenter` | Render committed public facts with personality | No |
| `WorldModel` | Produce viewer-relative candidate worlds and uncertainty | No |
| `MemorySystem` | Retain typed working, episodic, belief, commitment, and semantic records | No |
| `RAG` | Retrieve verified external or cross-game lessons | No |
| `AuditSystem` | Record inputs, decisions, boundaries, and commits | No |

The authority order is:

1. ruleset snapshot, RuleEngine state, legal window, and visibility policy;
2. the active revision-pinned `AgentTurn` and its strict schemas;
3. viewer-visible current-game facts;
4. the player's commitments and typed beliefs;
5. tool observations and possible-world hypotheses;
6. verified cross-game lessons and RAG cases;
7. persona, style, and presentation preferences.

Lower layers may affect judgment or language. They may not change facts,
permissions, legal actions, evidence ownership, or event semantics.

## 7. Top-Level Architecture

```text
Game Runtime
  -> PlayerTurnRunner
      -> ObservationProjector
      -> AgentLoop
          -> ToolGateway
          -> MemoryGateway
          -> ReflectionGateway
          -> WorldModelGateway
          -> Proposal
      -> HostBoundary
          -> VisibilityPolicy
          -> ProposalValidator
          -> RuleEngine
          -> EventCommitter
          -> CriticalAuditRecorder
  -> ProjectionWorkers
      -> player documents
      -> memories and beliefs
      -> evaluation projections
  -> JudgePresenter
      -> public broadcast
```

The only game-truth loop is:

```text
legal proposal
  -> host validation
  -> RuleEngine resolution
  -> atomic GameEvent commit
  -> projections
  -> next ObservationFrame
```

There is no path from model prose, reflection, RAG, memory, world hypotheses,
or judge narration directly to `GameState`.

### 7.1 Canonical Version and Commit Protocol

`GameRevision` is the strictly increasing, per-game integer assigned by the
transaction that appends a committed `GameEvent`. Revision zero is the
ruleset-and-roster snapshot before the first event. A committed event advances
the revision exactly once; derived projections, audit records, renderer output,
and retried deliveries never advance it themselves. The authoritative event
sequence and `GameRevision` are the same ordered stream; an optional event
sequence field in an in-memory object is not an acceptable source of truth.

The runtime uses four distinct versioned values:

| Value | Meaning | Used by |
| --- | --- | --- |
| `game_revision` | Canonical committed-event position | proposal CAS and replay |
| `window_version` | Ruleset-defined legal-action window and target set at that revision | action legality |
| `view_fingerprint` | Hash of actor ID, visibility policy version, active grants, role/capability snapshot, and visible record hashes | read/result reuse |
| `workspace_revision` | Monotonic version of one generated player projection | document/cache reads |

`base_revision` in every proposal means `game_revision`, never a rendered-text
counter or a document cache version. A proposal also carries the host-bound
`window_id`, `window_version`, and `view_fingerprint`. The terminal gateway,
not the model, supplies the actor identity and host idempotency key. A model may
echo those values only; a mismatch is `security_violation`.

A turn records a read set of immutable `(record_id, revision, content_hash)`
references plus its legal-window snapshot. A later global revision does not by
itself invalidate a turn. It is stale only when the active window, a referenced
record, a visibility grant, the actor's alive/capability state, or a declared
write conflict changes. This distinction is required for future independent
multi-player windows. Stage 1 declares all public speech writes conflicting,
so it opens exactly one active public-speech turn at a time.

### 7.2 `CommitTurn` Transaction

The persistence boundary exposes one transaction, conceptually:

```text
CommitTurn(
  base_game_revision,
  turn_id,
  idempotency_key,
  read_set,
  proposal,
  rule_result,
  critical_audit_records,
  projection_outbox_records
) -> CommitResult
```

It performs, in one database transaction:

1. lock the game stream and load its canonical revision;
2. return the previous `CommitResult` when `(turn_id, idempotency_key)` already
   committed;
3. compare the proposal's base revision with the locked head; an intervening
   revision is rejected for serial windows, while a future declared
   commutative window may proceed only after its read set, write set, active
   window, actor state, grants, and RuleEngine result all validate unchanged;
4. assign the next event sequence/revision and append the `GameEvent` plus
   immutable `PublicSpeechRecord` when applicable;
5. append all critical audit records, persist the idempotency result, and append
   projection-outbox records; then commit.

Any validation failure rolls back all writes. Projection workers consume the
outbox after commit and are idempotent by outbox ID; their failure cannot roll
back truth. `save_game()` followed by `append_events()` is not a substitute for
this transaction. The storage contract must add explicit turn, audit, outbox,
and compare-and-swap methods before a new runtime path is enabled.
Stage 1 requires equivalent semantics from the in-memory test repository,
SQLite, and PostgreSQL; other repository implementations report the capability
as unsupported and cannot enable the new runtime.

### 7.3 Action-Window Scheduling

Every `AgentTurn` belongs to a `LegalActionWindow` with a ruleset snapshot,
open/close conditions, conflict class, participant snapshot, and deadline.
The scheduler supports only these conflict classes:

- `serial_public`: exactly one active terminal turn; used by stage-1 daytime
  speech and any public action whose output may be answered by later speakers;
- `serial_private`: one active turn because its result changes another
  participant's legal view, such as a wolf kill followed by a witch decision;
- `commutative_private`: multiple terminal proposals may be accepted only when
  the ruleset declares their writes disjoint and their read sets remain valid;
- `team_coordinator`: team members may write discussion/proposal records, while
  exactly one mechanically selected coordinator owns the terminal action.

Opening, re-opening, and closing a window are RuleEngine/Host operations, not
agent tools. A cancelled turn is never silently rebased: the host creates a new
turn with a new observation and idempotency key when the window remains open.

### 7.4 Durable External Dispatch

Model and external read calls use a durable `DispatchAttempt` state machine:

```text
PENDING -> DISPATCHING -> DISPATCHED -> RESULT_RECORDED
                         \-> UNKNOWN_OUTCOME
PENDING | DISPATCHING -> CANCELLED
```

Before a network request, the host persists `DISPATCHING` with a unique
`dispatch_id`, lease hash, request hash, deadline, and provider/tool
idempotency key. It records `DISPATCHED` immediately after handing off the
request and records the typed result in a separate transaction. On restart the
host never blindly repeats an unresolved request:

- a provider/tool with an idempotency lookup is queried or reissued with the
  same key;
- a provider/tool without that guarantee becomes `UNKNOWN_OUTCOME`, consumes
  its call budget, and follows the task's cancellation/fallback policy;
- a result arriving after cancellation, lease replacement, deadline, or window
  closure is audited and discarded.

This protocol gives submit calls, model calls, and external tools different but
explicit exactly-once/at-most-once semantics. It also makes provider billing and
tool budgets auditable after a crash.

## 8. Proposed Package Boundaries

```text
werewolf_agent/player_agents/
  contracts/
    turns.py
    observations.py
    compaction.py
    tools.py
    memory.py
    proposals.py
    errors.py
  runtime/
    agent.py
    loop.py
    turn_runner.py
    budgets.py
    model_lease.py
  context/
    manifest.py
    projector.py
    workspace.py
    compaction.py
  tools/
    gateway.py
    context_tools.py
    evidence_tools.py
    world_tools.py
    memory_tools.py
    role_tools.py
    submission_tools.py
  memory/
    working.py
    beliefs.py
    commitments.py
    episodic.py
    semantic.py
    gateway.py
  cognition/
    hypotheses.py
    world_model.py
    evidence_graph.py
  roles/
    capabilities.py
    manifests.py
  host/
    visibility.py
    validator.py
    committer.py
    cancellation.py

werewolf_agent/judge_presenter/
  contracts.py
  projector.py
  presenter.py
  renderer.py
  fallback.py

werewolf_agent/audit/
  schemas.py
  recorder.py
  store.py
  chain.py
  keys.py
  redaction.py
  retention.py
  access.py
  export.py
  verifier.py
  queries.py
  evaluation_projection.py
```

Files may be combined while small, but these dependency directions and
responsibility boundaries must remain explicit.

## 9. Per-Player Document Workspace

Each player receives an isolated logical workspace:

```text
players/{player_id}/
  PLAYER.md
  ROLE.md
  GAME.md
  BELIEFS.md
  COMMITMENTS.md
  MEMORY.md
  WORKING.md
  INDEX.md
```

These are model-readable projections, not arbitrary physical files exposed to
the model and not canonical storage.

### 9.1 `PLAYER.md`

Contains stable identity, personality, expression preferences, risk appetite,
and verified long-term tendencies. It never contains current hidden roles,
other players' private information, or unverified strategic conclusions.

Its canonical source is a versioned `PersonaProfile` owned by authenticated
project configuration or an explicitly authorized human operator. At game
start the host pins a profile version and projects an immutable `PLAYER.md`
snapshot for that game. A player model, reflection, RAG result, win/loss, or
automated ability score cannot edit this snapshot during live play.

Post-game analysis writes observed ability metrics, verified tendencies, and
persona-change candidates to `PlayerProfile` or semantic memory, not directly
to `PersonaProfile`. A candidate changes future `PLAYER.md` only through an
audited between-game promotion policy or explicit human approval. Core
identity, voice, and risk appetite never drift automatically from model
self-assessment.

### 9.2 `ROLE.md`

Contains the player's current role facts, faction, available abilities,
ability state, and mechanical restrictions derived from the ruleset. It says
what the role may do, not what strategy the role should choose.

### 9.3 `GAME.md`

Contains a bounded current-game checkpoint: day, phase, living players,
important public events, public vote ledger summary, recent relevant speech,
and current revision. It is not the complete transcript.

### 9.4 `BELIEFS.md`

Contains typed, revisable hypotheses with confidence, supporting evidence,
counter-evidence, open questions, and invalidation conditions. It cannot turn a
hypothesis into a fact.

### 9.5 `COMMITMENTS.md`

Contains the player's committed public claims, positions, promises, vote
intentions, final votes, and retractions. It is generated from committed public
records and cannot be freely edited by the model.

### 9.6 `MEMORY.md`

Acts as a concise index of approved cross-game lessons. Detailed lesson topics
are loaded on demand. Every lesson carries role, phase, situation, evidence,
quality, applicability limits, and success/failure history.

### 9.7 `WORKING.md`

Contains current-turn questions, candidate hypotheses, bounded tool summaries,
reflection candidates, and an intended next step. It expires with the turn
unless typed records are promoted by the host.

### 9.8 `INDEX.md`

Provides section ID, revision, content hash, token estimate, visibility, short
summary, and last-update metadata. The initial prompt includes this manifest,
not every detailed document.

Every cache key and document lookup is bound to at least:

```text
game_id + player_id + view_fingerprint + workspace_revision + section_id
```

The requested document content is additionally checked against the immutable
record IDs and hashes in the turn read set. A cache hit with a matching key but
a missing, changed, expired, or newly invisible record is rejected.

## 10. Observation and Progressive Context

An action starts with a minimal immutable `ObservationFrame`:

```text
turn identity
role and phase
legal actions and targets
critical private facts
bounded public summary
recent commitments
document manifest
tool manifest
budget and deadline
```

The initial model context includes the static system contract, the current
`ObservationFrame`, essential `PLAYER.md` and `ROLE.md` fragments, a compact
`GAME.md` checkpoint, and `INDEX.md`. It does not automatically include the
full transcript, all beliefs, all memories, all RAG hits, or all skill content.

The player reads detailed sections through revision-pinned tools. Tool results
state whether they contain facts, observations, hypotheses, advice, or errors.

### 10.1 Context Budget and Compaction Trigger

Every model lease pins a `ContextBudgetPolicy` containing the model context
limit, output-token reserve, tool-schema reserve, estimator version, automatic
compaction threshold, post-compaction target, and maximum compactions per turn.
The initial defaults are:

```text
auto_compact_threshold = 0.80
post_compact_target    = 0.55
max_compactions        = 2
```

The threshold uses total active context, not only newly appended history. Before
every model request, the host computes predicted occupancy as:

```text
(
  immutable prefix tokens
  + active turn-history tokens
  + exposed tool-schema tokens
  + reserved output tokens
) / model context limit
```

Use the provider tokenizer when available. Otherwise use a versioned
conservative estimator and add a ten-percent safety margin. If predicted
occupancy is at least 80 percent, compaction completes before another model
request starts. Turn admission fails before model work when the immutable
prefix plus output and tool reserves alone cannot fit below the 55-percent
target. The player cannot change thresholds, reserves, estimator, or limits.

### 10.2 Two-Layer Compaction Checkpoint

Codex's public handoff-summary pattern is necessary but insufficient for a
hidden-information game. This runtime uses two distinct layers:

1. a host-built, strict `CompactionCheckpoint` that carries exact references
   and is the only resumable state;
2. an optional model-built `CompactionHandoff` that helps the same or successor
   model continue but remains untrusted advisory text.

The host checkpoint preserves:

- game, player, turn, task, base revision, event cursor, visibility fingerprint,
  model-lease hash, and checkpoint lineage;
- legal action and target snapshots, remaining deadline, and every remaining
  model/tool/reflection/repair/failover/compaction budget;
- current confirmed fact references and hashes, active disclosure-grant
  references, and the latest relevant public-record references;
- typed hypotheses with confidence, supporting and counter-evidence references,
  open questions, commitments, retractions, and accepted or pending reflection
  candidate references;
- tool-call IDs, tool/result kinds, revision, visibility, result hashes,
  evidence references, warnings, truncation state, and whether the result was
  adopted, superseded, or still unresolved;
- the current intended operation as a bounded enum plus authorized target,
  evidence, section, skill, or memory references.

Exact facts, legal sets, grants, budgets, and commitments are regenerated from
canonical host stores. They are never copied from model prose. Hypotheses and
open questions retain their type and provenance and cannot be promoted during
compaction.

The optional `CompactionHandoff` mirrors the public Codex prompt categories:
current progress, decisions already made, important constraints, remaining
steps, and critical references. It is strict structured output, bounded to
2,000 estimated tokens, labelled `UNTRUSTED_DATA`, and may mention only IDs
already present in the checkpoint. It may not add facts, evidence, grants,
legal targets, memories, tool results, public speech, or a terminal proposal.
It uses the same model lease and consumes compaction and lease cost/token
budgets, but it is not an agent-selectable reasoning step. Its absence must not
prevent the turn from continuing.

Compaction deliberately drops:

- hidden chain-of-thought, reasoning tokens, and unfiltered provider output;
- full prior prompt copies, superseded wording, repeated summaries, and
  redundant observations;
- full tool, RAG, skill, world-model, transcript, and rendered-speech payloads
  whose typed result or public-record references remain addressable;
- rejected proposal prose and any item that fails current viewer, revision,
  provenance, or integrity validation.

### 10.3 Compaction Transaction and Rehydration

Compaction is a host lifecycle operation, not a player tool and not a memory
write. It runs in this order:

1. after every in-flight tool result is recorded, predict the next-request
   occupancy and atomically move the turn from `THINKING` to `COMPACTING`;
2. freeze new model and tool dispatch, snapshot the active history, and record
   a `compaction_started` audit item;
3. build the exact checkpoint from revision-pinned host stores;
4. ask the leased model for the optional strict handoff without tools,
   failover, semantic repair, or authority to mutate the checkpoint;
5. validate every carried reference against player ownership, visibility,
   revision, content hash, grant state, and checkpoint schema;
6. persist checkpoint, lineage, token accounting, and audit record in one
   turn-state transaction;
7. replace model history with the static system contract, current `AgentTurn`,
   latest minimal `ObservationFrame`, current manifests, host checkpoint, and
   optional handoff;
8. recompute token usage, require occupancy at or below 55 percent, then resume
   `THINKING`.

This mirrors Codex's separation between a compacted summary and reinjected
initial context, but the game's reinjected prefix is generated from current
host authority. A compaction never changes `GameState`, appends `GameEvent`,
accepts memory, consumes a disclosure grant, or submits an action.

If the active window, declared read set, player-alive/capability state, or
visibility fingerprint changes before the transaction commits, the host
discards the checkpoint and cancels the stale turn. A global game-revision
change alone follows the conflict rules in section 7.1. If optional handoff
generation fails, rehydration uses the host checkpoint alone. If checkpoint
persistence or validation fails, the old history remains active and the host
may retry compaction once while the deadline permits; it never continues from a
partially persisted checkpoint.
After a failed retry the host cancels the model turn and applies the declared
provider-failure fallback policy while the rule window remains open.

If rehydration remains above the 55-percent target, the host removes the
optional handoff first and then replaces adopted tool summaries with their
typed references. It must not remove authority, legal sets, active grants,
budgets, commitments, unresolved evidence, or the intended operation. A second
compaction rebuilds from canonical state plus the newest uncompressed segment;
it does not summarize only the previous summary. After the per-turn maximum,
the host forbids additional context-expanding reads and builds a terminal-only
context from the exact checkpoint with only the required submit schema exposed.
It requests the best current strict terminal proposal only when this reduced
request predicts less than 80-percent occupancy. If that cannot fit or
validate, normal neutral fallback policy applies; stale or closed windows still
receive no fallback.

## 11. Bounded Agent Turn Lifecycle

Each legal opportunity creates a separate `AgentTurn`:

```text
OPEN -> OBSERVING -> THINKING
THINKING <-> WAITING_TOOL
THINKING -> COMPACTING -> THINKING
THINKING -> SUBMITTED -> VALIDATING -> COMMITTED

validation repair -> REPAIRING -> SUBMITTED
state change       -> CANCELLED
deadline           -> EXPIRED
```

The turn records game, player, role, phase, task, `base_revision`,
`window_id`, `window_version`, read set, legal actions, legal targets,
`view_fingerprint`, model lease, budget, deadline, status, and idempotency
key. `event_cursor` is recorded only as a replay convenience; it is not a
substitute for the canonical `base_revision` or read set.

The model may choose one of three operations at each step:

1. invoke an authorized read or analysis tool;
2. invoke `reflect` to propose a typed short-term memory update;
3. invoke the task's terminal submit tool.

Plain assistant text cannot commit an action. A successful terminal submit
ends the loop. A validation repair may alter only rejected fields on the same
model lease and cannot reopen context, RAG, or skills.

Stage 1 exposes only `serial_public` daytime speech windows. The scheduler
opens the next speaker only after the previous speech commit, cancellation, or
deadline. Votes, private role actions, team coordination, and any truly
concurrent window remain disabled until their contract-specific tests prove the
conflict class declared in section 7.3.

Initial configurable budgets are bounded by task type. A reasonable starting
profile is eight model steps, twelve total tool calls, six context reads, four
evidence queries, two world-model queries, two RAG queries, three reflections,
two context compactions, one semantic repair, and one provider failover. These
are experimental defaults, not game strategy rules.

## 12. Tool System

Tools are exposed through a role- and phase-scoped whitelist. The agent chooses
which available tools to call and in which order. The host derives tool identity
from the active turn; the model cannot supply another player identity.

Core tool families are:

```text
context:
  read_context
  read_public_timeline
  read_visible_player_history
  read_private_role_facts
  read_commitments
  read_vote_ledger

evidence and analysis:
  get_evidence
  compare_claim_timeline
  compare_vote_history
  check_public_consistency
  query_relation_graph
  generate_possible_worlds
  simulate_public_consequence

memory:
  search_current_memory
  search_cross_game_memory
  query_rag
  reflect
  review_memory_candidate

skills:
  list_applicable_skills
  consult_skill

role and team:
  query_role_ability_state
  read_wolf_team_discussion
  propose_wolf_target
  review_team_disagreement

terminal submission:
  submit_speech
  submit_vote
  submit_night_action
  submit_role_ability
  submit_wolf_team_action
```

Read tools do not create game side effects. `reflect` submits a candidate;
skills, RAG, and world-model tools return reference material; terminal tools
submit proposals. Only the host and RuleEngine can produce committed events.

A `ToolResult` identifies its revision, visibility, result kind, evidence
references, warnings, and truncation. It never returns an unlabelled verdict.

`query_rag` accepts a bounded situation signature and optional public evidence
references. The host derives player, role, phase, ruleset, and visibility
filters from the active turn; the model cannot override them. Results have
`ADVICE` or `EXAMPLE` kind and cannot be cited as current-game evidence.

`list_applicable_skills` returns metadata only. `consult_skill` accepts one
listed skill ID plus a bounded focus made of authorized target and evidence
references. It returns a versioned `ADVICE` result with intended use,
counter-signals, forbidden use, provenance, and expiry. The initial per-turn
budget allows at most two RAG queries and two skill consultations, configured
independently so disabling one does not affect the other.

## 13. Role Capabilities Without Strategy Hard-Coding

All roles share the same AgentLoop. A host-generated capability manifest
defines private facts, available tools, available actions, and mechanical
constraints for the active ruleset.

A capability manifest may say that a living seer can inspect its own previous
check results and submit one legal night check. It may not say when to claim,
whom to check, which vote is optimal, or how a smart seer should speak.

Existing directives and strategy handlers are classified as follows:

- rules such as potion count, alive-only targeting, or phase restrictions move
  to RuleEngine or capability state;
- reusable observations or strategy lessons may enter reviewed RAG content;
- code that selects the best target, enforces a standard tactic, or generates
  fixed role strategy is removed.

Reasonable non-optimal decisions, incorrect beliefs, risk-taking, persuasion,
and strategy changes are allowed if they respect facts, visibility, and rules.

## 14. Wolf-Team Coordination

Werewolves retain separate personal workspaces. A distinct team-visible store
contains only wolf discussion, shared proposals, agreed commitments, and
team-visible evidence. One wolf cannot read another wolf's hidden chain of
thought, private working notes, or unrelated memories.

The team store does not decide a target. A `wolf_team` legal window snapshots
the living wolves, allowed targets, deadline, and a ruleset-defined mechanical
coordinator selector. The initial selector is the lexicographically first
living wolf ID in the pinned roster; a ruleset may replace that selector only
with another deterministic, auditable function. Selecting a coordinator gives
one agent terminal authority; it does not select a target or encode strategy.

The protocol is:

1. every living wolf may create bounded, team-visible discussion and
   non-terminal `WolfTargetStance` records until the discussion deadline;
2. the coordinator receives the same team-visible records and may submit the
   sole `WolfTeamProposal` for the window;
3. a coordinator death, loss of capability, or stale view cancels its turn and
   deterministically selects the next eligible coordinator from the original
   participant snapshot; it never transfers a private workspace;
4. non-coordinator disagreement is evidence for the coordinator, not a host
   vote and not a RuleEngine target-selection policy;
5. on coordinator deadline or infrastructure failure, the ruleset's declared
   neutral mandatory-action fallback applies. It uses the stable selector from
   section 26 and records the participant snapshot and selector inputs.

`WolfTeamProposal` is idempotent by team-window ID and may be committed only
by the current coordinator. The RuleEngine validates the submitted action and
resolves the night outcome; it never chooses among disagreeing targets. Team
windows are stage-2 work and are not enabled by the daytime-speech vertical
slice.

## 15. World Model

The world model is viewer-relative and hypothesis-only. It combines visible
evidence, role-count constraints, known self/private role facts, current
beliefs, and contradiction records to produce candidate worlds with normalized
probabilities, supporting evidence, counter-evidence, and uncertainty warnings.

Deterministic code enforces role counts, evidence visibility, identity
canonicalization, probability validity, and candidate bounds. The player agent
decides which worlds deserve attention, how evidence should be weighted, and
whether a hypothesis should affect its action.

Possible worlds never reveal the true hidden state, become RuleEngine facts,
or directly produce an action proposal.

## 16. Memory and Reflection

The memory system separates five layers:

| Layer | Lifetime | Content |
| --- | --- | --- |
| Working | one turn | temporary questions, hypotheses, and tool summaries |
| Episodic game | one game | visible events and the player's action history |
| Belief | one game, revisable | hypotheses, confidence, evidence, contradictions |
| Commitment | one game | public claims, promises, intentions, votes, retractions |
| Semantic | cross-game | verified lessons and failure patterns |

The player may call `reflect` during a turn. It submits a candidate classified
as observation, hypothesis, commitment note, or lesson candidate with evidence
references, confidence, applicability, and expiry. The host accepts, rejects,
or downgrades it. The player cannot assign factual authority.

Live acceptance is structural and deterministic, never a host judgment about
strategy quality:

- an `observation` must cite visible immutable records and may only store a
  bounded, source-labelled extract; it cannot add a new fact predicate;
- a `hypothesis` must remain player-private, cite at least one visible record,
  carry a bounded confidence bucket and expiry, and is stored in the belief
  layer only;
- a `commitment_note` is accepted only when it resolves to the actor's already
  committed `PublicSpeechRecord` or final vote record; `COMMITMENTS.md` is
  always regenerated from those records rather than from the candidate text;
- a `lesson_candidate` is stored only in the game-local review queue and is
  ineligible for semantic memory until the post-game transaction verifies its
  claimed outcomes.

The host rejects dangling, invisible, stale, duplicate, over-length, or
wrong-layer candidates. Duplicate detection uses a canonical typed payload hash
within `(game_id, player_id, layer, source_refs)`. It never scores a candidate
as tactically good or bad during live play. Any non-structural quality policy
is a versioned, post-game review policy with its own audit record.

At game end, a separate review transaction compares structured claims against
committed events and final roles. It produces verified lessons, disproved
hypotheses, preserved strengths, failure patterns, and applicability limits.
Only quality-approved lessons enter live cross-game memory.

Hidden chain-of-thought and raw provider output are never stored as memory.

## 17. RAG

RAG remains valuable for verified external cases, reviewed cross-game lessons,
failure patterns, role/phase-specific examples, and high-quality strategic
cases. It is not a source of rules, current-game truth, other-player private
facts, or final decisions.

Retrieval first applies hard filters for ruleset, role, phase, visibility,
source quality, approval state, and applicability. Semantic ranking occurs only
within the allowed candidate set. Results are bounded, source-labelled, and
returned as lessons or examples, never verdicts.

RAG is invoked on demand rather than injected into every action. Ablation must
demonstrate that it improves relevant decision quality or reduces repeated
mistakes. If it does not, it remains disabled rather than surviving solely as
architectural complexity.

### 17.1 Skills and Procedural Knowledge

The legacy skill handlers are not reused as decision code. A retained skill is
a versioned, read-only procedural reference package with concise metadata,
role/phase/ruleset applicability, intended use, counter-signals, forbidden use,
source quality, and optional detailed examples loaded on demand.

Skills may help the player ask better questions or examine evidence from a
particular angle. They cannot select a target, mutate belief, become evidence,
submit an action, or instruct the host. Any procedural content that merely
encodes a fixed "best move" is removed or converted into a reviewed RAG case
with explicit limitations. Skill text is untrusted model input and receives
the same visibility, injection, length, provenance, and audit treatment as RAG.

Skill usefulness is measured through ablation and tool-result adoption. A
skill system that adds calls or tokens without improving calibrated decisions
or reducing repeated mistakes is not enabled in live play.

## 18. New Structured Proposal System

The rewrite does not use `SpeechAct`. It introduces new independent contracts:

```text
SpeechProposal
VoteProposal
NightActionProposal
RoleAbilityProposal
WolfTeamProposal
```

All contracts use strict discriminated schemas, reject extra fields, carry
schema version, turn ID, player ID, and base revision, and cannot carry a
`GameEvent` or declare the next phase or winner.

Immutable contract models expose no unvalidated copy-update path. Their
`model_copy(update=...)` implementation merges into a round-trip dump and
validates the complete result again. The deprecated `copy()` API delegates to
that same validated path and rejects partial include/exclude copies, which
could otherwise produce structurally incomplete contract instances.

Provider support for strict tool calling or structured output is checked before
a player route is leased. The live system does not fall back to free-form JSON,
regular-expression extraction, or prose-to-action recovery.

The versioned schema fixture produced by the contract foundation is the
canonical proposal schema and the input to later provider adapters. It is not,
by itself, proof that a provider accepts that exact JSON Schema dialect.
Provider adapters may only apply deterministic, versioned dialect transforms;
their transformed fixtures, content hashes, and live provider acceptance tests
are part of the later tool-gateway plan and must pass before a route is leased.

Schema compliance is followed by host semantic, visibility, revision, and
RuleEngine legality validation. The guaranteed property is that every final
committed action is structured and legal, not that every first model attempt is
valid.

### 18.1 Common Terminal Envelope

All terminal tools accept a strict, `extra=forbid` envelope. The gateway binds
`turn_id`, `player_id`, `window_id`, `window_version`, `base_revision`,
`view_fingerprint`, and `idempotency_key` from the active turn. They are not
agent-selectable arguments. The model supplies only the proposal body and the
pinned `schema_version`; an echoed bound value must match exactly.

```text
TerminalProposalEnvelope
  schema_version             # exact proposal-schema version pinned by the turn
  turn_id                    # host-bound
  player_id                  # host-bound
  window_id / window_version # host-bound
  base_revision              # host-bound GameRevision
  view_fingerprint           # host-bound
  body                       # one strict discriminated proposal body
```

The host assigns the idempotency key when it opens the turn. It assigns
`proposal_id`, request/dispatch IDs, and timestamps after schema parsing. The
model cannot manufacture IDs that appear committed. Proposal schemas reject
unknown fields, duplicate logical move IDs, unknown discriminators, and omitted
required fields before any semantic validation starts.

### 18.2 Action Proposal Bodies

The following bodies are the complete first-contract surface. Their enum
values are pinned by the ruleset snapshot and proposal-schema version; a
ruleset cannot silently reinterpret an existing value.

```text
VoteProposal
  kind = "vote"
  choice = "target" | "abstain"
  target_id: required only for "target"

NightActionProposal
  kind = "night_action"
  action = "wolf_kill" | "wolf_no_kill" | "seer_check" |
           "witch_antidote" | "witch_poison" | "witch_pass"
  target_id: required only for wolf_kill, seer_check, witch_poison
  subject_ref: required only for witch_antidote and identifies the
               RuleEngine-provided current kill intent

RoleAbilityProposal
  kind = "role_ability"
  ability_id: a ruleset capability ID exposed in this exact turn
  operation: a discriminator defined by that capability's immutable schema
  args: the strict discriminated union selected by (ability_id, operation)

WolfTeamProposal
  kind = "wolf_team"
  team_window_id
  coordinator_id
  action = "kill" | "no_kill"
  target_id: required only for "kill"
  considered_stance_refs: zero to eight team-visible stance/discussion refs
```

`args` is not an arbitrary JSON map: the `CapabilityManifest` embeds the
exact JSON Schema and content hash for each enabled `(ability_id, operation)`.
The gateway exposes that schema only for the active role/action window and
validates it before it calls the RuleEngine. Adding a role capability therefore
requires a new schema version, schema fixtures, legality fixtures, and an
explicit migration rule.

Shared invariants are:

- `target_id` must be in the current legal-target snapshot when present;
- an abstention, pass, or no-kill may be submitted only if the ruleset exposes
  that exact action as legal;
- `subject_ref` and every team stance reference must be visible, active, and in
  the turn read set;
- a wolf-team coordinator ID must equal the Host-selected current coordinator;
- an ability proposal cannot carry a capability unavailable to the actor or a
  field belonging to a different operation;
- proposal bodies never carry a `GameEvent`, phase transition, winner,
  rendered text, raw rationale, or arbitrary metadata.

### 18.3 Schema Versions and Errors

`ProposalSchemaVersion` is a semantic version plus a content hash. A game pins
the ruleset snapshot and the set of accepted proposal-schema versions at game
creation. A new field or enum value is accepted only by a new minor/major
version with explicit validators; an incompatible interpretation requires a
new major version and cannot be enabled for an existing game. Historical
proposals retain their original schema version and are decoded by versioned
readers during replay.

The validator returns stable machine-readable errors:

```text
schema_invalid
bound_context_mismatch
unknown_schema_version
unknown_capability
wrong_action_window
stale_read_set
target_not_legal
invisible_reference
grant_inactive
semantic_mismatch
rule_illegal
idempotency_conflict
```

Each error includes a JSON-pointer field path and never includes hidden values.
Only `schema_invalid` and a field-local `semantic_mismatch` are eligible for
the one same-lease repair attempt. A stale read set, inactive grant, or changed
window always cancels the turn and requires a new observation.

This safety rule is enforced at construction time, not left to caller
convention. Human-readable messages come from a closed, code-keyed safe-message
catalog; callers cannot attach arbitrary message text. A validator derives the
JSON Pointer only from its canonical schema/model error location (field names
and collection indexes), never from record contents, player identifiers, role
values, or other observed data. The failure contract rejects `repairable=true`
unless the code is `schema_invalid`, or the code is `semantic_mismatch` with a
non-root field path.

### 18.4 Private Disclosure Grants

Self `RoleClaim` is a public statement and may be true or false; it does not
prove a private fact and requires no disclosure grant. A
`PrivateResultDisclosure` asserts that an actor's actual private check or
ability result is being published. Before the terminal request, the visibility
policy may create a one-time `DisclosureGrant` containing:

```text
grant_id, actor_id, turn_id, window_id, game_revision,
fact_kind, fact_record_id, fact_hash, target_id, timing_ref, expiry
```

The host matches every field exactly and consumes the grant in the same
`CommitTurn` transaction as the public record. A grant is never reusable after
commit, cancellation, expiry, or a changed fact revision. The host controls
whether a ruleset permits a private result to be disclosed; the agent never
creates or broadens a grant.

## 19. Public Speech From Scratch

Speech uses three separate contracts:

```text
SpeechProposal
  private turn-scoped proposal submitted by the player

PublicSpeechRecord
  host-validated committed public semantics

RenderedUtterance
  human-facing language projection
```

A `SpeechProposal` is the `body` of a terminal envelope and contains typed
public moves rather than arbitrary public prose. Bound envelope fields carry
the turn/player/revision identity; the speech body is fixed before implementation
planning:

```text
SpeechProposalBody
  kind = "speech"
  objective
  moves[]
  response_record_refs[]
  delivery_plan
```

`objective` is one of `state_case`, `challenge_claim`, `answer_question`,
`ask_question`, `defend_self`, `declare_vote_position`, `retract_or_correct`,
`express_uncertainty`, or `no_new_information`. It is used for evaluation and
rendering, not as an instruction to consumers. `moves` contains one to eight
uniquely identified public moves.

`delivery_plan` contains only `tone`, `length_class`, `address_style`,
`move_order`, `emphasis_move_ids`, and `connector_ids`. Tone is one of `calm`,
`firm`, `skeptical`, `urgent`, `defensive`, or `conciliatory`; length is
`brief`, `standard`, or `extended`; address style is `room`, `targeted`, or
`mixed`. Move order contains every move ID exactly once, emphasis is a unique
subset, and connector IDs must exist in the pinned renderer catalog. The plan
cannot contain prose or alter move semantics. `response_record_refs` must equal
the unique set of external public-record IDs used by `ResponseMove` or quoted
moves.

Every move contains `move_id`, `move_type`, `modality`, and zero or more
viewer-visible `evidence_refs`. `modality` is one of `asserted`, `suspected`,
`conditional`, `hypothetical`, or `quoted`. The public move union is:

```text
AlignmentRead
RoleClaim
PrivateResultDisclosure
PublicEvidenceCitation
PlayerComparison
QuestionMove
ResponseMove
VotePosition
ConditionalCommitment
RetractionMove
UncertaintyStatement
```

The required fields and invariants are:

| Move | Required semantic fields | Additional invariants |
| --- | --- | --- |
| `AlignmentRead` | `target_id`, `alignment`, `strength` | alignment is `good`, `wolf`, or `uncertain`; strength is `leaning`, `probable`, or `committed` |
| `RoleClaim` | `claimant_id`, `role`, `claim_mode` | claim/denial requires claimant = the host-bound actor; quotation requires a visible source record; private role disclosure requires an active grant |
| `PrivateResultDisclosure` | `fact_kind`, `fact_ref`, `disclosure_grant_id`, `timing_ref`, typed result fields | grant must match actor, turn, revision, fact hash, result type, target, and timing exactly |
| `PublicEvidenceCitation` | `evidence_refs`, `relation`, `subject_ids`, `supports_move_ids` | relation is `supports`, `contradicts`, or `contextualizes`; it cannot introduce a conclusion outside the referenced moves |
| `PlayerComparison` | two-to-four `player_ids`, `dimension`, ordered `assessments` | dimension is claim, vote, evidence, commitment, or timeline consistency; every assessment cites public evidence |
| `QuestionMove` | `target_id`, `topic`, `requested_fields` | topic and requested fields are bounded enums; questions cannot request private or moderator facts |
| `ResponseMove` | `source_record_id`, `response_kind` | response is agree, disagree, clarify, or challenge and may cite only visible records |
| `VotePosition` | `target_id`, `commitment` | commitment is leaning, provisional, or committed; this remains distinct from the final vote action |
| `ConditionalCommitment` | typed `condition`, typed `consequence`, `expires_at_phase` | condition references public state or records; consequence is a bounded support, challenge, vote-position, or self-disclosure commitment |
| `RetractionMove` | `prior_public_move_ref`, optional `replacement_move_id` | actor may retract only its own prior move; the original record remains auditable |
| `UncertaintyStatement` | `subject_id`, `dimension`, two-to-four typed `alternatives` | alternatives carry value, confidence bucket, and public support refs; probabilities are presentation buckets, not game truth |

Roles, alignments, fact kinds, question topics, comparison dimensions,
conditions, consequences, and alternative values are versioned enums derived
from the active ruleset and proposal schema. Free-form interpretation,
explanation, rationale, summary, and public text fields are forbidden.

Move references must resolve either to another move in the proposal or to a
visible committed public record. Evidence references must exist at the base
revision and be visible to the actor. A proposal cannot quote itself, create a
reference cycle, retract another player's move, add a target absent from the
active turn, or publish a private fact without an exact one-time disclosure
grant. Host normalization preserves move IDs and semantics and may only add
event metadata, canonical enum values, and verified provenance.

After validation the host creates an immutable `PublicSpeechRecord`:

```text
record_id
schema_version
game_id / turn_id / actor_id
day / phase / committed_revision
normalized_moves[]
source_evidence_refs[]
disclosure_grant_refs[]
origin
renderer_contract_version
rendered_utterance_hash
```

`origin` distinguishes model submission, repaired submission, and neutral
terminal fallback. Private deliberation, rejected moves, working memory, RAG
text, and reflection candidates never enter the public record.

The record enforces every provenance relation that is locally provable: move
IDs are unique; `source_evidence_refs` is a unique superset of every move's
direct `evidence_refs` (the Host may add other verified provenance); and
`disclosure_grant_refs` is the unique exact set of grant IDs used by
`PrivateResultDisclosure` moves. Record existence, visibility, grant field
matching, expiry, and one-time transactional consumption remain Host checks.

`RenderedUtterance` contains `record_id`, sentence-plan version, renderer
version, text, content hash, and fallback status. It carries no independent
semantic fields and cannot be consumed as evidence.
Its text must contain at least one non-whitespace character; validation does
not trim or otherwise rewrite text because the exact bytes are hash input.

Other player agents, beliefs, summaries, relations, vote reasoning, and game
documents consume `PublicSpeechRecord`, never `RenderedUtterance`.

First-version rendering does not accept free-form model prose. The model may
submit only a strict `SentencePlan` containing move order, approved template
IDs, approved connector IDs, address style, emphasis move IDs, and pacing.
Critical player IDs, role names, check results, vote targets, attribution,
negation, modality, and commitments are host-filled slots. The host renders
the final text from a versioned template catalog and verifies that every move
is represented exactly once unless the template contract explicitly combines
compatible moves.

A future neural language realizer is an offline, feature-gated experiment. It
must pass entity, number, attribution, negation, modality, required-move, and
forbidden-claim checks and fall back to the deterministic renderer. Because a
free-form semantic-equivalence checker cannot formally guarantee absence of
new meaning, neural text is not enabled on a live path whose acceptance gate
requires zero renderer deviations. Rendered text never becomes a second
semantic authority.

## 20. Host Validation and Commit

The host validates, in order:

1. strict proposal schema;
2. active turn, player, task, and idempotency key;
3. current revision and action window;
4. target and evidence existence;
5. viewer visibility and evidence ownership;
6. memory, RAG, skill, and world-model references;
7. private disclosure authorization;
8. task-specific semantic invariants;
9. RuleEngine legality;
10. critical audit and event commit transaction.

Stable errors include unknown target, invisible evidence, stale revision,
illegal action, disclosure denied, rule window closed, schema invalid, semantic
mismatch, and security violation. Error text is explanatory; error codes and
field paths drive repair.

Before commit, the host rechecks that the player is alive, the phase/window
remains legal, the declared read and write sets are conflict-free under the
locked head revision, and all referenced grants and evidence are active.

## 21. Host Runtime and RuleEngine

`HostRuntime` is not a player or model. It creates turns, projects observation,
enforces ACLs and budgets, executes tools, manages leases and deadlines,
validates proposals, invokes RuleEngine, commits events, schedules projections,
and records audit.

`RuleEngine` remains the sole authority for role assignment, phase order, legal
targets, potion and ability state, checks, kills, exile, self-destruct, sheriff
mechanics, death triggers, victory, and deterministic replay.

The rewrite removes strategy policy from the host. The host knows what is legal
and visible, not what is strategically best.

## 22. Personality-Aware Judge Presenter

The judge remains as an expressive public presenter but is split from rules
authority:

```text
RuleEngine + HostRuntime -> objective authority
JudgePresenter           -> public personality and narration
```

The judge has read-only `JUDGE.md`, `STYLE.md`, `SHOW.md`, and public
`CURRENT.md` projections. It receives a `JudgeBroadcastRequest` containing
committed public event references, required facts, forbidden inference fields,
style profile, and length limit. It cannot read complete `GameState` or any
player-private workspace.

Critical facts such as phase, day, deaths, vote counts, exile, revealed roles,
badge state, and winner are host-filled slots. The judge may control opening,
closing, pacing, tone, and atmosphere. It cannot analyze player credibility,
suggest strategy, infer hidden roles, or reinterpret outcomes.

The first live version uses the same constrained-rendering principle as player
speech. The judge submits broadcast template IDs, public event references,
approved connector IDs, pacing, tone, and emphasis; the host fills every fact
slot from committed public events. Arbitrary opening or closing prose is not a
live model field. A free-form personality realizer may be evaluated offline but
cannot replace the template renderer until factual-deviation gates define and
demonstrate an acceptable non-zero policy; the default gate remains zero.

Judge model failure, schema failure, hidden inference, or missing required
facts falls back to a deterministic personality template. Narration failure
never blocks or rolls back a committed game event.

## 23. Audit Architecture

Audit is a top-level subsystem shared by player agents, tools, memory, RAG,
world model, host, RuleEngine commit, human operators, and judge presenter.

Each immutable `AuditRecord` contains audit and schema IDs, game and turn IDs,
actor ID and kind, event kind, sequence, timestamp, ruleset ID, context
revision, correlation and causation IDs, visibility class, redacted payload,
payload hash, previous hash, and record hash.

Actor kinds include player agent, host, RuleEngine, tool, memory, RAG, world
model, judge presenter, and human operator. Event kinds include turn lifecycle,
observation build, model lease, context reads, tool calls and results, RAG and
world queries, reflection candidates, memory decisions, proposal submission,
validation decisions, action commit, compaction, judge render/fallback, and
security violations.

### 23.1 Audit Privacy

Audit visibility classes are public, player-private, moderator-secure, and
security-only. Players cannot query the audit stream during live play. A
post-game reflection receives only a sanitized own-player projection.

Audit never stores hidden chain-of-thought, unfiltered provider output,
credentials, authentication headers, complete system prompts, another player's
private documents, or unbounded raw RAG text.

It stores prompt-contract version, document section IDs/revisions/hashes, tool
schema version, model lease, structured proposal, evidence references,
validation result, and committed event references. This is sufficient to
reconstruct the observable decision boundary without retaining hidden thought.

Every compaction audit record stores trigger, policy and estimator versions,
threshold, context limit, reserved tokens, token estimates before and after,
checkpoint and source-history hashes, checkpoint lineage, carried reference
IDs by type, dropped data classes, handoff presence and hash, validation
result, failure code, and rehydrated-context hash. It never stores hidden
reasoning or the discarded raw payloads. This lets audit replay reconstruct the
exact decision boundary after compaction without treating the handoff as fact.

### 23.2 Tamper Evidence and Atomicity

Records use RFC 8785 JSON canonicalization encoded as UTF-8. Binary fields are
base64url without padding. The previous hash is a 32-byte value, not a text
concatenation, and the chain uses an explicit domain separator:

```text
record_hash = SHA-256(
  "wofkill-audit-v1\0" || previous_hash || canonical_record_without_record_hash
)
```

The genesis previous hash is 32 zero bytes. `previous_hash` is present in the
canonical record and must match the separately framed input; verification
rejects a mismatch. These framing rules, the canonicalization version, and hash
algorithm are stored in the game audit manifest and never inferred from the
current application version.

GameEvent, critical AuditRecord, and projection-outbox request are committed in
one transaction or not at all. Critical audit includes accepted proposals,
RuleEngine resolution, event commit, visibility/security rejection, operator
intervention, and turn cancellation. Non-critical telemetry failure does not
block a game.

GameEvent replay reconstructs objective game state. Audit replay reconstructs
what an agent was allowed to see, which tools it used, which proposal it
submitted, and why the host accepted or rejected it. Audit does not reproduce
private model reasoning.

### 23.3 Audit Queries

Authorized moderator and evaluation services may retrieve a turn trace, verify
the audit chain, find security incidents, explain an action commit, compare
ablations, and measure memory, RAG, world-model, and judge effectiveness.
Evaluation consumes read-only audit projections and never changes game truth.

### 23.4 Encryption, Access, Retention, and Deletion

Audit access is granted by authenticated service role, never by a caller-
supplied visibility string. Public records follow public replay permissions;
player-private records are available only to the owning player's sanitized
post-game view and authorized moderator services; moderator-secure records are
available only to moderator audit services; security-only records require the
security-auditor role. Every non-public audit read and export is itself audited.

Player-private, moderator-secure, and security-only payloads use AEAD envelope
encryption at rest with a per-game data-encryption key and a versioned
`AuditKeyProvider`. The associated data binds game ID, audit ID, sequence,
visibility, schema version, and key ID. The canonical audit record contains the
random nonce, ciphertext, authentication tag, key ID, and algorithm; its
`payload_hash` is SHA-256 of that encrypted envelope, never of low-entropy
private plaintext. The repository stores key IDs and ciphertext, never key
material. Key rotation creates a new key version for subsequent records and
keeps old versions only for their authorized retention window.

Chain hashes remain unkeyed canonical SHA-256 so a verifier can prove byte
integrity without decrypting payloads. Every 100 critical records and at least
once every five minutes while a game is active, the writer sends the chain
head, game ID, sequence, and timestamp to an append-only `AuditAnchorSink`.
Its signing/HMAC key and storage are outside the AuditStore/database
administrative boundary. The game-close transaction appends a final-anchor
outbox request atomically with the terminal event. The game result is not
rolled back if the external sink is unavailable, but audit export and the
`audit_finalized` status remain blocked until the signed anchor receipt is
persisted. Verification reports unanchored intervals rather than silently
treating them as storage-administrator-resistant.

The default retention policy is explicit and configurable only by deployment
policy:

- public audit follows the corresponding GameEvent/replay retention;
- encrypted player-private and moderator-secure payloads remain through
  post-game review and the evaluation window, default 30 days;
- security-only incident payloads default to 90 days;
- after payload expiry, destroy the applicable data-encryption key and retain
  only the now-undecryptable encrypted envelope, non-identifying event kind,
  timing, schema version, hashes, chain linkage, and aggregate metrics when
  policy permits;
- credentials, authentication headers, and hidden chain-of-thought have zero
  retention because they are never accepted into audit payloads.

An authorized game/player deletion request destroys the relevant private data
keys, making the retained ciphertext cryptographically unreadable, and appends
an audited tombstone. Retaining the original opaque envelope is required to
recompute the historical chain; it is never exported as user data. A deployment
whose policy requires physical ciphertext removal must use a separately signed
redaction manifest and accepts that verification proves the original record
hash and authorized deletion, not the removed payload bytes. Objective public
events and non-identifying integrity records follow the configured game-data
policy. Exports are scope-bounded, redacted by the requester's role, include a
schema/chain manifest, and never expose ciphertext keys or another player's
private records.

The `AuditStore` allocates each per-game sequence number inside the storage
transaction using an atomic counter or row lock; callers cannot choose a
sequence. Parallel turns may finish in any order, but the committed audit order
is total and matches transactional commit order. Critical GameEvent/outbox
transactions reserve and append their audit record atomically. Verification
reports missing sequences, duplicate sequences, invalid chain links, unknown
key versions, and unsigned chain-head intervals.

## 24. Prompt Injection and Security

Public speech, external cases, cross-game lessons, Markdown content, tool
results, and retrieved text are untrusted data. Static system contracts and
host permissions always outrank them.

The system applies typed envelopes, source and visibility labels, bounded
length, content sanitization, section allowlists, stable tool schemas, and
identity derived from the active turn. It audits instruction-like content,
sanitization policy version, truncation, model-context inclusion, attempted
privileged tool calls, and final host decisions.

No player tool can guess or supply a different player ID, read moderator state,
edit Markdown, append events, switch models, request retries, override
visibility, change budgets, or invoke arbitrary code.

## 25. Concurrency, Persistence, and Recovery

All reads are revision-pinned. Before commit, the host repeats legality,
visibility, player-alive, phase, window, declared read-set, write-conflict, and
grant checks under the game-stream lock. A changed global revision cancels the
turn only when one of those declared dependencies changed. Stage-1 public
speech is `serial_public`, so any intervening public speech is a declared write
conflict and cancels the stale turn. If the RuleEngine still offers the action,
the runtime may open a new turn; this is rescheduling, not semantic retry.

Turn state and idempotency keys survive process restart. Duplicate delivery of
the same submit call returns the existing result and cannot append a second
event. Pause prevents new model work at a safe boundary and invalidates or
suspends turns according to their deadline policy. Player death, phase close,
self-destruct, and game termination cancel affected turns without fallback.

Every model and external-tool request follows the durable dispatch protocol in
section 7.4. Recovery first reconciles `DISPATCHING` and `DISPATCHED` attempts;
it never starts new model work until every earlier attempt is either
`RESULT_RECORDED`, `UNKNOWN_OUTCOME`, or `CANCELLED`. A recorded result is
reused only when its lease, view fingerprint, read set, and active window still
match. Unknown outcomes never receive a fresh provider call under a new ID just
to hide an ambiguous prior charge.

Committed `CompactionCheckpoint` records are part of recoverable turn state.
On restart the host verifies checkpoint lineage, hashes, revision, visibility,
deadline, and remaining budgets before reconstructing the model context. It
does not rerun or reinterpret the optional handoff. An interrupted compaction
with no committed checkpoint resumes from the pre-compaction history or
cancels through the declared failure path; partial replacement history is never
accepted.

Objective replay uses initial seed, ruleset snapshot, and GameEvents. Decision
review additionally uses structured turn and audit records. Projection workers
are rebuildable from canonical stores.

## 26. Model Lease, Failures, and Fallback

A turn receives an immutable model lease containing provider, model, strict
output mode, reasoning setting, temperature policy, context limit, cost limits,
the pinned `ContextBudgetPolicy`, and a configuration snapshot. Capability and
turn-admission context fit are checked before execution.

Failure classes are tool error, budget exhaustion, provider error, compaction
error, schema error, semantic error, security error, stale revision, and
rule-window close. Only declared provider/infrastructure failures may invoke
provider failover. A failed optional compaction handoff does not invoke
failover because the host checkpoint is sufficient. Schema and semantic errors
never switch models. One field-level semantic repair stays on the same lease
and cannot reopen context.

Budget exhaustion asks for the current best legal terminal proposal when time
permits. Deterministic fallback is task- and ruleset-specific, introduces no
new private disclosure or unsupported public fact, and is measured separately.
Closed or stale action windows receive no fallback.

Fallback is a liveness mechanism, not a strategy engine. It follows this fixed
order:

1. cancel when the action is optional or the rule window is closed;
2. choose a legal no-op, abstention, or no-new-information action when the
   ruleset permits it;
3. for speech, restate only the actor's latest still-valid public commitment or
   emit a typed no-new-information record;
4. for a mandatory targeted action, preserve the actor's latest explicit
   public target commitment when it is still legal and applicable;
5. when no commitment applies, select from the canonical legal-target list by
   a stable turn-seeded neutral selector.

Fallback cannot score evidence, inspect RAG or private beliefs, apply role
strategy, prefer a faction, infer a target, or call a model. The canonical
target order and stable seed are recorded in audit so matched runs can measure
fallback effects. Every fallback has a distinct origin and is excluded from
first-pass agency-quality metrics.

## 27. Evaluation and Acceptance

### 27.1 Safety Gates

- final committed proposal structured rate: 100%;
- RuleEngine legal rate: 100%;
- cross-player private leaks: 0;
- stale turn commits: 0;
- free-text live action recovery: 0 calls;
- objective replay consistency: 100%;
- judge factual deviations: 0;
- direct model/tool state mutation: 0 paths.
- compaction-created fact, evidence, grant, commitment, or memory: 0;
- post-compaction invisible or stale retained references: 0;
- successful compaction above 55-percent predicted occupancy: 0.

### 27.2 Agency and Cognition

Measure problem-driven tool selection, useful information gain, unnecessary
calls, belief changes after new evidence, evidence citation correctness,
fact/hypothesis separation, probability calibration, contradiction detection,
commitment consistency, and proposal diversity.

### 27.3 Memory, RAG, Reflection, and World-Model Value

Run ablations for full agent, no RAG, no in-turn reflection, no cross-game
memory, no world model, and checkpoint-only context. Compare repeated-mistake
rate, decision quality in matched situations, retrieval relevance, memory
contamination, lesson reuse, calibration, commitment consistency, and
cross-game improvement.

Components that show no measurable benefit stay disabled or are removed.

### 27.4 Human-Like Game Quality

Use blinded human review and automated diagnostics for speech repetition,
strategy diversity, persona differentiation, evidence use, uncertainty,
reasonable belief revision, coherent public commitments, risk preferences,
template artifacts, and overall resemblance to real play. Win rate is a result
metric, not the only optimization objective.

### 27.5 Performance

Measure per-turn p50/p95 latency, tokens, tool calls, RAG and world queries,
repair rate, fallback rate, compaction count, per-game cost, concurrent-player
rate limiting, and judge overhead.

Track compaction trigger occupancy, before/after tokens, compression ratio,
handoff failures, validation failures, retries, repeated-compaction rate,
latency, checkpoint-only continuation rate, and decision-quality delta across
equivalent compacted and non-compacted turns.

### 27.6 Adversarial and Length Tests

Test similar player IDs, contradictory public speech, instruction injection,
malicious RAG content, inaccessible private data, reordered documents, stale
evidence, duplicate submits, pause/resume, player death during calls, provider
timeouts, and semantically equivalent 2K/4K/8K/16K context ladders. Force the
80-percent boundary immediately before model work, during tool-result growth,
after a failed handoff, and for two successive compactions. Verify exact
authority retention, optional-handoff removal, stale-turn cancellation,
checkpoint-only continuation, and deterministic rehydrated-context hashes.

Default enablement requires safety gates, replay, adversarial privacy, long-
context behavior, acceptable fallback and latency thresholds, measurable
ablation benefit, and human review not worse than the retired system.

### 27.7 Executable Stage-1 Gates

The foundation/daytime-speech feature flag may be merged disabled after all of
these CI gates pass:

- generated JSON Schema fixtures match the checked-in schema hashes and every
  accepted/rejected speech fixture has one stable expected error code;
- 100 percent of successful terminal submissions produce exactly one event,
  one idempotency result, required critical audit records, and one or more
  projection-outbox records in the same transaction;
- fault injection at every boundary before and after model dispatch, result
  recording, RuleEngine validation, audit append, event append, idempotency
  persistence, outbox append, and commit produces either the complete expected
  transaction or no truth change;
- 50 concurrent duplicate submissions on each supported repository backend
  produce one commit result and no sequence gaps or duplicate audit sequences;
- 30 seeded scripted games replay to byte-equivalent canonical GameEvents and
  PublicSpeechRecords before and after process restarts;
- adversarial visibility fixtures, including similar player IDs and stale
  disclosure grants, produce zero private leaks and zero stale commits;
- deterministic rendering passes every entity, attribution, negation,
  modality, move-coverage, and forbidden-field fixture with zero deviations;
- the complete existing rule, event, replay, visibility, persistence, API, and
  runtime-flow test suites remain green.

Default enablement for real games additionally requires at least 100 seeded
full games with a controlled provider profile: schema-repair rate at or below
5 percent, neutral fallback rate at or below 3 percent, no safety-gate failure,
and host-runtime p95 overhead excluding provider latency at or below 100 ms per
turn on the reference CI machine. Three blinded reviewers each score at least
60 paired old/new utterances using the fixed evidence fidelity, coherence,
persona consistency, repetition, and template-artifact rubric. The new system
must have no category whose paired median is more than 0.5 points worse on a
five-point scale, and no factual-fidelity regression is permitted.

Stage-3 cognition components use at least 200 frozen, visibility-valid decision
fixtures plus 100 paired seeded games. A component is beneficial only when its
predeclared primary metric improves and the paired bootstrap 95-percent
confidence interval excludes zero, while safety, calibration, fallback rate,
latency, and cost remain within their declared non-inferiority margins. The
evaluation manifest pins fixtures, seeds, provider/model snapshot, prompts,
schema hashes, scorer versions, and exclusion rules before a run starts.

## 28. Implementation Sequence

1. Characterize preserved RuleEngine, event, replay, visibility, persistence,
   API, and runtime-flow behavior. Add an import-boundary test proving the new
   package cannot import rejected player-decision modules; do not delete the
   legacy path yet.
2. Define `GameRevision`, `LegalActionWindow`, terminal envelopes,
   `SpeechProposalBody`, stable errors, audit envelopes, dispatch attempts, and
   schema fixtures without routing any live game through them.
3. Extend each repository behind an explicit capability flag with durable turn,
   compare-and-swap `CommitTurn`, critical audit, idempotency, and projection
   outbox transactions. Backends without the capability cannot enable the new
   runtime.
4. Implement stage-1 `serial_public` scheduling, HostRuntime turn lifecycle,
   cancellation, durable model dispatch, restart reconciliation, and fake-
   provider fault-injection tests.
5. Implement isolated player document projections, `ObservationFrame`, context
   budget accounting, structured compaction, and checkpoint rehydration.
6. Implement the minimal stage-1 ToolGateway, structurally validated working
   reflection candidates, and one real AgentLoop for ordinary daytime speech.
7. Complete the first vertical slice through `SpeechProposal`,
   `PublicSpeechRecord`, atomic event/audit/outbox commit, commitment
   projections, deterministic player rendering, and deterministic judge
   narration.
8. Run all stage-1 CI gates, then add a per-game feature gate that selects
   exactly one player-decision runtime at game creation. Persist that choice in
   the ruleset/config snapshot; never switch an active game in place.
9. Add vote and private role capabilities one legal-window family at a time,
   followed by declared commutative windows and the wolf-team coordinator
   protocol.
10. Add world model, RAG, verified post-game reflection, skills, and cross-game
    memory behind independent feature gates and predeclared ablation metrics.
11. Complete optional presenter experiments only after the deterministic live
    renderers meet zero-deviation gates.
12. Run seeded, adversarial, length, replay, privacy, performance, ablation,
    and human-review gates before default enablement.
13. After no new games use the legacy runtime and historical replay readers are
    versioned, delete rejected SpeechAct code/tests and the old `PlayerAgent`,
    prompt, directive, quality heuristic, strategy handler, parser, retry, and
    fallback live paths in a separate reversible commit.

The first playable vertical slice is one daytime speech turn: build isolated
documents, open a turn, let the agent inspect context and evidence, reflect,
submit a strict speech proposal, validate and commit it, update commitments and
game projections, and produce a personality-aware judge broadcast.

## 29. Rejected Artifacts and Compatibility Policy

The following artifacts are superseded and must not guide implementation:

- `docs/superpowers/specs/2026-07-28-player-context-speech-act-design.md`;
- `docs/superpowers/plans/2026-07-28-structured-speech-foundation.md`;
- `werewolf_agent/agents/speech_act_schemas.py`;
- `tests/agents/test_speech_act_schemas.py`;
- old `PlayerAgent` public API and action-generation protocol;
- old player prompt, directive, quality, retry, fallback, and strategy tests.

Existing game-rule, event, replay, visibility, persistence, API, and runtime
flow tests remain valuable where they verify preserved host behavior rather
than old agent behavior.

No compatibility facade may silently route new decisions through the old
player implementation. Migration uses explicit feature gates at the runtime
entry point, with only one authoritative player-decision path per game.

Every game snapshot pins `player_runtime_version`. Existing or restored games
continue on their pinned legacy runtime until completion; the host never
switches an active game to the new runtime. New-runtime code may read
versioned, immutable legacy public events through a historical replay adapter,
but it may not call a legacy decision generator or convert rendered legacy
speech into new semantic claims. Historical replays retain their original
event/schema versions and remain readable after legacy live code is deleted.

Cutover requires an inventory showing zero active games on the legacy runtime,
successful replay of the supported historical corpus, a rollback procedure
that changes only the default for newly created games, and a signed record of
the stage-1 gates. Deletion of legacy code and tests occurs only after this
cutover record exists.

## 30. Final Invariants

1. RuleEngine alone adjudicates game mechanics and victory.
2. HostRuntime alone grants tool permissions and commits side effects.
3. Every player sees only a revision-pinned, viewer-specific workspace.
4. Markdown is a generated projection, never authority or direct write path.
5. The agent chooses authorized tools; tools do not make final decisions.
6. Facts, hypotheses, advice, commitments, and lessons are distinct types.
7. Reflection proposes memory; the host assigns scope and quality.
8. RAG and possible worlds never become game truth.
9. All final actions are strict structured proposals; no live prose recovery
   exists.
10. Public semantics come from `PublicSpeechRecord`, not rendered language.
11. Judge personality affects presentation, never facts or strategy.
12. Critical audit and GameEvent commits are atomic and tamper-evident.
13. No hidden chain-of-thought, raw secret-bearing output, or cross-player
   private context is persisted or exposed.
14. Stale, dead-player, closed-window, and duplicate turns cannot commit.
15. The rejected SpeechAct and old PlayerAgent contracts are not imported,
    adapted, or treated as compatibility requirements.
16. A game pins `PersonaProfile`; models and automated reviews cannot rewrite
    live personality or silently change future core identity.
17. Fallback preserves liveness through neutral mechanical selection and never
    encodes evidence scoring or role strategy.
18. Private audit payloads are encrypted, access-audited, retention-bounded,
    deletable, and sequenced only by transactional storage.
19. First-version live player and judge rendering uses strict sentence plans
    and host-filled templates, not free-form model prose.
20. At 80-percent predicted context occupancy the host compacts before the next
    model request; canonical context is reinjected and only typed checkpoint
    state, never a model summary, may preserve authority across the boundary.
21. `GameRevision` advances only in `CommitTurn`; sequential repository calls
    cannot emulate a committed action.
22. A global revision mismatch is not automatically stale: the legal window,
    read set, visibility grants, actor state, and declared conflict class decide
    whether a turn remains valid.
23. Every model/external-tool dispatch is durable before network I/O; unknown
    outcomes are explicit and never hidden by an untracked retry.
24. Wolf-team target selection belongs to the current coordinator agent;
    RuleEngine validates and resolves but never selects among strategic targets.
25. Private audit integrity commits to randomized ciphertext, and key deletion
    has an explicit, verifiable crypto-shredding or signed-redaction meaning.
26. Legacy decision code remains isolated until a working replacement passes
    cutover gates, then is removed in a separate reversible change.
