# Deep Agent Player Executor Integration Design

Date: 2026-07-31
Status: Draft; architectural direction approved
Owner: Codex development session
Parent: `docs/superpowers/specs/2026-07-28-autonomous-player-agent-runtime-design.md`

## 1. Purpose

This document defines how the new autonomous-player runtime may use the
LangChain Deep Agents SDK without transferring game authority, persistence
authority, or recovery authority into the framework.

The selected direction is:

> Each player has one logically isolated deep-agent identity, while all
> players share one host-owned `PlayerCognitionExecutor` implementation and
> compiled harness template. A deep-agent invocation exists only for one
> admitted `AgentTurn`; it is not a process, a game-wide conversation, or an
> authority over game state.

Deep Agents is an implementation of the bounded `AgentLoop` from the parent
design. It does not replace `HostRuntime`, the durable active-turn fence,
`ObservationProjectionService`, the future `ContextBudgetPolicy`,
`ToolGateway`, proposal validation, `RuleEngine`, or `CommitTurn`.

This design is intentionally written before context-compaction and AgentLoop
implementation. It prevents Deep Agents' convenient filesystem, summarization,
checkpoint, and subagent defaults from silently becoming a second runtime with
weaker hidden-information and recovery semantics.

## 2. Stack Relationship

The product names are not three nested application layers. Their technical
relationship is:

- LangGraph is the graph runtime used for agent execution and checkpointing;
- LangChain supplies model, tool, middleware, and `create_agent` abstractions;
- Deep Agents is an opinionated agent harness built from those LangChain and
  LangGraph components.

The application boundary is therefore:

```text
HostRuntime and repository authority
  -> PlayerCognitionExecutor
      -> DeepAgentPlayerExecutor
          -> Deep Agents harness
              -> LangChain model/tool abstractions
              -> LangGraph execution runtime
```

The rejected shape is:

```text
top-level LangGraph
  -> one unconstrained LangChain supervisor agent
      -> player Deep Agents
```

That shape duplicates scheduling, lets one model supervise multiple private
views, and makes graph checkpoints compete with the repository as recovery
authority. A future application-level graph may call `HostRuntime` methods as
ordinary deterministic nodes, but it must not hold player-private state or
decide whether a model/tool call is authorized.

## 3. External Capability Basis

The design is based on the public Deep Agents and LangChain behavior available
on 2026-07-31:

- `create_deep_agent` builds an agent harness with model, tools, system prompt,
  middleware, backend, memory, skills, response-format, and subagent options;
- virtual filesystems support state, durable store, composite routing, and
  custom backends;
- filesystem permissions apply to built-in filesystem tools, not arbitrary
  custom or MCP tools;
- a general-purpose synchronous subagent may be installed by default unless it
  is explicitly disabled;
- automatic summarization is a harness feature and can be excluded through a
  harness profile;
- LangGraph checkpoint replay can re-execute later nodes and external calls;
  checkpoint persistence therefore does not provide application-level
  exactly-once semantics.

References:

- <https://docs.langchain.com/oss/python/deepagents/overview>
- <https://docs.langchain.com/oss/python/deepagents/customization>
- <https://docs.langchain.com/oss/python/deepagents/backends>
- <https://docs.langchain.com/oss/python/deepagents/permissions>
- <https://docs.langchain.com/oss/python/deepagents/subagents>
- <https://docs.langchain.com/oss/python/deepagents/profiles>
- <https://docs.langchain.com/oss/python/langgraph/persistence>
- <https://docs.langchain.com/oss/python/langchain/tools>
- <https://pypi.org/project/deepagents/>

The current `wofkill` environment contains `langchain 1.3.14` and
`langgraph 1.2.9`; it does not contain `deepagents`. The latest stable PyPI
release observed while writing this design is `deepagents 0.6.12`. No
dependency is added by this design. The implementation plan must run a narrow
compatibility spike against the repository's pinned LangChain/LangGraph
versions and pin one exact stable Deep Agents version before adding production
code. Compatibility is not inferred from version numbers alone. Alpha releases
are excluded.

## 4. Scope

This design specifies:

1. logical per-player deep-agent identity and physical runtime reuse;
2. the framework-neutral `PlayerCognitionExecutor` port;
3. the Deep Agents adapter and its allowed dependencies;
4. immutable runtime context and player profile injection;
5. a viewer-isolated virtual filesystem over observation projections;
6. durable fenced model and external-tool dispatch;
7. terminal structured proposal capture;
8. context-budget and checkpoint ownership;
9. failure, cancellation, restart, audit, and privacy behavior;
10. a bounded daytime-speech proof of concept and its gates.

## 5. Explicit Non-goals

This design does not authorize implementation of:

- a live-game path or integration with the old `PlayerAgent`;
- a supervisor model that controls multiple players;
- physical player workspace directories;
- arbitrary local filesystem, shell, code execution, browser, moderator, or
  unrestricted MCP access;
- Deep Agents long-term memory as current-game truth;
- Deep Agents automatic summarization as the host compaction protocol;
- synchronous or asynchronous Deep Agents subagents in stage 1;
- model-driven schedule, window, lease, budget, retry, provider, or feature-
  gate selection;
- proposal legality, visibility, RuleEngine resolution, `CommitTurn`, public
  rendering, or judge narration inside the Deep Agents adapter;
- vote, private-role, concurrent, or wolf-team windows;
- ToolResult Markdown projection ahead of its separate approved milestone; or
- installation of `deepagents` in this documentation-only change.

## 6. Authority Model

The parent design's authority hierarchy remains unchanged.

| Concern | Authority | Deep Agents role |
| --- | --- | --- |
| schedule and active turn | `HostRuntime` plus repository | receives opaque identity only |
| dispatch authorization | durable active-turn fence | no direct repository access |
| game facts and visibility | RuleEngine/Host authority stores | reads viewer projection only |
| observation bytes | `ObservationProjectionService` | reads immutable virtual files |
| context limits and compaction | host context lifecycle | consumes rehydrated context |
| tools and side effects | Host `ToolGateway` | chooses among granted wrappers |
| action semantics | strict proposal contracts | proposes model-controlled fields |
| legality and game mutation | Host validator, RuleEngine, `CommitTurn` | none |
| framework execution state | Deep Agents/LangGraph | disposable implementation state |

Deep Agents messages, TODOs, scratch files, summaries, model final text,
subagent output, and graph checkpoints are untrusted model-side artifacts.
They cannot add facts, permissions, evidence ownership, grants, legal targets,
budgets, committed records, or recovery authority.

## 7. One Logical Deep Agent Per Player

### 7.1 Logical identity

One player agent is identified by an immutable host-built key conceptually
equivalent to:

```text
game_id
+ player_id
+ player_runtime_version
+ pinned_persona_profile_hash
```

This key selects the player's profile and namespaces. It is not a provider
conversation ID and is never supplied by the model.

Each legal opportunity creates a separate execution identity:

```text
logical player identity
+ turn_id
+ model_lease_hash
+ active window identity
```

The LangGraph `thread_id` is derived from the opaque `turn_id`, not merely the
`player_id`. Replacement turns receive new turn IDs and new threads. This
prevents an unbounded conversation from surviving across windows and prevents
stale private context from being inherited by a replacement turn.

### 7.2 Shared physical implementation

The runtime should compile and reuse one Deep Agents harness template for each
compatible executor configuration. It must not create one Python process,
worker, model client, or compiled graph per player.

Per-player differences enter through strict runtime context and immutable
profile snapshots. Role and current-game facts do not belong in the permanent
profile; they arrive in the current `ObservationBundle`.

Stage-1 `serial_public` scheduling means only the active player's invocation is
authorized to perform model work. Logical identities for other players may
exist, but they are idle and have no readable active workspace.

### 7.3 Player profile

Introduce a new strict, frozen `PlayerAgentProfile` in a later contract change.
It should contain references and hashes, not arbitrary framework objects:

```text
profile_id
profile_version
persona_snapshot_id
persona_snapshot_hash
system_contract_version
model_policy_id
tool_policy_id
context_policy_id
subagent_policy = disabled
profile_hash
```

The per-turn `ModelLease` selects the actual provider/model. A player profile
may select a host policy but cannot pin provider credentials, instantiate a
model, or override the admitted lease.

Deep Agents `HarnessProfile` is provider/model harness configuration. It is
not the same concept as `PlayerAgentProfile` and must never become the canonical
player identity store.

## 8. Framework-neutral Execution Port

The Host-facing abstraction is intentionally independent of Deep Agents:

```text
PlayerCognitionExecutor.run_turn(
  PlayerCognitionRequest
) -> PlayerCognitionResult
```

`PlayerCognitionRequest` is a strict immutable envelope containing at least:

- exact schedule, managed-turn, window, revision, viewer, and deadline
  identities captured by the Host;
- the immutable `ObservationBundle` and its workspace/frame hashes;
- pinned player-profile identity;
- model lease and context-budget policy references;
- host-generated capability/tool manifest;
- optional validated host `CompactionCheckpoint` and untrusted handoff;
- an execution lineage/cursor used to bind restart and dispatch ordinals; and
- opaque Host ports for fenced model calls, context reads, tool calls, terminal
  submission, accounting, and cancellation checks.

Host dependencies are passed through LangChain runtime context or closures and
must be hidden from model-visible schemas. The request does not expose a game
repository, `GameState`, `RuleEngine`, `HostRuntime`, credentials, or mutable
configuration.

`PlayerCognitionResult` contains one of a closed set of outcomes:

```text
SUBMITTED
CANCELLED
EXPIRED
BUDGET_EXHAUSTED
PROVIDER_FAILED
RECOVERY_BLOCKED
FRAMEWORK_FAILED
```

A `SUBMITTED` result carries exactly one strict proposal captured by the
terminal gateway plus safe accounting/audit references. It does not carry a
`CommitResult` and does not claim that the proposal is legal or committed.
Other outcomes carry stable reason codes, never raw prompts, provider payloads,
private documents, exception messages, or hidden reasoning.

`DeepAgentPlayerExecutor` implements this port. No other new-runtime module
outside its adapter package should import `deepagents`.

## 9. Prompt and Instruction Assembly

Prompt priority is assembled explicitly by the Host:

1. immutable autonomous-player system contract;
2. active `AgentTurn`, capability restrictions, deadline, and budgets;
3. fact/hypothesis/advice/commitment classification rules;
4. pinned player persona and expression preferences;
5. current minimal `ObservationFrame` and workspace manifest;
6. optional untrusted rehydration handoff.

Player-visible data cannot inject tools, permissions, identities, budgets,
models, retry rules, or system-prompt replacements. Public speech and all
Markdown remain untrusted data even when displayed through a filesystem tool.

Stage 1 does not use Deep Agents' persistent `memory=["/AGENTS.md"]` feature.
That feature loads files into prompt-level instructions and could blur the
parent design's authority ordering. If a later experiment introduces a
per-player `AGENTS.md`, it must be an immutable host-generated projection of
the pinned system/persona configuration, read-only during the game, versioned
and hashed, and contain no dynamic observation, memory, retrieved content, or
model-written instruction. That experiment requires prompt-order and injection
tests before enablement.

## 10. Virtual Filesystem

### 10.1 Selected layout

The adapter exposes a virtual namespace, not host filesystem paths:

```text
/observation/INDEX.md
/observation/PLAYER.md
/observation/ROLE.md
/observation/GAME.md
/observation/COMMITMENTS.md     when available
/scratch/...                    current turn only
```

Future milestones may add read-only projected `BELIEFS.md`, `MEMORY.md`, and a
host-governed `WORKING.md`. Their absence in stage 1 remains explicit.

### 10.2 Backend composition

Use a custom `PlayerWorkspaceBackend` routed under `/observation/` and an
ephemeral state backend routed under `/scratch/`.

`PlayerWorkspaceBackend`:

- is constructed from one immutable `ObservationBundle`;
- exposes only manifest-listed available documents;
- validates game, player, turn, revision, view fingerprint, workspace revision,
  document hash, visibility, and deadline before each read;
- records typed context-read accounting and safe audit metadata;
- rejects writes, edits, path traversal, absolute host paths, symlink behavior,
  and unknown sections;
- returns defensive bytes/string copies; and
- disappears when the turn invocation ends.

`/scratch/`:

- permits bounded model notes and harness planning only;
- is scoped to the current turn/thread;
- is not projected back into observation or parsed into Host state;
- is not accepted as belief, reflection, memory, evidence, proposal, or
  compaction authority;
- is deleted or discarded after terminalization; and
- counts toward active context and configured local-tool budgets.

Do not use `FilesystemBackend`, `LocalShellBackend`, or a sandbox `execute`
tool in the live player runtime. Deep Agents filesystem permissions are useful
defense in depth, but the custom backend and Host gateways remain the actual
security boundaries.

### 10.3 Filesystem tools

The harness may expose bounded `ls`, `read_file`, `glob`, and `grep` over the
virtual routes. `write_file` and `edit_file` are permitted only under
`/scratch/`. Every tool is wrapped so identity, deadline, route, result size,
and accounting are checked outside model control.

Observation reads are local revision-pinned context operations, not external
network dispatches. They still consume the context-read/tool budget. An
external or durable read tool is not implemented through this backend; it goes
through the future `ToolGateway` and durable dispatch protocol.

## 11. Deep Agents Harness Configuration

The first implementation must construct the harness through one audited
factory. The conceptual configuration is:

```text
model              = FencedChatModelAdapter
tools              = HostToolGateway wrappers + terminal submit tool
system_prompt      = Host-built immutable prompt
backend            = PlayerWorkspaceBackend + ephemeral scratch backend
runtime context    = DeepAgentExecutionContext
checkpointer       = non-durable/disposable in stage 1
store              = none in stage 1
subagents          = none
default subagent   = explicitly disabled
summarization      = explicitly disabled
execute/shell      = excluded
long-term memory   = disabled
```

The default general-purpose subagent must be disabled through the supported
`GeneralPurposeSubagentProfile(enabled=False)` mechanism, with no synchronous
or asynchronous subagents supplied. The built-in summarization middleware must
be excluded. Tests must inspect the compiled tool manifest and middleware
behavior rather than trusting constructor intent.

Planning/TODO and scratch operations may be retained only if they are bounded,
player-local, non-authoritative, and included in accounting. If the chosen
Deep Agents version cannot hide or wrap a built-in tool without private API
patching, the tool is disabled or the integration is rejected; production code
must not monkey-patch Deep Agents internals.

## 12. Fenced Model Dispatch

Every model request originating from the harness must pass through a host-owned
`FencedModelDispatcher`. The Deep Agents adapter receives a LangChain-compatible
`FencedChatModelAdapter` whose only provider path delegates to that dispatcher.

For each request, the dispatcher:

1. checks cancellation, deadline, active status, remaining model-call budget,
   lease, and context occupancy;
2. computes a canonical request hash over the exact messages, tool schemas,
   strict-output settings, model parameters, observation/checkpoint lineage,
   and logical model-step ordinal;
3. asks `HostRuntime.create_active_turn_dispatch()` to create the attempt at
   the durable active-turn fence;
4. claims and hands off the attempt through the durable dispatch state machine;
5. invokes only the provider named by the immutable model lease, using the
   persisted provider idempotency key;
6. records the typed provider result before exposing it to the harness;
7. rechecks result binding to the active turn, lease, window, revision, view,
   deadline, and request hash; and
8. returns a sanitized LangChain model response or a stable execution failure.

The adapter never calls a provider client directly. Framework retries, model
fallback middleware, and provider-native retry behavior must be disabled or
configured so they cannot create an unrecorded second network request. Retry
and failover remain Host policies and receive new fenced attempts only when the
parent design permits them.

The future production dispatcher needs a narrow Host-facing API for claim,
handoff, result recording, and recovery reuse. It must not expose the raw
durable-dispatch repository to Deep Agents.

## 13. Tool Dispatch

Tools fall into three classes:

1. local harness operations such as bounded scratch writes;
2. revision-pinned local context reads from the observation bundle; and
3. Host `ToolGateway` operations, including any external or durable read.

Only class 3 may expose game-domain tools. Each wrapper derives actor, role,
phase, view, legal targets, grants, and budget from `DeepAgentExecutionContext`.
The model supplies only the bounded arguments allowed by the tool schema.

External class-3 calls use the durable active-turn dispatch protocol just like
model calls. Local deterministic calls do not create fake dispatch attempts,
but they still receive call IDs, accounting, output bounds, classification,
and audit records.

Deep Agents permission rules do not govern custom or MCP tools, so no custom
tool is considered safe merely because filesystem permissions are present.
There is no raw MCP client in the adapter. Any later MCP integration sits
behind `ToolGateway` and returns the project's structured `ToolResult`; its
Markdown is a model-facing projection only.

Parallel model-selected tool calls are disabled in stage 1 unless the
`ToolGateway` has an explicit deterministic ordering, budget reservation, and
per-call fence policy. A terminal submit tool cannot run in parallel with any
other tool.

## 14. Terminal Proposal

Stage 1 exposes exactly one terminal tool: `submit_speech`.

The tool schema contains only the model-controlled fields of the existing
strict speech proposal body. The Host supplies actor, turn, schedule, window,
revision, view fingerprint, schema version, proposal ID, idempotency key, and
timestamps. Echoed identity fields are either absent or compared exactly and
rejected as a security violation when mismatched.

The terminal tool:

1. parses the strict proposal schema;
2. captures exactly one proposal in executor state;
3. prevents further model, context, and tool dispatch;
4. returns a terminal `SUBMITTED` outcome to the Host; and
5. performs no RuleEngine call or game commit.

The full visibility, evidence, semantic, grant, legality, RuleEngine, audit,
and `CommitTurn` sequence remains outside `DeepAgentPlayerExecutor`.

Plain assistant text, Deep Agents final text, TODO completion, a scratch file,
or LangChain `structured_response` is not a terminal action. If the harness
ends without a successful terminal tool call, the Host may request another
bounded step while budget and deadline permit; otherwise it follows the
declared neutral fallback/cancellation policy. No prose parser or free-text
recovery is allowed.

LangChain `response_format` may be used only as an implementation aid after it
is proven not to bypass the terminal gateway. The default design keeps
`submit_speech` as the sole terminal boundary.

## 15. Context Budget and Compaction

Deep Agents automatic summarization is disabled because its generated summary
and offloaded history do not implement the parent design's authority-preserving
`CompactionCheckpoint` contract.

Before every model call, `FencedModelDispatcher` invokes the host context
lifecycle boundary. The Host owns:

- tokenizer/estimator selection and safety margin;
- immutable prefix, active history, tool-schema, and output reserves;
- the 80-percent automatic trigger and 55-percent post-compaction target;
- maximum compactions and all remaining budgets;
- transitions into and out of `COMPACTING`;
- exact checkpoint construction, validation, persistence, and lineage;
- optional untrusted handoff generation; and
- checkpoint-only terminal context and fallback decisions.

Deep Agents receives the rehydrated message/context sequence after the Host
completes compaction. Harness scratch files and TODO state may be discarded.
They can be preserved only as explicitly bounded untrusted attachments whose
hashes and token cost are recorded; they never enter the strict checkpoint by
implicit framework serialization.

The optional `CompactionHandoff` model call uses the same fenced model adapter
with a distinct operation kind, no tools, no framework retry, and the same
lease/cost budget. Deep Agents summarization middleware is not used to produce
it.

## 16. Checkpoint and Recovery Ownership

### 16.1 Stage-1 policy

Stage 1 uses no durable Deep Agents or LangGraph checkpointer. Framework state
is process-local and disposable. Restart recovery is:

```text
HostRuntime.recover_game
  -> reconcile durable model/tool dispatch
  -> verify active turn and durable fence
  -> load and validate Host CompactionCheckpoint when present
  -> rebuild current ObservationBundle
  -> reconstruct bounded model context
  -> invoke a fresh Deep Agents harness run for the same turn
```

Recorded dispatch results are reused only through the Host execution cursor and
their original binding. The framework cannot independently replay a provider
or tool request.

### 16.2 Future framework checkpoints

A later optimization may store LangGraph checkpoints only behind a
`ValidatedAgentCheckpointStore`. Such checkpoints are caches, not authority,
and must be:

- namespaced by exact game/player/turn/runtime/framework schema identity;
- bound to active-turn, model-lease, observation, context-checkpoint, tool-
  manifest, and budget hashes;
- validated by the Host before every resume;
- unable to reopen a terminal or stale turn;
- unable to restore a larger budget or invisible context;
- safe to delete and rebuild; and
- prevented from executing side effects except through original durable
  dispatch identities.

This future optimization requires a separate design and fault-injection gate.
It is not implied by installing Deep Agents.

## 17. Subagents

All Deep Agents subagents are disabled in stage 1. This is required both by the
parent design's rejection of planner/critic/actor model personas in one turn
and by the need to prove one bounded AgentLoop before adding delegation.

A Deep Agents subagent is never another game player. It must never receive a
different player identity, wider visibility, independent lease, independent
budget, terminal tool, or durable memory namespace.

Any future proposal to enable a subagent requires an amendment to the parent
design and a separate experiment proving that it:

- performs only advisory, context-isolated cognition;
- receives a strict subset of the parent player's tools and view;
- shares the parent deadline and reserved budget;
- returns typed references rather than hidden reasoning;
- cannot submit or commit an action; and
- improves a predeclared metric enough to justify its latency and cost.

Async subagents remain out of scope because their independent lifecycle,
mid-flight updates, and cancellation introduce another durable scheduling
problem that the current active-turn fence does not model.

## 18. Cancellation and Deadline Behavior

The executor checks an injected host cancellation/deadline guard:

- before harness invocation;
- before and after every model call;
- before and after every ToolGateway call;
- before an observation read that expands context; and
- before terminal proposal capture.

If the turn becomes cancelled, expired, replaced, stale, or terminal, new
dispatch is rejected by the persistent fence. A late result is recorded and
audited according to durable-dispatch rules but is not returned as usable
agent context.

Process-local cancellation flags may reduce wasted work, but they are only an
optimization. They cannot authorize or reject work independently of the
repository state.

## 19. Failure Mapping

Framework and provider details are sanitized at the adapter boundary.

| Failure | Stable treatment |
| --- | --- |
| unsupported/missing Deep Agents version | capability disabled before admission |
| harness construction/configuration error | `FRAMEWORK_FAILED` |
| active-turn fence rejection | stale/cancelled execution; no retry |
| provider unresolved outcome | recovery barrier and declared Host policy |
| model/provider error | Host lease failure/failover policy |
| unauthorized tool/path | security failure and audit |
| local scratch/backend failure | bounded tool error; no authority change |
| observation identity/hash mismatch | fail closed and cancel stale execution |
| terminal schema error | existing repairable proposal error policy |
| framework ends without terminal submit | bounded retry or Host fallback |
| context compaction failure | host compaction failure policy |

Raw framework exceptions, prompt fragments, private tool payloads, provider
responses, filesystem contents, and credentials are not placed in stable error
messages or public audit projections.

## 20. Package Boundaries

The eventual implementation should use cohesive modules resembling:

```text
werewolf_agent/player_agents/
  cognition/
    contracts.py                 framework-neutral request/result/profile
    executor.py                  PlayerCognitionExecutor protocol
    deep_agent/
      factory.py                 audited create_deep_agent configuration
      executor.py                DeepAgentPlayerExecutor adapter
      context.py                 hidden immutable runtime context
      model.py                   FencedChatModelAdapter
      workspace_backend.py       virtual observation/scratch routing
      terminal.py                submit_speech capture adapter
      errors.py                  sanitized framework error mapping
```

Host-owned model/tool dispatch services belong beside Host runtime or gateways,
not inside the framework adapter. Contracts must not import Deep Agents. The
adapter package may import `deepagents`, LangChain, and LangGraph, but it may
not import old `PlayerAgent`, old prompts, directives, strategies, parsers,
fallbacks, `_dispatch_agent`, or live `GameRunner` paths.

Deep Agents remains an optional runtime capability behind the future per-game
`player_runtime_version` gate. Importing the general contracts or running
legacy games must not require the package to be installed.

## 21. Data Flow

One ordinary daytime-speech turn is:

```text
Host admits serial-public turn
  -> ObservationProjectionService builds immutable player bundle
  -> Host pins profile, lease, budget, capabilities, and execution cursor
  -> PlayerCognitionExecutor.run_turn
      -> factory creates/configures bounded Deep Agents run
      -> model request -> fenced durable model dispatch
      -> optional local observation/scratch operations
      -> optional ToolGateway calls -> fenced when external
      -> submit_speech captures strict proposal
  -> Host validates active identity and proposal
  -> visibility/evidence/semantic validation
  -> RuleEngine resolution
  -> CommitTurn atomic transaction
  -> HostRuntime.complete_active_turn
  -> projection workers and deterministic presentation
```

There is no edge from Deep Agents directly to repository truth, schedule
terminalization, RuleEngine, `CommitTurn`, another player's workspace, or the
legacy runtime.

## 22. Proof-of-concept Sequence

The integration should not begin until the current context-budget/checkpoint
milestone defines the Host lifecycle consumed by the adapter.

Then implement in this order:

1. add framework-neutral cognition request/result/profile contracts;
2. add an optional-dependency capability probe and exact-version gate;
3. implement virtual observation plus ephemeral scratch backend;
4. implement the fenced LangChain chat-model adapter against a fake provider;
5. construct Deep Agents with summarization, subagents, memory, execute, and
   framework retries disabled;
6. expose only local context operations and `submit_speech`;
7. run one ordinary daytime-speech proposal without RuleEngine or live routing;
8. add restart/replay tests using durable dispatch results and Host checkpoint;
9. implement minimal ToolGateway and ToolResult presentation in their already
   declared later milestones; and
10. connect proposal validation/RuleEngine/`CommitTurn` only in the first
    vertical-slice milestone.

Each step remains mergeable while the runtime feature is disabled.

## 23. Test Strategy

### 23.1 Contract and identity tests

- strict/frozen player profile, request, result, execution context, and status;
- logical player identity differs for similar player IDs and games;
- replacement turn receives a new framework thread;
- role/current facts cannot enter the permanent profile; and
- model-visible schemas omit repositories, Host ports, credentials, identity
  selectors, budgets, and retry controls.

### 23.2 Harness configuration tests

- compiled tools contain no `execute`, shell, raw MCP, moderator, commit, model-
  switch, retry, or cross-player tool;
- general-purpose, custom synchronous, and async subagents are absent;
- Deep Agents automatic summarization and model fallback/retry are absent;
- no durable framework store/checkpointer is configured in stage 1; and
- unsupported package/API versions fail capability admission safely.

### 23.3 Workspace and privacy tests

- two players with similar IDs receive byte-distinct correct views;
- private role facts never cross namespaces, cache entries, scratch space, or
  error output;
- traversal, absolute paths, symlinks, unknown paths, and writes outside
  `/scratch/` fail closed;
- only manifest-available observation documents can be read;
- stale identity, view, source hash, workspace revision, or deadline rejects a
  read; and
- no physical player file or durable projection row is created.

### 23.4 Dispatch and recovery tests

- every harness model call produces one fenced durable dispatch before network
  I/O;
- framework retry settings cannot create an unrecorded provider call;
- external tool calls follow the same active-turn fence;
- recorded result reuse preserves original request/lease/view bindings;
- unresolved non-idempotent calls enter the existing unknown-outcome path;
- cancellation/terminalization races cannot admit a new call;
- restart rebuilds from Host checkpoint and recorded results without trusting
  a LangGraph checkpoint; and
- a forged or stale framework thread cannot resume work.

### 23.5 Terminal tests

- exactly one valid `submit_speech` becomes `SUBMITTED`;
- parallel or duplicate terminal calls cannot produce two proposals;
- plain final text and `structured_response` do not become proposals;
- Host-bound identity cannot be supplied or changed by the model;
- schema/semantic repair obeys the existing bounded repair policy; and
- the adapter never calls RuleEngine, `CommitTurn`, or lifecycle completion.

### 23.6 Context tests

- occupancy is checked before every model call;
- Deep Agents summarization never runs;
- 80-percent/55-percent behavior is owned by Host context tests;
- scratch/TODO state cannot appear as checkpoint authority;
- optional handoff failure still permits checkpoint-only continuation; and
- rehydration cannot restore stale visibility, grants, legal targets, or
  budgets.

### 23.7 Architecture and regression tests

- AST/import tests constrain `deepagents` imports to the adapter package;
- the adapter cannot import legacy player decision or live runtime modules;
- the new package has no raw provider or MCP construction path;
- Memory/SQLite/PostgreSQL fence conformance remains unchanged;
- focused autonomous-player tests, Ruff, mypy, `git diff --check`, and full
  pytest remain green.

## 24. Proof-of-concept Acceptance Criteria

The daytime-speech Deep Agents proof of concept is accepted only when:

1. one shared executor template serves multiple logically isolated players;
2. every run is scoped to exactly one active turn and viewer;
3. every model and external-tool network request is durably fenced;
4. there are zero direct provider, raw MCP, filesystem, shell, RuleEngine,
   `CommitTurn`, or `GameState` paths from the adapter;
5. observation files are virtual, read-only, hash/revision pinned, and never
   persisted as authority;
6. built-in summarization, durable framework checkpointing, long-term memory,
   and all subagents are disabled;
7. one strict terminal proposal is captured only through `submit_speech`;
8. crash recovery uses Host checkpoint plus durable results and never blind
   framework replay;
9. similar-player-ID, prompt-injection, stale-turn, deadline, duplicate-submit,
   and cancellation-race fixtures produce zero leaks and zero stale proposals;
10. the optional dependency is exact-version pinned and the feature remains
    disabled for games that did not select the new runtime at creation; and
11. all existing autonomous and repository gates remain green.

Passing these criteria proves an isolated cognition adapter. It does not prove
the playable vertical slice, which still requires ToolGateway, full proposal
validation, RuleEngine resolution, `CommitTurn`, projections, deterministic
renderers, presenters, feature gates, and stage-1 evaluation.

## 25. Final Invariants

1. One player means one logical agent identity, not one process or permanent
   provider conversation.
2. One admitted turn means one bounded Deep Agents thread.
3. All players share code and infrastructure while their profiles, views,
   scratch state, checkpoints, and future memories remain isolated.
4. Deep Agents is a replaceable `PlayerCognitionExecutor`, not Host authority.
5. LangGraph execution/checkpoint state never outranks repository state.
6. Every external model/tool call crosses the durable active-turn fence.
7. Deep Agents built-ins cannot bypass Host budgets, ACLs, deadlines, or audit.
8. Framework summaries, files, TODOs, and final text are untrusted model data.
9. Only a strict terminal gateway yields a proposal; only Host/RuleEngine/
   `CommitTurn` can make it game truth.
10. No Deep Agents integration reaches the old `PlayerAgent` or live game path.
