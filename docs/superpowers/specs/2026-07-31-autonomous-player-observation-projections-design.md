# Autonomous Player Observation Projections Design

Date: 2026-07-31
Status: Draft for written review; design direction approved
Owner: Codex development session
Parent: `docs/superpowers/specs/2026-07-28-autonomous-player-agent-runtime-design.md`

## 1. Purpose

This design adds the first model-readable context boundary for the new
autonomous player runtime. Given one valid active `serial_public` daytime
speech turn, the Host can build an immutable, viewer-specific bundle containing
typed player-document projections, `INDEX.md`, and a revision-pinned
`ObservationFrame`.

The documents resemble a modular `CLAUDE.md` workspace for one player, but
they are not editable instructions, canonical game storage, or arbitrary
physical files. They are deterministic read-only projections rebuilt from
Host authority.

The first slice establishes:

- strict projection and manifest contracts;
- dedicated projectors for `PLAYER.md`, `ROLE.md`, `GAME.md`, and, when a
  committed source is available, `COMMITMENTS.md`;
- explicit unavailable entries for `BELIEFS.md`, `MEMORY.md`, and
  `WORKING.md` until the new runtime owns their authoritative stores;
- deterministic `INDEX.md` generation;
- a Host-facing observation projection service that captures and rechecks the
  active turn; and
- privacy, revision, visibility, integrity, cache-isolation, and deterministic
  rebuild tests.

## 2. Scope Boundary

This stage targets exactly one active `serial_public` daytime speech turn. It
does not generalize to every future window type.

This stage does not:

- persist projection rows in Memory, SQLite, or PostgreSQL;
- create real `players/{player_id}/*.md` files;
- add canonical belief, cross-game memory, or working-reflection stores;
- implement context-threshold decisions, compaction checkpoints, rehydration,
  provider/model leases, AgentLoop, proposal validation, RuleEngine calls, or
  `CommitTurn` orchestration;
- implement the autonomous ToolGateway or ToolResult Markdown projection;
- connect to `GameRunner`, live-game entry points, the old `PlayerAgent`, old
  prompts, `ModelRouter`, `_dispatch_agent`, or legacy tool schemas; or
- claim the first playable vertical slice is complete.

Real PostgreSQL service validation remains a pre-production repository gate.
No new PostgreSQL schema or SQL is needed for this read-only projection stage.

## 3. Existing Gap

The durable active-turn fence now gives the Host an exact schedule, managed
turn, window, revision, lease, view, and deadline identity. The runtime still
lacks a safe way to turn that identity into player-visible model context.

Existing legacy prompts, player workspaces, world-model state, RAG, memory,
reflection, judge, and tool code are not accepted compatibility contracts for
the new runtime. Directly exposing the legacy `GameState` or filesystem would
allow the new player path to bypass viewer-specific visibility, revision pins,
read-set integrity, and deterministic replay.

The next boundary must therefore separate:

1. Host-owned typed authority;
2. viewer-specific deterministic projections;
3. the immutable observation envelope; and
4. the future model's interpretation.

Only the first three layers are built here.

## 4. Approaches Considered

### 4.1 Selected: typed section projectors plus a separate frame assembler

Each supported document has a dedicated typed projector. A workspace
projector combines their results and generates the manifest and `INDEX.md`.
An independent observation-frame assembler verifies the active-turn and
projection identities before creating the final bundle.

Advantages:

- authority, presentation, and active-turn authorization remain distinct;
- section-specific visibility and requiredness are explicit in types;
- missing future stores are represented honestly instead of by fake content;
- new beliefs, memory, and working sections can be added without changing the
  existing frame identity; and
- deterministic tests can reject cross-player, stale, reordered, or mutated
  inputs.

### 4.2 Rejected: generic document registry

A configuration-driven registry would reduce boilerplate, but it would hide
requiredness, visibility, source validation, and semantic rendering rules in
runtime configuration. Errors would surface late and type checking would not
prove that a role projector cannot receive a public-game payload.

### 4.3 Rejected: durable projection snapshots

Persisting every generated Markdown snapshot would require schema migrations,
invalidation, cleanup, and a second concurrency protocol. These projections
are rebuildable from authority and are not recoverable turn state. Durable
compaction checkpoints belong to the next milestone and must not be confused
with presentation caches.

## 5. Authority Hierarchy and Core Invariants

The representation hierarchy is:

1. canonical Host stores and committed records;
2. a revision-pinned, viewer-filtered `ObservationAuthoritySnapshot`;
3. typed projected documents and manifest;
4. immutable `ObservationFrame` and `ObservationBundle`; and
5. a future model's interpretation.

Mandatory invariants:

- projections never become game facts, role facts, evidence, commitments,
  memories, or legal-action authority;
- Markdown is never parsed back into Host state;
- projectors receive only one viewer's filtered snapshot and cannot load all
  roles and filter them locally;
- every available document is bound to game, player, base revision, view
  fingerprint, renderer version, source IDs, and source hashes;
- every cache lookup includes game, player, view fingerprint, workspace
  revision, and section ID;
- required projections fail closed; optional unavailable projections expose no
  hidden-record existence signal;
- identical authoritative input produces byte-identical output and hashes;
- a result built for a turn that changes during projection is discarded; and
- successful projection does not authorize a future model dispatch. The
  durable active-turn fence remains the dispatch authorization boundary.

## 6. Contract Model

All contracts are strict, frozen Pydantic models and reject unknown fields.
Mutable inputs and returned collections are defensively copied.

### 6.1 Section and availability enums

```text
WorkspaceSection
  PLAYER
  ROLE
  GAME
  BELIEFS
  COMMITMENTS
  MEMORY
  WORKING
  INDEX

ProjectionAvailability
  AVAILABLE
  UNAVAILABLE

ProjectionVisibilityClass
  PLAYER_PRIVATE
  ROLE_PRIVATE
  PUBLIC
  MIXED_VIEWER_FILTERED
  MANIFEST
```

The initial required sections are `PLAYER`, `ROLE`, `GAME`, and `INDEX`.
`COMMITMENTS` is available with an empty typed collection when its committed
source exists but contains no records. It is unavailable only when the source
capability itself is absent. `BELIEFS`, `MEMORY`, and `WORKING` are unavailable
in this stage.

### 6.2 Projection identity and source references

```text
ProjectionIdentity
  game_id
  player_id
  schedule_id
  turn_id
  schedule_state_version
  turn_state_version
  window_id
  window_version
  base_game_revision
  view_fingerprint

ProjectionSourceReference
  record_kind
  record_id
  record_revision
  content_hash
```

Source references are sorted by `(record_kind, record_id, record_revision,
content_hash)` before rendering and hashing. A duplicate identity with a
different hash is an integrity error.

### 6.3 Projected documents and manifest

```text
ProjectedDocument
  section_id
  identity
  renderer_version
  content_markdown
  content_hash
  token_estimate
  estimator_version
  visibility_class
  source_references

ManifestEntry
  section_id
  availability
  required
  identity
  renderer_version
  content_hash
  token_estimate
  estimator_version
  visibility_class
  source_references
  unavailable_reason
```

Available entries require a matching document, content hash, estimate, and
source references. Unavailable entries contain no document, content hash,
estimate, or source references. Their external reason is a bounded stable code
such as `source_capability_unavailable`; it does not reveal whether a hidden
record exists.

### 6.4 Workspace snapshot

```text
PlayerWorkspaceSnapshot
  identity
  workspace_revision
  documents
  manifest_entries
  workspace_hash
```

`workspace_revision` is a deterministic content-addressed digest over the
projection identity, ordered manifest, renderer versions, estimator version,
and source hashes. It is not a new database sequence. `workspace_hash` covers
the complete ordered document bytes plus the manifest.

To avoid a self-referential hash, `INDEX.md` renders the ordered entries for
`PLAYER` through `WORKING` plus the already-computed workspace revision. The
final manifest then appends the `INDEX` entry containing the resulting INDEX
content hash. INDEX never renders its own content hash or token estimate.

### 6.5 Observation frame and bundle

```text
ObservationFrame
  identity
  task_kind
  actor_id
  role_id
  phase
  legal_action_snapshot
  legal_target_snapshot
  critical_private_fact_references
  bounded_public_summary
  recent_commitment_references
  document_manifest
  tool_manifest
  workspace_revision
  workspace_hash
  deadline
  observed_at

ObservationBundle
  frame
  workspace
```

The first `task_kind` is the existing ordinary daytime speech task. Legal
actions, targets, private facts, summaries, commitments, and tool manifest are
typed bounded values supplied by Host authority. They are not free-form model
output and do not use `Any` payloads.

## 7. Component Boundaries

### 7.1 `projection_contracts.py`

Owns the immutable contracts, enums, validation, canonical ordering rules, and
stable public error codes. It does not render Markdown or access repositories.

### 7.2 `observation_authority.py`

Defines the narrow read-only `ObservationAuthorityReader` protocol and the
strict `ObservationAuthoritySnapshot` returned by it. The reader must prove:

- exact game and player identity;
- exact base revision;
- current viewer visibility fingerprint;
- pinned persona profile identity and version;
- current self-role facts only;
- bounded public game checkpoint;
- committed public records visible to the viewer;
- legal-action and target snapshots;
- critical private-fact references; and
- source IDs and hashes for every projected value.

The protocol is the only input seam for future canonical-state adapters. The
initial projection package does not import or adapt the old `PlayerAgent` or
live-game path.

### 7.3 `document_projectors.py`

Owns dedicated typed projectors for:

- `PLAYER.md`: pinned persona identity, expression preferences, risk appetite,
  and verified long-term tendencies; no current hidden roles;
- `ROLE.md`: the viewer's role facts, faction, abilities, ability state, and
  mechanical restrictions; no strategy advice;
- `GAME.md`: bounded phase, living-player list, public checkpoint, current
  revision, and viewer-authorized critical facts;
- `COMMITMENTS.md`: committed public claims, positions, promises, vote intents,
  final votes, and retractions from committed records; and
- `INDEX.md`: generated only from the ordered manifest.

`BELIEFS.md`, `MEMORY.md`, and `WORKING.md` have no projector in this stage.
The workspace projector emits explicit unavailable entries for them.

### 7.4 `workspace_projector.py`

Coordinates section projectors, validates their identities and references,
computes hashes and estimates, emits unavailable entries, generates
`INDEX.md`, and returns a frozen `PlayerWorkspaceSnapshot`.

An optional process-local cache sits behind a narrow cache protocol. Cache
failure is equivalent to a miss. Cache content is never authoritative and can
be deleted without affecting recovery.

### 7.5 `observation_service.py`

Owns active-turn capture, authority reads, workspace assembly, final active-
turn recheck, and `ObservationBundle` creation. It depends on the reviewed
active-turn repository capability and the new authority reader. It does not
change HostRuntime or expose provider callbacks.

## 8. Deterministic Projection Format

All Markdown follows these rules:

- UTF-8 text with LF newlines and exactly one trailing newline;
- fixed document title, section order, field order, list markers, and empty
  representation;
- canonical scalar formatting and no locale-dependent values;
- source records sorted by stable identity before rendering;
- no timestamps other than authoritative pinned values;
- renderer and estimator versions included in the manifest;
- SHA-256 hashes over canonical UTF-8 bytes; and
- no rendering of fields removed by the viewer-specific authority snapshot.

Untrusted public speech or labels are rendered inside fixed
`UNTRUSTED_PUBLIC_DATA` envelopes. Control characters are removed or escaped,
Markdown headings and code fences cannot escape the envelope, and free text
cannot create Host instruction sections. Structured record IDs, actor IDs,
revision, modality, negation, and retraction state remain outside free text.

The player-document renderer is independent of the future
`ToolResultMarkdownProjection`. They may later share low-level escaping helpers
only after their authority and truncation contracts remain separate.

## 9. Token Estimation

`INDEX.md` requires a token estimate even though context compaction is deferred.
This stage introduces a versioned conservative `TokenEstimator` protocol and
one deterministic fallback estimator.

The fallback estimator uses canonical document bytes, applies a documented
Unicode-aware conservative calculation, and adds the ten-percent safety margin
required by the parent design. It records its version in every document and
manifest entry.

This stage does not calculate context occupancy, enforce 80-percent or
55-percent thresholds, compact history, or create checkpoints. The estimator
is only projection metadata and a stable seam for the next milestone.

## 10. Active-Turn Capture and Build Flow

The Host-facing entry point is conceptually:

```text
build_serial_public_observation(schedule_id, observed_at)
  -> ObservationBundle
```

The exact implementation flow is:

1. load the schedule and require an active turn;
2. load that managed turn and validate game, actor, schedule, status, task,
   window, revision, lease, view, and deadline identity;
3. reload the schedule and require the same active turn and state version;
4. capture schedule and turn state versions plus the exact window, base
   revision, view fingerprint, and deadline;
5. require an aware `observed_at` strictly before the turn deadline;
6. ask `ObservationAuthorityReader` for the exact game, player, revision, and
   view snapshot;
7. reject any identity, revision, visibility, source-reference, or legal-
   snapshot mismatch before rendering;
8. build documents, ordered manifest, `INDEX.md`, workspace revision, and
   workspace hash;
9. assemble the immutable `ObservationFrame` and bundle;
10. reload schedule and managed turn and require every captured identity and
    state version to remain unchanged;
11. read a fresh aware completion time from the service clock, reject clock
    regression, and require completion strictly before the captured deadline;
    and
12. return the bundle only after the final recheck succeeds.

The first and final checks provide an optimistic read fence without adding a
new persistence transaction. Schedule and turn state versions are monotonic,
so cancellation, expiry, transition, or replacement cannot create an accepted
ABA snapshot.

The bundle describes captured context only. A later model dispatch must still
win the durable active-turn fence immediately before external execution.

## 11. Requiredness and Failure Policy

Required inputs and sections fail closed:

- active schedule and managed-turn identity;
- player persona projection;
- self-role projection;
- game checkpoint projection;
- legal action and target snapshots;
- critical private-fact reference integrity;
- base revision and view fingerprint; and
- generated `INDEX.md` integrity.

Optional sections fail independently only when their source capability is
absent. They become unavailable manifest entries with a generic reason.
Visibility denial, identity mismatch, changed sources, or invalid hashes are
security failures, not optional absence.

Stable public error codes are:

```text
active_observation_conflict
required_projection_unavailable
projection_identity_mismatch
projection_visibility_rejected
projection_source_changed
projection_integrity_failed
projection_render_failed
projection_build_failed
```

Expected errors contain only stable identifiers and codes. Unexpected reader,
cache, renderer, estimator, or repository failures are mapped to the relevant
safe projection error, using `projection_build_failed` when no narrower public
classification applies, with `from None`. Error strings and formatted tracebacks
must not expose role payloads, private facts, SQL, filesystem paths, cache
content, or untrusted public text.

No partial `ObservationFrame` or bundle is returned after a required failure.

## 12. Cache Semantics

The minimum cache key is:

```text
game_id
+ player_id
+ view_fingerprint
+ workspace_revision
+ section_id
+ renderer_version
+ estimator_version
```

A cache hit is accepted only when all source references and hashes still match
the authority snapshot. Missing, changed, expired, newly invisible, or cross-
player content is rejected. Cache returns are defensive copies of frozen
contracts.

A cache lookup, decode, integrity, or write failure becomes a miss and triggers
a rebuild. A cache must not convert a security or authority validation failure
into a miss. Cache behavior cannot change generated bytes or error priority.

## 13. Testing Strategy

### 13.1 Contract tests

Cover strict fields, frozen values, canonical ordering, unavailable/available
invariants, requiredness, hashes, estimator versions, manifest/document
agreement, workspace identity, and defensive copies.

### 13.2 Deterministic projector tests

Prove:

- identical typed snapshots yield byte-identical documents and hashes;
- input record reordering does not change output;
- changed source bytes or hashes change or reject the projection as specified;
- every document has one trailing LF and fixed section order;
- untrusted control characters, headings, links, and code fences cannot escape
  their data envelope;
- optional unavailable entries do not reveal hidden-record existence; and
- cached and uncached builds are identical.

### 13.3 Active-turn integration tests

Use real Memory and SQLite active-turn repositories with the same scenarios:

- a valid active daytime speech turn builds the expected bundle;
- schedule, turn, game, actor, window, revision, view, lease, and deadline are
  pinned consistently;
- cancellation, expiry, non-terminal transition, or replacement between the
  first capture and final recheck returns `active_observation_conflict`;
- required source absence returns no partial bundle;
- cross-player authority snapshots, stale revisions, and changed visibility
  fingerprints are rejected;
- restart and rebuild produce the same workspace revision and hash; and
- caller mutation cannot alter stored cache values or later results.

No new PostgreSQL write behavior is required. Focused protocol tests prove the
service depends only on the existing active-turn repository reads. Real
PostgreSQL remains outside this milestone.

### 13.4 Import and architecture boundary tests

Production projection modules must not import legacy `PlayerAgent`, legacy
tool schemas, live-game runners, provider adapters, ToolResult Markdown,
AgentLoop, or proposal execution code. Tests also prove that projection never
creates physical player Markdown files.

### 13.5 Verification commands

All Python commands run through the repository environment:

```bash
conda run -n wofkill python -m pytest tests/player_agents -o addopts='' -q
conda run -n wofkill python -m pytest \
  tests/player_agents \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_autonomous_turns.py \
  -o addopts='' -q
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents tests/player_agents
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/player_agents
git diff --check
conda run -n wofkill python -m pytest -q
```

## 14. Acceptance Criteria

The milestone is accepted only when:

1. one valid active `serial_public` daytime speech turn can produce
   `PLAYER.md`, `ROLE.md`, `GAME.md`, `COMMITMENTS.md` when supported,
   unavailable optional entries, `INDEX.md`, and an immutable
   `ObservationFrame`;
2. the frame and every document bind the exact game, player, schedule, turn,
   window, base revision, and visibility fingerprint;
3. deterministic rebuilds produce identical bytes, workspace revision, and
   workspace hash;
4. cross-player, stale, changed-source, changed-visibility, expired, cancelled,
   or replaced inputs return stable safe errors and no partial bundle;
5. hidden-role or other-player-private leakage is zero in adversarial fixtures;
6. generated content is read-only presentation and cannot alter Host state;
7. Memory and SQLite integration scenarios pass with the same observable
   behavior;
8. no physical workspace files, projection persistence, provider call,
   ToolResult Markdown, AgentLoop, live-game path, or legacy PlayerAgent
   integration is introduced; and
9. focused and full repository tests plus scoped Ruff, mypy, and diff checks
   exit successfully.

Passing this milestone establishes a safe observation boundary. It does not
complete context compaction or a playable autonomous-player vertical slice.

## 15. Next Milestone

After this design is implemented and verified, the next unique milestone is
context-budget accounting, the 80-percent compaction trigger, strict
`CompactionCheckpoint`, optional untrusted `CompactionHandoff`, the 55-percent
rehydration target, lineage validation, and restart rehydration.

ToolGateway, ToolResult Markdown projection, provider dispatch, and AgentLoop
remain sequenced after that context lifecycle foundation.
