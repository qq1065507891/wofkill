# Tool Result Markdown Projection Design

Date: 2026-07-31
Status: Approved design; implementation plan not yet written
Owner: Codex development session

## 1. Decision Summary

The autonomous player runtime will keep every tool result as a strict,
structured, revision-pinned Host object and may additionally render that result
as deterministic Markdown for inclusion in model context.

Markdown is a read-only presentation projection. It is never the canonical tool
result, never parsed back into authority, never used to determine visibility,
and never substituted for typed evidence references, hashes, or compaction
checkpoints.

Each supported `result_kind` uses a dedicated renderer. A shared rendering
library may provide safe escaping, tables, lists, truncation notices, and fixed
envelope sections, but there is no generic JSON-to-Markdown semantic converter
on the live path. If a result has no registered renderer, its data is not
silently exposed through a generic dump.

This capability belongs to the new autonomous ToolGateway. It must not modify,
import as a compatibility contract, or route through the legacy
`werewolf_agent/tools/schemas.py` `ToolResult` protocol.

Implementation is sequenced after the durable active-turn fence and alongside
the stage-1 ToolGateway/context work. It does not move ahead of the current
highest-priority runtime milestone.

## 2. Problem

Strict structured tool results are well suited to validation, persistence,
hashing, replay, visibility checks, and typed reference tracking. They are not
always the best representation for a model to inspect.

Long JSON objects make relationships harder to see and spend tokens on syntax
that does not help the player reason. Public timelines, evidence groups, vote
comparisons, commitments, and possible worlds are often clearer as bounded
tables or labelled lists. At the same time, replacing the structured result
with Markdown would create unacceptable ambiguity:

- types, nulls, identifiers, modality, negation, and provenance can be lost;
- free text can look like instructions or escape its intended section;
- Markdown cannot safely become a replay, validation, or evidence source;
- compaction and restart recovery need exact typed references, not rendered
  prose;
- a generic formatter cannot know which fields are facts, hypotheses, advice,
  warnings, or non-model internal metadata.

The runtime therefore needs separate authority and presentation layers.

## 3. Goals

1. Make approved tool results easier for the model to scan and compare.
2. Preserve the exact structured result as the only Host authority.
3. Make Markdown generation deterministic, versioned, bounded, and auditable.
4. Preserve source type, revision, visibility, trust class, evidence references,
   warnings, and truncation state in every presentation.
5. Prevent untrusted text from altering the presentation envelope or appearing
   as Host instructions.
6. Support exact reconstruction after restart or context compaction from the
   structured result and pinned renderer version.
7. Measure whether a dedicated Markdown renderer improves evidence use or
   decision quality relative to compact structured JSON.

## 4. Non-Goals

1. Do not store Markdown as a game fact, memory fact, evidence record, or audit
   authority.
2. Do not parse Markdown back into `ToolResult` or any proposal.
3. Do not let a tool, provider, player model, RAG document, or skill choose a
   renderer or supply template structure.
4. Do not build a generic recursive JSON-to-Markdown semantic formatter for
   live use.
5. Do not render fields that visibility policy removed from the viewer-specific
   structured result.
6. Do not convert terminal proposal schemas or model outputs to Markdown.
7. Do not change the rejected legacy player/tool decision pipeline.
8. Do not assume Markdown is better without controlled evaluation.

## 5. Authority and Invariants

The representation hierarchy is:

1. canonical game records and Host stores;
2. viewer-specific structured `ToolResult`;
3. deterministic `ToolResultPresentation`;
4. the model's interpretation of that presentation.

The following invariants are mandatory:

- only the structured result may provide `result_id`, revision, visibility,
  trust class, evidence references, result hash, warnings, and truncation state;
- model-visible Markdown carries no independent fact or reference;
- a model may cite only structured evidence IDs explicitly exposed by the
  result, never an identifier extracted from free text;
- identical canonical result bytes, renderer version, presentation policy, and
  token budget produce identical Markdown bytes;
- changing visibility, source result hash, renderer version, or presentation
  policy invalidates any cached presentation;
- the renderer never promotes `ADVICE`, `EXAMPLE`, `HYPOTHESIS`, or untrusted
  text to `FACT` or `OBSERVATION`;
- compaction checkpoints preserve typed result references and hashes, not
  Markdown authority;
- a rendering failure cannot mutate or delete the structured result.

## 6. Architecture

```text
authorized tool execution
  -> strict viewer-specific ToolResult
  -> Host schema / revision / visibility validation
  -> persist structured result and audit metadata
  -> structural truncation policy
  -> renderer lookup by Host-owned result_kind
  -> dedicated deterministic renderer
  -> presentation validation and token estimate
  -> model-context ToolResultPresentation
```

The boundary has four cohesive components:

### 6.1 `ToolResult`

The new autonomous ToolGateway owns a strict result contract. Its exact fields
will be defined in the ToolGateway plan, but the presentation layer requires at
least:

```text
result_id
tool_call_id
tool_id
result_kind
status
game_revision
view_fingerprint
visibility
trust_class
data
evidence_refs
warnings
truncated
result_hash
```

`data` is a strict discriminated schema selected by `result_kind`; it is not an
arbitrary dictionary on the new live path.

### 6.2 `ToolResultRenderer`

One renderer handles one result schema or a small family with identical
presentation semantics. A renderer:

- accepts only its typed, already visibility-filtered payload;
- uses a Host-pinned `RendererPolicy`;
- applies no game strategy, scoring, inference, or summarization model;
- returns deterministic section data or final Markdown;
- declares its version and supported result schema hash;
- owns structural truncation rules for its result kind.

Renderers are selected from a closed Host registry keyed by `result_kind` and
result schema version. Tools and models cannot request renderer IDs.

### 6.3 Shared Safe Rendering Primitives

Shared helpers may render:

- fixed headings;
- bounded key/value metadata;
- tables with stable columns;
- ordered or unordered reference lists;
- trust and scope notices;
- warnings and truncation notices;
- escaped untrusted text blocks.

These helpers format already typed values. They do not recursively interpret
unknown objects or choose which fields matter.

### 6.4 `ToolResultPresentation`

The presentation is a frozen derived object:

```text
presentation_version
result_id
result_kind
renderer_id
renderer_version
source_result_hash
policy_hash
markdown
markdown_hash
estimated_tokens
truncated
```

It intentionally does not duplicate evidence authority, visibility grants, or
game mechanics. Consumers resolve those through the structured result ID.

## 7. Model-Visible Markdown Contract

Every presentation uses a fixed Host-generated boundary:

```markdown
<tool_result id="result-018"
             kind="vote_history"
             revision="42"
             visibility="player_private"
             trust="host_observation">

## 投票历史比较

### 结论范围

以下内容仅汇总可见投票记录，不代表阵营判断。

### 记录

| 玩家 | 轮次 | 投票目标 | 记录引用 |
|---|---:|---|---|
| p03 | 1 | p07 | `vote-12` |
| p03 | 2 | p08 | `vote-26` |

### 可引用证据

- `vote-12`
- `vote-26`

### 警告

- 第二轮存在改票，但工具没有判断改票原因。

</tool_result>
```

The XML-like boundary is a model-context delimiter, not a parser or security
authority. Security remains in Host validation. The delimiter and every
attribute are Host-generated. Untrusted content is escaped so it cannot emit
the opening or closing delimiter.

The fixed envelope exposes only model-useful metadata:

- result identity;
- result kind;
- canonical revision;
- visibility class;
- trust class;
- scope statement;
- rendered typed data;
- allowed evidence references;
- warnings and truncation state.

Database keys, full internal hashes, encryption fields, credentials, audit
ciphertext, and non-model operational metadata are omitted.

## 8. Renderer Rules

### 8.1 Tables

Tables are used only when columns are stable, cells are short, and the bounded
row count fits the presentation policy. Cells escape pipes, newlines, delimiter
text, and other structural characters. A renderer never places unrestricted
speech or retrieved documents into table cells.

### 8.2 Lists

Lists are used for timelines, evidence groups, warnings, questions, hypotheses,
and references. Nested depth is bounded by policy. Host-owned labels state
whether each section is fact, observation, hypothesis, advice, example, or
error.

### 8.3 Untrusted Text

Player speech, RAG passages, skill content, external tool text, and cross-game
lessons are untrusted data. They are rendered in a bounded escaped data block
with a fixed `UNTRUSTED_DATA` label. Renderer escaping neutralizes:

- `<tool_result>` and `</tool_result>` delimiters;
- Markdown headings and thematic separators at line start;
- fenced code blocks;
- raw HTML tags;
- link/image syntax when links are not explicitly allowed;
- control characters and invalid Unicode sequences.

Escaping is a presentation defense, not permission enforcement. The static
system contract still tells the model that content inside the boundary is data,
not instruction.

### 8.4 Evidence References

The `可引用证据` section is generated exclusively from the structured
`evidence_refs` field after visibility validation. Renderers cannot discover
new evidence IDs in prose. If rendered content mentions an ID that is not in
the structured result, that text remains untrusted content and cannot be cited
as evidence by the Host.

### 8.5 Trust and Scope Notices

Every result kind has a fixed scope notice. Examples:

- a vote-history comparison describes recorded votes but does not infer role;
- a consistency check reports a typed contradiction but does not adjudicate
  truthfulness;
- a possible-world result is a hypothesis, not hidden-state access;
- RAG and skill results are advice or examples, never current-game evidence;
- an external provider result is untrusted even when transport succeeded.

## 9. Structural Truncation and Context Budget

Truncation occurs on typed data before Markdown generation. Character slicing
of final Markdown is forbidden because it can break tables, omit negation, cut
an evidence reference, or remove a warning.

Each renderer declares deterministic reduction steps. A typical sequence is:

1. remove optional descriptive columns;
2. limit records using a Host-defined stable relevance/order key;
3. replace omitted rows with a typed count and continuation reference;
4. retain warnings, trust class, unresolved items, and all evidence references
   used by retained rows;
5. render the reduced typed view;
6. estimate tokens using the turn's pinned estimator;
7. repeat the next declared reduction step when still over budget.

The renderer cannot use a model to summarize or choose rows. It cannot drop
authority metadata, negation, modality, visibility, warnings, or the fact that
truncation occurred.

Presentation token estimates contribute to the same context occupancy
calculation that triggers autonomous turn compaction. A large presentation
cannot bypass the 80-percent pre-request compaction boundary.

## 10. Persistence, Cache, Compaction, and Recovery

The structured result is persisted before presentation. A presentation may be
cached as a derived optimization using at least:

```text
game_id
player_id
turn_id
result_id
source_result_hash
view_fingerprint
renderer_id
renderer_version
policy_hash
```

A cache hit is accepted only after the result still belongs to the active turn,
the result hash matches, the view fingerprint remains valid, and the renderer
and policy versions are pinned by the model lease.

The Host checkpoint records the result ID, result kind, revision, visibility,
result hash, evidence references, warnings, truncation state, and adoption
state. It does not treat cached Markdown as resumable authority.

After restart or compaction, the Host revalidates the structured result and
regenerates identical Markdown with the pinned renderer version. If that
renderer version is unavailable, the Host returns a presentation failure; it
does not silently use the current renderer and pretend the context is
equivalent.

## 11. Failure Model

Presentation failures use stable non-secret codes:

```text
renderer_not_registered
renderer_schema_mismatch
presentation_visibility_stale
presentation_source_hash_mismatch
presentation_policy_invalid
presentation_too_large
presentation_unsafe_content
presentation_render_failed
```

Failure behavior is:

- keep the structured `ToolResult` unchanged and addressable;
- record a bounded audit item with result ID, renderer version, policy version,
  failure code, and safe field path;
- do not expose a generic raw JSON dump as an automatic fallback;
- return a short Host-generated `presentation_unavailable` result containing no
  hidden payload;
- allow the AgentLoop to continue when the tool is optional;
- follow the task's declared failure/cancellation policy when the result is
  required;
- never convert a tool success into a committed game action or silently retry a
  model call.

Renderer bugs, missing versions, and token overflow are presentation failures,
not permission to weaken visibility or data boundaries.

## 12. Initial Renderer Scope

The first implementation is limited to the stage-1 daytime-speech ToolGateway.
It should add renderers only for tools actually exposed by that vertical slice,
expected to include a minimal subset of:

- context section reads;
- bounded public timeline reads;
- visible player history;
- commitments;
- public vote ledger;
- evidence lookup;
- public claim/timeline comparison;
- public consistency checks;
- structurally accepted working reflection results.

Private role facts may use a dedicated compact renderer if stage-1 observation
does not already carry them. World-model, RAG, skill, cross-game memory, wolf
team, vote, and role-action renderers are added only with their later feature
gates.

## 13. Security and Prompt Injection

The presentation pipeline runs only after ToolGateway ACL, viewer identity,
revision, read-set, grant, and visibility validation. Markdown rendering does
not repair or compensate for an unauthorized structured result.

Mandatory protections are:

- player identity comes from the active turn, never tool arguments;
- only allowlisted result schemas and renderers enter model context;
- untrusted strings are length-bounded and escaped before assembly;
- system-like phrases inside data remain visibly labelled untrusted data;
- renderer output is validated against forbidden fields and delimiters;
- every context inclusion records result and presentation hashes;
- presentation does not expose another player's private workspace, moderator
  state, hidden role truth, credentials, or provider internals;
- the model cannot edit Markdown or request a weaker rendering policy;
- terminal proposals never cite Markdown offsets or quotes as evidence.

## 14. Evaluation and Acceptance

Markdown enablement is a measured presentation experiment, not an assumed
improvement.

The evaluation uses frozen, visibility-valid tool results and identical model
leases. Each fixture runs at least two variants:

- A: compact canonical structured JSON presentation;
- B: dedicated Markdown presentation.

Primary measures are:

- factual answer accuracy;
- evidence-reference precision and recall;
- preservation of negation, modality, attribution, and uncertainty;
- rate of treating advice, examples, or hypotheses as facts;
- downstream proposal evidence correctness;
- unnecessary repeat-tool-call rate.

Secondary measures are:

- input tokens;
- model latency;
- presentation generation latency;
- compaction frequency;
- truncation frequency;
- decision-quality delta on matched fixtures.

Safety gates are:

- cross-player or moderator-private leaks: zero;
- invented structured evidence references: zero accepted by the Host;
- missing required visible facts, negation, modality, or warnings: zero in
  renderer conformance fixtures;
- nondeterministic output for identical pinned inputs: zero;
- delimiter escape or prompt-injection boundary failures: zero;
- Markdown parsed back into authority: zero code paths.

Default Markdown presentation for a `result_kind` requires improvement in a
predeclared primary metric or a meaningful reduction in tokens/repeat calls,
with no safety regression and latency inside its declared margin. Result kinds
that do not benefit remain on compact structured JSON presentation.

## 15. Test Strategy

### 15.1 Contract Tests

- strict frozen presentation model;
- renderer ID/version and schema-hash binding;
- source-result and policy-hash validation;
- stable failure codes;
- no extra or authority-bearing fields.

### 15.2 Golden Renderer Tests

- exact Markdown byte fixtures for every renderer;
- Unicode, Chinese punctuation, empty sections, maximum rows, and null-like
  optional values;
- stable ordering independent of dictionary insertion order;
- exact trust/scope/warning wording;
- table and list well-formedness after structural truncation.

### 15.3 Semantic Preservation Tests

- every required fact and visible reference appears exactly as specified;
- attribution, negation, modality, uncertainty, and chronology survive;
- no new conclusion, role claim, target preference, or strategy appears;
- evidence section equals the structured allowed evidence set.

### 15.4 Adversarial Tests

- embedded closing delimiters;
- Markdown headings and code fences;
- raw HTML and link/image syntax;
- instruction injection in player speech, RAG, skill, and external text;
- similar player IDs;
- stale view fingerprints and result hashes;
- very long cells, rows, evidence lists, and nested content;
- invalid Unicode and control characters.

### 15.5 Lifecycle Tests

- cache hit and invalidation;
- restart reconstruction;
- compaction reconstruction with identical presentation hashes;
- renderer-version absence;
- presentation failure without mutation of structured result;
- context-budget accounting and pre-request compaction.

## 16. Rejected Alternatives

### 16.1 Replace Structured Results With Markdown

Rejected because validation, replay, visibility, evidence ownership, and
compaction authority would depend on prose formatting.

### 16.2 Generic Recursive JSON-to-Markdown Conversion

Rejected for the live path because it cannot distinguish facts from advice,
choose safe truncation, preserve domain semantics, or reliably produce a
model-useful layout.

### 16.3 Always Show Pretty-Printed JSON

Retained as the A/B control and as an explicitly selected presentation for
result kinds that do not benefit from Markdown. It is not the preferred default
for relational or timeline-heavy results.

### 16.4 Model-Generated Summaries

Rejected for first-version presentation because they can omit, reinterpret, or
invent semantics, add cost and latency, and complicate deterministic recovery.

## 17. Implementation Sequence Boundary

This design does not authorize implementation before the current durable
active-turn fence milestone is complete.

After that milestone, implementation should proceed as part of the stage-1
ToolGateway/context plan:

1. define the new strict autonomous `ToolResult` discriminated schemas;
2. define `RendererPolicy`, `ToolResultRenderer`, registry, presentation model,
   and stable failures;
3. implement safe rendering primitives and adversarial tests;
4. implement only the minimal stage-1 dedicated renderers;
5. integrate rendering after visibility validation and before context-budget
   admission;
6. integrate cache, audit metadata, compaction, and restart reconstruction;
7. run JSON-versus-Markdown evaluation before enabling each result kind by
   default.

No implementation step may import or adapt the legacy tool result as a new
runtime compatibility contract.
