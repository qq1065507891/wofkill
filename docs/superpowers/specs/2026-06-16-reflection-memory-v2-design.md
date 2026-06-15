# Reflection Memory V2 Design

Date: 2026-06-16
Status: Implemented in code; database migration pending operator run
Owner: Codex development session

## Problem

The reflection-memory subsystem is intended to make agents improve across
games by injecting prior post-game lessons into later player prompts. The
current implementation has the right high-level idea, but the practical value
is weak:

1. Reflections can exist in PostgreSQL while not being injected into live
   prompts because reflection injection is gated by profile availability.
2. New games restore memory by the new `game_id`, so they often miss the latest
   saved cross-game snapshot.
3. Persisted reflection text is mostly generic review summary text, not
   actionable learning material.
4. Error-pattern aggregation depends on section headers that current persisted
   reflections do not contain.
5. LLM self-review and deterministic review are stored through different paths
   without a single quality-controlled merge step.

The goal of this design is to upgrade reflection memory once as a complete
learning subsystem: generate, merge, score, store, retrieve, inject, and
evaluate reflections through one coherent path.

This does not train model weights. It makes the agent behave more capable by
placing high-quality, relevant, prompt-safe historical lessons into the next
game's context.

## Current Evidence

The current database contains 144 reflection rows across 12 games. The rows are
mostly short automatic review summaries:

- average text length is about 71 characters;
- section headers such as `【投票错误】`, `【信息缺失】`, `【悍跳分析】`, and
  `【保留的优点】` are absent;
- player-ID scrubbing is working on the sampled data;
- low-information text such as `复盘失败对局，关注关键转折点的信息缺失` appears
  repeatedly;
- `MemoryStore(repo=repo)` loads reflections but not profiles, so
  `profile_count == 0` while `reflection_count == 144`.

The practical consequence is that the system stores review artifacts, but does
not reliably deliver concise, role-relevant, actionable experience to the LLM.

## Goals

1. Inject reflection memory whenever useful reflections exist, without requiring
   a profile row.
2. Restore cross-game memory from the latest available snapshot for new games,
   while still supporting explicit snapshot IDs.
3. Introduce a structured V2 reflection schema that can express mistakes,
   successful patterns, trigger signals, and concrete next-game advice.
4. Add a quality gate so generic, duplicated, unsafe, or low-actionability
   reflections are not injected into player prompts.
5. Merge LLM self-review and deterministic review into one calibrated reflection
   entry.
6. Keep historical memory prompt-safe: no historical player-ID mapping, no
   hidden truth promoted as current public fact, no full game transcript dump.
7. Provide migration and cleanup tooling for existing reflections and embedded
   reflection copies inside memory snapshots.
8. Add tests and metrics proving reflection memory is present, relevant, and
   bounded.
9. Remove the old multi-source reflection write paths so production creates one
   calibrated V2 reflection per player per game.

## Non-Goals

1. Do not train or fine-tune model weights.
2. Do not change game rules, victory logic, or role abilities.
3. Do not move current-game public facts into reflection memory.
4. Do not introduce a mandatory external embedding service.
5. Do not make reflection memory a hard instruction. It remains reference
   context below current-game facts, role rules, strategy directives, and the
   output contract.
6. Do not expose historical hidden roles or historical concrete player IDs to
   live prompts.

## Design Principles

Reflection memory has value only when it changes future decisions. A useful
reflection must answer four questions:

1. What situation did this lesson come from?
2. What error or strength pattern was observed?
3. What signals should trigger the lesson next time?
4. What should the player do differently or preserve?

The live prompt should never receive raw review dumps. It should receive compact
reflection cards that are:

- role-relevant;
- high quality;
- explicit about applicability;
- safe to treat only as historical learning;
- short enough to fit under prompt pressure.

## Architecture

The upgraded subsystem has seven responsibilities.

1. `ReflectionGenerator`
   - Calls the player LLM after the game for role-family-specific self-review.
   - Produces raw subjective review text.

2. `ReviewGenerator`
   - Computes deterministic review facts from cognition matrix, relation graph,
     votes, role, outcome, and ground truth available only after game end.
   - Produces calibrated factual review data.

3. `ReflectionSynthesizer`
   - Combines LLM self-review and deterministic review into a single V2
     reflection entry.
   - Resolves conflicts in favor of deterministic review facts.

4. `ReflectionQualityGate`
   - Scores each candidate reflection.
   - Rejects or marks low-quality entries.
   - Scrubs and validates prompt-visible fields.

5. `ReflectionMemory`
   - Persists reflection entries.
   - Loads legacy and V2 entries.
   - Keeps backward compatibility with the existing `reflections` table.

6. `ReflectionRetriever`
   - Selects high-quality reflections for the current player, role, faction,
     phase, and situation.
   - Provides deterministic ranking first, optional semantic scoring second.

7. `PlayerPromptBuilder`
   - Renders only prompt-safe reflection cards and aggregate error patterns.
   - Keeps reflection memory inside the existing `跨局学习参考` section.

## Data Model

The existing PostgreSQL `reflections` table can remain unchanged:

```text
entry_id TEXT PRIMARY KEY
game_id TEXT
player_id TEXT
entry_json JSONB
```

The V2 schema lives inside `entry_json`. This avoids a risky schema migration
while allowing richer data.

### ReflectionEntryV2

```json
{
  "schema_version": 2,
  "entry_id": "reflection_g_123_p01",
  "game_id": "g_123",
  "player_id": "p01",
  "role": "seer",
  "faction": "good",
  "faction_won": false,
  "quality_score": 0.82,
  "quality_status": "approved",
  "quality_flags": [],
  "situation_signature": {
    "role": "seer",
    "faction": "good",
    "outcome": "loss",
    "phase_focus": ["sheriff_speech", "vote"],
    "game_patterns": ["seer_counterclaim", "badge_flow"]
  },
  "mistake_patterns": [
    {
      "category": "vote_mistake",
      "trigger": "对跳局中只看发言气势，没有核验警徽流兑现",
      "wrong_action": "过早站边悍跳位",
      "better_action": "先比较验人时间线、警徽流和票型承接",
      "fact_basis": "auto_review",
      "auto_verified": true,
      "corrected_from_llm": false
    }
  ],
  "preserved_strengths": [
    {
      "category": "speech_quality",
      "behavior": "能明确说明验人心路",
      "reuse_condition": "警上起跳或被质疑时"
    }
  ],
  "actionable_advice": [
    "对跳局投票前先列双方验人时间线、警徽流和票型承接。",
    "证据不足时不要把情绪化强势发言当成身份强信号。"
  ],
  "avoid_next_time": [
    "不要因为发言强势就默认预言家可信。"
  ],
  "prompt_card": {
    "theme": "对跳局先核验警徽流",
    "lesson": "你过去在对跳局中过早相信气势强的一方。下次先核验验人时间线、警徽流和票型承接。",
    "trigger_signals": ["双预言家对跳", "警徽流解释不完整"],
    "recommended_action": "发言或投票前先列对比表，再给站边结论。",
    "misuse_risk": "不要把历史对跳经验直接映射到本局玩家身份。",
    "fact_basis": "auto_review",
    "auto_verified": true
  },
  "source": {
    "llm_self_review": "...",
    "auto_review_summary": "...",
    "merged_by": "reflection_synthesizer_v2"
  }
}
```

### Compatibility

Existing `ReflectionEntry` remains loadable. Legacy entries are normalized into
an internal view:

- missing `schema_version` means legacy V1;
- legacy `text` remains available for audit;
- legacy entries receive a conservative quality score during migration or load;
- low-quality legacy entries are not injected unless explicitly configured.

## Generation Flow

At game end, each player produces two inputs.

### LLM Self-Review

The current role-family templates stay conceptually useful, but their output is
no longer persisted directly as the final long-term memory. The self-review
prompt should ask for structured sections:

- mistakes;
- missed signals;
- role execution;
- preserved strengths;
- next-game advice;
- no concrete player IDs.

The returned text is treated as subjective evidence, not final truth.

### Deterministic Review

`ReviewGenerator` produces factual calibration:

- role and faction;
- win/loss;
- judged role guesses versus ground truth;
- vote/deception analysis from relation graph;
- ability deltas;
- deterministic summary.

If deterministic review conflicts with LLM self-review, deterministic review
wins on factual claims.

### Synthesis

`ReflectionSynthesizer` receives:

- raw LLM self-review text;
- `ReviewReport`;
- role and faction metadata;
- optional compact public event summary;
- optional cognition and relation graph signals.

It outputs `ReflectionEntryV2`.

Rules:

1. Convert subjective review into structured mistake and strength patterns.
2. Use deterministic review to validate or correct factual claims.
3. Keep advice transferable, not tied to historical player IDs.
4. Generate `prompt_card` as the only player-prompt-facing summary.
5. Preserve raw sources under `source` for audit, with strict length caps.
6. Mark every pattern and prompt-card claim with provenance:
   - `fact_basis="auto_review"` for deterministic facts;
   - `fact_basis="llm_transferable"` only for fact-free tactical lessons;
   - `auto_verified=true` when the claim depends on deterministic review;
   - `corrected_from_llm=true` when the self-review contradicted the automatic
     review and was corrected.
7. A prompt card may use only auto-confirmed facts or fact-free transferable
   lessons. LLM-only claims about votes, roles, deaths, checks, or outcomes must
   be corrected or removed.

## Single Write Path

Production reflection memory must have exactly one write path after this
upgrade:

```text
LLM self-review + deterministic ReviewReport
  -> ReflectionSynthesizer
  -> ReflectionQualityGate
  -> ReflectionMemory.store_v2()
```

Existing direct stores are demoted:

- raw `_agent_reflection()` output is an input to synthesis only;
- `ReviewGenerator` summary is an input to synthesis only;
- direct `ReflectionMemory.store(text=...)` remains for tests, migration, and
  audit tooling, but production game-end code must not call it for live-learning
  entries;
- legacy direct writes must set `quality_status="review_only"` or
  `schema_version=1`, and retrieval must treat them as ineligible for live
  prompt cards.

After a completed game, each player should produce at most one calibrated V2
reflection for that game. If both old summary-node reflection and
`GameRunner._save_memory_snapshot()` are active, the implementation must merge
or route them through the same synthesizer instead of writing duplicate rows.

Tests must assert that a newly run game writes no new V1 live-learning
reflections and that every production reflection row has `schema_version == 2`
and `quality_status` populated.

## Quality Gate

`ReflectionQualityGate` decides whether a reflection can enter long-term prompt
memory.

### Scoring Inputs

Positive signals:

- has at least one mistake pattern or preserved strength;
- has concrete `trigger`, `better_action`, or `recommended_action`;
- has role and phase or situation tags;
- has an actionable prompt card;
- includes both mistake prevention and strength preservation when applicable;
- passes player-ID and hidden-truth safety checks.

Negative signals:

- text is too short;
- repeated generic phrases such as `关注关键转折点的信息缺失`;
- no actionable advice;
- no role-specific content;
- duplicate of an existing reflection;
- contains concrete player IDs;
- contains hidden role truth in prompt-visible fields;
- contains malformed JSON or missing V2 fields.

### Statuses

```text
approved       quality_score >= 0.70, eligible for prompt injection
review_only    0.40 <= quality_score < 0.70, stored for audit/offline stats only
rejected       quality_score < 0.40, not injected
```

Rejected entries may be retained for audit during migration, but must not be
rendered in live prompts.

`review_only` entries must not contribute to live prompt cards or live prompt
error-pattern aggregation. Offline reports may count them separately.

### Deterministic Scoring

The first implementation must use a reproducible score, not subjective judgment.

Hard rejects set `quality_status="rejected"` regardless of numeric score:

| Condition | Flag |
| --- | --- |
| prompt-visible player ID remains after scrubbing | `player_id_leak` |
| missing `prompt_card.theme` or `prompt_card.recommended_action` | `missing_prompt_card` |
| hidden role truth appears in prompt-visible fields without auto verification | `unsafe_truth_claim` |
| malformed V2 schema | `invalid_schema` |
| empty role or game_id | `missing_identity` |

Base score starts at `0.0` and is capped to `[0.0, 1.0]`.

Positive points:

| Signal | Points |
| --- | ---: |
| at least one mistake pattern with `trigger` and `better_action` | +0.25 |
| at least one preserved strength with `reuse_condition` | +0.15 |
| prompt card has theme, lesson, trigger signals, recommended action, misuse risk | +0.25 |
| role and phase/game pattern are present in `situation_signature` | +0.10 |
| at least one prompt-card claim is auto-verified or fact-free transferable | +0.10 |
| actionable advice has at least one imperative next-step sentence | +0.10 |
| text/card is role-specific rather than generic | +0.05 |

Negative points:

| Signal | Points |
| --- | ---: |
| generic phrases dominate the text | -0.25 |
| prompt card is shorter than 80 Chinese chars or equivalent content | -0.15 |
| no phase, situation, or trigger signal | -0.15 |
| near-duplicate of an approved reflection for the same player/role/category | -0.20 |
| source text exceeds caps and had to be heavily truncated | -0.05 |

Duplicate detection should compare this key:

```text
player_id + role + primary mistake category + normalized prompt_card.theme
```

If the key matches an approved entry and token Jaccard similarity of
`prompt_card.lesson + recommended_action` is above `0.70`, apply the duplicate
penalty. If the resulting score falls below `0.70`, keep the newer entry as
`review_only` unless it has strictly higher score than the existing approved
entry.

Legacy conversion minimum:

- legacy text must be at least 80 characters;
- it must contain either a concrete mistake, a concrete preserved strength, or a
  concrete next-game action;
- it must be role-specific;
- otherwise it becomes `review_only` or `rejected`, never `approved`.

Migration dry-run must output for each entry:

```json
{
  "entry_id": "...",
  "old_schema_version": 1,
  "score": 0.35,
  "decision": "review_only",
  "flags": ["generic_text"],
  "reason": "No concrete trigger or next-game action"
}
```

### Safety Checks

The quality gate applies the same player-ID scrubber used today and extends it
to every field that can influence the prompt or prompt aggregation:

- `prompt_card`;
- `mistake_patterns`;
- `preserved_strengths`;
- `actionable_advice`;
- `avoid_next_time`;
- legacy `text` if converted.

The gate must reject or scrub `p01`, `player_1`, `agent_1`, and similar IDs.

## Retrieval Design

Retrieval should rank by usefulness, not just recency.

### Hard Filters

1. `quality_status == approved`.
2. Same `player_id` for personal learning by default.
3. Visibility is prompt-safe.
4. Role and faction compatibility:
   - same role is strongest;
   - same explicit faction is allowed;
   - hybrid history is same-role-only unless master faction is known.

### Ranking

Score components:

```text
same_player         +4
same_role           +4
same_faction        +2
current_phase_match +2
situation_match     +2
quality_score       +0..2
recent_game         +0..1
duplicate_penalty   -3
```

The top result set should be capped:

- max 3 reflection cards in the live prompt;
- max 2 cards from the same role;
- max 1 card with the same mistake category unless the category is dominant.

### Optional Semantic Layer

The existing `BagOfWordsVectorIndex` can remain optional. The V2 prompt card and
structured fields give enough signal for deterministic retrieval. A future
embedding backend can score:

```text
prompt_card.theme
prompt_card.lesson
trigger_signals
recommended_action
mistake_patterns.category
situation_signature
```

No external embedding service is required for the V2 upgrade.

## Prompt Injection

Reflection memory stays inside `跨局学习参考`.

The live prompt must render only a strict whitelist. It must not dump raw V2
JSON and must fail closed on unknown fields.

Allowed live fields:

- `prompt_card.theme`
- `prompt_card.lesson`
- `prompt_card.trigger_signals`
- `prompt_card.recommended_action`
- `prompt_card.misuse_risk`
- approved aggregate error labels derived only from approved entries

Forbidden live fields:

- `source.*`
- `quality_score`
- `quality_status`
- `quality_flags`
- deterministic ground-truth details;
- raw `mistake_patterns`;
- raw `preserved_strengths`;
- raw `actionable_advice`;
- raw `avoid_next_time`;
- historical concrete player IDs;
- hidden historical role truth unless abstracted into fact-free transferable
  guidance.

The rendered shape is card-based:

```text
跨局反思记忆:
以下是历史经验，不代表本局任何玩家真实身份。不得把历史玩家、历史身份真相或历史决策链映射到本局。

反思 1: 对跳局先核验警徽流
- 触发信号: 双预言家对跳；警徽流解释不完整
- 历史教训: 你过去在对跳局中过早相信气势强的一方。
- 本局做法: 发言或投票前先列验人时间线、警徽流和票型承接，再给站边结论。
- 误用风险: 不要把历史对跳经验直接映射到本局玩家身份。
```

Aggregate error pattern remains above detailed reflections:

```text
【跨局错误模式】
你最常犯的错误: 投票错误(3次)、漏读信息(2次)。
本局优先自检: 投票前核验公开证据链，不要用历史经验替代当前事实。
```

Prompt priority remains reference tier. Reflection can influence reasoning, but
cannot override:

1. game rules;
2. role guide;
3. current-game public facts;
4. private role facts for this turn;
5. strategy directives;
6. output contract.

Live error-pattern aggregation must use only `quality_status="approved"` entries
that also pass prompt safety. `review_only` and `rejected` entries may be used in
offline audit dashboards, but never in live prompt cards or live prompt
aggregate labels.

## Profile And Snapshot Restoration

Reflection injection must not require profile data.

### New Injection Rule

Current behavior:

```text
profile exists and games_played > 0 -> build profile, reflections, error pattern
```

New behavior:

```text
if reflections exist -> build reflection cards and error pattern
if profile exists -> build profile hint
if cognition matrix exists -> build cognition hint
```

### Snapshot Restore Rule

New games should restore the latest available memory snapshot when there is no
snapshot for the current `game_id`.

`PersistentMemoryCoordinator` should expose:

```python
restore_memory(snapshot_id: str, *, fallback_to_latest: bool = False)
restore_for_new_game(game_id: str)
restore_latest_memory()
save_memory(..., snapshot_id=current_game_id)
save_memory(..., snapshot_id="latest")
```

`restore_memory(snapshot_id, fallback_to_latest=False)` is the explicit replay
and debugging API. If the requested snapshot is missing, it must return an empty
store or `None` according to the existing call contract; it must not silently
load `latest`.

`restore_for_new_game(game_id)` is the automatic new-game API. It may attempt
`game_id` first and then fall back to `latest`. `GameRunner._restore_memory_if_configured()`
should use this method.

Saving a completed game should update both:

1. `snapshot_id=current_game_id`
2. `snapshot_id="latest"`

These writes should happen in one coordinator operation. If the repository
cannot make them transactional, failures must be logged with enough detail to
tell whether the game-specific snapshot or `latest` alias failed.

The `reflections` table remains the durable long-term reflection source. The
snapshot provides profile, cognition, and relation graph continuity.

### Snapshot Reflection Boundary

After V2, memory snapshots must not be a second source of full reflection
truth.

Rules:

1. `memory_snapshots.snapshot_json.reflections` should be empty, omitted, or
   contain only lightweight reflection entry IDs.
2. Restoring a snapshot must not write snapshot-embedded reflection bodies back
   into the `reflections` table.
3. `MemoryStore(repo=repo)` or the V2 reflection loader should load approved V2
   reflections from the `reflections` table.
4. Dirty legacy snapshots must not rehydrate rejected or review-only reflection
   text into live prompt memory.

This closes the old two-source loop where rows cleaned from `reflections` could
reappear from `memory_snapshots`.

## Migration And Cleanup

Migration is required because existing rows are low-density legacy summaries.

### Backup

Before any cleanup:

1. export all rows from `reflections`;
2. export all `memory_snapshots`;
3. write timestamped backup JSON files under `tmp_analysis/reflection_memory_v2`;
4. do not overwrite prior backups.

### Legacy Row Processing

For each existing reflection:

1. load legacy entry;
2. score with `ReflectionQualityGate`;
3. if sufficient information exists, convert to minimal V2;
4. if generic or too short, mark `review_only` or `rejected`;
5. preserve original text under `source.legacy_text`;
6. update or delete according to migration mode.

Recommended default:

- keep `approved` and `review_only`;
- delete or archive `rejected` from live tables after backup;
- remove or replace embedded reflection bodies inside memory snapshots to
  prevent rehydration.

### Embedded Snapshot Cleanup

Memory snapshots currently contain accumulated reflection copies. Migration must
also scan `memory_snapshots.snapshot_json.reflections` and apply the same
quality decision:

- approved V2 reflection bodies should be replaced by entry IDs or omitted;
- review-only and rejected items are removed from snapshot reflection lists;
- snapshot restore must not use embedded reflection bodies as live memory even
  if an old snapshot still contains them.

This prevents cleaned rows from reappearing when a snapshot is restored.

## Observability

Add lightweight counters:

- `reflection_candidates_loaded`;
- `reflection_cards_rendered`;
- `reflection_rejected_low_quality`;
- `reflection_rejected_safety`;
- `reflection_legacy_loaded`;
- `reflection_v2_loaded`;
- `reflection_snapshot_fallback_used`;
- `reflection_prompt_chars`.

The prompt audit should record which reflection entry IDs were injected and why
they ranked above other candidates. The audit is for debugging only and must not
be rendered to live player prompts.

## Testing

### Unit Tests

Memory schema and quality:

- V2 reflection accepts complete schema.
- V2 reflection rejects missing prompt card fields.
- player IDs are scrubbed or rejected in all prompt-visible fields.
- generic text receives low quality.
- specific actionable reflection receives high quality.
- duplicate reflection is penalized.
- deterministic scoring produces expected status, flags, score, and reason for
  representative approved, review-only, and rejected entries.

Synthesis:

- LLM self-review plus deterministic review produces one V2 entry.
- deterministic review overrides conflicting LLM claims.
- preserved strengths survive synthesis.
- source text is capped and audit-only.
- LLM-only claims about votes, roles, checks, deaths, or outcome are corrected
  or removed when automatic review disagrees.
- prompt-card claims carry `fact_basis` and `auto_verified` where needed.

Retrieval:

- same player and same role outrank same faction.
- low-quality reflections are not retrieved for prompt injection.
- `review_only` and `rejected` entries do not contribute to live error-pattern
  aggregation.
- hybrid history is not reused cross-faction without known master faction.
- max card and per-role caps are enforced.

Restoration:

- `restore_for_new_game()` falls back to latest snapshot.
- explicit missing snapshot ID does not silently fall back to latest.
- saving a completed game updates the game-specific snapshot and `latest`.
- reflections inject even when profile is missing.
- dirty legacy snapshots do not rehydrate rejected reflection bodies.

Prompt rendering:

- reflection cards appear in `跨局学习参考`.
- cards do not contain historical player IDs.
- cards do not contain hidden historical identity truth as current fact.
- cards render only the prompt-card whitelist.
- `source.*`, `quality_*`, raw pattern arrays, and deterministic ground-truth
  details never appear.
- current-game grounding and output contract remain protected under budget.

Migration:

- existing generic legacy rows become rejected or review-only.
- approved rows convert to V2.
- embedded snapshot reflection cleanup removes rejected copies.
- new snapshots contain no full reflection bodies.
- dirty snapshots cannot rehydrate rejected reflection bodies.
- backup files are written before mutation.

### Integration Tests

Run targeted suites:

```powershell
python -m pytest tests/memory -q -o addopts='' -p no:cacheprovider -p no:xdist -p no:xdist.looponfail --basetemp E:\NLP\agent\wofkill\.pytest_tmp
python -m pytest tests/runtime tests/agents -q -o addopts='' -p no:cacheprovider -p no:xdist -p no:xdist.looponfail --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Run database migration verification against local PostgreSQL:

- `reflections` count by `quality_status`;
- no prompt-visible player IDs;
- no rejected rows inside `memory_snapshots`;
- no full reflection bodies inside new `memory_snapshots`;
- sample prompt for a player with reflections but no profile includes
  reflection cards.

## Rollout Plan

1. Add V2 schema helpers without changing storage tables.
2. Add quality gate and tests.
3. Add synthesizer and tests.
4. Collapse production reflection writes into the single V2 write path.
5. Update game-end storage to persist synthesized V2 reflections.
6. Update retrieval and prompt rendering to use prompt cards.
7. Fix profile-independent injection.
8. Fix latest-snapshot restore fallback and latest alias updates.
9. Stop storing full reflection bodies inside new snapshots.
10. Add migration and cleanup script.
11. Run migration in dry-run mode and inspect report.
12. Run migration with backup.
13. Verify database and prompt output.
14. Commit code, docs, and migration report.

## Success Criteria

The upgrade is complete only when:

1. a new game can inject reflection cards from prior games without requiring
   profile rows;
2. latest memory snapshot fallback works for new game IDs;
3. persisted new reflections are V2 with quality status;
4. low-quality legacy reflections do not reach player prompts;
5. LLM self-review and deterministic review are merged into one calibrated
   reflection;
6. historical player IDs and hidden truth do not appear in live prompt cards;
7. error-pattern aggregation produces non-empty categories for V2 reflections;
8. `review_only` and `rejected` entries do not affect live prompt cards or live
   prompt error-pattern aggregation;
9. new snapshots do not store full reflection bodies and dirty old snapshots do
   not rehydrate rejected reflections;
10. targeted memory, runtime, and prompt-builder tests pass;
11. local PostgreSQL verification confirms cleaned reflection data and no rejected
   snapshot copies.

## Open Decisions For Implementation Planning

1. Whether rejected legacy rows should be deleted from `reflections` or retained
   with `quality_status="rejected"`. Recommended: retain during first rollout,
   then delete after a verified backup and one successful game cycle.
2. Whether `ReflectionEntryV2` should be a new dataclass or a Pydantic model.
   Recommended: Pydantic, because the RAG V2 schema already uses Pydantic and
   prompt-visible field validation benefits from model validators.
3. Whether prompt cards should render as JSON or readable cards. Recommended:
   readable cards, because reflection advice is short and decision-oriented.
4. Whether semantic retrieval should be enabled immediately. Recommended: keep
   deterministic ranking for V2 and leave semantic indexing optional.

## Explicit Non-Regressions

1. RAG seed storage and retrieval are not changed by this work.
2. Current-game public summary, visible state, and recent transcript remain
   higher priority than cross-game reflections.
3. Existing legacy reflections remain loadable for audit and migration.
4. The live player prompt never receives raw `source.llm_self_review`,
   `source.auto_review_summary`, quality flags, or deterministic ground-truth
   details.
5. Production game-end code does not create new V1 live-learning reflection
   rows.

## Implementation Progress

Implemented on 2026-06-16:

1. Added `ReflectionEntryV2`, prompt-card schema, quality status, deterministic
   quality gate, synthesizer, and approved-only live query support.
2. Updated cross-game context injection so approved V2 reflections inject even
   when no `PlayerProfile` exists.
3. Updated prompt rendering so V2 live prompts render only prompt-card
   whitelist fields.
4. Added latest snapshot restore semantics and `latest` alias writes through
   `PersistentMemoryCoordinator`.
5. Changed MemoryStore snapshots to store reflection IDs only and ignore dirty
   legacy snapshot reflection bodies on restore.
6. Changed production game-end persistence to synthesize/gate/store V2
   reflections only; raw LLM self-review is retained as synthesis input, not
   written directly as live memory.
7. Added migration helpers and `scripts/migrate_reflection_memory_v2.py` for
   backup, dry-run legacy-row quality reports, and snapshot
   reflection-boundary cleanup.

Verification on 2026-06-16:

- `python -m pytest tests/runtime tests/agents -q -o addopts='' -p no:cacheprovider -p no:xdist -p no:xdist.looponfail --basetemp E:\NLP\agent\wofkill\.pytest_tmp`
  passed: 1614 tests.
- `python -m pytest tests/memory tests/storage/test_storage.py -q -o addopts='' -p no:cacheprovider -p no:xdist -p no:xdist.looponfail --basetemp E:\NLP\agent\wofkill\.pytest_tmp`
  passed: 257 tests.
- `python -m pytest tests/storage tests/skills -q -o addopts='' -p no:cacheprovider -p no:xdist -p no:xdist.looponfail --basetemp E:\NLP\agent\wofkill\.pytest_tmp`
  passed: 297 tests.

Operational follow-up:

- Run `scripts/migrate_reflection_memory_v2.py` against the target database in
  dry-run mode and inspect the JSON report for legacy row quality.
- Rerun with `--apply` only to clean dirty memory snapshots after backup. This
  helper reports legacy reflection rows but does not mutate or convert them;
  row conversion/deletion should be handled in a separately reviewed migration
  after the report is accepted.
