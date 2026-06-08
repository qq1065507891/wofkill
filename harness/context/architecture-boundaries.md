# Architecture Boundaries

## RuleEngine

Owns deterministic truth:

- role assignment;
- legal actions;
- night resolution;
- witch limits;
- seer results;
- hunter trigger;
- idiot reveal;
- voting and tie;
- exile;
- last words;
- sheriff election, badge passing, badge tearing;
- death records;
- victory.

## Runtime

Owns orchestration:

- phase transitions;
- node execution;
- checkpointing;
- pause/resume;
- replay;
- event reduction into state.

Runtime must call RuleEngine for rule decisions.

## Agents

Agents produce proposals:

- action choice;
- target;
- speech;
- private intent snapshot;
- confidence;
- reason.

Agents must never mutate game truth directly.

## RAG And Memory

RAG and memory help agents play better, but they do not own rules.

Long-term vector memory stores reflections and lessons. Relationship logic, votes, claims, and attack/defense edges belong in structured tables or graphs.

### Reflection Memory (LLM prompt layer)

- **Owner**: `runtime/context.py` + `agents/prompt_builder.py` (LLM prompt construction). NOT the `RuleEngine`.
- **Storage**: `memory/reflection.py:ReflectionMemory` (in-memory) + PostgreSQL `reflections` table (`storage/postgres_store.py`). Schema is stable: `(entry_id, game_id, player_id, entry_json)`.
- **Reflection generation** (`_agent_reflection` in `agent_adapter.py`): calls LLM at game end with a role-family-specific template (good / wolf / hybrid). Templates emit section headers like `【投票错误】` / `【悍跳分析】` / `【保留的优点】` — a generic prompt is forbidden.
- **Cross-game injection** (`build_agent_context` in `runtime/context.py`): three hint fields:
  - `reflection_memory_hints` — top 8 per-role entries, sort key `(-priority, -faction_won, neg_game_id, entry_id)`. priority: same-role=2, same-faction=1, else=0. faction_won=True ranks first within same priority (winning patterns > losing lessons).
  - `error_pattern_hint` — aggregated top-2 mistake categories from section-header regex, no LLM call. Rendered as a separate "你历史最常犯的错误" prompt section.
  - `profile_memory_hint` — long-term ability stats and per-role win-rate.
- **Boundaries**:
  - Reflection text is free-form LLM output. The schema does NOT add columns for `category` / `severity` / `preserved_strength` — those are derived at retrieval time from section headers via regex.
  - Cross-game learning is primarily **cross-player same-role** (priority=2) and **cross-player same-faction** (priority=1) because V1 board assigns roles randomly per game (same player + same role rate is ~1/12).
  - LLM prompt changes must not require schema migrations. If a future improvement needs new columns (e.g. explicit `category` field), that is a separate decision and should be tracked in `PROGRESS.md` with explicit user approval.

## Visibility

Every event must have a visibility label. Player contexts are built from allowed visibility only.

Allowed view modes:

- `public`;
- `player_view`;
- `moderator_full`.

Player agents cannot receive `moderator_full`.
