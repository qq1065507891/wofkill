# RAG V2 Transferable Knowledge Design

Date: 2026-06-14
Status: Approved for implementation
Owner: Codex development session

## Problem

The current RAG entry shape is centered on `summary` and `key_decisions`.
That is safe after the previous live-prompt hardening, but it still describes
what happened in a case. For Werewolf decisions, the agent needs transferable
tactical knowledge: when a pattern applies, when it does not apply, how to use
it in the current game, and how it can be misused.

The target is a long-term schema, not a small presentation tweak. RAG should
become a structured tactical knowledge base while preserving enough migration
compatibility to load old persisted entries.

## Goals

1. Replace live-player RAG dependence on `summary` / `key_decisions` with
   explicit transferable-knowledge fields.
2. Make retrieval score on the same tactical fields that the live prompt uses.
3. Migrate all bundled seed YAML entries to V2 fields in one pass.
4. Keep old persisted RAG data loadable through a compatibility migration.
5. Preserve live-prompt information boundaries: no audit metadata, source
   quality, relevance score, short quotes, or god-view content in player prompts.
6. Extend existing ingestion checks so new V2 fields cannot bypass forbidden
   keyword, player-slot, or rule-truth filters.

## Non-Goals

1. Do not change the rule engine or base game rules.
2. Do not inject full source quotes into live player prompts.
3. Do not add online retrieval or external network dependencies.
4. Do not remove audit metadata from `RAGHit`; it remains available for review
   and observability.

## V2 Entry Schema

`RAGEntry` remains the stored top-level object, but gains a V2 tactical frame.
The frame is optional at the type level only to support legacy persisted data.
The invariant is explicit:

- `schema_version == 2` requires a complete `tactical_frame`.
- missing `schema_version` is treated as legacy `schema_version == 1`.
- `schema_version == 1` may have `tactical_frame is None`, but every live or
  retrieval path must call the shared fallback helper before using prompt-visible
  content.

```python
class RAGTacticalFrame(BaseModel):
    situation_signature: str
    transferable_lesson: str
    applicability: list[str]
    counter_signals: list[str]
    recommended_use: str
    misuse_risk: str
```

`RAGEntry` fields:

- `entry_id`
- `title`
- `tactical_frame: RAGTacticalFrame | None`
- `summary: str = ""`
- `key_decisions: list[str] = []`
- `short_quotes: list[str] = []`
- `metadata`
- `content_type`
- `schema_version: int = 2`

`summary` and `key_decisions` are retained only for compatibility, audit, and
legacy storage loading. New seed entries must set `schema_version: 2` and a
complete `tactical_frame`. A `RAGEntry` model validator should reject any V2
entry whose frame is missing or incomplete.

`persistence.load_rag_entries()` should normalize incoming dicts before
constructing `RAGEntry`: if `schema_version` is absent, inject
`schema_version=1`. It must not auto-upgrade stored legacy data to V2 in-place.

## Hit And Prompt Shape

`RAGHit` should carry the V2 frame through retrieval:

- `tactical_frame: RAGTacticalFrame | None`
- legacy `summary` / `key_decisions` remain for older entries and audit tests.

The live prompt slim renderer should return only prompt-safe fields:

```json
{
  "type": "rag_hit",
  "title": "...",
  "situation_signature": "...",
  "transferable_lesson": "...",
  "applicability": ["..."],
  "counter_signals": ["..."],
  "recommended_use": "...",
  "misuse_risk": "..."
}
```

For migrated legacy entries without an authored frame, a shared helper should
create a conservative prompt-safe frame from legacy fields:

- `situation_signature`: role, phase, and tags from metadata
- `transferable_lesson`: legacy summary
- `applicability`: first 1-2 legacy key decisions, rewritten as weak conditions
- `counter_signals`: `["本局公开事实与案例局面不相似时不要参考"]`
- `recommended_use`: `仅作为低优先级参考，先依据本局公开事实判断。`
- `misuse_risk`: `误把历史案例当成本局事实或直接套用案例动作。`

This fallback exists for persisted data only. Bundled seed data should not rely
on it. The fallback must live in one shared helper, not in `PlayerPromptBuilder`
alone:

- `get_prompt_tactical_frame(entry_or_hit) -> RAGTacticalFrame | None`
- `build_rag_retrieval_text(entry_or_hit) -> str`

The retriever, reranker document construction, vector indexing, prompt renderer,
and deduplication should use these helpers so retrieval scores on the same
concepts that the live prompt displays.

## Retrieval Text

Retriever and reranker input should score on the tactical frame through
`build_rag_retrieval_text()`:

```text
title
situation_signature
transferable_lesson
applicability
counter_signals
recommended_use
metadata tags
```

Legacy `summary` and `key_decisions` are used only inside the shared fallback
when no tactical frame is available. This keeps new retrieval aligned with what
the live prompt will show.

Vector indexing in `RAGKnowledgeService` should use the same tactical text
helper, not a separate hand-built string.

Near-duplicate filtering in `dedup_hits_by_similarity()` should also tokenize
the shared tactical text, not `title + summary`, because V2 summaries may be
empty or purely legacy audit text.

## Live Prompt Rendering

`PlayerPromptBuilder._build_rag_hints()` should keep the current section
framing and warning/tail guards, but cards should prefer V2 fields:

```text
案例 1：...
- 适用局面：...
- 可迁移原则：...
- 适用条件：...
- 不适用信号：...
- 本局参考方式：...
- 误用风险：...
```

Field caps:

- title: 160 chars
- situation signature: 160 chars
- transferable lesson: 220 chars
- applicability: max 3 items, 120 chars each
- counter signals: max 3 items, 120 chars each
- recommended use: 180 chars
- misuse risk: 160 chars
- live card count: existing RAG live cap

If a V2 field is missing in a legacy fallback, render the fallback text rather
than dropping the whole RAG hint. If no prompt-safe tactical content remains,
drop that hint.

`prompt_renderer.hits_to_prompt_lines()` must emit V2 prompt-safe fields.
`PlayerPromptBuilder._slim_rag_hint_items()` must whitelist the same V2 fields;
otherwise the builder will strip the fields before `_render_rag_hint_cards()`
can use them. Tests must cover V2-only `ctx.rag_hints` with no legacy
`summary/key_decisions`.

## Seed Data Migration

`config/rag_seeds/seed_entries.yaml` currently has 27 entries. All bundled seeds
should be converted to `schema_version: 2` and get a complete `tactical_frame`.

Migration rules:

1. Keep `entry_id`, `title`, `metadata`, `content_type`.
2. Preserve legacy `summary` and `key_decisions` during the first migration so
   existing tests and audit tooling still have old fields.
3. Author tactical frames manually enough to avoid mechanical copies of
   `key_decisions`. The point is to express transferability, applicability, and
   misuse risk.
4. Do not put specific player IDs (`p01` style), hidden role truth, or rule
   adjudication claims in any V2 field.

`seed_data._build_entry()` must parse and pass through `content_type`,
`schema_version`, and `tactical_frame`. Tests should assert loaded
`RAGEntry` objects preserve these values, not only that raw YAML contains them.

## Validation

`CaseIngester` should validate all V2 frame strings using the same safety policy
as existing fields:

- forbidden keywords
- forbidden content-type checks
- player-slot regex checks
- rule-truth duplication checks
- source and visibility requirements

The validation should inspect:

- title
- summary
- key_decisions
- short_quotes
- tactical_frame.situation_signature
- tactical_frame.transferable_lesson
- tactical_frame.applicability
- tactical_frame.counter_signals
- tactical_frame.recommended_use
- tactical_frame.misuse_risk
- metadata tags

Repository-loaded data needs the same safety boundary. Since
`RAGKnowledgeService._load_entries()` currently calls `load_rag_entries()` and
then may retrieve entries without `CaseIngester.ingest()`, `load_rag_entries()`
or a service-load validation step must reject persisted entries whose
prompt-visible fields contain forbidden player IDs, rule-truth phrases, or
forbidden keywords. Add a regression test with a malicious persisted V2 frame.

## Persistence And Migration

`load_rag_entries()` should accept old dicts without `schema_version` or
`tactical_frame`. It should instantiate a legacy-compatible `RAGEntry` where
`schema_version` defaults to `1` and `tactical_frame` is generated lazily or by
a migration helper.

`save_rag_entries()` should serialize V2 entries with `schema_version` and
`tactical_frame`. No destructive migration of repository data is required in
this change.

## Tests

Add or update tests covering:

1. `RAGEntry` V2 schema accepts complete tactical frames.
2. Legacy stored dicts still load.
3. Seed YAML entries all have `schema_version: 2` and complete tactical frames.
4. Ingestion rejects forbidden player IDs and rule-truth phrases in every V2
   tactical field.
5. Retriever/reranker text includes tactical frame fields.
6. Vector indexing uses the shared tactical text helper.
7. Prompt renderer emits V2 fields and does not emit compact
   `title/summary/key_decisions` JSON.
8. Live prompt still strips audit-only metadata.
9. Legacy fallback renders safe cards when tactical frame is absent.
10. `PlayerPromptBuilder._slim_rag_hint_items()` preserves V2-only prompt fields.
11. `seed_data.create_seed_entries()` preserves `content_type`,
    `schema_version`, and `tactical_frame` from YAML.
12. Repository-loaded malicious V2 entries are rejected before live retrieval.
13. Dedup uses tactical text, so two V2 hits with different legacy summaries but
    the same tactical frame collapse as duplicates.

## Rollout

1. Add schema and migration helpers with tests.
2. Route retriever, vector indexing, and prompt renderer through shared V2 text
   helpers.
3. Migrate all bundled seed YAML entries to V2.
4. Run RAG, prompt-builder, runtime-context, and persistence-related tests.
5. Update `PROGRESS.md` and commit.
