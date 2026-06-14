# RAG V2 Transferable Knowledge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert RAG from case-summary injection to a V2 transferable tactical knowledge schema that powers retrieval and live prompts.

**Architecture:** Add a `RAGTacticalFrame` model and shared tactical text helpers, then route schema loading, ingestion validation, retriever/reranker text, vector indexing, deduplication, and live prompt rendering through those helpers. Preserve legacy `summary/key_decisions` for persisted data and audit compatibility, but make all bundled seed entries explicit V2 entries.

**Tech Stack:** Python, Pydantic, YAML seed data, pytest, existing `werewolf_agent.rag` modules.

---

## File Structure

- Modify `werewolf_agent/rag/schemas.py`: add `RAGTacticalFrame`, `schema_version`, `tactical_frame`, and V2 invariant validators on `RAGEntry` / `RAGHit`.
- Create `werewolf_agent/rag/tactical_text.py`: shared `get_prompt_tactical_frame()`, `build_rag_retrieval_text()`, and prompt dict helpers used by retrieval, vector indexing, prompt rendering, prompt builder, and deduplication. These helpers must accept `RAGEntry`, `RAGHit`, and already-slim dicts.
- Modify `werewolf_agent/rag/persistence.py`: normalize legacy dicts to `schema_version=1`, load V2 dicts, and run prompt-visible safety validation.
- Modify `werewolf_agent/rag/ingestion.py`: validate V2 tactical fields in the same forbidden-content and rule-truth checks as legacy fields.
- Modify `werewolf_agent/rag/seed_data.py`: parse `content_type`, `schema_version`, and `tactical_frame` from YAML.
- Modify `werewolf_agent/rag/retriever.py`: use shared tactical text for reranker documents and carry `tactical_frame` into `RAGHit`.
- Modify `werewolf_agent/rag/knowledge_service.py`: use shared tactical text for vector indexing and apply live-safe role/phase/ruleset/visibility filtering to vector hits.
- Modify `werewolf_agent/rag/prompt_renderer.py`: emit V2 prompt-safe fields and dedup on shared tactical text.
- Modify `werewolf_agent/agents/prompt_builder.py`: preserve V2 fields in `_slim_rag_hint_items()` and render V2 cards.
- Modify `config/rag_seeds/seed_entries.yaml`: add `schema_version: 2` and complete `tactical_frame` to all 27 entries.
- Modify tests in `tests/rag/test_schemas.py`, `tests/rag/test_ingestion.py`, `tests/rag/test_prompt_renderer.py`, `tests/rag/test_rag.py`, `tests/rag/test_knowledge_service.py`, and `tests/agents/test_prompt_builder.py`.
- Modify `PROGRESS.md`: record RAG V2 schema/data migration and verification results.

## Task 1: Schema And Tactical Text Helpers

**Files:**
- Modify: `werewolf_agent/rag/schemas.py`
- Create: `werewolf_agent/rag/tactical_text.py`
- Test: `tests/rag/test_schemas.py`
- Compatibility Test: `tests/rag/test_rag.py`
- Compatibility Test: `tests/rag/test_prompt_renderer.py`

- [ ] **Step 1: Write failing schema tests**

Add tests:

```python
def test_rag_entry_v2_requires_tactical_frame() -> None:
    with pytest.raises(ValueError, match="tactical_frame"):
        RAGEntry(
            entry_id="v2_missing_frame",
            title="V2 missing frame",
            summary="legacy",
            schema_version=2,
            metadata=_metadata(),
        )


def test_rag_entry_v2_accepts_complete_tactical_frame() -> None:
    entry = RAGEntry(
        entry_id="v2_ok",
        title="V2 ok",
        schema_version=2,
        tactical_frame=RAGTacticalFrame(
            situation_signature="role=seer phase=speech",
            transferable_lesson="对跳局优先把讨论拉回真假预言家主线。",
            applicability=["场上存在对跳预言家", "公开发言围绕验人和警徽流展开"],
            counter_signals=["已有更高优先级查杀或爆点"],
            recommended_use="先拆验人和警徽流，再给出归票理由。",
            misuse_risk="把对跳局经验套到无对跳局面会误导投票。",
        ),
        metadata=_metadata(),
    )
    assert entry.schema_version == 2
    assert entry.tactical_frame.transferable_lesson.startswith("对跳局")


def test_rag_entry_v2_can_omit_legacy_summary() -> None:
    entry = RAGEntry(
        entry_id="v2_no_legacy_summary",
        title="V2 no legacy summary",
        schema_version=2,
        tactical_frame=_complete_tactical_frame(),
        metadata=_metadata(),
    )
    assert entry.summary == ""


@pytest.mark.parametrize(
    "field,value",
    [
        ("situation_signature", ""),
        ("transferable_lesson", ""),
        ("applicability", []),
        ("counter_signals", []),
        ("recommended_use", ""),
        ("misuse_risk", ""),
    ],
)
def test_rag_tactical_frame_rejects_incomplete_fields(field: str, value) -> None:
    data = _complete_tactical_frame().model_dump()
    data[field] = value
    with pytest.raises(ValueError, match=field):
        RAGTacticalFrame(**data)
```

Add tests for helper behavior:

```python
def test_build_rag_retrieval_text_uses_v2_fields() -> None:
    entry = _v2_entry()
    text = build_rag_retrieval_text(entry, max_chars=1500)
    assert "transferable_lesson" not in text
    assert "对跳局优先把讨论拉回真假预言家主线" in text
    assert "已有更高优先级查杀" in text


def test_legacy_entry_gets_safe_fallback_frame() -> None:
    entry = _legacy_entry(summary="历史案例摘要", key_decisions=["只在相似局面参考"])
    frame = get_prompt_tactical_frame(entry)
    assert frame is not None
    assert frame.transferable_lesson == "历史案例摘要"
    assert "不要参考" in " ".join(frame.counter_signals)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/rag/test_schemas.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: fails because `RAGTacticalFrame`, `schema_version`, `tactical_frame`, `summary` defaulting, completeness validation, and helper functions do not exist.

- [ ] **Step 3: Implement minimal schema and helper code**

In `werewolf_agent/rag/schemas.py`:

```python
class RAGTacticalFrame(BaseModel):
    situation_signature: str
    transferable_lesson: str
    applicability: list[str]
    counter_signals: list[str]
    recommended_use: str
    misuse_risk: str

    @field_validator(
        "situation_signature",
        "transferable_lesson",
        "recommended_use",
        "misuse_risk",
    )
    @classmethod
    def non_empty_text(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("applicability", "counter_signals")
    @classmethod
    def non_empty_list(cls, value: list[str], info: ValidationInfo) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError(f"{info.field_name} must be non-empty")
        return cleaned


class RAGEntry(BaseModel):
    ...
    summary: str = ""
    tactical_frame: RAGTacticalFrame | None = None
    schema_version: int = 2

    @model_validator(mode="after")
    def validate_schema_version(self) -> "RAGEntry":
        if self.schema_version == 2 and self.tactical_frame is None:
            raise ValueError("schema_version=2 requires tactical_frame")
        return self
```

Add the same optional `tactical_frame` to `RAGHit`.

Important compatibility step: after changing the default `schema_version` to
`2`, update existing test helpers or direct `RAGEntry(...)` fixtures that are
intentionally legacy to pass `schema_version=1`. Do not paper over failures by
making incomplete V2 frames valid.

Create `werewolf_agent/rag/tactical_text.py`:

```python
def get_prompt_tactical_frame(item: Any) -> RAGTacticalFrame | None:
    frame = item.get("tactical_frame") if isinstance(item, dict) else getattr(item, "tactical_frame", None)
    if frame is not None:
        if isinstance(frame, dict):
            return RAGTacticalFrame(**frame)
        return frame
    # Build conservative legacy fallback from summary, key_decisions, metadata.


def tactical_frame_to_prompt_dict(item: Any) -> dict[str, Any] | None:
    frame = get_prompt_tactical_frame(item)
    # Return title plus prompt-safe V2 fields, or None if no safe content exists.


def build_rag_retrieval_text(
    item: Any,
    *,
    max_chars: int | None = None,
    legacy_summary_chars: int = 800,
) -> str:
    frame = get_prompt_tactical_frame(item)
    # Join title, frame fields, and metadata tags. Enforce max_chars last.
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/rag/test_schemas.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: schema/helper tests pass.

Then run compatibility tests before committing:

```powershell
pytest tests/rag/test_schemas.py tests/rag/test_rag.py tests/rag/test_prompt_renderer.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: existing legacy fixtures still pass because intentional legacy
fixtures now set `schema_version=1`.

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/rag/schemas.py werewolf_agent/rag/tactical_text.py tests/rag/test_schemas.py tests/rag/test_rag.py tests/rag/test_prompt_renderer.py
git commit -m "feat: add rag v2 tactical frame schema"
```

## Task 2: Persistence, Seed Loader, And Safety Validation

**Files:**
- Modify: `werewolf_agent/rag/persistence.py`
- Modify: `werewolf_agent/rag/seed_data.py`
- Modify: `werewolf_agent/rag/ingestion.py`
- Test: `tests/rag/test_ingestion.py`
- Test: `tests/rag/test_knowledge_service.py`

- [ ] **Step 1: Write failing persistence and loader tests**

Add tests:

```python
from werewolf_agent.rag.persistence import load_rag_entries
from werewolf_agent.rag.seed_data import create_seed_entries as load_seed_entries


def test_load_rag_entries_treats_missing_schema_version_as_legacy() -> None:
    loaded = load_rag_entries([_legacy_entry_dict_without_schema_version()])
    assert loaded[0].schema_version == 1
    assert loaded[0].tactical_frame is None


def test_seed_loader_preserves_v2_fields(tmp_path) -> None:
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(_single_v2_seed_yaml(), encoding="utf-8")
    entry = load_seed_entries(seed_path)[0]
    assert entry.schema_version == 2
    assert entry.content_type == "strategy"
    assert entry.tactical_frame.transferable_lesson
```

Add parametrized malicious data regressions for both ingestion and persisted
repository loading:

```python
@pytest.mark.parametrize(
    "field,bad_value,match",
    [
        ("situation_signature", "role=seer p01", "player"),
        ("transferable_lesson", "p01 是狼", "player"),
        ("applicability", ["rule_engine_says this works"], "Forbidden keyword"),
        ("counter_signals", ["女巫不能自救"], "rule"),
        ("recommended_use", "actual_role_is werewolf", "Forbidden keyword"),
        ("misuse_risk", "p02 直接出局", "player"),
    ],
)
def test_ingestion_rejects_forbidden_content_in_v2_tactical_fields(
    field: str,
    bad_value,
    match: str,
) -> None:
    entry = _v2_entry_with_tactical_update(field, bad_value)
    with pytest.raises(IngestionError, match=match):
        CaseIngester().ingest(entry)


@pytest.mark.parametrize(
    "field,bad_value,match",
    [
        ("situation_signature", "role=seer p01", "player"),
        ("transferable_lesson", "p01 是狼", "player"),
        ("applicability", ["rule_engine_says this works"], "Forbidden keyword"),
        ("counter_signals", ["女巫不能自救"], "rule"),
        ("recommended_use", "actual_role_is werewolf", "Forbidden keyword"),
        ("misuse_risk", "p02 直接出局", "player"),
    ],
)
def test_load_rag_entries_rejects_forbidden_content_in_v2_tactical_fields(
    field: str,
    bad_value,
    match: str,
) -> None:
    data = _v2_entry_dict_with_tactical_update(field, bad_value)
    with pytest.raises(ValueError, match=match):
        load_rag_entries([data])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/rag/test_ingestion.py tests/rag/test_knowledge_service.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp -k "schema_version or tactical_frame or malicious_v2"
```

Expected: fails because loader drops V2 fields, test imports need the direct
seed loader / persistence loader, and persistence does not normalize or validate
V2 fields.

- [ ] **Step 3: Implement loader and validation changes**

In `seed_data._build_entry()`, pass:

```python
schema_version=raw.get("schema_version", 2),
tactical_frame=raw.get("tactical_frame"),
content_type=raw.get("content_type", "strategy"),
```

In `persistence.load_rag_entries()`:

```python
normalized = []
for raw in data:
    item = dict(raw)
    item.setdefault("schema_version", 1)
    entry = RAGEntry(**item)
    validate_rag_entry_prompt_safe(entry)
    normalized.append(entry)
return normalized
```

In `ingestion.py`, add or extract `validate_rag_entry_prompt_safe(entry)` and make `CaseIngester._validate_forbidden_content()` / `_validate_not_rule_truth()` include every V2 tactical field.

If `werewolf_agent.rag.ingestion.create_seed_entries()` remains as a
backward-compatible wrapper, optionally update it to forward `yaml_path`; tests
that need custom paths should still import `create_seed_entries` from
`werewolf_agent.rag.seed_data` directly.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/rag/test_ingestion.py tests/rag/test_knowledge_service.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: ingestion and knowledge-service tests pass.

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/rag/persistence.py werewolf_agent/rag/seed_data.py werewolf_agent/rag/ingestion.py tests/rag/test_ingestion.py tests/rag/test_knowledge_service.py
git commit -m "feat: load and validate rag v2 entries"
```

## Task 3: Retriever, Reranker, Vector Indexing, And Dedup

**Files:**
- Modify: `werewolf_agent/rag/retriever.py`
- Modify: `werewolf_agent/rag/knowledge_service.py`
- Modify: `werewolf_agent/rag/prompt_renderer.py`
- Test: `tests/rag/test_rag.py`
- Test: `tests/rag/test_knowledge_service.py`
- Test: `tests/rag/test_prompt_renderer.py`

- [ ] **Step 1: Write failing retrieval alignment tests**

Add tests:

```python
def test_reranker_input_uses_tactical_text_and_cap() -> None:
    entry = _v2_entry_with_long_tactical_frame()
    captured = []
    retriever = StrategyRetriever([entry], reranker=_SpyReranker(captured))
    retriever.retrieve(RAGQuery(role="seer", phase="speech", max_results=1))
    text = captured[0][0]["text"]
    assert "可迁移原则标记" in text
    assert len(text) <= 1500


def test_entry_to_hit_carries_tactical_frame() -> None:
    hit = StrategyRetriever([_v2_entry()]).retrieve(RAGQuery(role="seer", phase="speech"))[0]
    assert hit.tactical_frame is not None
    assert hit.tactical_frame.recommended_use
```

Add vector live-safe regressions for role, phase, ruleset, and visibility:

```python
def test_vector_candidate_role_filter_blocks_werewolf_frame_for_villager(repo) -> None:
    service = RAGKnowledgeService(
        repository=repo,
        vector_store=_FakeVectorStoreReturning(["wolf_entry"]),
        seed_provider=lambda: [_v2_werewolf_entry(), _v2_villager_entry()],
    )
    service.ensure_seeded()
    hits = service.retrieve_live_hints(RAGQuery(role="villager", phase="speech"))
    assert all(hit.role_perspective in ("villager", "general", "any", "") for hit in hits)
    assert all(hit.entry_id != "wolf_entry" for hit in hits)


@pytest.mark.parametrize(
    "bad_entry_factory,query",
    [
        (_v2_entry_with_phase("night_action"), RAGQuery(role="villager", phase="speech")),
        (_v2_entry_with_ruleset("other_ruleset"), RAGQuery(role="villager", phase="speech", ruleset_id="pre_witch_hunter_idiot_mixed")),
        (_v2_entry_with_visibility(VisibilityBoundary.GOD_VIEW), RAGQuery(role="villager", phase="speech")),
        (_v2_entry_with_visibility(VisibilityBoundary.MODERATOR_ONLY), RAGQuery(role="villager", phase="speech")),
    ],
)
def test_vector_candidates_obey_phase_ruleset_and_visibility_filters(
    repo,
    bad_entry_factory,
    query,
) -> None:
    bad_entry = bad_entry_factory(entry_id="bad_vector_entry")
    service = RAGKnowledgeService(
        repository=repo,
        vector_store=_FakeVectorStoreReturning(["bad_vector_entry"]),
        seed_provider=lambda: [bad_entry, _v2_villager_entry(entry_id="safe_entry")],
    )
    service.ensure_seeded()
    hits = service.retrieve_live_hints(query)
    assert all(hit.entry_id != "bad_vector_entry" for hit in hits)
```

Add dedup test:

```python
def test_dedup_uses_tactical_text_not_legacy_summary() -> None:
    a = _hit(summary="legacy A", tactical_frame=_same_frame())
    b = _hit(summary="legacy B", tactical_frame=_same_frame())
    assert len(dedup_hits_by_similarity([a, b], max_items=3)) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/rag/test_rag.py tests/rag/test_knowledge_service.py tests/rag/test_prompt_renderer.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp -k "tactical or vector_candidate_role_filter or dedup"
```

Expected: fails because retriever/vector/dedup still use legacy summary/key decisions.

- [ ] **Step 3: Implement retrieval and vector routing**

In `retriever.py`:

- build reranker document `text` using `build_rag_retrieval_text(entry, max_chars=1500)`.
- keep the existing legacy summary cap behavior through the helper.
- pass `tactical_frame=entry.tactical_frame` in `_entry_to_hit()`.

In `knowledge_service.py`:

- `_index_entry()` indexes `build_rag_retrieval_text(entry, max_chars=<bounded value>)`.
- `_vector_candidates()` applies a shared metadata predicate before selected vector entries are returned. The predicate must check ruleset, role compatibility, phase compatibility, and live visibility.

In `prompt_renderer.py`:

- `dedup_hits_by_similarity()` tokenizes `build_rag_retrieval_text(hit, max_chars=1500)`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/rag/test_rag.py tests/rag/test_knowledge_service.py tests/rag/test_prompt_renderer.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: RAG retrieval, vector, and prompt-renderer tests pass.

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/rag/retriever.py werewolf_agent/rag/knowledge_service.py werewolf_agent/rag/prompt_renderer.py tests/rag/test_rag.py tests/rag/test_knowledge_service.py tests/rag/test_prompt_renderer.py
git commit -m "feat: route rag retrieval through tactical text"
```

## Task 4: Live Prompt V2 Card Rendering

**Files:**
- Modify: `werewolf_agent/rag/prompt_renderer.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Test: `tests/rag/test_prompt_renderer.py`
- Test: `tests/agents/test_prompt_builder.py`

- [ ] **Step 1: Write failing live prompt tests**

Add prompt-renderer test:

```python
def test_hits_to_prompt_lines_emits_v2_prompt_fields() -> None:
    line = hits_to_prompt_lines([_v2_hit()])[0]
    assert set(line) == {
        "type",
        "title",
        "situation_signature",
        "transferable_lesson",
        "applicability",
        "counter_signals",
        "recommended_use",
        "misuse_risk",
    }
```

Add prompt-builder test:

```python
def test_prompt_builder_preserves_v2_only_rag_fields() -> None:
    ctx = _make_villager_context().model_copy(update={
        "rag_hints": [{
            "type": "rag_hit",
            "title": "V2 案例",
            "situation_signature": "role=villager phase=speech",
            "transferable_lesson": "先用公开发言解释怀疑链。",
            "applicability": ["当前只有公开发言线索"],
            "counter_signals": ["已有明确查杀时不优先使用"],
            "recommended_use": "把怀疑写成可复盘的发言差异。",
            "misuse_risk": "把低视角怀疑说成铁狼。",
        }]
    })
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "适用局面：role=villager phase=speech" in prompt
    assert "可迁移原则：先用公开发言解释怀疑链。" in prompt
    assert '"summary"' not in prompt[prompt.find("知识库提示"):]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/rag/test_prompt_renderer.py tests/agents/test_prompt_builder.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp -k "v2_prompt_fields or preserves_v2_only"
```

Expected: fails because V2 fields are stripped or not rendered.

- [ ] **Step 3: Implement V2 live prompt rendering**

In `prompt_renderer.render_hit_for_prompt()`:

- call `get_prompt_tactical_frame(hit)`.
- return only V2 prompt-safe fields plus `type` and `title`.
- do not return audit fields, `summary`, `key_decisions`, `short_quotes`, `relevance`, or source metadata for V2 hits.

In `prompt_builder._slim_rag_hint_items()`:

- preserve V2 keys.
- field-cap strings and list items according to the spec.
- call the shared tactical prompt helper from `werewolf_agent.rag.tactical_text`
  for both V2 dicts and legacy dicts.
- do not implement a separate `summary/key_decisions` fallback inside
  `PlayerPromptBuilder`; fallback behavior belongs in the shared helper so
  retrieval, dedup, prompt renderer, and prompt builder stay aligned.

In `_render_rag_hint_cards()`:

- render `适用局面`, `可迁移原则`, `适用条件`, `不适用信号`, `本局参考方式`, and `误用风险`.
- keep warning before cards and tail after cards.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/rag/test_prompt_renderer.py tests/agents/test_prompt_builder.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: prompt renderer and prompt builder tests pass.

- [ ] **Step 5: Commit**

```powershell
git add werewolf_agent/rag/prompt_renderer.py werewolf_agent/agents/prompt_builder.py tests/rag/test_prompt_renderer.py tests/agents/test_prompt_builder.py
git commit -m "feat: render rag v2 tactical cards"
```

## Task 5: Seed YAML V2 Migration

**Files:**
- Modify: `config/rag_seeds/seed_entries.yaml`
- Test: `tests/rag/test_ingestion.py`
- Test: `tests/rag/test_rag.py`

- [ ] **Step 1: Write failing seed-completeness tests**

Add tests:

```python
def test_all_seed_entries_are_v2_with_tactical_frame() -> None:
    expected_ids = {
        "seed_ext_seer_claim_01",
        "seed_ext_wolf_deep_hook_01",
        "seed_rule_seer_badge_flow_01",
        "seed_seer_counterclaim_vote_push_01",
        "seed_tutorial_yumindao_seer_beginner_450",
        "seed_tutorial_yumindao_witch_beginner_450",
        "seed_tutorial_yumindao_hunter_idiot_civilian_488",
        "seed_tutorial_yumindao_wolf_roles_883",
        "seed_tutorial_yumindao_hybrid_beginner_488",
        "seed_speech_wolf_defense_01",
        "seed_ext_witch_poison_timing_01",
        "seed_godview_review_01",
        "seed_hybrid_survive_01",
        "seed_jingcheng_villager_fake_seer_250709",
        "seed_jingcheng_wolf_antiprophet_push_250415",
        "seed_jingcheng_review_double_bomb_badge_loss_241218",
        "seed_jingcheng_wolf_god_hunt_260227",
        "seed_foundation_seer_night1_blind",
        "seed_foundation_gold_water_strategy",
        "seed_foundation_speech_originality",
        "seed_foundation_peace_night",
        "seed_foundation_peace_night_wolf",
        "seed_foundation_vote_record_hardest_info",
        "seed_foundation_counterclaim_analysis",
        "seed_foundation_withdraw_tactics",
        "seed_hunter_evidence_vote_01",
        "seed_idiot_evidence_vote_01",
    }
    entries = create_seed_entries()
    assert {entry.entry_id for entry in entries} == expected_ids
    assert len(entries) == 27
    for entry in entries:
        assert entry.schema_version == 2, entry.entry_id
        assert entry.tactical_frame is not None, entry.entry_id
        assert entry.tactical_frame.situation_signature, entry.entry_id
        assert entry.tactical_frame.transferable_lesson, entry.entry_id
        assert entry.tactical_frame.applicability, entry.entry_id
        assert entry.tactical_frame.counter_signals, entry.entry_id
        assert entry.tactical_frame.recommended_use, entry.entry_id
        assert entry.tactical_frame.misuse_risk, entry.entry_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/rag/test_ingestion.py tests/rag/test_rag.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp -k "seed_entries_are_v2 or seeds_ingest_cleanly"
```

Expected: fails because YAML seeds do not yet include `schema_version` or `tactical_frame`.

- [ ] **Step 3: Migrate all 27 YAML seed entries**

For each entry in `config/rag_seeds/seed_entries.yaml`, add:

```yaml
  schema_version: 2
  tactical_frame:
    situation_signature: "role=<role> phase=<phase> tags=<main tags>"
    transferable_lesson: "..."
    applicability:
      - "..."
    counter_signals:
      - "..."
    recommended_use: "..."
    misuse_risk: "..."
```

Rules:

- Do not insert `p01` / `p02` style player IDs.
- Do not encode role truth or adjudication truth.
- Keep `summary` and `key_decisions` for compatibility.
- Make `transferable_lesson` abstract enough to be reused, not a copy of what one case did.

- [ ] **Step 4: Run seed tests**

Run:

```powershell
pytest tests/rag/test_ingestion.py tests/rag/test_rag.py tests/rag/test_knowledge_service.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: all seed ingestion, RAG, and knowledge-service tests pass.

- [ ] **Step 5: Commit**

```powershell
git add config/rag_seeds/seed_entries.yaml tests/rag/test_ingestion.py tests/rag/test_rag.py
git commit -m "data: migrate rag seeds to v2 tactical frames"
```

## Task 6: Integration Verification, Progress, And Final Commit

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Run focused integration tests**

Run:

```powershell
pytest tests/rag/test_schemas.py tests/rag/test_ingestion.py tests/rag/test_prompt_renderer.py tests/rag/test_rag.py tests/rag/test_knowledge_service.py tests/agents/test_prompt_builder.py tests/runtime/test_context.py -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp
```

Expected: all selected tests pass.

- [ ] **Step 2: Run compile and whitespace checks**

Run:

```powershell
python -m compileall -q werewolf_agent tests
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 3: Update progress document**

In `PROGRESS.md`, add a top section `rag-v2-transferable-knowledge` covering:

- schema addition and legacy migration behavior
- shared tactical text helper
- retrieval/vector/dedup routing
- live prompt V2 card rendering
- all 27 seeds migrated
- test commands run

- [ ] **Step 4: Commit progress doc**

```powershell
git add PROGRESS.md
git commit -m "docs: record rag v2 migration progress"
```

- [ ] **Step 5: Final verification after all commits**

Run:

```powershell
git status --short
git log --oneline -8
```

Expected: working tree clean except intentional ignored local artifacts; latest commits show the RAG V2 implementation series.
