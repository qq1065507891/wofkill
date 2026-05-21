# RAG Vector Seed Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production RAG knowledge path where cold-start seed cases, persisted RAG entries, and pgvector recall work through one service with deterministic fallback and safe live-player injection.

**Architecture:** Keep `create_seed_entries()` as the versioned canonical seed source, sync those seeds into repository storage and vector storage when available, and expose a single `RAGKnowledgeService` to runtime code. Runtime should not know whether knowledge came from code seed, Postgres, pgvector, or in-memory fallback; it only receives live-safe `RAGHit` context items.

**Tech Stack:** Python, Pydantic RAG schemas, existing `GameRepository` RAG persistence, PostgreSQL/pgvector via `PgVectorStore`, in-memory fallback via `StrategyRetriever`, pytest.

---

## Design Decisions

1. `create_seed_entries()` remains the source of truth for curated cold-start knowledge.
2. Docker/production startup should upsert seed entries into `rag_entries` and index them into `rag_vectors`.
3. Runtime retrieval should use a single service:
   - Postgres/pgvector available: restore entries from repository, ensure seed sync, retrieve via vector candidates plus rule-based reranking.
   - No database or vector backend: fall back to current in-memory seed retriever.
4. RuleEngine remains the only rule authority. RAG only suggests strategy.
5. Live-player injection must always use `InjectionContext.LIVE_PLAYER`; god-view/moderator-only entries never reach players.
6. Conflict resolution is metadata-driven:
   - `APPROVED` beats `PENDING` / `UNREVIEWED`.
   - `PRO_MATCH > EXPERT_REVIEW > HIGH_RANK_GAME > COMMUNITY_CASE > SELF_PLAY_CANDIDATE > UNREVIEWED`.
   - same `ruleset_id`, `phase`, and `role_perspective` beat generic matches.
   - code seed is not automatically highest priority; its quality metadata decides priority.

## File Structure

- Create: `werewolf_agent/rag/knowledge_service.py`
  - Owns seed sync, repository restore, vector indexing, retrieval, and fallback.
- Modify: `werewolf_agent/rag/vector_store.py`
  - Add metadata-filter-friendly query helpers if needed; keep existing interface stable.
- Modify: `werewolf_agent/storage/repository.py`
  - Ensure RAG save/load/delete contract is explicit enough for seed sync.
- Modify: `werewolf_agent/storage/sqlite_store.py`
  - Support seed metadata stored in `entry_json`; avoid schema churn unless necessary.
- Modify: `werewolf_agent/storage/postgres_store.py`
  - Ensure `save_rag_entries()` and `load_rag_entries()` work reliably in Docker mode.
- Modify: `werewolf_agent/api/app.py`
  - Construct `RAGKnowledgeService` at app startup when repository/vector env vars exist.
- Modify: `werewolf_agent/runtime/game_runner.py`
  - Add `rag_service` or `rag_injector` to `GameRunnerConfig` and runtime state.
- Modify: `werewolf_agent/runtime/agent_adapter.py`
  - Replace the current global seed-only injector with injected service/injector fallback.
- Test: `tests/rag/test_knowledge_service.py`
  - Service unit tests: seed sync, fallback, vector indexing, conflict ranking.
- Test: `tests/runtime/test_runtime.py`
  - Runtime context injection tests.
- Test: `tests/api/test_api.py`
  - Docker-like env wiring tests without needing real Docker.

---

### Task 1: Define RAGKnowledgeService Contract

**Files:**
- Create: `werewolf_agent/rag/knowledge_service.py`
- Test: `tests/rag/test_knowledge_service.py`

- [ ] **Step 1: Write failing tests for in-memory fallback**

```python
def test_service_falls_back_to_seed_entries_without_repository():
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService
    from werewolf_agent.rag.schemas import RAGQuery

    service = RAGKnowledgeService()
    hits = service.retrieve_live_hints(RAGQuery(
        role="werewolf",
        phase="night_discussion",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        max_results=3,
    ))

    assert hits
    assert any("京城大师赛" in hit.title for hit in hits)
    assert all(hit.allowed_in_live_context for hit in hits)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rag/test_knowledge_service.py::test_service_falls_back_to_seed_entries_without_repository -q`

Expected: FAIL because `RAGKnowledgeService` does not exist.

- [ ] **Step 3: Implement minimal service with seed fallback**

```python
class RAGKnowledgeService:
    def __init__(self, repository=None, vector_store=None, seed_provider=create_seed_entries):
        self._repository = repository
        self._vector_store = vector_store
        self._seed_provider = seed_provider
        self._injector = RAGInjector(StrategyRetriever(seed_provider()))

    def retrieve_live_hints(self, query: RAGQuery, *, game_id="", player_id="") -> list[RAGHit]:
        return self._injector.inject(
            query,
            injection_context=InjectionContext.LIVE_PLAYER,
            game_id=game_id,
            player_id=player_id,
        )

    def hits_to_context_items(self, hits: list[RAGHit], max_items: int = 3) -> list[dict]:
        return self._injector.hits_to_context_items(hits, max_items=max_items)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/rag/test_knowledge_service.py -q`

Expected: PASS.

---

### Task 2: Seed Sync Into Repository

**Files:**
- Modify: `werewolf_agent/rag/knowledge_service.py`
- Test: `tests/rag/test_knowledge_service.py`

- [ ] **Step 1: Write failing repository sync test**

```python
def test_service_ensure_seeded_upserts_seed_entries(repo):
    service = RAGKnowledgeService(repository=repo)

    result = service.ensure_seeded()
    loaded = repo.load_rag_entries()

    assert result["seed_count"] >= 11
    assert any(e["entry_id"] == "seed_jingcheng_wolf_god_hunt_260227" for e in loaded)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rag/test_knowledge_service.py::test_service_ensure_seeded_upserts_seed_entries -q`

Expected: FAIL because `ensure_seeded()` does not exist.

- [ ] **Step 3: Implement `ensure_seeded()`**

Implementation notes:
- Load seeds from `create_seed_entries()`.
- Serialize via `save_rag_entries(seeds)`.
- Upsert through `repository.save_rag_entries()`.
- Compute `content_hash` in memory and add it to serialized entry metadata if desired, but do not require DB schema changes yet.

- [ ] **Step 4: Run repository sync tests**

Run: `python -m pytest tests/rag/test_knowledge_service.py tests/storage/test_storage.py -q`

Expected: PASS.

---

### Task 3: Vector Indexing For Seed And Persisted Entries

**Files:**
- Modify: `werewolf_agent/rag/knowledge_service.py`
- Test: `tests/rag/test_knowledge_service.py`
- Optional Modify: `werewolf_agent/rag/vector_store.py`

- [ ] **Step 1: Write failing vector indexing test using `LocalVectorStore`**

```python
def test_ensure_seeded_indexes_seed_entries_in_vector_store(repo):
    from werewolf_agent.rag.vector_store import LocalVectorStore

    vector_store = LocalVectorStore()
    service = RAGKnowledgeService(repository=repo, vector_store=vector_store)

    service.ensure_seeded()

    assert vector_store.count() >= 11
    results = vector_store.query("狼人夜聊 抗推预言家 神牌信息", top_k=5)
    assert any("seed_jingcheng_wolf_god_hunt_260227" == r["doc_id"] for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rag/test_knowledge_service.py::test_ensure_seeded_indexes_seed_entries_in_vector_store -q`

Expected: FAIL because service does not index vectors.

- [ ] **Step 3: Implement vector indexing**

Index text format:

```python
text = "\n".join([
    entry.title,
    entry.summary,
    "\n".join(entry.key_decisions),
    " ".join(entry.metadata.tags),
])
metadata = {
    "entry_id": entry.entry_id,
    "ruleset_id": entry.metadata.ruleset_id,
    "phase": entry.metadata.phase,
    "role_perspective": entry.metadata.role_perspective,
    "quality_grade": entry.metadata.quality_grade.value,
    "review_status": entry.metadata.review_status.value,
    "visibility_boundary": entry.metadata.visibility_boundary.value,
}
vector_store.add(entry.entry_id, text, metadata)
```

- [ ] **Step 4: Run vector tests**

Run: `python -m pytest tests/rag/test_knowledge_service.py tests/rag/test_rag_hardening.py -q`

Expected: PASS.

---

### Task 4: Unified Retrieval With Vector Candidates And Metadata Reranking

**Files:**
- Modify: `werewolf_agent/rag/knowledge_service.py`
- Test: `tests/rag/test_knowledge_service.py`

- [ ] **Step 1: Write failing test for vector-backed retrieval**

```python
def test_vector_backed_retrieval_returns_full_rag_hits(repo):
    from werewolf_agent.rag.vector_store import LocalVectorStore
    from werewolf_agent.rag.schemas import RAGQuery

    service = RAGKnowledgeService(repository=repo, vector_store=LocalVectorStore())
    service.ensure_seeded()

    hits = service.retrieve_live_hints(RAGQuery(
        role="werewolf",
        phase="night_discussion",
        situation="抗推预言家后讨论神牌信息",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        max_results=3,
    ))

    assert any(h.entry_id == "seed_jingcheng_wolf_god_hunt_260227" for h in hits)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rag/test_knowledge_service.py::test_vector_backed_retrieval_returns_full_rag_hits -q`

Expected: FAIL because retrieval still uses only in-memory seed retriever or lacks vector candidate hydration.

- [ ] **Step 3: Implement retrieval pipeline**

Pipeline:
1. Ensure entries are available:
   - Prefer repository `load_rag_entries()`.
   - If empty, call `ensure_seeded()`.
   - If repository fails, use seed entries.
2. If vector store exists:
   - Query vector store with role + phase + situation.
   - Hydrate candidate `entry_id`s into `RAGEntry`s.
   - Add any exact role/phase seed candidates to avoid vector misses.
   - Run `StrategyRetriever(candidates).retrieve(query)`.
3. If vector store does not exist:
   - Use `StrategyRetriever(entries).retrieve(query)`.
4. Filter through `RAGInjector` live-player boundary.

- [ ] **Step 4: Run retrieval tests**

Run: `python -m pytest tests/rag/test_knowledge_service.py tests/rag/test_rag.py -q`

Expected: PASS.

---

### Task 5: Runtime Injection Uses RAGKnowledgeService

**Files:**
- Modify: `werewolf_agent/runtime/game_runner.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `tests/runtime/test_runtime.py`
- Test: `tests/runtime/test_game_runner.py`

- [ ] **Step 1: Write failing test for configured service injection**

```python
class FakeRAGService:
    def retrieve_live_hints(self, query, *, game_id="", player_id=""):
        self.last_query = query
        return []

    def hits_to_context_items(self, hits, max_items=3):
        return [{"type": "rag_hit", "entry_id": "fake", "title": "fake", "allowed_in_live": True}]

def test_build_agent_context_uses_passed_rag_service():
    service = FakeRAGService()
    ctx = build_agent_context(
        engine,
        gs,
        "p01",
        TaskType.SPEECH,
        rag_service=service,
    )
    assert any(item["entry_id"] == "fake" for item in ctx.salience_items)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/runtime/test_runtime.py::test_build_agent_context_uses_passed_rag_service -q`

Expected: FAIL because `build_agent_context()` does not accept `rag_service`.

- [ ] **Step 3: Add `rag_service` plumbing**

Implementation notes:
- Add `rag_service: Any = None` to `build_agent_context()`.
- Runtime nodes should pass `state.get("rag_service")`.
- Add `rag_service` to `RuntimeState`.
- Add `rag_service` to `GameRunnerConfig`.
- `_build_runtime_state()` should include it when configured.
- Keep `_default_rag_injector()` fallback only for CLI/dev if no service is passed.

- [ ] **Step 4: Run runtime tests**

Run: `python -m pytest tests/runtime/test_runtime.py tests/runtime/test_game_runner.py -q`

Expected: PASS.

---

### Task 6: API/Docker Startup Wires Repository And Vector Store

**Files:**
- Modify: `werewolf_agent/api/app.py`
- Test: `tests/api/test_api.py`

- [ ] **Step 1: Write failing API wiring test**

```python
def test_create_app_initializes_rag_service_from_env(monkeypatch):
    monkeypatch.setenv("WEREWOLF_VECTOR_BACKEND", "local")
    app = create_app(repository=InMemoryGameRepository())
    assert hasattr(app.state, "rag_service")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/api/test_api.py::test_create_app_initializes_rag_service_from_env -q`

Expected: FAIL because `app.state.rag_service` does not exist.

- [ ] **Step 3: Implement API startup wiring**

Implementation notes:
- In `create_app()`, after repository is created:
  - read `WEREWOLF_VECTOR_BACKEND`
  - call `create_vector_store(backend)` only if backend is set
  - catch `VectorStoreConfigError` and fall back to no vector store with warning
  - create `RAGKnowledgeService(repository=_repo, vector_store=vector_store)`
  - call `ensure_seeded()` if repository exists
  - store as `app.state.rag_service`
- When constructing `GameRunnerConfig`, pass `rag_service=app.state.rag_service`.

- [ ] **Step 4: Run API tests**

Run: `python -m pytest tests/api/test_api.py tests/integration/test_final_delivery.py -q`

Expected: PASS.

---

### Task 7: RAG Injection Audit Events

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_runtime.py`
- Test: `tests/api/test_api.py`

- [ ] **Step 1: Write failing audit event test**

```python
def test_rag_injection_audit_event_emitted_for_context_build():
    ctx = build_agent_context(..., rag_service=service)
    events = ctx.visible_world_state.get("pending_audit_events", [])
    assert any(e["type"] == "rag_injection_audit" for e in events)
```

- [ ] **Step 2: Prefer graph-level event emission**

Do not overload `visible_world_state` if graph nodes can append audit events after agent calls. Use service `last_audit()` or return audit payload alongside context.

- [ ] **Step 3: Implement audit event append**

Payload shape:

```python
{
    "player_id": player_id,
    "phase": phase,
    "task_type": task_type.value,
    "hits": [
        {
            "entry_id": "...",
            "title": "...",
            "quality": "...",
            "source_type": "...",
            "visibility": "...",
        }
    ],
}
```

- [ ] **Step 4: Run audit/API tests**

Run: `python -m pytest tests/runtime/test_runtime.py tests/api/test_api.py -q`

Expected: PASS.

---

### Task 8: Documentation And Operations

**Files:**
- Modify: `README.md`
- Modify: `docs/operations/deployment-guide.md`
- Modify: `PROGRESS.md`

- [ ] **Step 1: Document runtime modes**

Document:
- No DB: code seed + in-memory retriever.
- Docker with Postgres only: seed synced to `rag_entries`, in-memory/rule rerank fallback.
- Docker with pgvector: seed synced to `rag_entries` and `rag_vectors`; vector recall plus metadata rerank.

- [ ] **Step 2: Document conflict policy**

Include:
- RuleEngine > RAG.
- approved > pending.
- pro_match > expert_review > high_rank_game > community_case > self_play_candidate > unreviewed.
- same ruleset/role/phase wins.
- god-view never injected into live-player context.

- [ ] **Step 3: Document operational commands**

Examples:

```powershell
docker compose up --build
python -m pytest tests/rag/test_knowledge_service.py -q
```

- [ ] **Step 4: Run final focused verification**

Run:

```powershell
python -m pytest tests/rag/test_rag.py tests/rag/test_knowledge_service.py tests/rag/test_rag_hardening.py -q
python -m pytest tests/runtime/test_runtime.py tests/runtime/test_game_runner.py -q
python -m pytest tests/api/test_api.py -q
```

Expected: PASS.

---

## Acceptance Criteria

- Docker startup seeds `rag_entries` and indexes `rag_vectors` when pgvector is configured.
- Non-Docker startup still works with in-memory seed fallback.
- Runtime players receive RAG hints from one service, not ad hoc seed-only globals.
- Live-player RAG injection never includes `GOD_VIEW` or `MODERATOR_ONLY` entries.
- `rag_injection_audit` shows exactly which entries were injected.
- If pgvector fails, game startup still works and falls back to seed/in-memory retrieval.
- Conflicts are resolved by metadata priority and never override RuleEngine.
