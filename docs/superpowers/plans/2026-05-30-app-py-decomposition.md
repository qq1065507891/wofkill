# app.py Decomposition Plan

**Goal:** Split `api/app.py` (786 lines) into route modules + slim factory. Zero caller changes outside api/.

**Approach:** Each sub-module exports a `create_router(...)` factory that takes closure variables as params. `app.py` creates routers and includes them.

---

### Task 1: Extract customization routes → `api/routes/customization.py`

Move:
- Template download: `/templates/ruleset`, `/templates/persona-pack`
- Validation: `/customization/rulesets/validate`, `/customization/persona-packs/validate`
- Save: `/customization/rulesets`, `/customization/persona-packs`
- Marketplace: `/marketplace/rulesets`, `/marketplace/persona-packs`
- Helpers: `_require_customization_admin`, `_validation_result_to_dict`, `_record_to_public_dict`, `_record_to_storage_dict`, `_load_marketplace`

Router factory: `create_customization_router(authorized_callers, customization_repo, persist_fn, project_root) → APIRouter`

Modify: `api/app.py` — remove moved code, import and mount router

---

### Task 2: Extract game routes → `api/routes/games.py`

Move:
- CRUD: create, start, step, pause, resume
- Query: public-state, private-state, timeline, replay, evaluation, cognitive-diff, rag-audit, share-summary, list
- Helpers: `_get_game`, `_resolve_caller_role`, `_build_locked_config_snapshot`, `_event_is_public_for_share`, `_pick_public_mvp_candidate`

Router factory: `create_game_router(games, games_lock, runners, runners_lock, repo, rag_service, authorized_callers, auth, checker, executor, project_root) → APIRouter`

Modify: `api/app.py` — remove moved code, import and mount router

---

### Task 3: Final verification

- `app.py` should be ~200 lines (infrastructure + auth + dashboard)
- Full test suite
- Update PROGRESS.md
