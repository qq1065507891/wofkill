# Project Progress

This file is the control ledger for Claude/GLM development. Update it at the start and end of every development session.

## Current Status

- Current phase: Phase 9 - Evaluation Lab COMPLETE
- Active task: Phase 9 delivered. All planned phases complete.
- Task owner: Claude/GLM development session
- Last updated: 2026-05-16

## Non-Negotiable Context

- Primary design: `docs/design/werewolf-agent-v1-design.md`
- Current ruleset config: `config/rulesets/pre_witch_hunter_idiot_mixed.yaml`
- Ruleset YAML has been synchronized with the design document and is now ready to drive RuleEngine tests.

## Completed

- Design document reviewed and corrected for V1 rule consistency.
- Harness scaffold created for Claude/GLM-driven development.
- Synchronized `config/rulesets/pre_witch_hunter_idiot_mixed.yaml` with `docs/design/werewolf-agent-v1-design.md` Chapter 3.
- Synchronized wolf no-kill and timeout rules into design/ruleset: wolves may actively no-kill, wolf discussion timeout defaults to no-kill, and day speech timeout records `speech_timeout`.
- Created the initial RuleEngine pytest suite and a minimal importable RuleEngine skeleton.
- Added Conda development environment definition and setup instructions.
- Fixed `check_victory` god-slaughter: revealed idiot alive counts as living god.
- Normalized YAML `torn_badge_order_policy` value.
- Implemented `assign_roles` with seed-based reproducible random distribution.
- Implemented `choose_master` with master faction resolution and double-call guard.
- Implemented `resolve_night` pipeline: wolf kill → antidote → poison, with peace night and same-night double death support.
- Implemented `resolve_self_destruct` for wolf day self-destruct with no last words.
- Implemented full sheriff election: `sheriff_register`, `sheriff_withdraw`, `resolve_sheriff_vote`, badge transfer/tear with `target_id`.
- Implemented event reducer: `reduce_event` / `reduce_events` supporting all event types for deterministic replay.
- Extended `GameState` with `antidote_used`, `poison_used`, `deaths`, `events` fields.
- **58 rule tests passing**.
- **Phase 2**: Built LangGraph `StateGraph` with 27 nodes matching design doc §6.2.
- **Phase 2**: Implemented conditional edges for self-destruct, first-tie PK, second-tie no-exile, victory→finish/badge-transfer/continue-night.
- **Phase 2**: Sheriff election split nodes (registration, speech, withdraw, vote) on first day only.
- **Phase 2**: Night flow: wolf discussion → consensus → witch → seer → hybrid master → resolve.
- **Phase 2**: Day 4 scripted-mode safeguard forces good victory to prevent infinite loops.
- **Phase 2**: Event replay via `replay_from_events()` using RuleEngine reducer.
- **Phase 2**: Checkpoint support via `build_game_graph_with_checkpoint(MemorySaver)`.
- **Phase 2**: All route functions check `GameState` fields instead of ephemeral `_result` keys.
- **79 tests passing** (58 rule + 21 runtime).
- **Phase 3**: Created agent output schemas (`agents/schemas.py`) with Pydantic models: `PlayerAction`, `PrivateIntent`, `JudgeBroadcast`, `FallbackAction`, `RetryInfo`, `AgentContext`. Enums constrain legal `ActionType` and `TaskType`.
- **Phase 3**: Built `PlayerAgent` with schema-constrained output, illegal-output retry (3 attempts with correction hints), and fallback to first legal action after exhaustion. Code fence stripping for markdown-wrapped JSON.
- **Phase 3**: Built `JudgeAgent` for non-adjudicating broadcasts: phase announcements, death translations, vote results, sheriff results, speech summarization. Never contains ruling/winner fields.
- **Phase 3**: Implemented `PersonaRouter` with YAML-loaded persona profiles, per-task style resolution, and dynamic policy adjustments (when_suspected, when_teammate_exiled, when_trusted_by_good_players). Effective params clamped to [0,1].
- **Phase 3**: Created 12 anonymous jingcheng-style persona prototypes in `config/personas/jingcheng_style_prototypes.yaml`.
- **Phase 3**: Implemented `ModelRouter` Gateway with per-player `llm_profile`, per-task model selection, fallback chain, usage logging, config snapshots for reproducibility. No hardcoded API keys. Extensible `LLMProvider` protocol.
- **Phase 3**: Created model profiles and per-player assignments in `config/models.yaml` supporting Claude/GLM providers.
- **Phase 3**: `PrivateIntent` separation — private strategy stored separately from public speech, never written to timeline, only enters debug/audit views.
- **Phase 3**: `DefaultActionValidator` checks actions against RuleEngine-provided legal sets.
- **Phase 3**: Visibility boundaries enforced: `AgentContext` exposes only own role, no moderator_full, no other players' private state.
- **133 tests passing** (58 rule + 21 runtime + 54 agent).
- **Phase 4**: Implemented StructuredWorldState — converts GameEvents to typed structured facts (player_died, speech, vote, seer_check, hybrid_master_chosen, sheriff events, witch potions, claims).
- **Phase 4**: Implemented VisibilityPolicy — deterministic per-fact visibility by viewer role (public/wolf_team/seer_private/witch_private/hybrid_private/hunter_private/moderator_only). Hard boundary enforced before all downstream modules. Leak detection via check_no_leaks().
- **Phase 4**: Implemented AttentionFilter — prunes visible facts by role-specific attention rules, respects VisibilityPolicy hard boundary.
- **Phase 4**: Implemented SalienceEngine — weights visible facts by recency, phase relevance, role relevance, fact type priority. Buckets into high/medium/low for prompt inclusion decisions.
- **Phase 4**: Implemented BeliefUpdater — deterministic code-only belief updates: death removes player, self_destruct confirms wolf, idiot_reveal sets role=1.0, role claims shift probabilities, votes adjust trust. No LLM calls.
- **Phase 4**: Implemented ContradictionEngine — detects stance reversals (opposing claims on same target across days), vote conflicts (suspect claim ≠ vote target), claim conflicts (multiple players claim same role).
- **Phase 4**: Implemented StrategySelector — 8 predefined strategy packages (aggressive_defense, deep_hook, protect_seer, push_counter_wagon, find_wolves, survive_lay_low, claim_and_push, confuse_good) with role defaults and situation overrides.
- **Phase 4**: Implemented LocalContextBuilder — assembles token-budgeted AgentContext from all pipeline outputs. Enforces visibility boundaries, trims salience items and transcript to budget, no moderator_full or other private state.
- **Phase 4**: Implemented CognitivePipeline — top-level entry point wiring all cognition modules together.
- **203 tests passing** (58 rule + 21 runtime + 54 agent + 70 cognition).
- **Phase 5**: Created RAG schemas (`rag/schemas.py`) with Pydantic models: CaseMetadata, SourceMetadata, QualityGrade, ReviewStatus, CaseType, VisibilityBoundary, SourceType enums. RAGEntry with forbidden content_type validator. RAGHit with auto-computed allowed_in_live_context. RAGQuery with retrieval parameters.
- **Phase 5**: Built CaseIngester (`rag/ingestion.py`) with multi-layer validation: forbidden keyword detection, source metadata requirements for external cases, quality grade restrictions (project-internal cannot claim PRO_MATCH), base rule truth pattern rejection (witch self-save, seer hybrid, idiot reveal, hunter poison, second tie). Auto-timestamp on ingestion.
- **Phase 5**: Created 7 seed entries for cold start: seer claim/badge_flow template, wolf deep_hook tactics, seer badge_flow rule-derived template, wolf defense speech template, witch poison timing strategy, god-view review case (review-only), hybrid survival strategy.
- **Phase 5**: Implemented StrategyRetriever (`rag/retriever.py`) with priority ranking per design doc §9.2: external high-end first, external tactics second, project history third, speech templates last. Scoring: case type priority, quality grade, role match, phase match, tag overlap.
- **Phase 5**: Built RAGInjector (`rag/injector.py`) with context-aware visibility filtering: live player (no god-view), spectator (no god-view), review (all), moderator (all). Hits-to-context-items conversion with source/quality annotation for spectating audit.
- **268 tests passing** (58 rule + 21 runtime + 54 agent + 70 cognition + 65 rag).
- **Phase 6**: Created memory schemas (`memory/schemas.py`) with dataclasses: CognitionMatrixEntry (JSON-serializable per-player state), RelationEvent/RelationType (structured predicates: spoke_against, voted, claimed_role, defended, night_result_claimed), PlayerProfile (ability scores with growth), ReviewJudgment, ReviewReport (post-game analysis), ReflectionEntry (long-term experience), CrossGameQuery.
- **Phase 6**: Implemented CognitionMatrix (`memory/cognition_matrix.py`): per-viewer short-term state with role probabilities, faction read, trust, key evidence, open questions. Syncs from BeliefUpdater. JSON round-trip serialization.
- **Phase 6**: Implemented RelationGraph (`memory/relation_graph.py`): structured event table with typed predicates. Queryable by predicate, source, target, day. Import from StructuredWorldState with heuristic attack/defend detection from speech keywords. NOT vectors — all structured data stays queryable as events.
- **Phase 6**: Implemented ReflectionMemory (`memory/reflection.py`): long-term store for unstructured post-game reflections. Tag-based filtering, role/situation/player queries. Tag index for observability. Future vector search extension point.
- **Phase 6**: Implemented ProfileStore (`memory/profile.py`): player profiles with 6 ability scores (logic, deception, leadership, credibility, learning_rate, risk_preference) plus game stats. Clamped delta application. Top-by ranking. Aggregate summary.
- **Phase 6**: Implemented ReviewGenerator (`memory/review.py`): per-player post-game analysis comparing cognition matrix guesses against ground truth. Computes ability deltas from judgment accuracy. Deception analysis via relation graph (who pushed the player into voting good). Auto-generates improvement suggestions.
- **Phase 6**: Implemented MemoryStore (`memory/store.py`): unified coordinator managing all memory subsystems. Game lifecycle: init matrices → sync beliefs → add relations → generate reviews → update profiles → store reflections → reset for next game. Cross-game retrieval for pre-game experience loading.
- **344 tests passing** (58 rule + 21 runtime + 54 agent + 70 cognition + 65 rag + 76 memory).
- **Phase 7**: Created skill schemas (`skills/schemas.py`) with SkillName enum (12 core skills: bold_claim, counter_claim, push_vote, swing_vote, deep_hook, find_power, hide_identity, resist_push, wolf_pit, protect_power, last_words, review_correct), SkillDefinition with role/phase applicability, SkillInput (role, phase, situation, persona, experience, legal actions), SkillOutput (recommended action, speech structure, risk alerts, confidence).
- **Phase 7**: Built 12 werewolf skill definitions (`skills/werewolf_skills.py`) with per-skill deterministic suggestion handlers. Each handler returns structured output with recommended actions, speech structures, risk alerts, and confidence — agents make the final decision.
- **Phase 7**: Implemented SkillRegistry (`skills/registry.py`): register, lookup, dispatch skills by name/role/phase/tag. Dispatch applicable returns all matching skills for a given input.
- **Phase 7**: Created tool schemas (`tools/schemas.py`) with ToolSource (LOCAL vs MCP_EXTERNAL), InternalToolName (9 tools per design doc §11.2), ToolCall/ToolResult with source annotation, MCPProvider protocol, annotate_mcp_result enforcing suggestion-only status.
- **Phase 7**: Implemented LocalToolExecutor (`tools/local_tools.py`): 9 internal LangGraph tools — query_legal_actions, query_public_state, query_private_state, query_relation_graph, query_cognition_matrix, write_review, call_evaluator, read_experiment_config, generate_game_report. All deterministic, no RPC, no MCP.
- **Phase 7**: Implemented MCPRegistry (`tools/mcp_registry.py`): register/unregister external MCP providers. All MCP results annotated as suggestions with source annotation. MCP does not own game state truth. Includes MockMCPProvider, ExternalHistoryProvider, ExternalProfileProvider for testing.
- **Phase 7**: Implemented ToolCallLogger (`tools/tool_logger.py`): records all tool invocations with caller, parameters, result, duration, source. Queryable by tool name, caller, source type, errors. Summary for observability.
- **434 tests passing** (58 rule + 21 runtime + 54 agent + 70 cognition + 65 rag + 76 memory + 37 skill + 53 tool).
- **Phase 8**: Created API schemas (`api/schemas.py`) with ViewMode enum (public, player_view, moderator_full), CallerRole (player_agent, spectator, moderator, debugger), AuditEvent, request/response models for all endpoints (CreateGame, GameAction, PrivateState, Timeline, Replay, Evaluation, CognitiveDiff).
- **Phase 8**: Implemented PermissionChecker (`api/permissions.py`): three-tier view mode enforcement — public always allowed, player_view for own data only, moderator_full for moderator/debugger only. Spectators downgraded to public. Player agents denied moderator_full during live play. All denials logged as AuditEvents with caller, requested view, reason.
- **Phase 8**: Implemented view functions (`api/views.py`): build_public_state (no roles), build_private_state (role-specific fields, wolf teammates for own view), build_timeline (public events only for public mode), build_replay (moderator_full shows all roles), build_evaluation (hides roles in public mode), build_cognitive_diff (actual roles only in moderator_full, never during live play). private_intent stripped from all API responses.
- **Phase 8**: Built FastAPI app (`api/app.py`): 11 endpoints — POST /games, POST start/pause/resume, GET public-state/private-state/timeline/replay/evaluation/cognitive-diff, GET list games. Permission checks on every endpoint. Audit trail exposed via checker.
- **486 tests passing** (58 rule + 21 runtime + 54 agent + 70 cognition + 65 rag + 76 memory + 37 skill + 53 tool + 52 api).
- **Phase 9**: Created evaluation schemas (`evaluation/schemas.py`) with dataclasses: ActionRecord/ActionVerdict, LeakageRecord, CostRecord, GameResult (with initial_seed, ruleset_snapshot, event_log for replay), BatchConfig (with ExperimentDimension enum), MetricsSnapshot (FactionMetrics, PlayerMetrics, RoleMetrics, QualityMetrics, SafetyMetrics, CostMetrics, GrowthPoint), LeaderboardEntry, LeaderboardReport (with to_json_dict), ReplayRecord (with from_dict round-trip), ExperimentComparison.
- **Phase 9**: Implemented MetricsAggregator (`evaluation/metrics.py`): computes all §14 metrics from GameResult lists — faction win rates, per-player and per-role win rates, quality metrics (anti-push, vote accuracy, identity disguise, hybrid co-win, contradiction hit), safety metrics (leakage rate, illegal action rate, retry recovery, fallback rate), cost/latency statistics (per provider/task/player), growth curves (cumulative win rate evolution across games). Static compare_snapshots for experiment comparisons.
- **Phase 9**: Implemented BatchRunner (`evaluation/runner.py`): runs N games with fixed seed sets from BatchConfig, generates deterministic seed sets from batch_id hash, collects GameResult with ruleset_snapshot and event_log. Supports add_leakage_record, add_action_record, add_cost_record, import_usage_records from ModelRouter. verify_replay replays from initial_seed + ruleset_snapshot + event_log through RuleEngine reducer. Evaluation never mutates rule truth.
- **Phase 9**: Implemented ReportGenerator (`evaluation/reports.py`): generates LeaderboardReport with ranked entries (overall score from weighted win_rate + stance + anti_push + lie_detection + illegal_action + cost), experiment comparisons across all metric dimensions, growth curves, JSON round-trip serialization. Leaderboard entries include all §14 dimensions: werewolf win rate, good win rate, anti-push rate, lie detection rate, stance accuracy, illegal action rate, avg cost, avg latency.
- **542 tests passing** (56 evaluation + 52 api + 37 skill + 53 tool + 76 memory + 65 rag + 70 cognition + 54 agent + 58 rule + 21 runtime).

## Active Task Checklist

- [x] Read `docs/design/werewolf-agent-v1-design.md` Chapter 3.
- [x] Compare `config/rulesets/pre_witch_hunter_idiot_mixed.yaml` against Chapter 3.
- [x] Fix YAML fields that conflict with the design.
- [x] Add or scaffold deterministic `RuleEngine` models.
- [x] Add tests for critical V1 rules before relying on agent behavior.
- [x] Run verification commands.
- [x] Update this file with changed files and next task.
- [x] Fix `check_victory` god-slaughter logic: revealed-idiot alive counts as living god.
- [x] Fix YAML `torn_badge_order_policy` to match test expectation.
- [x] All 30 tests pass.
- [x] Create agent output schemas.
- [x] Build player agent with retry/fallback.
- [x] Build judge agent.
- [x] Implement Persona Router with dynamic modifiers.
- [x] Implement Model Router Gateway with fallback and cost tracking.
- [x] Add comprehensive agent/gateway tests.
- [x] Run full pytest suite (133 passed, 0 failed).
- [x] Create cognition package and structured world state.
- [x] Implement Visibility Policy with leak detection.
- [x] Implement Attention Filter with role-specific pruning.
- [x] Implement Salience Engine with weight/bucket system.
- [x] Implement Belief Updater with deterministic updates.
- [x] Implement Contradiction Engine with stance/vote/claim detection.
- [x] Implement Strategy Selector with role/situation mapping.
- [x] Implement Local Context Builder with token budget.
- [x] Implement Cognitive Pipeline top-level entry point.
- [x] Add comprehensive cognition tests (70 tests).
- [x] Run full pytest suite (203 passed, 0 failed).
- [x] Create RAG schemas with metadata, quality grades, visibility boundaries.
- [x] Implement CaseIngester with forbidden content and rule truth validation.
- [x] Create 7 seed entries for cold start (seer, wolf, witch, hybrid, god-view review).
- [x] Implement StrategyRetriever with priority ranking per design doc §9.2.
- [x] Implement RAGInjector with context-aware visibility filtering.
- [x] Add comprehensive RAG tests (65 tests).
- [x] Run full pytest suite (268 passed, 0 failed).
- [x] Create memory schemas with structured types.
- [x] Implement CognitionMatrix with BeliefUpdater sync.
- [x] Implement RelationGraph with typed predicates.
- [x] Implement ReflectionMemory with tag-based retrieval.
- [x] Implement PlayerProfile with growth tracking.
- [x] Implement ReviewGenerator with judgment evaluation and deception analysis.
- [x] Implement MemoryStore unified coordinator.
- [x] Add comprehensive memory tests (76 tests).
- [x] Run full pytest suite (344 passed, 0 failed).
- [x] Create skill schemas with 12 core werewolf skills.
- [x] Implement skill handlers with structured suggestion output.
- [x] Implement SkillRegistry with dispatch by role/phase/tag.
- [x] Create tool schemas with local vs MCP source classification.
- [x] Implement LocalToolExecutor with 9 internal LangGraph tools.
- [x] Implement MCPRegistry with external tool provider management.
- [x] Implement ToolCallLogger for observability.
- [x] Add comprehensive skill/tool tests (90 tests).
- [x] Run full pytest suite (434 passed, 0 failed).
- [x] Create API schemas with view modes (public, player_view, moderator_full).
- [x] Implement PermissionChecker with three-tier enforcement.
- [x] Implement view functions (public-state, private-state, timeline, replay, evaluation, cognitive-diff).
- [x] Build FastAPI app with 11 endpoints.
- [x] Add API permission tests (52 tests).
- [x] Run full pytest suite (486 passed, 0 failed).
- [x] Create evaluation schemas with GameResult, BatchConfig, MetricsSnapshot, LeaderboardReport, ReplayRecord.
- [x] Implement MetricsAggregator with all §14 metrics (faction, player, role, quality, safety, cost, growth).
- [x] Implement BatchRunner with fixed seed sets, ruleset_snapshot, event_log, replay verification.
- [x] Implement ReportGenerator with leaderboard JSON, experiment comparisons, growth curves.
- [x] Add comprehensive evaluation tests (56 tests).
- [x] Run full pytest suite (542 passed, 0 failed).

## Upcoming Tasks

All planned phases (1-9) are complete. Potential future work:
- Real LLM provider SDK integration.
- Vector search for RAG and reflection memory.
- Frontend observer UI (React/Vue).
- Real LLM-driven game execution (replacing mock actions in BatchRunner).

## Blockers

- None known.

## Recent Changes

- Created harness files for controlled development.
- Rewrote the V1 ruleset YAML to match the final design: night order, witch self-save ban, hybrid master/slaughter logic, idiot reveal state, sheriff badge transfer/tear, tie policy, last-words policy, visibility boundaries, and victory fields.
- Added pytest-based RuleEngine acceptance tests covering role distribution, night order, seer, witch, hunter, idiot, hybrid victory/slaughter, sheriff badge, tie, last words, and visibility.
- Added a minimal `werewolf_agent` package skeleton so tests collect and fail on unimplemented RuleEngine behavior instead of import errors.
- Added `.gitignore` entries for Python and pytest cache files.
- Added `environment.yml` and Conda setup documentation for a reproducible development environment.
- Fixed `check_victory` god-slaughter: revealed idiot that is still alive now correctly counts as a living god (only counts as god-out after actual death).
- Normalized YAML `torn_badge_order_policy` value to `random_start_then_seat_order` matching the no-sheriff policy (same behavior, consistent naming).

## Changed Files In Current Session

- `werewolf_agent/api/__init__.py` — new package
- `werewolf_agent/api/schemas.py` — API schemas: ViewMode (public/player_view/moderator_full), CallerRole, AuditEvent, request/response models for all endpoints
- `werewolf_agent/api/permissions.py` — PermissionChecker: three-tier view mode enforcement, audit logging for all denials, spectator downgrade, player agent moderator_full ban during live play
- `werewolf_agent/api/views.py` — View functions: public-state (no roles), private-state (role-specific fields), timeline (public events filter), replay (moderator_full with all roles), evaluation (role hiding), cognitive-diff (actual roles only in moderator_full), private_intent stripping
- `werewolf_agent/api/app.py` — FastAPI app: 11 endpoints with permission checks, game lifecycle management
- `tests/api/__init__.py` — new test package
- `tests/api/test_api.py` — 52 tests: permissions, endpoints, private state access, timeline/replay/evaluation views, cognitive diff, information leak prevention, audit trail
- `werewolf_agent/evaluation/__init__.py` — new package
- `werewolf_agent/evaluation/schemas.py` — Evaluation schemas: ActionRecord/ActionVerdict, LeakageRecord, CostRecord, GameResult (replayable from initial_seed + ruleset_snapshot + event_log), BatchConfig (ExperimentDimension), MetricsSnapshot (FactionMetrics, PlayerMetrics, RoleMetrics, QualityMetrics, SafetyMetrics, CostMetrics, GrowthPoint), LeaderboardEntry, LeaderboardReport (to_json_dict), ReplayRecord (round-trip), ExperimentComparison
- `werewolf_agent/evaluation/metrics.py` — MetricsAggregator: all §14 metrics computation, growth curves, compare_snapshots
- `werewolf_agent/evaluation/runner.py` — BatchRunner: fixed seed sets, deterministic game execution, replay verification, cost/leakage/action record injection
- `werewolf_agent/evaluation/reports.py` — ReportGenerator: leaderboard with ranked entries, experiment comparisons, growth curves, JSON round-trip
- `tests/evaluation/__init__.py` — new test package
- `tests/evaluation/test_evaluation.py` — 56 tests: schemas, metrics aggregation, batch runner, report generator, leaderboard JSON, replay verification, safety metrics, integration
- `PROGRESS.md`

## Verification Log

- Parsed `config/rulesets/pre_witch_hunter_idiot_mixed.yaml` with Python `yaml.safe_load`.
- Confirmed `player_count=12` and role count sum is `12`.
- Confirmed witch `can_self_save=false`.
- Confirmed hybrid conditional slaughter requirements for good master vs wolf master.
- Confirmed night order is wolf, witch, seer, first-night hunter/idiot status, first-night hybrid master.
- Confirmed old risky fields such as legacy master-field naming, configurable self-save, static hybrid slaughter boundary, and RAG rule explanation flags no longer remain.
- Ran `pytest` after adding tests only: initial red failed with `ModuleNotFoundError: No module named 'werewolf_agent'`, confirming tests caught missing implementation.
- Added minimal package skeleton and reran `pytest`: tests now collect and run; result is `3 passed, 27 failed`, with failures coming from `NotImplementedError` in unimplemented RuleEngine behavior.
- Attempted to remove generated `__pycache__` directories, but the cleanup command was blocked by the sandbox approval reviewer. `.gitignore` now excludes those generated caches.
- Added Conda environment file; not created locally because environment creation may require dependency downloads.
- 2026-05-15: Ran `pytest tests/rules/test_rule_engine_v1.py` — initial result 28 passed, 2 failed.
- Fixed `check_victory` idiot logic and YAML torn_badge_order_policy. Re-ran: **30 passed, 0 failed**.
- 2026-05-15: Implemented all Phase 1 missing features. Final result: **58 passed, 0 failed**.
- 2026-05-15: Phase 2 runtime implemented. LangGraph graph with 27 nodes, conditional edges, checkpoint, replay. **79 passed, 0 failed**.
- 2026-05-15: Phase 3 agents implemented. Schema-constrained outputs, private_intent separation, retry/fallback, Persona Router, Model Router Gateway, Judge Agent. **133 passed, 0 failed** (54 agent + 58 rule + 21 runtime).
- 2026-05-15: Phase 4 cognition implemented. StructuredWorldState, VisibilityPolicy, AttentionFilter, SalienceEngine, BeliefUpdater, ContradictionEngine, StrategySelector, LocalContextBuilder, CognitivePipeline. **203 passed, 0 failed** (70 cognition + 54 agent + 58 rule + 21 runtime).
- 2026-05-15: Phase 5 RAG implemented. RAG schemas, CaseIngester with forbidden content/rule truth validation, 7 seed entries, StrategyRetriever with priority ranking, RAGInjector with visibility-filtered injection. **268 passed, 0 failed** (65 rag + 70 cognition + 54 agent + 58 rule + 21 runtime).
- 2026-05-15: Phase 6 Memory/Review implemented. CognitionMatrix, RelationGraph, ReflectionMemory, ProfileStore, ReviewGenerator, MemoryStore. **344 passed, 0 failed** (76 memory + 65 rag + 70 cognition + 54 agent + 58 rule + 21 runtime).
- 2026-05-15: Phase 7 Skill/MCP implemented. 12 skill definitions, SkillRegistry, 9 local LangGraph tools, MCPRegistry for external tools, ToolCallLogger. **434 passed, 0 failed** (37 skill + 53 tool + 76 memory + 65 rag + 70 cognition + 54 agent + 58 rule + 21 runtime).
- 2026-05-15: Phase 8 API/Observer implemented. FastAPI with 11 endpoints, three-tier permission enforcement (public/player_view/moderator_full), cognitive diff view, audit logging, information leak prevention. **486 passed, 0 failed** (52 api + 37 skill + 53 tool + 76 memory + 65 rag + 70 cognition + 54 agent + 58 rule + 21 runtime).
- 2026-05-15: Phase 9 Evaluation Lab implemented. Evaluation schemas, MetricsAggregator (all §14 metrics), BatchRunner (fixed seed sets, replay from initial_seed + ruleset_snapshot + event_log), ReportGenerator (leaderboard JSON, experiment comparisons, growth curves). Evaluation never mutates rule truth. **542 passed, 0 failed** (56 evaluation + 52 api + 37 skill + 53 tool + 76 memory + 65 rag + 70 cognition + 54 agent + 58 rule + 21 runtime).

## Open Risks

- Model Router currently uses MockProvider for testing; real provider implementations need SDK integration when running live games.
- Persona dynamic policy triggers are based on simple boolean flags; Phase 4 cognitive pipeline now provides richer context for adjustments.
- VisibilityPolicy hardcoded role mappings may need config-driven overrides for future rulesets.
- RAG retriever currently uses exact-match filtering; semantic/vector search needed for scaling beyond seed data.
- RelationGraph speech-to-predicate extraction uses simple keyword heuristics; richer NLP can improve attack/defend detection.
- ReflectionMemory uses exact tag/role matching; vector semantic search needed for scaling.
- Review ability deltas use fixed coefficients; may need calibration from real game data.
- MCP providers are mock implementations; real external service integrations need SDK connectors.
- Skill handlers return deterministic suggestion skeletons; LLM integration needed for dynamic speech generation.
- API uses in-memory game storage; production needs database persistence.
- BatchRunner uses mock/simplified game actions for evaluation; real LLM agent integration needed for live benchmark runs.
- Quality metrics like lie_detection_rate and bold_claim_success_rate currently default to 0.0 without real LLM-generated speech; need LLM integration for meaningful measurement.
- Growth curves currently track only win_rate; could be extended with per-player ability score curves from ProfileStore.

## Next Step

All 9 planned phases complete. **542 tests pass** (56 evaluation + 52 api + 37 skill + 53 tool + 76 memory + 65 rag + 70 cognition + 54 agent + 58 rule + 21 runtime). Potential next steps: real LLM provider SDK, vector search, frontend observer UI, real LLM-driven game execution.
