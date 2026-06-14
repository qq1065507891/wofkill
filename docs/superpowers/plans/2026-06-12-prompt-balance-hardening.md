# Prompt Balance Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the confirmed prompt-layer balance defects while leaving the wolf-team captain planning prompt unchanged.

**Architecture:** Keep the existing prompt-builder, directive, persona, skill, RAG, and memory boundaries. Add focused role/task filtering and one shared directive-priority policy; make current-game grounding survive budget pressure; keep `ActionContract` as the only field-level output schema.

**Tech Stack:** Python 3.11+, pytest, Pydantic v2, YAML prompt configuration.

## Status Update

Completed on 2026-06-14. All planned prompt-balance hardening tasks are implemented, including output-contract isolation, protected current-game grounding, shared directive priority, role prompt corrections, persona sanitization, skill/RAG updates, memory/cognition/judge wording, and static prompt audit coverage.

Verification completed:

- `pytest tests/agents tests/runtime tests/skills tests/rag tests/memory tests/persona_runtime -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp`
- `pytest -q -n 0 --basetemp E:\NLP\agent\wofkill\.pytest_tmp`
- `git diff --check`

---

## File Structure

Primary files:

- `werewolf_agent/agents/prompt_builder.py` — system/user contract, section priority, budget, JSON rendering.
- `werewolf_agent/runtime/context.py` — directive capping, RAG mapping, reflection/profile/cognition injection.
- `werewolf_agent/runtime/agent_adapter.py` — witch, hunter, vote prompt wording.
- `werewolf_agent/runtime/directives/{seer,witch,_shared,idiot,villager}.py` — role prompt corrections.
- `werewolf_agent/persona_runtime/router.py` — role-aware persona sanitization.
- `werewolf_agent/skills/{registry,werewolf_skills}.py` and `werewolf_agent/skills/*/SKILL.md` — post-append cap and factual corrections.
- `config/rag_seeds/seed_entries.yaml` — live RAG eligibility and vote coverage.
- `config/personas/judge_profiles.yaml`, `werewolf_agent/agents/judge.py` — non-causal judge wording.

Tests stay in existing focused suites. No new runtime architecture is introduced.

## Task 1: Output Contract And Valid Prompt Serialization

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Test: `tests/agents/test_prompt_mode_isolation.py`
- Test: `tests/agents/test_prompt_builder.py`

- [x] Add failing tests asserting the system prompt does not enumerate
  `action_type`, `choice`, or `intent` fields and instead delegates to the
  current `ActionContract`.
- [x] Add a failing test asserting `_compact_json` always returns parseable
  JSON when the source exceeds `_MAX_JSON_CONTEXT_CHARS`.
- [x] Add a failing test asserting the information-boundary prompt does not
  state an incorrect section count.
- [x] Run the focused tests and confirm the expected failures.
- [x] Replace the system field list with protocol invariants only.
- [x] Make oversized JSON render as a valid explicit truncation envelope.
- [x] Remove the hard-coded section count.
- [x] Run the focused tests and the existing action-contract suite.

## Task 2: Prompt Budget Grounding Priority

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Test: `tests/agents/test_prompt_builder.py`

- [x] Add failing budget-pressure tests showing public summary, visible state,
  salience, phase context, and latest transcript survive while persona,
  profile, RAG, reflection, cognition, and error history are droppable.
- [x] Add a failing test asserting persona is not in `_NEVER_DROP`.
- [x] Run the tests and confirm they fail under the current persona-first
  policy.
- [x] Reclassify current-game grounding as protected compact sections.
- [x] Demote persona and historical/reference sections in the eviction order.
- [x] Keep local section caps and the global 6,250-character budget.
- [x] Run prompt-builder tests.

## Task 3: Directive Priority And Cap Symmetry

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Modify: `werewolf_agent/runtime/context.py`
- Test: `tests/agents/test_prompt_builder.py`
- Test: `tests/runtime/test_directive_role_gating.py`

- [x] Add failing tests for explicit classification of seer, witch, hunter,
  idiot, villager, hybrid, and wolf speech/vote keys.
- [x] Add failing tests proving `must_address_alerts`, `role_alerts`,
  `vote_pressure`, and task-critical role directives survive the strategy cap.
- [x] Run the tests and confirm current reference fallback/drop behavior.
- [x] Add the missing known keys to hard/suggestion/reference sets using
  symmetric role semantics.
- [x] Change the cap drop order to use the same classifications and drop
  explicit round/reference data before hard constraints or current-turn role
  facts.
- [x] Run prompt and runtime directive tests.

## Task 4: Role Prompt Factual Corrections

**Files:**
- Modify: `werewolf_agent/runtime/directives/seer.py`
- Modify: `werewolf_agent/runtime/directives/witch.py`
- Modify: `werewolf_agent/runtime/directives/_shared.py`
- Modify: `werewolf_agent/runtime/directives/idiot.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `tests/runtime/test_strategy_directives.py`
- Test: `tests/runtime/test_night_flow.py`

- [x] Add a failing test where a late-position seer has a good N1 result and
  must report "好人", never "查杀".
- [x] Add a failing test where no check result exists and the prompt must not
  fabricate one.
- [x] Add failing assertions for evidence-sensitive antidote/poison wording,
  legal hunter voting, evidence-first no-sheriff voting, and consistent idiot
  reveal cost.
- [x] Run the focused tests and confirm failures.
- [x] Implement the minimal wording/branch changes.
- [x] Run strategy, night, witch, vote, and idiot tests.

## Task 5: Role-Aware Persona Sanitization

**Files:**
- Modify: `werewolf_agent/persona_runtime/router.py`
- Modify: `werewolf_agent/agents/player.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Test: `tests/agents/test_persona_router.py`
- Test: `tests/agents/test_player_agent.py`
- Test: `tests/agents/test_prompt_builder.py`

- [x] Add failing tests assigning `bold_pretender` and `deep_hooker` to
  villager/witch/seer contexts; live snapshots must not contain fake-seer,
  manipulation, or hook task styles.
- [x] Add a control test showing the same profile may retain compatible
  deceptive style for a werewolf.
- [x] Run the tests and confirm raw persona leakage.
- [x] Sanitize the live snapshot using `GameContext.own_role` and task type.
- [x] Render role-neutral expression fields only.
- [x] Run persona and player-agent tests.

## Task 6: Skill Prompt Cap And Content Corrections

**Files:**
- Modify: `werewolf_agent/skills/registry.py`
- Modify: `werewolf_agent/skills/deep-hook/SKILL.md`
- Modify: `werewolf_agent/skills/counter-claim/SKILL.md`
- Modify: `werewolf_agent/skills/bold-claim/SKILL.md`
- Modify: `werewolf_agent/skills/resist-push/SKILL.md`
- Modify: `werewolf_agent/skills/hide-identity/SKILL.md`
- Modify: `werewolf_agent/skills/last-words-analysis/SKILL.md`
- Modify: `werewolf_agent/skills/find-power/SKILL.md`
- Test: `tests/skills/test_registry.py`
- Test: `tests/skills/test_werewolf_skills.py`

- [x] Add a failing test proving the final prompt, including appended markdown,
  never exceeds `PROMPT_INJECTABLE_CAP`.
- [x] Add content regression tests rejecting the known false statements.
- [x] Run the tests and confirm failures.
- [x] Apply the cap after markdown append using the existing cap helper.
- [x] Correct the skill text and add private-analysis/no-public-exposure
  wording for shared power-role analysis.
- [x] Run all skill tests.

## Task 7: RAG Phase, Tags, And Coverage

**Files:**
- Modify: `werewolf_agent/runtime/context.py`
- Modify: `config/rag_seeds/seed_entries.yaml`
- Test: `tests/runtime/test_context.py`
- Test: `tests/rag/test_rag.py`
- Test: `tests/rag/test_ingestion.py`

- [x] Add failing tests asserting vote maps to `vote` and self-destruct maps to
  werewolf/self-destruct tags.
- [x] Add a failing ingestion/retrieval test proving the villager fake-seer
  case is not delivered as default villager live advice.
- [x] Add failing seed coverage tests for hunter and idiot vote cases.
- [x] Run the tests and confirm failures.
- [x] Correct mappings and reframe/exclude the conflicting case.
- [x] Add compact approved hunter/idiot vote seeds.
- [x] Run RAG and context tests.

## Task 8: Reflection, Profile, Cognition, And Judge Prompts

**Files:**
- Modify: `werewolf_agent/runtime/context.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Modify: `werewolf_agent/agents/judge.py`
- Modify: `config/personas/judge_profiles.yaml`
- Test: `tests/runtime/test_context.py`
- Test: `tests/memory/test_reflection.py`
- Test: `tests/agents/test_judge_agent.py`
- Test: `tests/runtime/test_judge_flow.py`

- [x] Add failing reflection tests proving hybrid history is not treated as
  generic good-faction history.
- [x] Add failing prompt tests excluding ability-rank labels, duplicate
  cognition suspect/trust lists, and unresolved evidence hashes.
- [x] Add failing judge tests forbidding guard/protection implications in
  peace-night wording and requiring fact-only public narration.
- [x] Run focused tests and confirm failures.
- [x] Use an explicit good-role set for reflection ranking.
- [x] Slim profile/cognition live hints.
- [x] Strengthen judge system/user wording and correct persona patterns.
- [x] Run memory, context, and judge tests.

## Task 9: Static Prompt Audit And Regression

**Files:**
- Create: `tests/agents/test_prompt_balance_static_audit.py`
- Modify only if failures expose an unclassified key or remaining conflict.

- [x] Add static tests for known directive-key classification, forbidden false
  prompt phrases, system/dynamic output-contract isolation, valid JSON
  compaction, and explicit exclusion of wolf-team-plan files from this change.
- [x] Run the new audit test.
- [x] Run targeted suites:
  `pytest tests/agents tests/runtime tests/skills tests/rag tests/memory tests/persona_runtime -q`.
- [x] Run the repository's standard full test command if the targeted suites
  are green.
- [x] Review `git diff` and confirm no wolf-team-plan behavior changed.

## Non-Goal Guard

Do not modify:

- `agent_wolf_team_plan`;
- `WolfTeamPlan` schema or tool;
- wolf-team-plan graph node;
- night discussion aggregation;
- the existence of centralized wolf planning.
