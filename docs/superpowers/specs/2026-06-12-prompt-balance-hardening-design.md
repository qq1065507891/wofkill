# Prompt Balance Hardening Design

## Scope

Correct the prompt-layer defects identified in the June 11 audit while keeping
the wolf-team captain planning prompt unchanged.

This work is limited to:

- player system and user prompt assembly;
- directive text, priority, truncation, and role/task gating;
- persona fields rendered into player prompts;
- skill prompt bodies and prompt length enforcement;
- RAG query tags, live seed content, and role/phase coverage;
- player-facing reflection, profile, cognition, and judge prompt wording.

This work does not change game rules, role distribution, model routing, action
semantics, or the wolf-team planning architecture.

## Design Principles

The live prompt follows one precedence order:

1. game rules, information boundaries, legal actions, and the current
   `ActionContract`;
2. current-game public facts and task-relevant private role facts;
3. role-critical directives;
4. tactical suggestions;
5. skills, RAG, cross-game memory, profile, cognition, and persona style.

Lower layers may refine a decision but may not contradict or replace a higher
layer. Prompt budget pressure must remove historical and stylistic context
before removing current-game evidence.

## Changes

### 1. Factual Role Directives

The late-position seer directive will require an immediate claim and an exact
report of the first available check result. It will never use the word
"查杀" unless that result is actually `wolf`, and it will not invent a result
when no check exists.

Witch, hunter, villager, idiot, and no-sheriff vote guidance will be rewritten
as evidence-sensitive recommendations:

- antidote use is evaluated from target value and information, not described
  as an unconditional duty;
- poison use requires evidence even in low-player-count states;
- hunter guidance never recommends abstention when abstention is illegal;
- no-sheriff voting follows evidence, not an unverified check-kill claimant;
- idiot guidance consistently treats reveal as survival with a vote-cost, not
  as automatically beneficial.

### 2. Directive Priority And Capping

Known role-critical keys will be explicitly classified. Good-side role
directives will no longer fall through to the reference tier while equivalent
wolf directives receive stronger labels.

The directive cap will use the same priority knowledge as the prompt renderer:

- explicit round-specific, reference, and duplicated context are removed first;
- unknown keys are considered only after known droppable context, so
  forward-compatible current-turn role facts are not discarded ahead of known
  reference noise;
- suggestions, hard constraints, legal-action rules, contradiction alerts, and
  role-critical task directives are preserved for cap purposes.

Current-turn role facts such as `gold_water_duty`, `unreported_checks`, and
`my_check_history` are classified as hard directive keys so vote/speech turns do
not lose private role-critical context under budget pressure.

Tests will fail when a newly produced critical directive key is left
unclassified.

### 3. Prompt Budget

The budget trimmer will preserve compact current-turn grounding:

- phase and legal-task context;
- public summary;
- visible state;
- salient events;
- the latest transcript;
- hard directive content and the strict output contract.

Persona, profile, error history, cognition, reflection, RAG, belief summaries,
and other historical or stylistic sections are droppable. Each protected
current-game section keeps its existing local size cap so the global prompt
does not grow without bound.

Persona is therefore no longer a never-drop section.

> **决议更新（2026-06-18，`prompt-sanitizer-fix`）**：实现中 persona 仍是 never-drop
> （`prompt_builder.py` `_SectionSpec("_build_persona", ..., _NEVER_DROP_TIER)`，
> 测试 `test_persona_core_survives_under_budget_pressure` 锁定），与上文"no longer
> never-drop"的原始提议相反。保留 never-drop 的理由：persona 已行为化为短行
> （`_clean_prompt_text` + `_MAX_PERSONA_LINE_CHARS` 限长），预算压力下很少触发裁剪；
> 保持 persona 稳定可避免 LLM 在预算紧张时丢失人格/发言风格一致性。若未来要改回可裁剪，
> 需同步更新该测试并评估对发言风格一致性的影响。此为对上文提议的覆盖性决议。

### 4. Persona Isolation

Persona injection will contain role-neutral expression guidance only. Raw
factional labels such as `full_fake_seer`, `manipulative_push`,
`seer_claim_aggressive`, and `deep_hook` will not be shown to incompatible
roles.

The router may retain the full profile for audit and evaluation, but the live
prompt receives a sanitized snapshot derived from the current role and task.
Personality may control tone, brevity, confidence, and organization; it may
not instruct a good role to fabricate information or imitate a wolf role.

### 5. One Output Contract

The system prompt will state only the invariant protocol:

- produce one structured result;
- obey the current turn's `ActionContract`;
- do not add prose or Markdown.

It will not enumerate `action_type`, `choice`, or `intent` fields. The dynamic
strict contract and provider schema, both generated from `ActionContract`, are
the sole field-level source of truth.

### 6. Skills

Skill prompt length is capped after the optional `SKILL.md` body is appended.
Incorrect or contradictory skill statements are corrected, including:

- wolves cannot receive a good result from a normal seer check;
- wolves can and often should kill a credible real seer;
- idiot reveal has a voting-power cost;
- hiding identity cannot override mandatory seer reporting;
- last words are evaluated against public evidence, not trusted by death type
  alone.

Skills shared with good roles will explicitly say that power-role detection is
private defensive analysis and must not become public role exposure.

### 7. RAG

RAG task mapping will distinguish voting from speech. The `self_destruct`
action will use werewolf/self-destruct tags rather than idiot-reveal tags.

The villager fake-seer case will no longer be delivered as a default live
recommendation that conflicts with the villager role guide. It will be
reframed as a risk-analysis case or excluded from live player retrieval.

The seed corpus will add compact vote guidance for hunter and idiot so these
roles do not rely only on generic or wolf/seer-heavy examples. Live RAG remains
reference-only and cannot override rules or current facts.

### 8. Memory, Cognition, JSON, And Judge Wording

Cross-game reflection ranking will treat only explicit good roles as
same-faction good experience. Hybrid reflections are reused for another role
only when their actual faction is available; with the current schema they
remain same-role-only.

The live profile hint will omit judgmental ability ranks. Cognition will not
repeat the same suspect/trust list already present in the belief section, and
unresolvable `salience_items#hash` references will not be rendered.

JSON compaction must always return valid JSON. When content is too large, the
renderer will emit a valid, explicitly truncated representation rather than
cutting through a JSON token.

Judge prompts and persona patterns may add tone but may not imply a hidden
cause. In particular, a peace-night message must not use protection or guard
language in a ruleset without a guard.

The information-boundary section will not claim an incorrect number of user
prompt sections.

## Testing

Every behavior change follows red-green-refactor:

- prompt-builder tests for budget survival, persona sanitization, valid JSON,
  and mode-isolated output contracts;
- directive tests for exact seer result wording and evidence-sensitive
  witch/hunter/idiot/vote guidance;
- context tests for priority-aware capping and reflection/profile/cognition
  rendering;
- skill tests for post-append caps and corrected prompt bodies;
- RAG tests for vote mapping, self-destruct tags, seed eligibility, and
  hunter/idiot vote coverage;
- judge tests for non-causal public wording.

Targeted suites run after each task. The complete agents, runtime, skills, RAG,
memory, and persona test suites run before completion.

## Explicit Non-Goal

The wolf-team captain prompt, role assignment plan, night discussion
aggregation, and its asymmetric team-planning capability are not modified in
this change.
