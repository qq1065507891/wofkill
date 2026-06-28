# Game Balance Repair Design

Date: 2026-06-28
Status: Proposed
Owner: Codex development session

## Problem

The latest live-game sample shows a severe werewolf-side skew that is not
explained by deterministic rule errors.

Primary evidence from the seven newest saved games
(`game_g_850282163.json` through `game_g_3224969250.json`):

- Werewolf win rate: 7/7.
- Day-1 true seer exile rate: 4/7.
- Sheriff held by a werewolf: 5/7.
- Fallback / structured-output failure rate: 14.75%.
- Sheriff-vote action traces marked fallback: 36/36.
- Hunter shot hit werewolf: 2/7; hunter shot hit good roles: 5/7, including
  witch 3 times.
- Weak-evidence wolf plans still produced 8 wolf kills.

Hard rule auditing was clean. `find_boundary_violations` returned zero
violations for the same seven games: no dead-player actions, no pending hunter
shots, no self-votes, no empty vote reasons, and no dead-role wake broadcasts.
The fix should therefore focus on prompt contracts, fallback behavior,
decision-support signals, and evaluation guardrails rather than changing the
ruleset or victory conditions.

## Goals

1. Stop sheriff election from being polluted by exile-vote audit fields and
   schema fallbacks.
2. Make seer-line credibility visible, auditable, and strong enough to reduce
   fake-seer black-check cascades without weakening supported true-seer lines.
3. Prevent weak or fallback wolf-team plans from becoming automatic night kills.
4. Reduce good-side friendly fire from hunter, idiot, and witch decision paths.
5. Expand balance audit metrics so future live batches catch the same failure
   modes automatically.
6. Preserve deterministic rules and private-information boundaries.

## Non-Goals

1. Do not change the V1 role distribution, victory rules, sheriff weight, or
   witch / hunter / idiot base rules.
2. Do not make the game artificially 50/50 by hardcoding faction outcomes.
3. Do not let hidden roles or moderator-only truth enter live player prompts.
4. Do not remove legitimate wolf coordination. Strong evidence-based wolf
   consensus should still produce night kills.
5. Do not require large LLM batch runs for unit-test success.

## Design Principles

### Rules Stay Deterministic

RuleEngine remains the authority for actions, deaths, role skills, sheriff
badge state, and victory. This repair only changes how agents are prompted,
how invalid outputs are recovered, and how strategy signals are audited.

### Separate Election From Exile Voting

Sheriff voting is not day exile voting. It should not request or accept the
six moderator-only vote-audit fields used by `VotePlayerAction`:
`seer_stance`, `vote_basis`, `standing_with_seer`, `suspect_reason`,
`not_voting_reason`, and `private_reason`.

The sheriff-vote contract should include only:

- `action_type`
- `target_id`
- `speech`
- `reason`
- `confidence`

If a model includes exile-vote audit fields during sheriff voting, the retry
hint must explicitly say that those fields are forbidden for sheriff voting.

### Credibility Is Evidence, Not Truth

Seer-line credibility remains visible-evidence-only. It must not read hidden
roles in live play. It may score public seer claims, public check claims,
badge-flow claims, vote consistency, and public contradictions.

The live effect should be conservative:

- contested or weak black checks should not directly hard-mark good players as
  wolf-leaning;
- supported true-looking lines can still increase suspicion against their
  black-check targets;
- claimant credibility must be auditable through compact evidence codes.

### Weak Wolf Plans Are Advisory

`evidence_quality="none"` means no plan kill target is eligible.
`evidence_quality="weak"` means the plan can guide individual wolf actions, but
it must not become an automatic `_planned_wolf_kill` unless there is matching
target evidence and a fresh living-wolf confirmation path.

### Power Roles Need Anti-Friendly-Fire Support

Hunter, idiot, and witch already have rule-correct behavior, but the live
sample shows poor strategic outcomes. Their prompts and fallback behavior
should prefer evidence thresholds over dramatic role claims:

- hunter shot should require at least two independent public evidence sources
  before targeting a power-role-looking player;
- idiot should avoid following sheriff herds before reveal and should treat
  reveal as survival with a vote-cost;
- witch poison should remain evidence-gated, with late-game pressure only when
  candidates have public support.

## Architecture

### 1. Balance Audit Guardrails

Extend `werewolf_agent.evaluation.balance_audit` with metrics for the observed
failure modes:

- `sheriff_werewolf_rate`
- `sheriff_vote_fallback_rate`
- `d1_seer_exile_rate`
- `hunter_friendly_fire_rate`
- `weak_plan_kill_rate`
- `power_role_fallback_rate`

These metrics consume saved JSON logs only. They do not call model providers or
mutate game state.

### 2. Sheriff Vote Contract Isolation

`werewolf_agent.agents.action_contract` should distinguish
`TaskType.SHERIFF_VOTE` from `TaskType.VOTE` in both full JSON and target-choice
schemas. `PlayerPromptBuilder` should likewise only render the vote-audit field
guidance for exile votes, not sheriff votes.

Retry hints should classify schema errors caused by extra exile-vote fields in
a sheriff-vote response and give a targeted correction.

### 3. Seer Credibility Runtime Closure

The existing `SeerClaimCredibilityEngine` and prompt section should be wired
into runtime observability:

- `build_agent_context()` computes `seer_credibility`.
- `PlayerPromptBuilder._build_seer_credibility()` renders it when present.
- The runtime emits a moderator-only `seer_credibility_audit` event when at
  least one seer line exists.

The audit event should include player-visible claimant IDs, statuses, scores,
confidence, and compact public evidence strings. It must not include hidden
truth.

### 4. Wolf Plan Kill Gating

`_planned_wolf_kill` should reject:

- `evidence_quality="none"`;
- weak plans whose target has no matching `evidence_from_discussion`;
- stale targets that are dead, revealed-idiot exile-immune where inappropriate,
  or otherwise no longer legal.

Static fallback plans may preserve role assignments and public story, but must
not introduce fresh kill targets.

### 5. Good-Side Role Decision Hardening

The strategy directive layer should add explicit, testable language for:

- hunter shot evidence thresholds;
- hunter no-shot consistency when evidence is low;
- idiot pre-reveal anti-herd caution;
- witch poison public-source requirements and late-game pressure.

Fallback target selection should prefer evidence-aware candidates and should be
audited separately from successful model decisions.

## Data Flow

```text
saved/live game events
  -> public evidence extraction
  -> seer credibility / contradiction / vote quality
  -> AgentContext strategy + prompt sections
  -> model action contract
  -> schema validation and targeted retry
  -> fallback only if retries fail
  -> audit events and balance metrics
```

## Testing Strategy

1. Unit tests for new balance metrics on saved-style synthetic logs.
2. Contract tests proving sheriff-vote prompts and schemas omit exile-vote audit
   fields.
3. Replay-style parser tests using raw sheriff-vote examples from recent logs.
4. Cognition and prompt tests proving seer credibility is computed, rendered,
   and audited without hidden truth.
5. Wolf-strategy tests proving weak/no-evidence plans cannot produce automatic
   kills.
6. Strategy directive tests for hunter, idiot, and witch evidence thresholds.
7. A final saved-log audit over the seven recent games to confirm the guardrail
   warnings reproduce the current baseline before live fixes are evaluated.

## Acceptance Criteria

The repair is successful when deterministic tests pass and a fresh live batch
shows all of the following:

- `sheriff_vote_fallback_rate < 0.05`.
- `weak_plan_kill_rate == 0` for automatic planned kills.
- Day-1 true seer exile rate is materially below the current 4/7 baseline.
- Hunter friendly-fire rate is materially below the current 5/7 baseline.
- Wolf win rate in a 20-game batch is outside the current failure state; target
  observation band is 45%-65%, with review required above 70%.
- No new `find_boundary_violations` findings.

The 20-game live batch is an evaluation gate, not a unit-test gate. If model
availability blocks live runs, deterministic tests and saved-log audits still
must pass, and the live-batch risk remains open.

## Risks

1. Reducing wolf-plan fallback kills too much may create too many peace nights.
   Mitigation: keep strong evidence-based consensus unchanged.
2. Seer credibility weights can overfit recent games. Mitigation: use small
   deterministic deltas and scenario tests, not hidden truth.
3. Extra audit events may bloat logs. Mitigation: emit compact summaries only
   when seer claim lines exist.
4. Prompt text can drift from schema. Mitigation: add tests that compare
   contract required fields with rendered prompt guidance.

