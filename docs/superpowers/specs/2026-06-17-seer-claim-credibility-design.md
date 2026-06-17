# Seer Claim Credibility Design

Date: 2026-06-17
Status: Proposed design
Owner: Codex development session

## Problem

The current belief chain can consume public seer claims, votes, and role claims,
but the seer-claim logic is too coarse for counterclaim games.

The recent P0/P1 belief experiment showed a useful split:

1. public vote signals improved trust separation;
2. counterclaim dampening did not improve belief quality.

The counterclaim failure has two causes:

1. When two players claim seer, the current dampening reduces both true and fake
   check claims. This weakens real wolf detection.
2. A black-check claim still sets the target to `wolf_lean`, even when the seer
   claimant is contested. This means a fake seer can still mark a good player as
   a suspect.

The system needs a deterministic credibility layer between structured public
facts and belief updates:

```text
visible facts -> claim credibility -> belief updates -> possible worlds -> simulation
```

The credibility layer must evaluate the claimant and the whole seer line before
belief treats a check result as strong evidence.

## Current Baseline

Relevant components:

- `werewolf_agent.cognition.world_state` extracts `claimed_role`,
  `seer_check_claim`, `claimed_suspect`, `claimed_good`, `badge_flow_claim`, and
  `vote` facts from public events.
- `werewolf_agent.cognition.belief.BeliefUpdater` updates role probabilities,
  faction lean, and trust from visible facts.
- `werewolf_agent.cognition.contradiction.ContradictionEngine` detects claim,
  vote, and stance conflicts.
- `werewolf_agent.runtime.cognition_state.CognitionStateManager` maintains
  per-viewer live belief state and syncs it into cognition matrices.
- `werewolf_agent.cognition.worlds.PossibleWorldsEngine` scores hidden-role
  worlds from the prompt belief summary.
- `werewolf_agent.agents.prompt_builder.PlayerPromptBuilder` can render compact
  cognition summaries to the live prompt.

Known implementation hazards:

- Belief updates must process evidence in chronological order. A later public
  claim must not affect an earlier vote.
- The credibility layer must use only facts visible to the viewer.
- Runtime paths use `build_agent_context()` and `CognitionStateManager`; any new
  scoring must be wired into both recompute and persistent cognition paths.
- `CognitionStateManager` updates from only new events. Seer-line state must
  persist per viewer across update calls; otherwise later votes cannot be judged
  against earlier claims.

## Goals

1. Score seer claimants deterministically from visible public facts.
2. Distinguish claimant credibility, seer-line coherence, and check-result
   impact.
3. Prevent fake or low-credibility black checks from directly setting good
   players to `wolf_lean`.
4. Preserve strong updates for high-credibility seer checks.
5. Keep all scoring explainable through evidence and penalties.
6. Feed better belief summaries into possible worlds and simulation.
7. Render concise, prompt-safe seer-line credibility when useful.
8. Add evaluation scenarios that prove the change improves counterclaim games
   without degrading single-seer games.

## Non-Goals

1. Do not use hidden truth, final roles, or moderator-only state during live
   belief updates.
2. Do not ask the LLM to decide credibility inside the core belief updater.
3. Do not build a universal role-claim credibility system in the first version.
   The first implementation focuses on seer lines because they have the largest
   belief impact.
4. Do not replace the contradiction engine. Reuse its outputs as evidence where
   useful.
5. Do not make credibility a hard truth label. It remains a probabilistic,
   visible-evidence signal.
6. Do not let RAG, reflection, persona, or skills alter credibility scores.

## Design Principles

1. Public evidence only.
   - The same viewer should reach the same score from the same visible facts.
   - Private seer night results remain visible only to that seer.

2. Time-ordered updates.
   - Scores are updated as facts arrive.
   - A vote can only be judged against claims known before that vote.

3. Weak until anchored.
   - A contested black check is not enough to set `wolf_lean`.
   - Strong belief updates require credibility above a threshold.

4. Explainability.
   - Every score change records a compact evidence code.
   - Prompt summaries can explain why one line is stronger than another.

5. Conservative blast radius.
   - Start with seer claims only.
   - Keep the existing `BeliefUpdater` API stable where possible.

## Architecture

Add a new cognition component:

```text
werewolf_agent.cognition.claim_credibility
```

Primary classes:

```python
@dataclass
class ClaimCredibility:
    claimant: str
    claimed_role: str
    score: float
    confidence: float
    status: str
    evidence: list[str]
    penalties: list[str]


@dataclass
class SeerCheckClaim:
    claimant: str
    target: str
    result: str
    day: int
    evidence_ref: str


@dataclass
class SeerLine:
    claimant: str
    checks: list[SeerCheckClaim]
    badge_flow: list[str]
    vote_consistency: float
    contradiction_count: int
    credibility: ClaimCredibility


class SeerClaimCredibilityEngine:
    def observe(self, fact: StructuredFact) -> None: ...
    def score_for(self, claimant: str) -> ClaimCredibility: ...
    def line_for(self, claimant: str) -> SeerLine | None: ...
    def prompt_summary(self, limit: int = 3) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, Any]: ...
    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "SeerClaimCredibilityEngine": ...
```

`observe()` is called in fact order. It updates role claims, check claims,
badge-flow claims, stance claims, and votes. It does not read `GameState`
roles.

The snapshot methods are required for persistent cognition. A runtime manager
that processes only new events must be able to restore the viewer's prior
seer-line state before applying the next event batch.

## Scoring Model

Scores are deterministic and bounded to `[0.0, 1.0]`.

Base score:

```text
seer claimant starts at 0.50
```

Positive signals:

| Signal | Delta | Notes |
| --- | ---: | --- |
| First seer claim in visible facts | +0.05 | weak, not enough alone |
| Has at least one structured check claim | +0.08 | target and result present |
| Check chain has no duplicate target conflict | +0.05 | repeated same result is neutral |
| Vote follows own black check | +0.10 | claimant votes target they called wolf |
| Does not attack own gold-water target | +0.05 | consistency signal |
| Badge flow exists before vote/speech pressure | +0.05 | weak planning signal |
| Public exiled/revealed result supports a prior claim | +0.15 | only if public result exists |

Negative signals:

| Signal | Delta | Notes |
| --- | ---: | --- |
| Multiple seer claimants exist | -0.15 | contested line |
| Claimant votes away from own black check | -0.15 | strong inconsistency |
| Claimant attacks own gold-water target | -0.15 | strong inconsistency |
| Same claimant changes claimed role | -0.20 | broken line |
| Check result contradicts public revealed role | -0.25 | severe, public only |
| Stance reversal on a check target | -0.10 | from public claims |
| Missing vote support after pushing a black check | -0.05 | weak, phase-sensitive |

Confidence is separate from score. It reflects how much evidence exists:

```text
confidence = min(1.0, 0.25 + 0.15 * evidence_count)
```

Low confidence prevents over-strong downstream updates even when score is high.

## Status Bands

The engine maps score and contest state to stable status labels:

| Status | Condition | Meaning |
| --- | --- | --- |
| `uncontested` | one seer claimant, no severe penalty | single line, still not truth |
| `supported` | score >= 0.65 and confidence >= 0.55 | strong enough for belief lean |
| `contested` | multiple seer claimants and no line is supported | do not hard-lean check targets |
| `weak` | score < 0.45 | weak line |
| `broken` | severe public contradiction | line should mostly stop influencing belief |

## Belief Integration

`BeliefUpdater` should process visible facts in order with a credibility engine:

```text
for fact in visible_facts_in_order:
    credibility.observe(fact)
    apply fact using current credibility snapshot
```

This fixes the future-anchor problem: later claims do not affect earlier votes.

There are two supported call patterns:

1. Recompute fallback: build a fresh credibility engine and replay all visible
   facts in chronological order.
2. Persistent cognition: load the viewer's saved credibility snapshot, apply
   only new visible facts, then save the updated snapshot.

For `seer_check_claim`, belief update depends on the claimant credibility at the
time of the check:

```text
effective_strength = credibility.score * credibility.confidence
```

Black-check target update:

| Condition | Role probability | Faction lean | Open question |
| --- | --- | --- | --- |
| supported line | strong boost | may set `wolf_lean` | optional |
| uncontested but low evidence | medium boost | no hard lean unless score >= 0.65 | yes |
| contested line | weak boost | do not set `wolf_lean` | yes |
| weak line | tiny boost or none | no hard lean | yes |
| broken line | none | no hard lean | yes, source suspect |

Gold-water target update:

Gold-water claims are less dangerous than black checks but still should depend
on credibility. A supported line may set `good_lean`; a contested or weak line
only raises trust slightly and records an open question.

Claimant update:

Claimants receive role-probability and trust changes from their credibility,
not from claiming seer alone:

```text
seer_role_boost = 0.30 * score * confidence
trust_delta = (score - 0.50) * 0.08
```

This replaces one-size-fits-all claim boosts for seer claims.

## Contradiction Integration

The credibility engine should not duplicate every contradiction rule. Instead:

1. It performs lightweight local checks needed for streaming updates.
2. `ContradictionEngine.detect()` remains the full public contradiction pass.
3. Runtime can pass high-value contradiction alerts into credibility scoring
   when available.

Initial integration points:

- `claim_conflict` lowers all involved seer lines.
- `vote_conflict` lowers the claimant if the voter is the claimant and the
  conflict is against their own black check.
- `stance_reversal` lowers the claimant when it concerns their own check
  target.

The first implementation can work without alert injection by deriving the same
basic signals from facts in order. Alert injection is a second step if tests
show duplicated logic drifting.

## Prompt Integration

Add a compact seer-line section only when at least one seer claim exists:

```text
【辅助】 预言家线可信度
- p08 supported score=0.72: 查杀p03; 投票跟随查杀; 无明显矛盾
- p01 weak score=0.31: 对跳; 查杀后未投查杀; 与公开结果冲突
```

Rules:

- This section is private reasoning support, not public truth.
- Cap at three lines.
- Include only public evidence visible to the viewer.
- Drop before current-game public facts, persona, strategy directives, and
  output contract under prompt pressure.

## Runtime Integration

1. `CognitionStateManager.update_from_events()`
   - Uses the same `BeliefUpdater` path, so persistent matrices receive the
     credibility-aware updates.
   - Stores one seer-credibility snapshot per viewer alongside the existing
     in-memory `BeliefState`.
   - Loads the snapshot before applying the next incremental event batch.

2. `runtime.context.build_agent_context()`
   - Recompute fallback uses the same updater by replaying all visible facts.
   - Prompt belief summaries remain compact.
   - Optional seer-line prompt summary can be added to `AgentContext` later, or
     folded into `strategy_directive`/`belief_state` for the first version.

3. `PossibleWorldsEngine`
   - No direct changes in the first version.
   - It benefits through better `my_suspects` and `my_trusted` summaries.

4. `BoundedSimulator`
   - No direct changes in the first version.

## Evaluation Plan

Add deterministic belief-layer scenarios:

1. `single_seer_true_black_check`
   - One seer claimant black-checks a true wolf.
   - Expected: wolf probability and suspect recall do not regress.

2. `counterclaim_fake_black_check_good`
   - Fake seer black-checks a good player while true seer also claims.
   - Expected: good player is not set to `wolf_lean`; `good_false_sus` drops.

3. `counterclaim_true_line_supported_by_vote`
   - True seer black-checks a wolf and votes that target; fake seer avoids
     their own black check.
   - Expected: true line score > fake line score; wolf probability separation
     improves or stays neutral.

4. `claimant_votes_against_own_black_check`
   - Claimant black-checks p03 but votes p04.
   - Expected: claimant credibility drops; p03 does not get hard `wolf_lean`.

5. `future_anchor_guard`
   - Vote happens before a later seer claim.
   - Expected: the earlier vote is not judged against the later claim.

Prompt-layer tests:

1. seer-line credibility section renders only when seer claims exist;
2. section caps at three lines;
3. section contains no hidden role truth;
4. section is droppable under budget pressure.

Runtime tests:

1. `CognitionStateManager` persists credibility-aware belief changes;
2. `build_agent_context()` fallback and manager path produce compatible belief
   summaries;
3. visible-fact filtering still blocks private seer checks from other players.
4. incremental manager updates remember a previous seer claim when scoring a
   later vote.

Evaluation metrics:

```text
trust_sep = avg(good_trust) - avg(wolf_trust)
wolf_prob_sep = avg(wolf_werewolf_prob) - avg(good_werewolf_prob)
wolf_suspect_recall = true wolves marked suspect / true wolves
good_false_suspect = true good players marked suspect / true good players
seer_line_rank = true seer line rank in post-game evaluation only
```

Post-game `seer_line_rank` may use ground truth for evaluation only. Live
credibility scoring must not.

## Migration And Compatibility

No database migration is required for the first version.

- `BeliefState` and `PlayerBelief` can remain compatible.
- If open questions are already available on `PlayerBelief`, store low
  credibility check notes there.
- Runtime can keep seer-line snapshots in `CognitionStateManager` memory for
  the first implementation. Persisting them to repository snapshots is optional
  and can be added later if cross-process resume needs exact seer-line state.
- If prompt seer-line summaries need persistence later, add them to
  `CognitionMatrixEntry.metadata` or a dedicated runtime-only context field in
  a separate change.
- Existing tests that hand-construct `StructuredFact` objects remain valid.

## Risks And Mitigations

1. Overfitting to heuristic weights.
   - Keep weights small and tested by scenario.
   - Prefer status thresholds over many special cases.

2. Duplicating contradiction logic.
   - Keep only streaming essentials in the credibility engine.
   - Reuse `ContradictionEngine` outputs later if drift appears.

3. Weakening true seer too much in contested games.
   - Do not use global counterclaim dampening.
   - Reward line consistency, especially vote/check alignment.

4. Prompt bloat.
   - Cap summaries.
   - Make the section droppable.

5. Hidden information leaks.
   - Accept only visible structured facts.
   - Add explicit tests for villager and wolf perspectives.

## Rejected Alternatives

### A. Keep Current Counterclaim Dampening

Rejected because the experiment showed it weakens true and fake checks together
and does not reduce false suspect marking.

### B. Move Credibility Entirely Into Prompt Text

Rejected because possible worlds and simulation depend on structured belief.
Prompt-only hints do not fix upstream cognition.

### C. Build Universal Role Credibility First

Rejected for scope. Seer lines dominate belief pollution. A general system can
reuse this architecture later after the seer-specific path is proven.

## Acceptance Criteria

1. A contested fake black check no longer directly marks a good target as
   `wolf_lean`.
2. A supported true black check can still mark the target as `wolf_lean`.
3. A claimant who votes away from their own black check loses credibility.
4. Earlier votes are not reinterpreted using later claims.
5. `CognitionStateManager` and `build_agent_context()` both use the same
   credibility-aware belief path.
6. Focused cognition, runtime, and prompt tests pass.
7. Incremental cognition updates preserve earlier seer-line state when judging
   later votes.
8. The offline belief evaluation shows:
   - `good_false_suspect` improves in the counterclaim fake-check scenario;
   - single-seer regression stays neutral;
   - vote-signal gains from P0 remain.
