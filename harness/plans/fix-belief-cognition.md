# Fix Plan - Belief Vote Signal → Role Probabilities

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement task-by-task. Keep exactly one active task in `PROGRESS.md`. This plan is **independent** from `fix-reflection-memory.md` — do not mix them in one PR; their acceptance metrics differ. Peer-reviewed 2026-06-21; revised to apply review findings.

## Relationship to fix-reflection-memory

Both plans cooperate on `approved > 0` (see the other plan's score analysis). **This plan's defect B fix makes deterministic `error_analysis` non-empty → `mistake_patterns` non-empty → stable `quality_score ≥ 0.70`**, because deterministic mistakes (`"误判X为Y，最佳角色概率0.80"`) are long enough to avoid the `short_prompt_card` penalty that can stall 1b at 0.55. This plan also improves in-game cognition. Execute after or in parallel with 1a/1b.

## Goal

Improve in-game role judgment accuracy and unblock deterministic review's `error_analysis`. Currently `role_probabilities` only update on seer/werewolf/idiot hard claims; votes and speeches only touch `trust`, leaving `best_prob ≈ 0.143` for most players.

**Corrected causal chain (peer-review):** `role_probabilities` directly feeds (a) `review.py` `best_prob` → `error_analysis` → reflection `mistake_patterns` → `approved`, and (b) `top_role_prob` → `worlds.py:146-159` belief scoring. `my_suspects` list membership is driven by `faction_read`/`trust` (`cognition_state.py:177-182`), **not** `role_probabilities` directly — `my_suspects` benefits only indirectly.

## Root Cause (diagnosed 2026-06-21, peer-reviewed)

`_apply_vote` (`werewolf_agent/cognition/belief.py:224-254`) and `_apply_speech_signal` (`:256-266`) update `trust` only — verified, no `role_probabilities` mutation. `_CLAIM_ROLE_BOOST` (`:57-59`) has only `werewolf: {werewolf: 0.4}`; non-werewolf role claims get no boost. So the matrix never learns role probabilities from the richest public signal (vote patterns). Initial prior is `1/len(roles)` (`belief.py:91,97`; `cognition_matrix.py:34-43`) = `1/7 ≈ 0.143 < 0.3` → `_evaluate_judgments` (`review.py:106`) records zero errors.

## Deliverables

- **Dependency (path b, peer-review):** `target_claim_status(target, credibility)` helper. `PublicEvidenceIndex.EvidenceRef` does **not** store credibility (`public_evidence.py:24-30`); credibility lives in `SeerClaimCredibilityEngine` (`claim_credibility.py:69`). So: add a `credibility` parameter to `_apply_vote`, thread it through `_apply_fact` (`belief.py:145`); the helper queries `credibility.score_for(ref.source_player)` for the target's `checked_wolves` / gold-water refs and applies the credible threshold. **`EvidenceRef` schema unchanged** (avoids snapshot/from_dict round-trip churn).
- **"Credible seer" threshold (peer-review required):** `status in {"supported", "uncontested"} AND score >= 0.65` — reuse the existing `wolf_lean` gate (`belief.py:293`).
- **Helper:** `_shift_role(belief, role, delta)` with normalization — **reuse** the normalize block at `belief.py:218-221` / `:300-303` (do NOT write a third copy).
- **Core:** `_apply_vote` extension — shift the voter's `role_probabilities` based on target's public claim status.
- **confirmed_wolves branch (peer-review):** target is a confirmed wolf → current code early-returns at `belief.py:244-246`; the new role shift must STILL trigger (voting a confirmed wolf is a good-side signal).
- Unit tests: vote→role shift; credible-vs-non-credible gating; normalization; confirmed-wolf target.
- Acceptance: scripted games showing `best_prob` median rise, accuracy rise, `error_analysis` non-empty.

## Required Boundary

- **HARD (do not violate):** vote→role update must use **public anchors only** (declared-seer credibility + public vote pattern), **never ground truth** (`belief.py:61`). Note: the belief code path has **no `ground_truth` parameter** (verified) — "no ground-truth read" is structurally guaranteed; the primary compliance test is therefore **credible-vs-non-credible gating** (below), not ground-truth mutation.
- **Primary compliance test (peer-review):** construct two scenarios — non-credible seer checks target X as wolf vs credible seer checks target X as wolf, same vote — assert the role shift fires ONLY for the credible case. This is what actually proves "public anchor = credibility-gated, not any-claim-counts".
- Boost magnitudes **small and symmetric** (0.05–0.08) to avoid oscillation from wolf reverse-voting (倒钩).
- `_apply_vote` must remain **additive** to the existing `vote_delta` trust update. Trigger condition: role shift fires whenever a claim status exists, **including when `vote_delta == 0`** (target neither checked_wolf nor gold_water in `vote_delta`'s table — peer-review minor).
- `RuleEngine` untouched — this is agent cognition only.

## Required Order

1. **Confirm dependency + select path (b).** Read `public_evidence.py`; confirm `EvidenceRef` has no credibility and `vote_delta` (`:72-85`) returns a fixed delta. Add `credibility` param to `_apply_vote`; thread through `_apply_fact` (`belief.py:145`). Unit test: `_apply_vote` accepts credibility without breaking existing trust update.
2. **`target_claim_status` + credible threshold.** `status in {"supported","uncontested"} AND score >= 0.65`. Unit test: returns `wolf_by_credible_seer`/`good_by_credible_seer`/`none` correctly.
3. **`_shift_role` + normalization test.** Reuse `belief.py:218-221` normalize logic. Assert probabilities stay in [0,1] and sum to 1.
4. **`wolf_by_credible_seer` rule.** Voter who votes a target publicly checked as wolf by a credible seer → shift voter's `werewolf` probability **down**. Unit test.
5. **`good_by_credible_seer` rule.** Voter who votes a gold-water target → shift voter's `werewolf` probability **up**, smaller boost. Unit test. **Decision point:** keep or drop based on whether scripted-game accuracy improves — this rule may be too noisy.
6. **confirmed_wolves branch.** Ensure role shift still triggers when target is a confirmed wolf (do not skip via the `:244-246` early return). Unit test.
7. **Primary compliance test (credible vs non-credible, same target+vote → role shift differs).** Secondary: mutate ground truth, assert belief updates identical (guards future regressions).
8. **Scripted-game acceptance.** `best_prob` median > 0.2; accuracy > 2/11; `error_analysis` non-empty in review.

## Files

- Modify: `werewolf_agent/cognition/public_evidence.py` — `target_claim_status` helper (reads existing `checked_wolves`/gold-water refs).
- Modify: `werewolf_agent/cognition/belief.py` — `_apply_vote` (~L224, add `credibility` param), `_apply_fact` (~L145, thread credibility), new `_shift_role` (reusing normalize at L218-221).
- Test: `tests/cognition/test_belief_*.py` — **Glob first** to list actual filenames before adding; vote→role shift, credible gating, normalization, confirmed-wolf.
- Docs: `PROGRESS.md`; design doc §belief-integration note.

## Done Means

- `best_prob` median across a scripted game's matrices > 0.2 (was ~0.143).
- Review accuracy rises above 2/11 baseline over scripted games.
- **`error_analysis` non-empty in deterministic review** (cross-plan effect: unblocks reflection `approved`).
- Primary compliance test passes (credible seer triggers shift, non-credible does not).
- Existing belief tests still pass; `RuleEngine` unaffected; `git diff --check` clean.

## Risks

- **Signal pollution.** Wolves reverse-vote to poison belief. Mitigate with small symmetric boosts; if scripted-game accuracy degrades, reduce magnitudes or drop rule 5.
- **Rule 5 (`good_by_credible_seer`) is noisy.** Voting a gold-water player may be a misled good, not a wolf. May need to drop after tuning — step 5 marks this as a decision point.
- **Acceptance cost.** LLM games are slow (1–2h each). Use scripted games for belief-logic acceptance; only minimal real-game confirmation. Do not ship without scripted-game numbers.
- **`delta == 0` trigger.** Role shift must fire on claim status even when `vote_delta` returns 0 — clarify in `_apply_vote` to avoid the shift being skipped by an existing early return.
