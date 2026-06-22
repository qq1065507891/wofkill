# Fix Plan - Reduce Fallback Rate (0.18 → ~0.08-0.10)

> **For agentic workers:** Single-task plan — 3 self-contained fixes, execute in order. Keep one active task in `PROGRESS.md`.

## Goal

Reduce per-game fallback rate from 0.18 (g_3819861640 benchmark) to ~0.08-0.10 by fixing the three dominant error classes, ranked by ROI.

## Root Cause (g_3819861640 fallback analysis, 2026-06-23)

28 fallback events (fallback_rate=0.180):

| last_error | count | root cause |
|---|---|---|
| `empty_response` | 18 (64%) | p01/p09 use `ark_deepseek`→`minimax-m3`; the 6 agents on `minimax_default` have **0 empty** |
| `speech_quality` | 5 (18%) | all SPEECH forced to 4-field check (stance/suspicion/vote_leaning/evidence); sheriff/PK/D1 shouldn't require vote_leaning |
| `schema_validation` | 4 (14%) | LLM outputs illegal action_type; retry hint is too vague ("你提交的 action_type 不在合法动作内" — doesn't list what IS legal) |
| `illegal_action` | 1 | rare, same root as schema_validation |

## Deliverables

### Fix 1 (highest ROI, yaml-only): swap p01/p09 profile
- File: `config/models.yaml`
- Change: `p01` and `p09` `llm_profile` from `ark_deepseek` → `minimax_default`
- Rationale: the 6 agents already on `minimax_default` (p02/p04/p06/p08/p10/p12) had **0 empty_response** in g_3819861640. This swap should eliminate ~80% of empty_response.
- `minimax_default` uses `ark_deepseek_v4_flash` model (different ark endpoint, more stable).

### Fix 2 (medium ROI, ~30 lines): per-phase speech_quality required-set
- File: `werewolf_agent/runtime/speech_quality.py` (`_required_components`)
- Change:
  - Sheriff speech / PK speech: only `{"stance", "evidence"}` (no vote_leaning before voting)
  - Day-1 SPEECH: `{"stance", "suspicion_target"}` (evidence hard to require on D1)
  - D2+ SPEECH: keep existing 4-field default
- `_required_components(intent, *, phase="", day=0)` — add phase/day params.

### Fix 3 (low ROI, bonus, ~5 lines): schema_validation hint lists legal actions
- File: `werewolf_agent/agents/player.py` (retry hint for illegal_action ~L556-562)
- Change: correction_hint from generic "不在合法动作内" to include the legal set:
  ```python
  correction_hint=(
      f"你提交的 action_type '{parsed_action.action_type}' 不在当前合法动作内。"
      f"合法 action_type: {', '.join(str(a) for a in (context.legal_actions or []))}。"
      f"请从上述动作中选择并重新提交。"
  )
  ```

## Required Boundary

- Fix 1: `minimax_default` profile configuration must exist and be stable (verified — it's already used by 6 agents with 0 empty in g_3819861640).
- Fix 2: Does NOT remove evidence requirement for D2+ voting-phase speeches; only trims sheriff/PK/D1.
- Fix 3: Retry hint text only — no behavior change, no new code paths.
- All fixes: no `RuleEngine` changes, no game-logic changes, no schema changes.

## Files

- Modify: `config/models.yaml` — `player_assignments` (Fix 1)
- Modify: `werewolf_agent/runtime/speech_quality.py` — `_required_components` signature + body (Fix 2)
- Modify: `werewolf_agent/agents/player.py` — illegal_action retry hint (Fix 3)
- Test: existing tests pass; no new tests needed (Fix 3 hint: grep for tests that assert on the old hint string and update if any).

## Done Means

- Fallback rate in a new real game drops to ~0.08-0.12 (from 0.18 baseline).
- `empty_response` count for p01/p09 drops from 18 → ~0-2.
- `speech_quality` fallback count drops from 5 → ~1-2 (sheriff/PK/D1 no longer penalized for missing vote_leaning).
- `schema_validation` count drops from 4 → ~1-2.
- All existing tests pass; `git diff --check` clean; `compileall` clean.

## Risks

- Fix 1: `minimax_default`'s model (`ark_deepseek_v4_flash`) may behave differently on wolf/seer personas than `minimax-m3`. If game quality degrades (weird speeches, wrong votes), revert p01/p09 individually.
- Fix 2: Relaxing D1 requirements may let through lower-quality speeches. Monitor D1 speech quality in next game.
- Fix 3: No risk (hint text change only — existing tests that assert on the old hint string may need updating; grep first).

## Self-Review

- Spec coverage: Fix 1 (profile swap) ✓ Fix 2 (per-phase required-set) ✓ Fix 3 (hint string) ✓
- No placeholders
- Type consistency: `_required_components(intent, *, phase="", day=0)` ✓
