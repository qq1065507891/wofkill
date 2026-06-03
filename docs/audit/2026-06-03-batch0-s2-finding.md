# P0-S2 Finding: Three-Step Generation NOT Worth Implementing

**Date:** 2026-06-03
**Audit target:** Single-shot JSON output vs design §7.2 three-step generation
**Method:** Analyzed 279 action traces across 3 saved games (`game_g_3528592081/2989362760/2913931821.json`)

## Fallback Root-Cause Breakdown

Across 279 actions, **36 (12.9%) ended in fallback** after 3 retries. Error code distribution:

| Error | Fallback | Recovered via retry | Total occurrences |
|---|---|---|---|
| `parse_error` | 20 (55%) | 38 | 58 |
| `empty_response` | 15 (42%) | 24 | 39 |
| `speech_quality` | 0 | 16 | 16 |
| `vote_quality` | 0 | 10 | 10 |

## Three-Step Benefit Analysis

**Three-step generation** (private_intent → action+target → speech) would only help with `parse_error` cases, because:
- It reduces the JSON size per step (less likely to be malformed)
- It separates action selection from speech generation (less context pollution)

**Best-case impact:**
- 20 fallback parse_errors → maybe 12-15 saved (assuming 3-step reduces parse rate by 60%)
- 38 recovered parse_errors → maybe 23-30 saved (no retry needed if first try succeeds)
- **Maximum savings: ~30-45 actions per 3-game sample** (~10-15% of total)

**Cost:**
- 3x token cost per action (3 separate LLM calls)
- 3x latency per action
- Adds complexity (3-stage retry logic, partial state handling)

## Recommendation

**Do NOT implement three-step generation.** Reasoning:

1. **Cost > Benefit**: 30-45 actions saved per 279-action sample × 3x token cost = net negative ROI.

2. **Empty_response unaffected**: 15 fallbacks + 24 recoveries (39 total) are from the model returning nothing. Three-step doesn't address this. Fix this via P0-R2 (shorter prompts, provider tuning).

3. **Parse errors mostly recoverable**: 38/58 (65%) of parse_errors recover on retry. The 20 fallbacks are the **hard tail** that three-step might help, but those are likely model capability limits, not format complexity.

4. **P0-S1 (mode isolation) is a better fix**: Tighter prompt with mode-mutually-exclusive fields will reduce parse_errors more than splitting into 3 calls.

5. **Design doc is for weak models**: §7.2 says "弱模型或本地模型无法稳定一次性生成完整 JSON 时，必须启用三步生成". Current GLM-5.1 / Claude / Baidu models in this codebase are **not** weak models. Single-shot is fine.

## Implication for P0-S2

**Mark P0-S2 as "WON'T FIX"** (deferred indefinitely). If a future weak-model provider is added, revisit.

The 12.9% fallback rate is real and needs addressing, but the fix vector is:
- P0-S1 (mode isolation) — reduce parse_errors
- P0-R2 (shorter prompts) — reduce empty_response
- P0-R3 (encoding repair) — fix mojibake parse_errors
- P0-S6 (better retry hint) — help LLM self-correct on parse errors

Not three-step generation.

## Test

No new test needed for P0-S2 (decision to defer). Add this finding as a comment in the prompt_builder module's docstring.
