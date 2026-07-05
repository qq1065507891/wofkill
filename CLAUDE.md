# Claude Development Harness

This repository is the Werewolf Agent V1 project. Claude/GLM development must follow the design document and the progress ledger before making code changes.

## Required Reading Order

Before every development session, read these files in order:

1. `PROGRESS.md`
2. `docs/design/werewolf-agent-v1-design.md`
3. `harness/context/project-brief.md`
4. `harness/context/rule-authority.md`
5. The current phase plan under `harness/plans/`

Do not start implementation until the current task in `PROGRESS.md` is clear.

Use the shared Conda environment before running tests or development commands:

```powershell
conda env create -f environment.yml
conda activate wofkill
```

If the environment already exists, use `conda env update -f environment.yml --prune`.

## Authority

The V1 rule authority order is:

1. `docs/design/werewolf-agent-v1-design.md`, especially Chapter 3
2. `config/rulesets/pre_witch_hunter_idiot_mixed.yaml`, after it has been synchronized with the design document
3. `RuleEngine` unit tests

External links, RAG content, examples, and model output are strategy references only. They must never override deterministic rules.

## Progress Discipline

`PROGRESS.md` is mandatory.

- Read it before editing.
- Keep exactly one active task.
- Do only the active task unless the user explicitly changes scope.
- Update it after every meaningful change.
- Record changed files, verification commands, open risks, and next steps.
- If implementation conflicts with the design document, stop and record the conflict. Do not silently invent a new rule.

## Development Boundaries

- `RuleEngine` owns identities, legal actions, night resolution, voting, exile, hunter shots, idiot reveal, sheriff badge, last words, deaths, and victory.
- LLM agents only propose actions, reasons, and public speech.
- RAG never answers base rules or adjudicates game state.
- Player agents must never see `moderator_full`, other players' private state, other players' `private_intent`, hidden identities, or forbidden night information.
- All state mutations must be represented as auditable events and reduced deterministically into `GameState`.

## Reflection Memory (LLM Prompt Layer)

Reflection / cross-game memory is owned by the LLM prompt layer, NOT the `RuleEngine`. The RuleEngine never reads or writes reflection entries.

- `ReflectionMemory` (`memory/reflection.py`) stores per-`(game_id, player_id, role, faction_won)` entries with a free-form `text` field.
- `_agent_reflection` (`agent_adapter.py`) calls LLM at game end with a **role-family-specific template** (good-side / wolf-side / hybrid) — not a generic prompt. The template emits section headers like `【投票错误】` / `【悍跳分析】` / `【保留的优点】` that downstream aggregation relies on.
- `build_agent_context` (`runtime/context.py`) injects three cross-game hint fields:
  - `reflection_memory_hints` (top 8 per-role entries, winning-first within priority tier);
  - `error_pattern_hint` (top-2 mistake categories + preserved-strength count aggregated from section headers, no extra LLM call);
  - `profile_memory_hint` (long-term ability stats, current-role win-rate).
- V1 board assigns roles randomly per game, so cross-game learning is primarily **cross-player same-role** (priority=2) and **cross-player same-faction** (priority=1), not same-player same-role. Hint ranking reflects this.

**Schema boundary**: the `reflections` PostgreSQL table (`entry_id, game_id, player_id, entry_json`) is NOT modified by LLM-prompt-layer changes. All reflection quality improvements go into `entry_json.text`; the column layout is stable. See `docs/design/werewolf-agent-v1-design.md` §10.

## Critical V1 Rules

- 12 players plus 1 judge.
- Roles: 4 werewolves, 3 villagers, seer, witch, hunter, idiot, hybrid.
- Seer sees hybrid as good.
- Witch has one antidote and one poison, cannot use both in the same night, and cannot self-save at any time.
- Hunter can shoot if killed by wolves or exiled; cannot shoot if poisoned by witch.
- Idiot reveals only when exiled, remains alive, can speak, loses vote, cannot receive badge, cannot be exiled again, and only counts as dead god after later wolf night kill.
- Hybrid chooses a master on first night, does not know master's role or faction, and wins with the master's original faction.
- Hybrid slaughter rule is conditional: if the master is good, wolves must eliminate the 3 villagers plus hybrid to win by villager slaughter; if the master is wolf, wolves only need to eliminate the 3 villagers.
- Sheriff can pass badge or tear badge after any death cause. If badge is torn, no sheriff remains for the game and later speech starts randomly.
- First tie enters PK speech and revote. Second tie means no exile and immediate night.
- Day flow:
  - D1: 天亮 → 警长竞选 (sheriff_registration → sheriff_speech → sheriff_withdraw → sheriff_vote) → 死讯广播 → 遗言 → 警长指定发言顺序 → 自由讨论。
  - D2+: 死讯广播 → 遗言 → 警长指定发言顺序 → 自由讨论。
  - D1 self-destruct during sheriff election: N1 deaths still broadcast before continuing (route_after_self_destruct → announce_deaths).

## Model Usage

Claude may use GLM-5.1 through the user's configured environment. Do not hardcode API keys or vendor-specific credentials. Keep provider/model choices in configuration and snapshots, not in core rules.

## File Modification Discipline

- When modifying a file, the module docstring (and class/function docstrings where applicable) must be updated to reflect the changes.
- The **修改日期** field in the module docstring must be updated to the current date for any non-trivial change.
- If the file lacks a docstring, add one following the template in the global rules (功能描述 / 作者 / 创建日期 / 修改日期 / 使用示例).
- Do not silently edit a file and leave stale documentation behind.

## Verification Before Done

Before claiming a task is complete:

1. Run the relevant tests or explain why they cannot run.
2. Check that `PROGRESS.md` is current.
3. Check that changed behavior still matches `docs/design/werewolf-agent-v1-design.md`.
4. Report remaining risks plainly.
