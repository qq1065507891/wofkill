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
- Day flow is death announcement, last words, then first-day sheriff election.

## Model Usage

Claude may use GLM-5.1 through the user's configured environment. Do not hardcode API keys or vendor-specific credentials. Keep provider/model choices in configuration and snapshots, not in core rules.

## Verification Before Done

Before claiming a task is complete:

1. Run the relevant tests or explain why they cannot run.
2. Check that `PROGRESS.md` is current.
3. Check that changed behavior still matches `docs/design/werewolf-agent-v1-design.md`.
4. Report remaining risks plainly.
