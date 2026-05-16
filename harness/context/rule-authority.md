# Rule Authority

## Authority Order

1. `docs/design/werewolf-agent-v1-design.md`, Chapter 3.
2. `config/rulesets/pre_witch_hunter_idiot_mixed.yaml`, after synchronization.
3. `RuleEngine` unit tests.

If these conflict, stop and update `PROGRESS.md`. Do not silently choose a rule.

## RAG Boundary

RAG may provide:

- high-level strategy;
- examples from high-end games;
- public speech patterns;
- review lessons;
- tactical references.

RAG must not provide:

- base rules;
- role-skill truth;
- victory truth;
- current live state;
- hidden identities;
- final adjudication.

## Critical Rules To Preserve

- Hybrid is seen as good by seer.
- Hybrid wins with the original faction of the first-night master.
- Hybrid master good: wolves must eliminate 3 villagers plus hybrid for villager-side slaughter.
- Hybrid master wolf: wolves only need to eliminate 3 villagers for villager-side slaughter.
- Idiot reveal is not death. Revealed idiot is alive, cannot vote, can speak, cannot receive badge, cannot be exiled again.
- Witch cannot self-save at any time.
- Witch cannot use antidote and poison in the same night.
- Hunter cannot shoot if poisoned.
- Sheriff may pass or tear badge after death by any cause.
- Torn badge means no sheriff for the rest of the game.
- Second tie means no exile and immediate night.
