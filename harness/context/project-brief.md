# Project Brief

Werewolf Agent V1 is a multi-agent engineering training project built around 12-player Werewolf: `pre_witch_hunter_idiot_mixed`.

The goal is not a simple game demo. The project is a controlled environment for learning:

- deterministic rule engines;
- multi-agent orchestration;
- private/public information boundaries;
- structured LLM output;
- persona routing;
- model routing;
- cognitive context compression;
- RAG for strategy and examples;
- memory and replay;
- evaluation.

## V1 Scope

V1 uses:

- 12 players and 1 judge;
- 4 werewolves;
- 3 villagers;
- seer;
- witch;
- hunter;
- idiot;
- hybrid.

V1 should first build a deterministic rules loop. Agents, RAG, memory, and UI must not be allowed to change rule truth.

## Engineering Priority

1. Rules must be correct.
2. Private information must not leak.
3. State must be replayable from events.
4. Agents must use schema-constrained outputs.
5. Progress must be tracked in `PROGRESS.md`.
