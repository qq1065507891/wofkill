# Phase 1 Plan - RuleEngine

## Goal

Build the deterministic V1 rules loop before LLM agents are trusted.

## Deliverables

- synchronized `pre_witch_hunter_idiot_mixed.yaml`;
- core enums and domain models;
- event model and reducer;
- legal action generator;
- night resolution;
- voting and tie resolution;
- exile and post-exile skills;
- sheriff election and badge transfer;
- last-words logic;
- victory logic;
- tests for critical rules.

## Required Order

1. Synchronize YAML with the design document.
2. Define core data model.
3. Write tests for the high-risk rules.
4. Implement the minimum RuleEngine to pass tests.
5. Add replay/reducer checks.

## High-Risk Tests

- hybrid master good vs wolf slaughter condition;
- seer checks hybrid as good;
- witch cannot self-save;
- witch cannot use both potions in one night;
- hunter poisoned cannot shoot;
- idiot reveal state;
- revealed idiot cannot be exiled again;
- sheriff death by each cause can pass or tear badge;
- torn badge means no sheriff;
- first tie PK and second tie no exile;
- first-night night deaths have last words, later night deaths do not;
- day death announcement, last words, first-day sheriff election order.

## Done Means

- Tests pass.
- `PROGRESS.md` records changed files and verification.
- No LLM/RAG path can affect RuleEngine decisions.
