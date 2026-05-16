# Phase 2 Plan - Runtime

## Goal

Implement orchestration around the deterministic RuleEngine.

## Deliverables

- main game graph skeleton;
- checkpoint persistence;
- pause/resume;
- replay timeline;
- night wolf discussion and consensus nodes;
- first-night hybrid master node;
- sheriff-election split nodes;
- day discussion and vote nodes;
- conditional edges for self-destruct, tie, no exile, sheriff death, and victory.

## Required Boundary

Runtime must never adjudicate rules in natural language. It calls RuleEngine and routes based on deterministic results.

## Done Means

- A scripted non-LLM game can run through setup, night, day, voting, and victory checks.
- Event log can replay into the same `GameState`.
- Player-visible contexts are not built from full private history.
