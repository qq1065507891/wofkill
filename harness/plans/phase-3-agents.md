# Phase 3 Plan - Agents

## Goal

Add LLM agents after the rule loop is deterministic.

## Deliverables

- player agent interface;
- judge agent interface;
- model router gateway for Claude/GLM and other profiles;
- schema-constrained action outputs;
- private-intent separation;
- illegal-output retry and fallback;
- persona router integration.

## Agent Output Contract

Every player action must be parseable and schema-valid:

- `action_type`;
- `target_id` when required;
- `private_intent` for deceptive or strategic tasks;
- `speech`;
- `reason`;
- `confidence`.

RuleEngine decides whether the action is legal.

## Done Means

- Agents cannot choose illegal targets.
- Private intent is not written to public timeline.
- Model provider details are isolated behind configuration.
