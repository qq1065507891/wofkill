# Architecture Boundaries

## RuleEngine

Owns deterministic truth:

- role assignment;
- legal actions;
- night resolution;
- witch limits;
- seer results;
- hunter trigger;
- idiot reveal;
- voting and tie;
- exile;
- last words;
- sheriff election, badge passing, badge tearing;
- death records;
- victory.

## Runtime

Owns orchestration:

- phase transitions;
- node execution;
- checkpointing;
- pause/resume;
- replay;
- event reduction into state.

Runtime must call RuleEngine for rule decisions.

## Agents

Agents produce proposals:

- action choice;
- target;
- speech;
- private intent snapshot;
- confidence;
- reason.

Agents must never mutate game truth directly.

## RAG And Memory

RAG and memory help agents play better, but they do not own rules.

Long-term vector memory stores reflections and lessons. Relationship logic, votes, claims, and attack/defense edges belong in structured tables or graphs.

## Visibility

Every event must have a visibility label. Player contexts are built from allowed visibility only.

Allowed view modes:

- `public`;
- `player_view`;
- `moderator_full`.

Player agents cannot receive `moderator_full`.
