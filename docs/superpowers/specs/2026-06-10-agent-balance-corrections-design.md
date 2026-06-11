# Agent Balance Corrections Design

## Scope

Fix the implementation-layer biases identified from the five saved games while
leaving model routing, persona/seat rotation, game rules, and large batch
experiments unchanged.

## Design

1. Speech parsing must preserve model meaning. Post-processing may normalize
   structure, but it must not invent a suspicion or vote target. Vote reasons
   that explicitly name a different legal player than the mapped choice are
   replaced with a neutral target-consistent explanation.
2. Role-claim conflicts are capacity-aware. A conflict exists only when unique
   claimers exceed the configured role count; the default distribution is used
   when no ruleset-specific capacity map is supplied.
3. Sheriff prompts use public claims and the assigned fake-seer role, not hidden
   roles of all candidates. A true seer or assigned fake seer stays in the race
   after making the role-critical commitment.
4. Hybrid prompts expose the master id but never the master's hidden faction,
   including after the master dies. Cross-game memory still records the faction
   for moderator-side evaluation only.
5. Speech validation is intent-aware. Informational, defensive, and seer-support
   speech is not forced to include an unrelated suspicion and vote. Filler
   speech remains invalid.
6. Evaluation reads the event names and payload shapes actually emitted by the
   runtime. Public daily summaries are generated from public transcript data and
   are not marked public when they contain player-private interpretation.

## Verification

Each behavior receives a regression test that fails before implementation.
Targeted suites run after every change, followed by the full test suite.
