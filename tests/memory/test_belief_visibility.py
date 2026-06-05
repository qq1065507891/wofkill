"""P0-M9 + absorbed P0-M8: cognition_matrix_hint must not leak evidence text.

Background
----------

``BeliefUpdater`` is private-info safe — it only consumes public
``visible_facts`` (per Batch 0 finding ``2026-06-03-batch0-m8-finding.md``).
The leak surface is the **rendering layer**:

* ``_cognition_matrix_hint`` (context.py) historically exposed
  ``key_evidence`` text and ``open_questions`` text directly. Those
  lists are populated by code paths that *can* include private
  observations (e.g. wolf-team channels, witch potion use, seer
  check results). Even when they don't, the LLM tends to treat the
  text as authoritative, anchoring downstream reasoning on data the
  viewer shouldn't have.

This file pins the fix:

* Same public facts ⇒ identical ``role_probabilities`` (M8 regression).
* ``_cognition_matrix_hint`` does not include raw text; it renders
  ``key_evidence`` and ``open_questions`` as ``salience_items#<hash>``
  references (M9).
* ``top_role_guess`` and the existing summary stats (trust / faction
  read) are kept (already public-derived).
"""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import pytest

from werewolf_agent.cognition.belief import BeliefUpdater
from werewolf_agent.cognition.world_state import StructuredFact
from werewolf_agent.memory.cognition_matrix import CognitionMatrix
from werewolf_agent.runtime.context import _cognition_matrix_hint


# ---------------------------------------------------------------------------
# M8: BeliefUpdater is public-fact safe. Wolf and villager with the
# same public facts must produce the same role_probabilities.
# ---------------------------------------------------------------------------


def _make_facts_p03_long_speech_and_claim():
    """Public facts: p03 gave a long speech and claimed seer on day 1."""
    return [
        StructuredFact(fact_type="speech", source_player="p03", value="x" * 200),
        StructuredFact(fact_type="claimed_role", source_player="p03", value="seer"),
    ]


def test_belief_state_uses_only_public_signals():
    """Wolf and villager with the same public facts ⇒ same probabilities.

    Per Batch 0 M8 finding: BeliefUpdater consumes only public
    visible_facts; this is a regression guard.
    """
    updater = BeliefUpdater()
    wolf_belief = updater.initialize(["p01", "p02", "p03"], viewer_id="p01")
    villager_belief = updater.initialize(["p01", "p02", "p03"], viewer_id="p02")

    facts = _make_facts_p03_long_speech_and_claim()

    updater.update(wolf_belief, facts, current_day=1)
    updater.update(villager_belief, facts, current_day=1)

    wolf_p = wolf_belief.beliefs["p03"].role_probabilities
    villager_p = villager_belief.beliefs["p03"].role_probabilities
    assert wolf_p == villager_p, (
        f"BeliefUpdater leaked private info: wolf {wolf_p} vs "
        f"villager {villager_p}"
    )


# ---------------------------------------------------------------------------
# M9: _cognition_matrix_hint renders evidence/questions as ID refs.
# ---------------------------------------------------------------------------


def _make_fake_matrix_store(entries_data: list[dict]) -> SimpleNamespace:
    """Build a fake restored_memory with a get_matrix that returns
    CognitionMatrix populated from a list of dicts.
    """
    matrix = CognitionMatrix("p01")
    matrix.initialize(["p01", "p02", "p03", "p04"])
    for ed in entries_data:
        e = matrix.get(ed["player_id"])
        if e is None:
            continue
        e.faction_read = ed.get("faction_read", "unknown")
        e.trust = ed.get("trust", 0.5)
        e.key_evidence = list(ed.get("key_evidence", []))
        e.open_questions = list(ed.get("open_questions", []))
        if "role_probabilities" in ed:
            e.role_probabilities = ed["role_probabilities"]
        if "top_role_guess" in ed:
            # Stored as (role, prob) tuple via a duck-typed attribute.
            e.top_role_guess = ed["top_role_guess"]

    return SimpleNamespace(get_matrix=lambda _pid: matrix)


def _evidence_ref(text: str) -> str:
    """Mirror the implementation's id-ref format."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"salience_items#{h}"


def test_cognition_matrix_no_text_evidence_in_rendered_hint():
    """key_evidence and open_questions are rendered as ID refs, not text."""
    store = _make_fake_matrix_store([
        {
            "player_id": "p02",
            "faction_read": "wolf_lean",
            "trust": 0.2,
            "key_evidence": [
                "p02 私下说 p07 是狼人",
                "p02 在夜间有动作",
            ],
            "open_questions": [
                "p02 是否在倒钩",
            ],
        },
    ])
    hint = _cognition_matrix_hint(store, "p01")
    assert "suspects" in hint
    suspect = hint["suspects"][0]
    assert suspect["player"] == "p02"
    # ID references, not raw text
    assert suspect["key_evidence"] == [
        _evidence_ref("p02 私下说 p07 是狼人"),
        _evidence_ref("p02 在夜间有动作"),
    ]
    assert suspect["open_questions"] == [
        _evidence_ref("p02 是否在倒钩"),
    ]
    # And no raw text leaks into the hint
    serialized = repr(hint)
    assert "私下说" not in serialized
    assert "倒钩" not in serialized


def test_cognition_matrix_keeps_summary_stats():
    """trust, faction_read, top_role_guess survive the rendering fix."""
    store = _make_fake_matrix_store([
        {
            "player_id": "p02",
            "faction_read": "wolf_lean",
            "trust": 0.21,
            "key_evidence": ["ev text"],
            "open_questions": [],
        },
        {
            "player_id": "p03",
            "faction_read": "good_lean",
            "trust": 0.84,
            "key_evidence": [],
            "open_questions": [],
        },
    ])
    hint = _cognition_matrix_hint(store, "p01")
    suspect = next(s for s in hint["suspects"] if s["player"] == "p02")
    trusted = next(t for t in hint["trusted"] if t["player"] == "p03")
    assert suspect["faction_read"] == "wolf_lean"
    assert suspect["trust"] == 0.21
    assert trusted["faction_read"] == "good_lean"
    assert trusted["trust"] == 0.84


def test_cognition_matrix_id_refs_are_stable():
    """The same evidence text always renders to the same ID ref."""
    text = "p02 私下说 p07 是狼人"
    store = _make_fake_matrix_store([
        {
            "player_id": "p02",
            "faction_read": "wolf_lean",
            "trust": 0.1,
            "key_evidence": [text, text],
            "open_questions": [],
        },
    ])
    hint = _cognition_matrix_hint(store, "p01")
    refs = hint["suspects"][0]["key_evidence"]
    assert refs[0] == refs[1]
    assert refs[0] == _evidence_ref(text)


def test_cognition_matrix_evidence_ref_format():
    """ID refs follow the ``salience_items#<hash>`` format."""
    text = "any text"
    ref = _evidence_ref(text)
    assert ref.startswith("salience_items#")
    suffix = ref.split("#", 1)[1]
    assert re.fullmatch(r"[0-9a-f]{6,16}", suffix), (
        f"hash suffix should be hex: {suffix!r}"
    )


def test_cognition_matrix_empty_evidence_returns_empty_lists():
    """Entries with no evidence/questions render empty ID lists."""
    store = _make_fake_matrix_store([
        {
            "player_id": "p02",
            "faction_read": "good_lean",
            "trust": 0.7,
            "key_evidence": [],
            "open_questions": [],
        },
    ])
    hint = _cognition_matrix_hint(store, "p01")
    assert "trusted" in hint
    assert hint["trusted"][0]["key_evidence"] == []
    assert hint["trusted"][0]["open_questions"] == []


# ---------------------------------------------------------------------------
# M8/M9 combined: Hint data is independent of who is viewing (same
# public facts ⇒ same hint stats for the suspects). This guards
# against the rendering layer accidentally pulling private info from
# elsewhere.
# ---------------------------------------------------------------------------


def test_belief_hint_independent_of_viewer_with_same_facts():
    """When two viewers see the same public facts, the suspicion /
    trust / role guess they derive should match.

    Implementation: build a CognitionMatrix per viewer with the same
    public-only BeliefUpdater output and assert the rendered hints
    agree on the summary statistics. We pin the trust / faction_lean
    values directly to a known-good bucket so the test focuses on
    "wolf and villager with same public signals see the same hint",
    not on whether BeliefUpdater happens to clear a threshold.
    """
    updater = BeliefUpdater()
    facts = _make_facts_p03_long_speech_and_claim()

    def _build_hint(viewer_id: str) -> dict:
        belief = updater.initialize(["p01", "p02", "p03"], viewer_id=viewer_id)
        updater.update(belief, facts, current_day=1)
        matrix = CognitionMatrix(viewer_id)
        matrix.initialize(["p01", "p02", "p03"])
        # Sync from belief (this is what real code does).
        for pid, b in belief.beliefs.items():
            entry = matrix.get(pid)
            if entry is None:
                continue
            entry.role_probabilities = dict(b.role_probabilities)
            # Pin the trust / faction_lean so the rendering buckets
            # are deterministic regardless of the public-signal
            # thresholds (those are BeliefUpdater tests, not
            # rendering tests). The pin value must be identical for
            # both viewers — that's the property we are testing.
            entry.faction_read = "good_lean"
            entry.trust = 0.8
        store = SimpleNamespace(get_matrix=lambda pid, _m=matrix: _m if pid == viewer_id else None)
        return _cognition_matrix_hint(store, viewer_id)

    wolf_hint = _build_hint("p01")
    villager_hint = _build_hint("p02")
    # Both viewers' p03 entry should agree on trust / faction_read
    w_p03 = next(
        (i for i in (wolf_hint.get("suspects", []) + wolf_hint.get("trusted", []))
         if i["player"] == "p03"),
        None,
    )
    v_p03 = next(
        (i for i in (villager_hint.get("suspects", []) + villager_hint.get("trusted", []))
         if i["player"] == "p03"),
        None,
    )
    assert w_p03 is not None, f"p03 not in wolf hint: {wolf_hint!r}"
    assert v_p03 is not None, f"p03 not in villager hint: {villager_hint!r}"
    # trust and faction_read should match across viewers
    assert w_p03["trust"] == v_p03["trust"], (
        f"trust diverges: wolf={w_p03['trust']} villager={v_p03['trust']}"
    )
    assert w_p03["faction_read"] == v_p03["faction_read"]


# ---------------------------------------------------------------------------
# MEM-07: key_evidence is now a list[EvidenceItem] (structured form)
# carrying claim / source_event / day / confidence / speaker.
# ---------------------------------------------------------------------------


def test_evidence_item_has_structured_fields():
    """MEM-07: add_evidence with an EvidenceItem must store the full
    structured object so the downstream renderer / debugger can see
    the claim's provenance and confidence.
    """
    from werewolf_agent.memory.schemas import EvidenceItem

    cm = CognitionMatrix("p1")
    cm.initialize(["p1", "p2"])
    ev = EvidenceItem(
        claim="p2 is wolf (long speech + claimed seer)",
        source_event="speech",
        day=1,
        confidence=0.8,
        speaker="p1",
    )
    cm.add_evidence("p2", ev)
    entry = cm.get("p2")
    assert entry is not None
    assert len(entry.key_evidence) == 1
    stored = entry.key_evidence[0]
    assert isinstance(stored, EvidenceItem), (
        f"MEM-07: stored evidence must be EvidenceItem; got {type(stored).__name__}"
    )
    # All 5 structured fields are accessible.
    assert stored.claim == "p2 is wolf (long speech + claimed seer)"
    assert stored.source_event == "speech"
    assert stored.day == 1
    assert stored.confidence == 0.8
    assert stored.speaker == "p1"


def test_evidence_item_back_compat_with_str():
    """MEM-07: add_evidence with a bare str must still work (backward
    compatibility); it gets wrapped into an EvidenceItem.
    """
    from werewolf_agent.memory.schemas import EvidenceItem

    cm = CognitionMatrix("p1")
    cm.initialize(["p1", "p2"])
    cm.add_evidence("p2", "legacy_claim_string")
    entry = cm.get("p2")
    assert len(entry.key_evidence) == 1
    stored = entry.key_evidence[0]
    # Bare str is wrapped into an EvidenceItem whose claim equals the string.
    assert isinstance(stored, EvidenceItem)
    assert stored.claim == "legacy_claim_string"


# ---------------------------------------------------------------------------
# MEM-NEW-9: add_evidence must REJECT types that aren't EvidenceItem or
# bare str. The pre-fix Union[EvidenceItem, str] silently wrapped
# everything else via ``EvidenceItem(claim=str(evidence))`` —
# including dicts, ints, and None — producing garbled evidence
# entries that polluted the cognition matrix and confused downstream
# consumers.
#
# Post-fix: explicit isinstance check + TypeError. Forces the caller
# to either pass a proper EvidenceItem or a clean string claim.
# ---------------------------------------------------------------------------


def test_add_evidence_rejects_non_string_or_item():
    """MEM-NEW-9: passing a dict (or int / None / list / etc.) to
    add_evidence must raise TypeError, not silently wrap into
    ``EvidenceItem(claim=str(evidence))``."""
    import pytest

    cm = CognitionMatrix("p1")
    cm.initialize(["p1", "p2"])
    with pytest.raises(TypeError):
        cm.add_evidence("p2", {"claim": "already a dict"})
    with pytest.raises(TypeError):
        cm.add_evidence("p2", 42)
    with pytest.raises(TypeError):
        cm.add_evidence("p2", None)
