# -*- coding: utf-8 -*-
"""
验证语义修复 V2 只拒绝不受支持的公开声明及语义关系篡改。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-19
"""

from __future__ import annotations

import pytest

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    SpeechPlayerAction,
    TaskType,
)


def _context() -> AgentContext:
    return AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p02"],
        public_claim_ledger=[
            {"speaker": "p05", "text": "我认为p02是狼人"},
            {"speaker": "p06", "text": "我认为p02是狼人"},
        ],
    )


def _action(speech: str) -> SpeechPlayerAction:
    return SpeechPlayerAction(
        target_id="p02",
        speech=speech,
        reason="引用公开发言",
        confidence=0.5,
    )


def test_claim_keys_keep_speakers_independent_for_the_same_target() -> None:
    from werewolf_agent.evaluation.balance_public_claims import (
        public_claim_audit_keys,
    )

    claims, _ = public_claim_audit_keys(
        "p05声称p02是狼人；p06声称p02是狼人。",
        [("p05", "我认为p02是狼人"), ("p06", "我认为p02是狼人")],
    )

    assert len(claims) == 2
    assert {claim.target for claim in claims} == {"p02"}
    assert {claim.speaker_attribution for claim in claims} == {"p05", "p06"}


def test_semantic_repair_rejects_changed_existing_speaker_attribution() -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        validate_semantic_repair,
    )

    source = _action("p05声称p02是狼人，我怀疑p02。")
    changed = _action("p06声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(_context(), source, changed)

    assert result.accepted is False
    assert result.reason_codes == ("speaker_attribution_changed",)
    assert result.audit["speaker_attribution_preserved"] is False


def test_semantic_repair_rejects_removed_existing_speaker_attribution() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    source = _action("p05声称p02是狼人，我怀疑p02。")
    unattributed = _action("有人声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(_context(), source, unattributed)

    assert result.accepted is False
    assert result.reason_codes == (
        "unsupported_public_claim",
        "speaker_attribution_changed",
    )
    assert result.audit["speaker_attribution_preserved"] is False


def test_semantic_repair_allows_dropping_attributed_duplicate_when_missing_exists() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    source = _action(
        "p05声称p02是狼人，有人声称p02是狼人，我怀疑p02。"
    )
    negated_unattributed = _action("有人并未声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(_context(), source, negated_unattributed)

    assert result.accepted is False
    assert result.reason_codes == (
        "unsupported_public_claim",
        "negation_changed",
    )
    assert result.audit["speaker_attribution_preserved"] is True


def test_semantic_repair_rejects_changed_negation_relation() -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        validate_semantic_repair,
    )

    source = _action("p05声称p02是狼人，我怀疑p02。")
    changed = _action("p05并未声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(_context(), source, changed)

    assert result.accepted is False
    assert result.reason_codes == (
        "unsupported_public_claim",
        "negation_changed",
    )
    assert result.audit["unsupported_public_claim_count"] == 1
    assert result.audit["negation_preserved"] is False


def test_semantic_repair_matches_negation_to_the_exact_source_speaker() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "我认为p02是狼人"},
            {"speaker": "p06", "text": "我认为p02是狼人"},
        ],
    })
    source = _action(
        "p05声称p02是狼人，p06并未声称p02是狼人，我怀疑p02。"
    )
    changed = _action("p06声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, changed)

    assert result.accepted is False
    assert result.reason_codes == ("negation_changed",)
    assert result.audit["negation_preserved"] is False


def test_semantic_repair_does_not_compare_negation_across_speakers() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "我认为p02是狼人"},
            {"speaker": "p06", "text": "p02不是狼人"},
        ],
    })
    source = _action("p05声称p02是狼人，我怀疑p02。")
    changed = _action("p06并未声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, changed)

    assert result.accepted is False
    assert result.reason_codes == ("speaker_attribution_changed",)
    assert result.audit["negation_preserved"] is True


def test_semantic_repair_preserves_same_speaker_negation() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "我认为p02是狼人"},
            {"speaker": "p06", "text": "p02不是狼人"},
        ],
    })
    source = _action(
        "p05声称p02是狼人，p06并未声称p02是狼人，我怀疑p02。"
    )
    retained = _action("p06并未声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, retained)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["negation_preserved"] is True


def test_semantic_repair_rejects_affirmative_claim_from_negated_evidence() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "p02不是狼人"}],
    })
    source = _action("我怀疑p02。")
    final = _action("p05声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)
    assert result.audit["unsupported_public_claim_count"] == 1


def test_semantic_repair_rejects_negated_claim_from_affirmative_evidence() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "我认为p02是狼人"},
        ],
    })
    source = _action("我怀疑p02。")
    final = _action("p05并未声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)
    assert result.audit["unsupported_public_claim_count"] == 1


def test_semantic_repair_allows_negated_claim_from_negated_evidence() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "p02不是狼人"}],
    })
    source = _action("我怀疑p02。")
    final = _action("p05并未声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["unsupported_public_claim_count"] == 0


@pytest.mark.parametrize("negative_relation", ["不是", "并非", "不为"])
def test_semantic_repair_rejects_direct_attributed_polarity_flip(
    negative_relation: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p03", "text": "p05是狼人"},
            {"speaker": "p03", "text": f"p05{negative_relation}狼人"},
        ],
    })
    source = _action("p03声称p05是狼人，我怀疑p02。")
    final = _action(f"p03声称p05{negative_relation}狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is False
    assert "negation_changed" in result.reason_codes
    assert result.audit["unsupported_public_claim_count"] == 0


@pytest.mark.parametrize(
    ("source_text", "final_text"),
    [
        ("p04知道狼刀信息", "p04不知道狼刀信息"),
        ("p04不知道狼刀信息", "p04知道狼刀信息"),
    ],
)
def test_semantic_repair_rejects_night_info_polarity_flip(
    source_text: str,
    final_text: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p04", "text": "p04知道狼刀信息"},
            {"speaker": "p04", "text": "p04不知道狼刀信息"},
        ],
    })
    result = validate_semantic_repair(
        context,
        _action(source_text),
        _action(final_text),
    )

    assert result.accepted is False
    assert result.reason_codes == ("negation_changed",)
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_rejects_affirmative_self_role_from_denial() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "我没有说我是预言家"},
        ],
    })
    source = _action("我怀疑p02。")
    final = _action("p05声称自己是预言家，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)
    assert result.audit["unsupported_public_claim_count"] == 1


def test_semantic_repair_allows_denied_self_role_from_denial() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "我没有说我是预言家"},
        ],
    })
    source = _action("我怀疑p02。")
    final = _action("p05并未声称自己是预言家，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_rejects_denied_self_role_from_affirmation() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "我是预言家"}],
    })
    source = _action("我怀疑p02。")
    final = _action("p05并未声称自己是预言家，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)
    assert result.audit["unsupported_public_claim_count"] == 1


def test_semantic_repair_allows_affirmed_self_role_from_affirmation() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "我是预言家"}],
    })
    source = _action("我怀疑p02。")
    final = _action("p05声称自己是预言家，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_rejects_attributed_claim_laundered_by_ledger_speaker() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "p06声称p02是狼人"},
        ],
    })
    source = _action("我怀疑p02。")
    final = _action("p05声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)
    assert result.audit["unsupported_public_claim_count"] == 1


def test_semantic_repair_allows_explicit_first_person_attributed_claim() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "我声称p02是狼人"}],
    })
    source = _action("我怀疑p02。")
    final = _action("p05声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_merges_direct_and_first_person_evidence_by_clause() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "p06声称p02是狼人；我声称p02是狼人"},
        ],
    })
    source = _action("我怀疑p02。")
    final = _action("p05声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_scopes_first_person_denial_to_its_clause() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "我不知道情况；p02是狼人"},
        ],
    })
    source = _action("我怀疑p02。")
    final = _action("p05声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_keeps_same_clause_first_person_denial_negated() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "我没有说p02是狼人"}],
    })
    source = _action("我怀疑p02。")
    final = _action("p05并未声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_rejects_ledger_speaker_for_punctuated_report() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "p06声称，p02是狼人"}],
    })
    source = _action("我怀疑p02。")
    final = _action("p05声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)
    assert result.audit["unsupported_public_claim_count"] == 1


def test_semantic_repair_allows_reported_speaker_for_punctuated_report() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "p06声称，p02是狼人"}],
    })
    source = _action("我怀疑p02。")
    final = _action("p06声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_allows_prefixed_affirmative_self_role() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "其实我是预言家"}],
    })
    source = _action("我怀疑p02。")
    final = _action("p05声称自己是预言家，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_rejects_prefixed_affirmative_for_denied_self_role() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "其实我是预言家"}],
    })
    source = _action("我怀疑p02。")
    final = _action("p05并未声称自己是预言家，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)
    assert result.audit["unsupported_public_claim_count"] == 1


def test_semantic_repair_allows_prefixed_denied_self_role() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "其实我没有说我是预言家"},
        ],
    })
    source = _action("我怀疑p02。")
    final = _action("p05并未声称自己是预言家，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_rejects_prefixed_denial_for_affirmative_self_role() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": "其实我没有说我是预言家"},
        ],
    })
    source = _action("我怀疑p02。")
    final = _action("p05声称自己是预言家，我怀疑p02。")

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)
    assert result.audit["unsupported_public_claim_count"] == 1


@pytest.mark.parametrize("delimiter", ("，", ",", "：", ":"))
def test_semantic_repair_preserves_reported_attribution_across_soft_delimiters(
    delimiter: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": f"p06声称{delimiter}p02是狼人"},
        ],
    })
    source = _action("我怀疑p02。")

    reported = validate_semantic_repair(
        context,
        source,
        _action("p06声称p02是狼人，我怀疑p02。"),
    )
    ledger_speaker = validate_semantic_repair(
        context,
        source,
        _action("p05声称p02是狼人，我怀疑p02。"),
    )

    assert reported.accepted is True
    assert reported.reason_codes == ()
    assert ledger_speaker.accepted is False
    assert ledger_speaker.reason_codes == ("unsupported_public_claim",)


def test_semantic_repair_stops_reported_attribution_at_sentence_terminator() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "p06声称。p02是狼人"}],
    })
    source = _action("我怀疑p02。")

    reported = validate_semantic_repair(
        context,
        source,
        _action("p06声称p02是狼人，我怀疑p02。"),
    )
    ledger_speaker = validate_semantic_repair(
        context,
        source,
        _action("p05声称p02是狼人，我怀疑p02。"),
    )

    assert reported.accepted is False
    assert reported.reason_codes == ("unsupported_public_claim",)
    assert ledger_speaker.accepted is True
    assert ledger_speaker.reason_codes == ()


def test_semantic_repair_does_not_bridge_soft_delimiter_across_terminator() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": "p06声称：。p02是狼人"}],
    })
    source = _action("我怀疑p02。")

    reported = validate_semantic_repair(
        context,
        source,
        _action("p06声称p02是狼人，我怀疑p02。"),
    )
    ledger_speaker = validate_semantic_repair(
        context,
        source,
        _action("p05声称p02是狼人，我怀疑p02。"),
    )

    assert reported.accepted is False
    assert reported.reason_codes == ("unsupported_public_claim",)
    assert ledger_speaker.accepted is True
    assert ledger_speaker.reason_codes == ()


@pytest.mark.parametrize(
    ("delimiter", "quoted_claim"),
    [
        ("：", "“p02是狼人”"),
        ("，", "“p02是狼人”"),
        (":", '"p02是狼人"'),
        (",", '"p02是狼人"'),
    ],
)
def test_semantic_repair_preserves_quoted_reported_target_attribution(
    delimiter: str,
    quoted_claim: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": f"p06声称{delimiter}{quoted_claim}"},
        ],
    })
    source = _action("我怀疑p02。")

    reported = validate_semantic_repair(
        context,
        source,
        _action("p06声称p02是狼人，我怀疑p02。"),
    )
    ledger_speaker = validate_semantic_repair(
        context,
        source,
        _action("p05声称p02是狼人，我怀疑p02。"),
    )

    assert reported.accepted is True
    assert reported.reason_codes == ()
    assert ledger_speaker.accepted is False
    assert ledger_speaker.reason_codes == ("unsupported_public_claim",)


@pytest.mark.parametrize(
    ("delimiter", "quoted_claim"),
    [
        ("：", "“我是预言家”"),
        ("，", "“我是预言家”"),
        (":", '"我是预言家"'),
        (",", '"我是预言家"'),
    ],
)
def test_semantic_repair_preserves_quoted_reported_self_role_attribution(
    delimiter: str,
    quoted_claim: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": f"p06声称{delimiter}{quoted_claim}"},
        ],
    })
    source = _action("我怀疑p02。")

    reported = validate_semantic_repair(
        context,
        source,
        _action("p06声称自己是预言家，我怀疑p02。"),
    )
    ledger_speaker = validate_semantic_repair(
        context,
        source,
        _action("p05声称自己是预言家，我怀疑p02。"),
    )

    assert reported.accepted is True
    assert reported.reason_codes == ()
    assert ledger_speaker.accepted is False
    assert ledger_speaker.reason_codes == ("unsupported_public_claim",)


@pytest.mark.parametrize(
    ("delimiter", "quoted_claim"),
    [
        ("：", "“p02不是狼人”"),
        ("，", "“p02不是狼人”"),
        (":", '"p02不是狼人"'),
        (",", '"p02不是狼人"'),
    ],
)
def test_semantic_repair_preserves_quoted_reported_target_polarity(
    delimiter: str,
    quoted_claim: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": f"p06声称{delimiter}{quoted_claim}"},
        ],
    })
    source = _action("我怀疑p02。")

    negated = validate_semantic_repair(
        context,
        source,
        _action("p06并未声称p02是狼人，我怀疑p02。"),
    )
    affirmative = validate_semantic_repair(
        context,
        source,
        _action("p06声称p02是狼人，我怀疑p02。"),
    )
    ledger_speaker = validate_semantic_repair(
        context,
        source,
        _action("p05并未声称p02是狼人，我怀疑p02。"),
    )

    assert negated.accepted is True
    assert negated.reason_codes == ()
    assert affirmative.accepted is False
    assert affirmative.reason_codes == ("unsupported_public_claim",)
    assert ledger_speaker.accepted is False
    assert ledger_speaker.reason_codes == ("unsupported_public_claim",)


@pytest.mark.parametrize(
    ("delimiter", "quoted_claim"),
    [
        ("：", "“我不是预言家”"),
        ("，", "“我不是预言家”"),
        (":", '"我不是预言家"'),
        (",", '"我不是预言家"'),
    ],
)
def test_semantic_repair_preserves_quoted_reported_self_role_polarity(
    delimiter: str,
    quoted_claim: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": f"p06声称{delimiter}{quoted_claim}"},
        ],
    })
    source = _action("我怀疑p02。")

    negated = validate_semantic_repair(
        context,
        source,
        _action("p06并未声称自己是预言家，我怀疑p02。"),
    )
    affirmative = validate_semantic_repair(
        context,
        source,
        _action("p06声称自己是预言家，我怀疑p02。"),
    )
    ledger_speaker = validate_semantic_repair(
        context,
        source,
        _action("p05并未声称自己是预言家，我怀疑p02。"),
    )

    assert negated.accepted is True
    assert negated.reason_codes == ()
    assert affirmative.accepted is False
    assert affirmative.reason_codes == ("unsupported_public_claim",)
    assert ledger_speaker.accepted is False
    assert ledger_speaker.reason_codes == ("unsupported_public_claim",)


@pytest.mark.parametrize("quoted_claim", ("p02并非狼人", "p02不为狼人"))
def test_semantic_repair_preserves_quoted_reported_target_negation_vocabulary(
    quoted_claim: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": f"p06声称：“{quoted_claim}”"},
        ],
    })
    source = _action("我怀疑p02。")

    negated = validate_semantic_repair(
        context,
        source,
        _action("p06并未声称p02是狼人，我怀疑p02。"),
    )
    affirmative = validate_semantic_repair(
        context,
        source,
        _action("p06声称p02是狼人，我怀疑p02。"),
    )
    ledger_speaker = validate_semantic_repair(
        context,
        source,
        _action("p05并未声称p02是狼人，我怀疑p02。"),
    )

    assert negated.accepted is True
    assert negated.reason_codes == ()
    assert affirmative.accepted is False
    assert affirmative.reason_codes == ("unsupported_public_claim",)
    assert ledger_speaker.accepted is False
    assert ledger_speaker.reason_codes == ("unsupported_public_claim",)


@pytest.mark.parametrize(
    "quoted_claim",
    ("我并非预言家", "我不为预言家", "我没有说我是预言家"),
)
def test_semantic_repair_preserves_quoted_reported_self_role_negation_vocabulary(
    quoted_claim: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [
            {"speaker": "p05", "text": f"p06声称：“{quoted_claim}”"},
        ],
    })
    source = _action("我怀疑p02。")

    negated = validate_semantic_repair(
        context,
        source,
        _action("p06并未声称自己是预言家，我怀疑p02。"),
    )
    affirmative = validate_semantic_repair(
        context,
        source,
        _action("p06声称自己是预言家，我怀疑p02。"),
    )
    ledger_speaker = validate_semantic_repair(
        context,
        source,
        _action("p05并未声称自己是预言家，我怀疑p02。"),
    )

    assert negated.accepted is True
    assert negated.reason_codes == ()
    assert affirmative.accepted is False
    assert affirmative.reason_codes == ("unsupported_public_claim",)
    assert ledger_speaker.accepted is False
    assert ledger_speaker.reason_codes == ("unsupported_public_claim",)


@pytest.mark.parametrize(
    ("prefix", "ledger_text", "final_text", "reason_codes"),
    [
        ("其实", "我是预言家", "p05声称自己是预言家，我怀疑p02。", ()),
        ("其实", "我是预言家", "p05并未声称自己是预言家，我怀疑p02。", ("unsupported_public_claim",)),
        ("其实", "我没有说我是预言家", "p05并未声称自己是预言家，我怀疑p02。", ()),
        ("其实", "我没有说我是预言家", "p05声称自己是预言家，我怀疑p02。", ("unsupported_public_claim",)),
        ("不过", "我是预言家", "p05声称自己是预言家，我怀疑p02。", ()),
        ("不过", "我是预言家", "p05并未声称自己是预言家，我怀疑p02。", ("unsupported_public_claim",)),
        ("不过", "我没有说我是预言家", "p05并未声称自己是预言家，我怀疑p02。", ()),
        ("不过", "我没有说我是预言家", "p05声称自己是预言家，我怀疑p02。", ("unsupported_public_claim",)),
    ],
)
def test_semantic_repair_bounds_discourse_prefix_polarity(
    prefix: str,
    ledger_text: str,
    final_text: str,
    reason_codes: tuple[str, ...],
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": f"{prefix}{ledger_text}"}],
    })

    result = validate_semantic_repair(
        context,
        _action("我怀疑p02。"),
        _action(final_text),
    )

    assert result.reason_codes == reason_codes
    assert result.accepted is (not reason_codes)


@pytest.mark.parametrize("ledger_text", ("随后我是预言家", "p06说我是预言家"))
def test_semantic_repair_rejects_unbounded_or_third_party_prefixes(
    ledger_text: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = _context().model_copy(update={
        "public_claim_ledger": [{"speaker": "p05", "text": ledger_text}],
    })

    result = validate_semantic_repair(
        context,
        _action("我怀疑p02。"),
        _action("p05声称自己是预言家，我怀疑p02。"),
    )

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)


def test_claim_key_records_prefix_negation_without_flipping_double_negation() -> None:
    from werewolf_agent.evaluation.balance_public_claims import (
        public_claim_audit_keys,
    )

    negated, _ = public_claim_audit_keys(
        "不能说p05声称p02是狼人。", [("p05", "我认为p02是狼人")]
    )
    assert {claim.negated for claim in negated} == {True}

    assertive, _ = public_claim_audit_keys(
        "不能否认p05声称p02是狼人。", [("p05", "我认为p02是狼人")]
    )
    assert {claim.negated for claim in assertive} == {False}


def test_semantic_repair_allows_supported_missing_attribution_completion() -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        validate_semantic_repair,
    )

    source = _action("有人声称p02是狼人，我怀疑p02。")
    attributed = _action("p05声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(_context(), source, attributed)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["speaker_attribution_preserved"] is True
    assert result.audit["negation_preserved"] is True
    assert result.audit["introduced_claim_count"] == 0
    assert result.audit["semantic_gate_version"] == 2
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_rejects_unsupported_public_claim() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    source = _action("p05声称p02是狼人，我怀疑p02。")
    unsupported = _action(
        "p05声称p02是狼人，p07声称自己是猎人，我怀疑p02。"
    )

    result = validate_semantic_repair(_context(), source, unsupported)

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)
    assert result.audit["unsupported_public_claim_count"] == 1


def test_semantic_repair_reports_all_rejection_reasons_in_stable_order() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    source = _action(
        "p05声称p02是狼人，p06声称p03是预言家，我怀疑p02。"
    )
    changed = _action(
        "p05并未声称p02是狼人，p07声称p03是预言家，我怀疑p02。"
    )

    result = validate_semantic_repair(_context(), source, changed)

    assert result.accepted is False
    assert result.reason_codes == (
        "unsupported_public_claim",
        "speaker_attribution_changed",
        "negation_changed",
    )
    assert result.audit["rejection_reason_codes"] == list(result.reason_codes)


@pytest.mark.parametrize(
    ("reason_code", "safe_explanation"),
    [
        ("unsupported_public_claim", "公开证据"),
        ("speaker_attribution_changed", "说话人归属"),
        ("negation_changed", "否定关系"),
    ],
)
def test_semantic_repair_fixed_messages_explain_each_reason_safely(
    reason_code: str,
    safe_explanation: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        semantic_repair_correction_hint,
        semantic_repair_rejection_message,
    )

    message = semantic_repair_rejection_message((reason_code,))
    hint = semantic_repair_correction_hint((reason_code,))

    assert safe_explanation in message
    assert safe_explanation in hint
    assert hint.startswith("请")


@pytest.mark.parametrize(
    ("reason_codes", "included_terms", "excluded_terms"),
    [
        (
            ("unsupported_public_claim", "negation_changed"),
            ("公开证据", "否定关系"),
            ("说话人归属",),
        ),
        (
            ("speaker_attribution_changed",),
            ("说话人归属",),
            ("公开证据", "否定关系"),
        ),
    ],
)
def test_semantic_repair_fixed_messages_disclose_only_requested_reasons(
    reason_codes: tuple[str, ...],
    included_terms: tuple[str, ...],
    excluded_terms: tuple[str, ...],
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        semantic_repair_correction_hint,
        semantic_repair_rejection_message,
    )

    text = " ".join((
        semantic_repair_rejection_message(reason_codes),
        semantic_repair_correction_hint(reason_codes),
    ))

    assert all(term in text for term in included_terms)
    assert all(term not in text for term in excluded_terms)
    assert "SPEECH_SENTINEL_SHOULD_NOT_LEAK" not in text
    assert "p08是狼人" not in text
    assert "provider_error" not in text
    assert "不得新增任何事实" not in text
    assert "保留全部论点" not in text
