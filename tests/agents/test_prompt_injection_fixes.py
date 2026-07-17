"""Tests for M4 prompt-injection audit fixes.

Tasks 1, 2, 3, 4, 7, 8 all live in this module — one per issue in
``docs/superpowers/plans/2026-06-09-prompt-injection-audit-fixes.md``.

The first test (M4-1) was the only one in scope for the initial commit;
later tasks will add their own tests below.
"""

from __future__ import annotations


def test_reflection_hints_slice_uses_card_budget_3() -> None:
    """Reflection live prompt rendering must cap cards at the V2 card budget."""
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.schemas import AgentContext, TaskType

    # Build 10 reflection hints.
    hints = [
        {
            "role": "seer",
            "result": "胜" if i % 2 == 0 else "负",
            "text": f"反思 {i}",
            "situation": "{}",
        }
        for i in range(10)
    ]
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=5,
        night_number=5,
        own_role="seer",
        reflection_memory_hints=hints,
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    text = builder._build_reflection_memory_hints()
    # Count hint entries in the rendered JSON array. The header itself
    # contains "反思" once, so plain ``text.count("反思")`` would be
    # off-by-one; instead count the unique per-entry prefix
    # ``"text":"反思 `` (note trailing space — the header has no JSON
    # ``"text":"反思 "``).
    item_count = text.count('"text":"反思 ')
    assert item_count == 3, (
        f"Expected 3 reflection hints in output, got {item_count}. "
        "Reflection V2 live prompt card budget is 3."
    )


def test_output_schema_constants_used_by_both_renderers():
    """M2-3: both renderers must reference the same field constants."""
    from werewolf_agent.agents import prompt_builder
    assert hasattr(prompt_builder, "_OUTPUT_SCHEMA_VOTE_FIELDS")
    assert hasattr(prompt_builder, "_OUTPUT_SCHEMA_SPEECH_FIELDS")
    assert hasattr(prompt_builder, "_OUTPUT_SCHEMA_SKILL_FIELDS")
    assert len(prompt_builder._OUTPUT_SCHEMA_VOTE_FIELDS) >= 5
    assert len(prompt_builder._OUTPUT_SCHEMA_SPEECH_FIELDS) >= 3
    assert len(prompt_builder._OUTPUT_SCHEMA_SKILL_FIELDS) >= 3


def test_vote_audit_fields_derived_from_constant():
    """M2-3: 投票审计字段必须从 _OUTPUT_SCHEMA_VOTE_FIELDS 派生,不重复字面。"""
    from werewolf_agent.agents import prompt_builder
    # 必须有 _VOTE_AUDIT_FIELDS 常量
    assert hasattr(prompt_builder, "_VOTE_AUDIT_FIELDS")
    # 必须等于 VOTE 减去 {choice, reason, confidence}
    vote = prompt_builder._OUTPUT_SCHEMA_VOTE_FIELDS
    audit = prompt_builder._VOTE_AUDIT_FIELDS
    expected = tuple(f for f in vote if f not in ("choice", "reason", "confidence"))
    assert audit == expected, (
        f"_VOTE_AUDIT_FIELDS 必须是 VOTE 减去 {{choice, reason, confidence}}. "
        f"Got: {audit!r}, expected: {expected!r}"
    )
    # 必须 non-empty (VOTE 9 fields - 3 = 6 期望)
    assert len(audit) >= 3, f"_VOTE_AUDIT_FIELDS 必须 >= 3 fields, got {len(audit)}"


def test_vote_basis_guidance_only_in_vote_or_speech_prompts():
    """M2-2: _VOTE_BASIS_GUIDANCE must NOT appear in night_action prompts."""
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    # Night task (wolf kill = NIGHT_ACTION): vote_basis guidance
    # should NOT be in role_guide (stable system section).
    ctx_night = AgentContext(
        agent_id="p01", task_type=TaskType.NIGHT_ACTION,
        phase="night", day_number=1, night_number=1,
        own_role="werewolf",
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx_night
    role_guide = builder._build_role_guide()
    assert "vote_basis" not in role_guide.lower(), (
        f"M2-2: vote_basis guidance leaked into role_guide (stable system "
        f"section). Got: {role_guide!r}"
    )


def test_vote_basis_guidance_present_in_speech_via_strategy_directive():
    """M2-2: vote_basis guidance must be injected per-turn via
    strategy_directive (the adapter does it, role_guide doesn't)."""
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.runtime.agent_adapter import VOTE_BASIS_GUIDANCE

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
        strategy_directive={"vote_basis_hint": VOTE_BASIS_GUIDANCE},
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    text = builder._build_strategy_directive()
    assert "vote_basis" in text.lower()


def test_vote_basis_hint_registers_in_hard_or_suggestion_tier():
    """M2-2 follow-up: vote_basis_hint must not silently fall to
    【参考】 tier — the guidance contains '不要用 seer_check' which
    is a command, not a suggestion. Budget trimmer must preserve it.
    """
    from werewolf_agent.agents import prompt_builder
    hard = prompt_builder.HARD_CONSTRAINT_KEYS
    sugg = prompt_builder.SUGGESTION_KEYS
    ref = prompt_builder.REFERENCE_KEYS
    in_hard = "vote_basis_hint" in hard
    in_sugg = "vote_basis_hint" in sugg
    in_ref = "vote_basis_hint" in ref
    assert in_hard or in_sugg, (
        f"M2-2 follow-up: vote_basis_hint must register in HARD or "
        f"SUGGESTION tier (command-language: '不要用 seer_check'). "
        f"Currently in_hard={in_hard} in_sugg={in_sugg} in_ref={in_ref}. "
        f"Falling through to 【参考】 allows budget trimmer to drop the "
        f"seer_check prohibition under tight token budgets."
    )
    assert not in_ref, "vote_basis_hint should NOT be in REFERENCE tier"


def test_vote_basis_hint_renders_in_hard_section():
    """M2-2 follow-up: when strategy_directive contains vote_basis_hint,
    the rendered output must place it under the 【硬约束】 header, not
    【参考】. Verifies the actual render path, not just the registry."""
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.runtime.agent_adapter import VOTE_BASIS_GUIDANCE

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
        strategy_directive={"vote_basis_hint": VOTE_BASIS_GUIDANCE},
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    text = builder._build_strategy_directive()
    # The hard-constraint section header is 【硬约束】 (defined in
    # _STRATEGY_GROUP_ORDER). The reference header is 【参考】.
    hard_pos = text.find("【硬约束】")
    ref_pos = text.find("【参考】")
    vote_basis_pos = text.find("vote_basis")
    assert vote_basis_pos != -1, "vote_basis should appear in the rendered text"
    # If both headers exist, the vote_basis text must come after the
    # hard header and (if reference exists) before the reference header.
    if hard_pos != -1 and ref_pos != -1:
        assert hard_pos < vote_basis_pos < ref_pos, (
            f"vote_basis must be rendered under 【硬约束】, not 【参考】. "
            f"hard_pos={hard_pos} vote_basis_pos={vote_basis_pos} ref_pos={ref_pos}"
        )
    elif ref_pos != -1:
        # Hard section didn't render (no hard keys other than this) — but
        # vote_basis IS a hard key, so this branch shouldn't be reached.
        # If it is, the registry still has it under REFERENCE — fail loudly.
        assert False, (
            f"vote_basis_hint rendered under 【参考】 at pos {ref_pos}, "
            f"but it should be in 【硬约束】. Text: {text!r}"
        )


def test_inject_vote_basis_helper_centralized():
    """M2-2 follow-up: 6 injection sites consolidated to a helper."""
    from werewolf_agent.runtime import agent_adapter
    from werewolf_agent.core.models import GameState, PlayerState
    assert hasattr(agent_adapter, "_inject_vote_basis_hint")
    gs = GameState(
        game_id="t", phase="day", day_number=2, night_number=2,
        players={
            "p01": PlayerState(id="p01", role="seer", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
        },
    )
    sd: dict = {}
    # Seer: no injection
    agent_adapter._inject_vote_basis_hint(sd, gs, "p01")
    assert "vote_basis_hint" not in sd, (
        "Seer should be exempt from vote_basis_hint injection"
    )
    # Villager: injected
    agent_adapter._inject_vote_basis_hint(sd, gs, "p02")
    assert "vote_basis_hint" in sd, (
        "Non-seer should get vote_basis_hint injection"
    )
    assert "speech_logic" in sd["vote_basis_hint"] or "vote_basis" in sd["vote_basis_hint"].lower()


def test_generation_request_records_only_unconfirmed_assembly_debug_for_retries():
    """Initial、structured retry、semantic retry 都从最终 messages 组装取证。"""
    from werewolf_agent.agents.player_generation_request import build_player_generation_request
    from werewolf_agent.agents.schemas import AgentContext, RetryInfo, TaskType
    from werewolf_agent.evaluation.trace_identity import DecisionIdentity
    from werewolf_agent.model_gateway.structured_output import StructuredOutputMode
    from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector

    class _Agent:
        def _player_action_tool(self, context):
            return {"name": "submit_player_action"}

        def _build_prompt(self, context, retry):
            return f"user attempt {retry.attempt}"

        def _build_system_prompt(self, context):
            from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
            return (
                PlayerPromptBuilder(context).build_system_prompt()
                + "\npersona-final-fragment"
            )

        def _build_persona_prompt(self, context):
            return "persona-final-fragment"

    collector = ModuleExposureAuditCollector()
    context = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH, phase="day", day_number=1,
        persona_snapshot={"profile_id": "calm"},
        decision_identity=DecisionIdentity("g1", "p01", "day", 1, 0, "speech", 1),
        exposure_collector=collector,
    )
    retries = (
        RetryInfo(attempt=1),
        RetryInfo(attempt=2, error_code="missing_tool_call"),
        RetryInfo(attempt=3, error_code="speech_quality"),
    )

    requests = [
        build_player_generation_request(_Agent(), context, retry, StructuredOutputMode.NATIVE_TOOL)
        for retry in retries
    ]
    events = collector.flush_events()
    proofs = [event.payload["proof"] for event in events]

    assert all(request.messages[0]["role"] == "system" for request in requests)
    assert all(request.messages[0]["content"] == request.system_prompt for request in requests)
    assert all(request.persona_confirmed_in_system for request in requests)
    assert all(event.type == "persona_request_assembly_audit" for event in events)
    assert all(proof["confirmed_injection"] is False for proof in proofs)
    assert [proof["attempt_kind"] for proof in proofs] == [
        "initial", "structured_retry", "semantic_retry",
    ]


def test_provider_fallback_reuses_final_persona_system_message_proof(monkeypatch):
    from types import SimpleNamespace

    from werewolf_agent.agents import player_generation_request as generation_request
    from werewolf_agent.agents.schemas import AgentContext, RetryInfo, TaskType
    from werewolf_agent.evaluation.trace_identity import DecisionIdentity
    from werewolf_agent.model_gateway.generation_attempt_context import GenerationAttemptContext
    from werewolf_agent.model_gateway.final_prompt_observer import FinalPromptAssembly
    from werewolf_agent.model_gateway.structured_output import StructuredOutputMode
    from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector

    class _Agent:
        agent_id = "p01"
        model_router = object()

        def _player_action_tool(self, context):
            return {"name": "submit_player_action"}

        def _build_prompt(self, context, retry):
            return "user action"

        def _build_system_prompt(self, context):
            from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
            return (
                PlayerPromptBuilder(context).build_system_prompt()
                + "\npersona-final-fragment"
            )

        def _build_persona_prompt(self, context):
            return "persona-final-fragment"

    collector = ModuleExposureAuditCollector()
    identity = DecisionIdentity("g1", "p01", "day", 1, 0, "speech", 2)
    context = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH, phase="day", day_number=1,
        persona_snapshot={"profile_id": "calm"},
        decision_identity=identity,
        exposure_collector=collector,
    )
    request = generation_request.build_player_generation_request(
        _Agent(), context, RetryInfo(attempt=1), StructuredOutputMode.NATIVE_TOOL,
    )
    collector.flush_events()
    attempt_context = GenerationAttemptContext(run_scope="p001")

    def _fake_generate(*args, **kwargs):
        observer = kwargs["final_prompt_observer"]
        observer(FinalPromptAssembly(
            system_bytes=(request.system_prompt + "\nprimary-assembly").encode("utf-8"),
            final_system_location="messages",
            final_system_message_index=0,
            provider="primary",
            model="primary-model",
            attempt_kind="primary",
            attempt_ordinal=1,
        ))
        observer(FinalPromptAssembly(
            system_bytes=(request.system_prompt + "\nfallback-assembly").encode("utf-8"),
            final_system_location="system",
            final_system_message_index=None,
            provider="anthropic",
            model="fallback-model",
            attempt_kind="provider_fallback",
            attempt_ordinal=2,
        ))
        attempt_context.attempts = (
            SimpleNamespace(route_kind=SimpleNamespace(value="primary"), ordinal=1),
            SimpleNamespace(route_kind=SimpleNamespace(value="provider_fallback"), ordinal=2),
        )
        return SimpleNamespace(text="ok")

    monkeypatch.setattr(generation_request, "_generate_player_response", _fake_generate)

    generation_request.call_player_generation_request(
        _Agent(), context, request, attempt_context,
    )

    events = collector.flush_events()
    assert len(events) == 2
    assert all(event.payload["trace_id"] == identity.trace_id() for event in events)
    fallback = events[1].payload["proof"]
    assert fallback["attempt_kind"] == "provider_fallback"
    assert fallback["final_system_location"] == "system"
    assert fallback["final_system_message_index"] is None
    assert fallback["confirmed_injection"] is True


def test_provider_prompt_proof_records_recomputable_contract_hmac_without_raw_data() -> None:
    import hashlib
    import hmac

    from werewolf_agent.evaluation.trace_identity import DecisionIdentity
    from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector

    secret = b"fixed-task-13-test-key"
    system_bytes = b"contract-header\nrules\nwolf-semantics"
    collector = ModuleExposureAuditCollector(prompt_proof_secret=secret)
    identity = DecisionIdentity("g1", "w1", "night", 1, 1, "wolf_discussion", 1)

    collector.record_provider_persona_prompt_proof(
        identity,
        system_bytes,
        "",
        "primary",
        attempt_ordinal=1,
        provider="openai",
        model="m",
        final_system_location="messages",
        final_system_message_index=0,
        prompt_contract_id="player-system",
        prompt_contract_version="2026-07-18",
        required_section_confirmations={
            "contract_header": True,
            "wolf_semantics": True,
        },
    )

    proof = collector.flush_events()[0].payload["proof"]
    assert proof["prompt_contract_id"] == "player-system"
    assert proof["prompt_contract_version"] == "2026-07-18"
    assert proof["system_byte_length"] == len(system_bytes)
    assert proof["system_hmac_sha256"] == hmac.new(
        secret,
        system_bytes,
        hashlib.sha256,
    ).hexdigest()
    assert proof["required_section_confirmations"] == [
        {"section_id": "contract_header", "confirmed": True},
        {"section_id": "wolf_semantics", "confirmed": True},
    ]
    serialized = str(proof)
    assert system_bytes.decode() not in serialized
    assert secret.decode() not in serialized


def test_player_generation_enforces_versioned_contract_without_persona(monkeypatch) -> None:
    from types import SimpleNamespace

    from werewolf_agent.agents import player_generation_request as generation_request
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.prompt_sections import (
        PLAYER_SYSTEM_PROMPT_CONTRACT_ID,
        PLAYER_SYSTEM_PROMPT_CONTRACT_VERSION,
    )
    from werewolf_agent.agents.schemas import AgentContext, RetryInfo, TaskType
    from werewolf_agent.evaluation.trace_identity import DecisionIdentity
    from werewolf_agent.model_gateway.final_prompt_observer import FinalPromptAssembly
    from werewolf_agent.model_gateway.structured_output import StructuredOutputMode
    from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector

    collector = ModuleExposureAuditCollector(prompt_proof_secret=b"fixed-contract-key")
    context = AgentContext(
        agent_id="w1",
        task_type=TaskType.WOLF_DISCUSSION,
        phase="night",
        night_number=1,
        own_role="werewolf",
        decision_identity=DecisionIdentity(
            "g1", "w1", "night", 0, 1, "wolf_discussion", 1,
        ),
        exposure_collector=collector,
    )

    class _Agent:
        agent_id = "w1"
        model_router = object()

        def _player_action_tool(self, _context):
            return {"name": "submit_player_action"}

        def _build_prompt(self, _context, _retry):
            return "user action"

        def _build_system_prompt(self, current_context):
            return PlayerPromptBuilder(current_context).build_system_prompt()

    request = generation_request.build_player_generation_request(
        _Agent(), context, RetryInfo(attempt=1), StructuredOutputMode.NATIVE_TOOL,
    )

    def _fake_generate(*_args, **kwargs):
        kwargs["final_prompt_observer"](FinalPromptAssembly(
            system_bytes=request.system_prompt.encode("utf-8"),
            final_system_location="messages",
            final_system_message_index=0,
            provider="openai",
            model="m",
            attempt_kind="primary",
            attempt_ordinal=1,
        ))
        return SimpleNamespace(text="ok")

    monkeypatch.setattr(generation_request, "_generate_player_response", _fake_generate)

    generation_request.call_player_generation_request(_Agent(), context, request)

    proof_event = collector.flush_events()[-1]
    assert proof_event.type == "final_prompt_contract_audit"
    proof = proof_event.payload["proof"]
    assert proof["prompt_contract_id"] == PLAYER_SYSTEM_PROMPT_CONTRACT_ID
    assert proof["prompt_contract_version"] == PLAYER_SYSTEM_PROMPT_CONTRACT_VERSION
    assert all(
        row["confirmed"] for row in proof["required_section_confirmations"]
    )


def test_villager_role_guide_is_concise():
    """M2-1: villager guide was 4 paragraphs (~400 chars), other
    roles are 1 paragraph. Token waste + over-guidance."""
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    guide = builder._build_role_guide()
    # Compress to ~200 chars while keeping the 2 key behaviors
    assert len(guide) < 200, (
        f"Villager role_guide too long: {len(guide)} chars. "
        f"Other roles are ~100 chars; this is over-guidance."
    )
    # But must keep the 2 key behaviors
    assert "N1" in guide or "解药" in guide, "should keep N1 解药 cue"
    assert "票型" in guide or "证据" in guide, "should keep evidence-based voting cue"


def test_learning_context_priority_wraps_rag_and_reflection():
    """Merged cross-game learning renders as one 【参考】 section."""
    from werewolf_agent.agents import prompt_builder
    prios = prompt_builder.PlayerPromptBuilder._SECTION_PRIORITIES
    assert prios["_build_learning_context"] == "【参考】"
    assert "_build_reflection_memory_hints" not in prios
    assert "_build_rag_hints" not in prios


def test_skill_policy_distinguishes_from_identity_rules():
    """M5-1: _build_skill_policy should clearly state that identity
    rules (above in role_guide) outrank skill advice on conflict.

    Without this 边界, LLM may conflate 'skill said vote X' with
    'role said vote X'.
    """
    import re
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    policy = builder._build_skill_policy()
    # Must mention both concepts
    assert "身份" in policy and "技能" in policy, (
        f"Skill policy must mention both '身份' and '技能' "
        f"to establish the precedence 边界. Got: {policy!r}"
    )
    # Precedence direction: 身份规则 followed by 优先 (the precedence verb)
    assert re.search(r"身份规则.*优先", policy) is not None, (
        f"Skill policy must state 身份规则 优先 (over 技能建议); got: {policy!r}"
    )
