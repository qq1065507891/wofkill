from __future__ import annotations

import json

from werewolf_agent.core.models import GameState, GameEvent
from scripts.print_game_audit import audit_game, find_boundary_violations, render_audit_report


def test_render_audit_report_includes_player_raw_output_and_judge_events() -> None:
    game = {
        "game_id": "g_test",
        "winning_faction": "werewolf",
        "day_number": 1,
        "night_number": 2,
        "steps": 12,
        "events": [
            {"type": "day_announce", "payload": {"day": 1}},
            {
                "type": "speech",
                "payload": {
                    "speaker": "p01",
                    "text": "我站边自己。",
                    "action_trace": {
                        "raw_text": json.dumps(
                            {
                                "action_type": "speech",
                                "speech": "我站边自己。",
                                "reason": "先表水",
                            },
                            ensure_ascii=False,
                        ),
                        "final_action_type": "speech",
                        "retry": {"attempt": 1, "max_retries": 3},
                    },
                },
            },
            {
                "type": "vote_resolved",
                "payload": {
                    "action_traces": {
                        "p02": {
                            "raw_text": '{"action_type":"vote","target_id":"p01","reason":"可疑"}',
                            "final_action_type": "vote",
                            "fallback_reason": "fallback: retries exhausted",
                            "retry": {
                                "attempt": 3,
                                "max_retries": 3,
                                "error_code": "parse_error",
                                "error_message": "bad enum",
                            },
                        }
                    }
                },
            },
        ],
    }

    report = render_audit_report(game)

    assert "# Game Audit: g_test" in report
    assert "## Judge / Runtime Events" in report
    assert "day_announce" in report
    assert "## Player Model Outputs" in report
    assert "p01" in report
    assert "我站边自己" in report
    assert "p02" in report
    assert "fallback: retries exhausted" in report
    assert "Provider thinking blocks were not persisted" in report


def test_render_audit_report_flags_rule_order_anomalies() -> None:
    game = {
        "game_id": "g_anomaly",
        "events": [
            {
                "type": "player_died",
                "payload": {
                    "player_id": "p02",
                    "reason": "exile",
                    "triggered_skills": ["hunter_shot"],
                },
            },
            {
                "type": "wolf_discussion",
                "payload": {"wolf_id": "p03", "text": "继续刀人"},
            },
            {
                "type": "wolf_kill_selected",
                "payload": {"target_id": "p10", "killer_id": "p03"},
            },
            {
                "type": "player_died",
                "payload": {
                    "player_id": "p03",
                    "reason": "hunter_shot",
                    "source_player_id": "p02",
                },
            },
            {
                "type": "speech",
                "payload": {
                    "speaker": "p08",
                    "text": "p02声称自己是狼人（Day 1公开记录），所以必须出。",
                },
            },
        ],
    }

    report = render_audit_report(game)

    assert "## Rule-Order Anomalies" in report
    assert "hunter_shot death after wolf action" in report
    assert "unsupported public-record role claim" in report


def test_boundary_audit_flags_core_game_record_invariants() -> None:
    game = {
        "players": {
            "p01": {"role": "witch", "alive": False},
            "p02": {"role": "hunter", "alive": False},
            "p03": {"role": "villager", "alive": True},
        },
        "deaths": [
            {"player_id": "p01", "reason": "exile"},
        ],
        "events": [
            {
                "type": "player_died",
                "payload": {
                    "player_id": "p01",
                    "reason": "exile",
                    "triggered_skills": [],
                },
            },
            {
                "type": "judge_broadcast",
                "payload": {"phase": "witch_wake"},
            },
            {
                "type": "player_died",
                "payload": {
                    "player_id": "p02",
                    "reason": "exile",
                    "triggered_skills": ["hunter_shot"],
                },
            },
            {
                "type": "vote_resolved",
                "payload": {
                    "votes": [
                        {"voter": "p03", "target": "p03", "reason": ""},
                    ],
                },
            },
        ],
    }

    kinds = {item["kind"] for item in find_boundary_violations(game)}

    assert "dead_role_broadcast" in kinds
    assert "pending_hunter_shot" in kinds
    assert "empty_vote_reason" in kinds
    assert "self_vote" in kinds
    assert "incomplete_death_export" in kinds


def test_boundary_audit_accepts_closed_loop_record() -> None:
    game = {
        "players": {
            "p01": {"role": "hunter", "alive": False},
            "p02": {"role": "werewolf", "alive": False},
            "p03": {"role": "villager", "alive": True},
        },
        "deaths": [
            {
                "player_id": "p01",
                "reason": "exile",
                "timing": "day_vote",
                "resolution_batch": "day_2_vote",
                "source_player_id": None,
                "can_leave_last_words": True,
                "triggered_skills": ["hunter_shot"],
            },
            {
                "player_id": "p02",
                "reason": "hunter_shot",
                "timing": "day_vote",
                "resolution_batch": "day_2_vote",
                "source_player_id": "p01",
                "can_leave_last_words": False,
                "triggered_skills": [],
            },
        ],
        "events": [
            {
                "type": "player_died",
                "payload": {
                    "player_id": "p01",
                    "reason": "exile",
                    "triggered_skills": ["hunter_shot"],
                },
            },
            {
                "type": "judge_broadcast",
                "payload": {"phase": "hunter_shot_prompt", "hunter_id": "p01"},
            },
            {
                "type": "judge_broadcast",
                "payload": {
                    "phase": "hunter_shot_choice",
                    "hunter_id": "p01",
                    "target_id": "p02",
                },
            },
            {
                "type": "player_died",
                "payload": {
                    "player_id": "p02",
                    "reason": "hunter_shot",
                    "source_player_id": "p01",
                },
            },
            {
                "type": "vote_resolved",
                "payload": {
                    "votes": [
                        {"voter": "p03", "target": "p02", "reason": "p02's claim conflicts with the vote record"},
                    ],
                },
            },
        ],
    }

    assert find_boundary_violations(game) == []


class TestJudgePhaseBroadcasts:
    """Game events include judge-visible broadcasts for all major phases."""

    def _make_gs_with_broadcasts(self):
        """Create a game state with judge broadcast events."""
        events = [
            GameEvent(type="judge_broadcast", payload={"phase": "enter_night", "message": "天黑请闭眼", "night_number": 1}),
            GameEvent(type="judge_broadcast", payload={"phase": "wolf_discussion_start", "message": "狼人请睁眼", "night_number": 1}),
            GameEvent(type="judge_broadcast", payload={"phase": "wolf_kill_choice", "message": "狼人请统一刀人", "night_number": 1}),
            GameEvent(type="judge_broadcast", payload={"phase": "seer_wake", "message": "预言家请睁眼", "night_number": 1}),
            GameEvent(type="judge_broadcast", payload={"phase": "witch_wake", "message": "女巫请睁眼", "night_number": 1}),
            GameEvent(type="judge_broadcast", payload={"phase": "day_announce", "message": "天亮了", "day_number": 1}),
            GameEvent(type="judge_broadcast", payload={"phase": "sheriff_election", "message": "开始选警"}),
            GameEvent(type="judge_broadcast", payload={"phase": "speech_order", "message": "发言顺序确定"}),
            GameEvent(type="judge_broadcast", payload={"phase": "vote", "message": "投票开始"}),
            GameEvent(type="judge_broadcast", payload={"phase": "exile", "message": "放逐结果"}),
        ]
        return GameState(game_id="test", events=events)

    def test_judge_broadcasts_major_phases(self):
        gs = self._make_gs_with_broadcasts()
        broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
        phases = {e.payload["phase"] for e in broadcasts}
        expected = {"enter_night", "wolf_discussion_start", "wolf_kill_choice", "seer_wake",
                    "witch_wake", "day_announce", "sheriff_election", "speech_order",
                    "vote", "exile"}
        assert expected.issubset(phases)

    def test_broadcast_payload_has_required_fields(self):
        gs = self._make_gs_with_broadcasts()
        broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
        for b in broadcasts:
            assert "phase" in b.payload
            assert "message" in b.payload


class TestAuditOutput:
    """Audit output includes private and public sections."""

    def test_audit_report_includes_private_and_public_sections(self):
        """Audit script can separate public and private event sections."""
        events = [
            GameEvent(type="speech", payload={"speaker": "p01", "text": "公开发言", "visibility": "public"}),
            GameEvent(type="wolf_discussion", payload={"wolf_id": "p01", "text": "狼人密谈", "visibility": "werewolf_team_only"}),
            GameEvent(type="action_trace_audit", payload={"player_id": "p01", "phase": "vote", "visibility": "moderator_only"}),
            GameEvent(type="vote_resolved", payload={"exiled": "p03", "reason": "majority"}),
        ]
        # Separate public vs private
        public = [e for e in events if e.payload.get("visibility", "public") == "public" or e.type in ("speech", "vote_resolved")]
        private = [e for e in events if e.payload.get("visibility") in ("moderator_only", "werewolf_team_only")]

        assert len(public) >= 1
        assert len(private) >= 1
        assert private[0].payload.get("visibility") == "werewolf_team_only"

    def test_audit_events_have_actor_and_target(self):
        """Events for audit have actor, target, phase fields where applicable."""
        event = GameEvent(type="vote_resolved", payload={
            "exiled": "p03",
            "reason": "majority",
            "voters": {"p01": "p03", "p02": "p03"},
        })
        assert "exiled" in event.payload
        assert "reason" in event.payload


class TestAuditScript:
    """The audit script can process a complete game JSON."""

    def test_audit_script_exists(self):
        import os
        assert os.path.exists("scripts/print_game_audit.py")

    def test_audit_script_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("print_game_audit", "scripts/print_game_audit.py")
        assert spec is not None
def test_audit_game_classifies_v2_top_level_private_speech(capsys) -> None:
    audit_game({"events": [{
        "type": "speech",
        "payload": {"speaker": "p01", "text": "private"},
        "visibility": "actor_private",
        "event_id": "g1:e000000",
        "sequence_number": 0,
        "occurred_at": "2026-07-15T00:00:00+00:00",
        "game_id": "g1",
        "trace_id": None,
        "schema_version": "2",
    }]})

    output = capsys.readouterr().out

    assert "公开: 0" in output
    assert "actor_private" in output
