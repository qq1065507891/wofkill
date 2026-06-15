from __future__ import annotations

from werewolf_agent.memory.migration import (
    clean_snapshot_reflection_boundary,
    dry_run_legacy_reflection,
)


def test_generic_legacy_reflection_dry_run_is_not_approved() -> None:
    report = dry_run_legacy_reflection({
        "entry_id": "legacy_1",
        "game_id": "g1",
        "player_id": "p01",
        "role": "seer",
        "text": "复盘失败对局，关注关键转折点的信息缺失",
    })

    assert report["old_schema_version"] == 1
    assert report["decision"] in {"review_only", "rejected"}
    assert report["score"] < 0.70
    assert "generic_text" in report["flags"]


def test_snapshot_cleanup_keeps_only_reflection_ids() -> None:
    cleaned = clean_snapshot_reflection_boundary({
        "cognition_matrices": {},
        "reflections": [
            {
                "entry_id": "body_1",
                "text": "must disappear",
                "quality_status": "approved",
            },
            {
                "entry_id": "body_2",
                "text": "must disappear",
                "quality_status": "rejected",
            },
            "already_id",
        ],
    })

    assert cleaned["reflections"] == ["body_1", "already_id"]
    assert "must disappear" not in str(cleaned)
