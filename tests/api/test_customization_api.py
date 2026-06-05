"""API tests for customization templates and validation endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app


def test_download_ruleset_template() -> None:
    client = TestClient(create_app())

    resp = client.get("/templates/ruleset")

    assert resp.status_code == 200
    assert "ruleset_id" in resp.text


def test_validate_ruleset_upload_returns_summary() -> None:
    client = TestClient(create_app())
    template = client.get("/templates/ruleset").text

    resp = client.post("/customization/rulesets/validate", content=template)

    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["summary"]["player_count"] == 12
    assert "diff_against_default" in data


def test_download_persona_template() -> None:
    client = TestClient(create_app())

    resp = client.get("/templates/persona-pack")

    assert resp.status_code == 200
    assert "players:" in resp.text


def test_validate_persona_upload_returns_summary_and_preview() -> None:
    client = TestClient(create_app())
    template = client.get("/templates/persona-pack").text

    resp = client.post("/customization/persona-packs/validate", content=template)

    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["summary"]["player_count"] == 12
    assert "persona_preview" in data
    assert "villager_opening" in data["persona_preview"]["p01"]


def test_ruleset_validation_rejects_malicious_yaml_tag() -> None:
    client = TestClient(create_app())

    resp = client.post(
        "/customization/rulesets/validate",
        content="!!python/object/apply:os.system ['echo bad']",
    )

    assert resp.status_code == 200
    assert resp.json()["valid"] is False


def test_save_ruleset_requires_authorized_caller() -> None:
    client = TestClient(create_app())
    template = client.get("/templates/ruleset").text

    resp = client.post("/customization/rulesets", content=template)

    assert resp.status_code == 403


def test_save_ruleset_stores_validated_metadata() -> None:
    client = TestClient(create_app())
    template = client.get("/templates/ruleset").text

    resp = client.post("/customization/rulesets?caller_id=mod1&caller_role=moderator", content=template)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "playable"
    assert data["config_type"] == "ruleset"
    assert data["content_hash"]


def test_marketplace_rulesets_lists_playable_and_display_only_items() -> None:
    client = TestClient(create_app())

    resp = client.get("/marketplace/rulesets")

    assert resp.status_code == 200
    data = resp.json()
    items = data["items"]
    assert any(item["ruleset_id"] == "pre_witch_hunter_idiot_mixed" and item["status"] == "playable" for item in items)
    assert any(item["ruleset_id"] == "wolf_king_guard_classic" and item["status"] == "display_only" for item in items)


def test_marketplace_persona_packs_lists_default_pack() -> None:
    client = TestClient(create_app())

    resp = client.get("/marketplace/persona-packs")

    assert resp.status_code == 200
    data = resp.json()
    assert any(item["profile_pack_id"] == "default_12_ai_players" for item in data["items"])


def test_share_summary_is_public_safe() -> None:
    client = TestClient(create_app())
    created = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "seed": 717,
        "caller_id": "mod1",
        "caller_role": "moderator",
    }).json()
    game_id = created["game"]["game_id"]
    client.post(
        f"/games/{game_id}/start",
        json={"caller_id": "mod1", "caller_role": "moderator"},
    )

    resp = client.get(
        f"/games/{game_id}/share-summary"
        f"?caller_id=mod1&caller_role=moderator"
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["game_id"] == game_id
    assert data["public_only"] is True
    assert "share_title" in data
    assert "highlight_events" in data
    assert "mvp_candidate" in data
    assert "winning_faction" in data
    assert data["leak_audit_summary"] == {
        "leak_check_status": "passed",
        "private_role_leaks": 0,
        "illegal_view_references": 0,
        "forbidden_event_exposures": 0,
    }


def test_create_game_human_seat_mode_requires_human_seat() -> None:
    client = TestClient(create_app())

    resp = client.post("/games", json={
        "experience_mode": "human_seat",
        "caller_id": "mod1",
        "caller_role": "moderator",
    })

    assert resp.status_code == 400
    assert "human_seat" in resp.json()["detail"]


def test_create_game_stores_locked_customization_snapshot() -> None:
    client = TestClient(create_app())

    resp = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "profile_pack_id": "default_12_ai_players",
            "experience_mode": "human_seat",
            "human_seat": 3,
            "share_code": "abc123",
            "seed": 818,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["game"]["profile_pack_id"] == "default_12_ai_players"
    assert data["game"]["experience_mode"] == "human_seat"
    assert data["game"]["human_seat"] == 3
    game_id = data["game"]["game_id"]
    state = client.app.state.games[game_id]
    snapshot_events = [event for event in state.events if event.type == "config_snapshot_locked"]
    assert snapshot_events
    snapshot = snapshot_events[0].payload["config_snapshot"]
    assert snapshot["ruleset_id"] == "pre_witch_hunter_idiot_mixed"
    assert snapshot["profile_pack_id"] == "default_12_ai_players"
    assert snapshot["human_seat"] == 3
