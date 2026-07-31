# -*- coding: utf-8 -*-
"""
验证自主玩家观察投影的严格不可变契约和跨对象一致性。

作者: Project contributors
创建日期: 2026-07-31
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.observation import (
    ActiveObservationConflict,
    ObservationBundle,
    PlayerWorkspaceSnapshot,
    ProjectionAvailability,
    ProjectionBuildFailed,
    ProjectionIdentity,
    ProjectionIdentityMismatch,
    ProjectionIntegrityFailed,
    ProjectionRenderFailed,
    ProjectionSourceChanged,
    ProjectionUnavailableReason,
    ProjectionVisibilityClass,
    ProjectionVisibilityRejected,
    RequiredProjectionUnavailable,
    WorkspaceSection,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
NOW = datetime(2026, 7, 31, 12, tzinfo=timezone.utc)


def _identity() -> ProjectionIdentity:
    return ProjectionIdentity(
        game_id="game-1",
        player_id="p01",
        schedule_id="schedule-1",
        turn_id="turn-1",
        schedule_state_version=1,
        turn_state_version=0,
        window_id="speech-d1",
        window_version=1,
        base_game_revision=4,
        view_fingerprint=HASH,
    )


def _entry_payload(section: WorkspaceSection) -> dict[str, object]:
    return {
        "section_id": section,
        "availability": ProjectionAvailability.AVAILABLE,
        "required": section is WorkspaceSection.INDEX,
        "identity": _identity().model_dump(),
        "renderer_version": "renderer-v1",
        "content_hash": HASH,
        "token_estimate": 1,
        "estimator_version": "estimator-v1",
        "visibility_class": ProjectionVisibilityClass.PLAYER_PRIVATE,
        "source_references": [],
        "unavailable_reason": None,
    }


def _document_payload(section: WorkspaceSection) -> dict[str, object]:
    return {
        "section_id": section,
        "identity": _identity().model_dump(),
        "renderer_version": "renderer-v1",
        "content_markdown": f"# {section.value}",
        "content_hash": HASH,
        "token_estimate": 1,
        "estimator_version": "estimator-v1",
        "visibility_class": ProjectionVisibilityClass.PLAYER_PRIVATE,
        "source_references": [],
    }


def _bundle_payload() -> dict[str, object]:
    entries = [_entry_payload(section) for section in WorkspaceSection]
    documents = [_document_payload(section) for section in WorkspaceSection]
    identity = _identity().model_dump()
    return {
        "frame": {
            "identity": identity,
            "task_kind": "day_speech",
            "actor_id": "p01",
            "role_id": "villager",
            "phase": "day_discussion",
            "legal_action_snapshot": ["speech"],
            "legal_target_snapshot": ["p02"],
            "critical_private_fact_references": [
                {"record_id": "fact-1", "revision": 0, "content_hash": HASH},
            ],
            "bounded_public_summary": ["公开摘要"],
            "recent_commitment_references": [],
            "document_manifest": entries,
            "tool_manifest": ["speak"],
            "workspace_revision": HASH,
            "workspace_hash": OTHER_HASH,
            "deadline": NOW + timedelta(minutes=1),
            "observed_at": NOW,
        },
        "workspace": {
            "identity": identity,
            "workspace_revision": HASH,
            "documents": documents,
            "manifest_entries": entries,
            "workspace_hash": OTHER_HASH,
        },
    }


def _invalid_contract_for(mutation: str) -> ObservationBundle:
    payload = _bundle_payload()
    frame = payload["frame"]
    workspace = payload["workspace"]
    assert isinstance(frame, dict)
    assert isinstance(workspace, dict)
    entries = workspace["manifest_entries"]
    assert isinstance(entries, list)

    if mutation == "available_without_hash":
        entries[0]["content_hash"] = None
    elif mutation == "unavailable_required":
        entry = entries[0]
        entry.update({
            "availability": ProjectionAvailability.UNAVAILABLE,
            "required": True,
            "renderer_version": None,
            "content_hash": None,
            "token_estimate": None,
            "estimator_version": None,
            "unavailable_reason": (
                ProjectionUnavailableReason.SOURCE_CAPABILITY_UNAVAILABLE
            ),
        })
        workspace["documents"] = workspace["documents"][1:]
    elif mutation == "duplicate_section":
        entries[-1]["section_id"] = WorkspaceSection.PLAYER
    elif mutation == "document_entry_hash_mismatch":
        workspace["documents"][0]["content_hash"] = OTHER_HASH
    elif mutation == "frame_workspace_identity_mismatch":
        frame["identity"] = {**frame["identity"], "turn_id": "turn-2"}
    elif mutation == "naive_deadline":
        frame["deadline"] = NOW.replace(tzinfo=None)
    elif mutation == "observation_at_deadline":
        frame["observed_at"] = frame["deadline"]
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    return ObservationBundle.model_validate(payload)


def test_projection_identity_is_strict_frozen_and_complete() -> None:
    identity = _identity()
    assert identity.model_dump(mode="json") == {
        "game_id": "game-1",
        "player_id": "p01",
        "schedule_id": "schedule-1",
        "turn_id": "turn-1",
        "schedule_state_version": 1,
        "turn_state_version": 0,
        "window_id": "speech-d1",
        "window_version": 1,
        "base_game_revision": 4,
        "view_fingerprint": HASH,
    }
    with pytest.raises(ValidationError):
        ProjectionIdentity.model_validate({**identity.model_dump(), "extra": True})
    with pytest.raises(ValidationError):
        identity.player_id = "p02"  # type: ignore[misc]


def test_projection_error_codes_are_stable() -> None:
    assert ActiveObservationConflict.code == "active_observation_conflict"
    assert RequiredProjectionUnavailable.code == "required_projection_unavailable"
    assert ProjectionIdentityMismatch.code == "projection_identity_mismatch"
    assert ProjectionVisibilityRejected.code == "projection_visibility_rejected"
    assert ProjectionSourceChanged.code == "projection_source_changed"
    assert ProjectionIntegrityFailed.code == "projection_integrity_failed"
    assert ProjectionRenderFailed.code == "projection_render_failed"
    assert ProjectionBuildFailed.code == "projection_build_failed"


@pytest.mark.parametrize(
    "mutation",
    [
        "available_without_hash",
        "unavailable_required",
        "duplicate_section",
        "document_entry_hash_mismatch",
        "frame_workspace_identity_mismatch",
        "naive_deadline",
        "observation_at_deadline",
    ],
)
def test_observation_contracts_reject_invalid_combinations(mutation: str) -> None:
    with pytest.raises(ValidationError):
        _invalid_contract_for(mutation)


def test_observation_bundle_defensively_freezes_nested_values() -> None:
    payload = _bundle_payload()
    bundle = ObservationBundle.model_validate(payload)
    frame = payload["frame"]
    assert isinstance(frame, dict)
    actions = frame["legal_action_snapshot"]
    assert isinstance(actions, list)
    actions.append("vote")
    assert bundle.frame.legal_action_snapshot == ("speech",)


def test_workspace_rejects_duplicate_and_conflicting_source_references() -> None:
    payload = _bundle_payload()
    workspace = payload["workspace"]
    assert isinstance(workspace, dict)
    documents = workspace["documents"]
    assert isinstance(documents, list)
    documents[0]["source_references"] = [
        {
            "record_kind": "commitment",
            "record_id": "record-1",
            "record_revision": 1,
            "content_hash": HASH,
        },
        {
            "record_kind": "commitment",
            "record_id": "record-1",
            "record_revision": 1,
            "content_hash": OTHER_HASH,
        },
    ]
    with pytest.raises(ValidationError):
        PlayerWorkspaceSnapshot.model_validate(workspace)


def test_errors_accept_no_private_payload() -> None:
    assert str(ProjectionBuildFailed())
    with pytest.raises(TypeError):
        ProjectionBuildFailed("private payload")


def test_manifest_and_documents_are_defensively_copied() -> None:
    payload = _bundle_payload()
    copied = deepcopy(payload)
    bundle = ObservationBundle.model_validate(copied)
    workspace = copied["workspace"]
    assert isinstance(workspace, dict)
    entries = workspace["manifest_entries"]
    assert isinstance(entries, list)
    entries[0]["content_hash"] = OTHER_HASH
    assert bundle.workspace.manifest_entries[0].content_hash == HASH
