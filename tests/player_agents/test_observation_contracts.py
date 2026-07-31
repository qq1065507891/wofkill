# -*- coding: utf-8 -*-
"""
验证自主玩家观察投影的严格不可变契约和跨对象一致性。

作者: Project contributors
创建日期: 2026-07-31
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.observation import (
    ActiveObservationConflict,
    ManifestEntry,
    ObservationBundle,
    PlayerWorkspaceSnapshot,
    ProjectedDocument,
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
REQUIRED_SECTIONS = frozenset({
    WorkspaceSection.PLAYER,
    WorkspaceSection.ROLE,
    WorkspaceSection.GAME,
    WorkspaceSection.INDEX,
})
UNAVAILABLE_SECTIONS = frozenset({
    WorkspaceSection.BELIEFS,
    WorkspaceSection.MEMORY,
    WorkspaceSection.WORKING,
})


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
    unavailable = section in UNAVAILABLE_SECTIONS
    return {
        "section_id": section,
        "availability": (
            ProjectionAvailability.UNAVAILABLE
            if unavailable else ProjectionAvailability.AVAILABLE
        ),
        "required": section in REQUIRED_SECTIONS,
        "identity": _identity().model_dump(),
        "renderer_version": None if unavailable else "renderer-v1",
        "content_hash": None if unavailable else _document_hash(section),
        "token_estimate": None if unavailable else 1,
        "estimator_version": None if unavailable else "estimator-v1",
        "visibility_class": ProjectionVisibilityClass.PLAYER_PRIVATE,
        "source_references": [],
        "unavailable_reason": (
            ProjectionUnavailableReason.SOURCE_CAPABILITY_UNAVAILABLE
            if unavailable else None
        ),
    }


def _document_payload(section: WorkspaceSection) -> dict[str, object]:
    content_markdown = f"# {section.value}\n"
    return {
        "section_id": section,
        "identity": _identity().model_dump(),
        "renderer_version": "renderer-v1",
        "content_markdown": content_markdown,
        "content_hash": hashlib.sha256(content_markdown.encode("utf-8")).hexdigest(),
        "token_estimate": 1,
        "estimator_version": "estimator-v1",
        "visibility_class": ProjectionVisibilityClass.PLAYER_PRIVATE,
        "source_references": [],
    }


def _document_hash(section: WorkspaceSection) -> str:
    """在测试侧独立计算固定文档字节的 SHA-256。"""

    return hashlib.sha256(f"# {section.value}\n".encode()).hexdigest()


def _literal_workspace_hash(
    entries: list[dict[str, object]],
    documents: list[dict[str, object]],
) -> str:
    """按设计文字独立计算有序 manifest 与文档字节哈希。"""

    manifest_bytes = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(manifest_bytes).to_bytes(8, "big"))
    digest.update(manifest_bytes)
    for document in documents:
        content_bytes = str(document["content_markdown"]).encode("utf-8")
        digest.update(len(content_bytes).to_bytes(8, "big"))
        digest.update(content_bytes)
    return digest.hexdigest()


def _bundle_payload() -> dict[str, object]:
    entries = [_entry_payload(section) for section in WorkspaceSection]
    documents = [
        _document_payload(section)
        for section in WorkspaceSection
        if section not in UNAVAILABLE_SECTIONS
    ]
    identity = _identity().model_dump()
    workspace_hash = _literal_workspace_hash(entries, documents)
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
            "workspace_hash": workspace_hash,
            "deadline": NOW + timedelta(minutes=1),
            "observed_at": NOW,
        },
        "workspace": {
            "identity": identity,
            "workspace_revision": HASH,
            "documents": documents,
            "manifest_entries": entries,
            "workspace_hash": workspace_hash,
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


def test_manifest_enforces_binding_section_policy() -> None:
    bundle = ObservationBundle.model_validate(_bundle_payload())
    entries = {entry.section_id: entry for entry in bundle.workspace.manifest_entries}
    assert {
        section
        for section, entry in entries.items()
        if entry.required and entry.availability is ProjectionAvailability.AVAILABLE
    } == REQUIRED_SECTIONS
    assert {
        section
        for section, entry in entries.items()
        if entry.availability is ProjectionAvailability.UNAVAILABLE
    } == UNAVAILABLE_SECTIONS

    for section in REQUIRED_SECTIONS:
        payload = _entry_payload(section)
        payload["required"] = False
        with pytest.raises(ValidationError):
            ManifestEntry.model_validate(payload)

        payload = _entry_payload(section)
        payload.update({
            "availability": ProjectionAvailability.UNAVAILABLE,
            "required": False,
            "renderer_version": None,
            "content_hash": None,
            "token_estimate": None,
            "estimator_version": None,
            "unavailable_reason": (
                ProjectionUnavailableReason.SOURCE_CAPABILITY_UNAVAILABLE
            ),
        })
        with pytest.raises(ValidationError):
            ManifestEntry.model_validate(payload)

    for section in UNAVAILABLE_SECTIONS:
        payload = _entry_payload(section)
        payload.update({
            "availability": ProjectionAvailability.AVAILABLE,
            "renderer_version": "renderer-v1",
            "content_hash": HASH,
            "token_estimate": 1,
            "estimator_version": "estimator-v1",
            "unavailable_reason": None,
        })
        with pytest.raises(ValidationError):
            ManifestEntry.model_validate(payload)

    commitments = _entry_payload(WorkspaceSection.COMMITMENTS)
    commitments["required"] = True
    with pytest.raises(ValidationError):
        ManifestEntry.model_validate(commitments)


@pytest.mark.parametrize("container", ["entry", "document"])
def test_workspace_rejects_non_workspace_projection_identity(
    container: str,
) -> None:
    payload = _bundle_payload()
    workspace = payload["workspace"]
    assert isinstance(workspace, dict)
    entries = workspace["manifest_entries"]
    documents = workspace["documents"]
    assert isinstance(entries, list)
    assert isinstance(documents, list)
    changed_identity = {**_identity().model_dump(), "turn_id": "turn-2"}
    if container == "entry":
        entries[0]["identity"] = changed_identity
        documents[0]["identity"] = changed_identity
    else:
        documents[0]["identity"] = changed_identity
    with pytest.raises(ValidationError):
        PlayerWorkspaceSnapshot.model_validate(workspace)


def test_observation_frame_rejects_actor_outside_projection_identity() -> None:
    payload = _bundle_payload()
    frame = payload["frame"]
    assert isinstance(frame, dict)
    frame["actor_id"] = "p02"
    with pytest.raises(ValidationError):
        ObservationBundle.model_validate(payload)


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("renderer_version", "renderer-v2"),
        ("estimator_version", "estimator-v2"),
        ("visibility_class", ProjectionVisibilityClass.PUBLIC),
    ],
)
def test_workspace_rejects_document_manifest_metadata_mismatch(
    field: str,
    replacement: object,
) -> None:
    payload = _bundle_payload()
    workspace = payload["workspace"]
    assert isinstance(workspace, dict)
    documents = workspace["documents"]
    assert isinstance(documents, list)
    documents[0][field] = replacement
    with pytest.raises(ValidationError):
        PlayerWorkspaceSnapshot.model_validate(workspace)


@pytest.mark.parametrize(
    "content_markdown",
    ("# PLAYER.md", "# PLAYER.md\n\n", "# PLAYER.md\r\n"),
)
def test_projected_document_rejects_noncanonical_markdown(
    content_markdown: str,
) -> None:
    payload = _document_payload(WorkspaceSection.PLAYER)
    payload["content_markdown"] = content_markdown
    payload["content_hash"] = hashlib.sha256(content_markdown.encode()).hexdigest()
    with pytest.raises(ValidationError):
        ProjectedDocument.model_validate(payload)


def test_projected_document_rejects_hash_not_matching_utf8_bytes() -> None:
    payload = _document_payload(WorkspaceSection.PLAYER)
    payload["content_hash"] = OTHER_HASH
    with pytest.raises(ValidationError):
        ProjectedDocument.model_validate(payload)


def test_workspace_rejects_forged_ordered_workspace_hash() -> None:
    payload = _bundle_payload()
    workspace = payload["workspace"]
    assert isinstance(workspace, dict)
    workspace["workspace_hash"] = OTHER_HASH
    with pytest.raises(ValidationError):
        PlayerWorkspaceSnapshot.model_validate(workspace)


def test_workspace_rejects_cross_document_source_hash_conflicts() -> None:
    payload = _bundle_payload()
    workspace = payload["workspace"]
    assert isinstance(workspace, dict)
    entries = workspace["manifest_entries"]
    documents = workspace["documents"]
    assert isinstance(entries, list)
    assert isinstance(documents, list)
    first_source = [{
        "record_kind": "commitment",
        "record_id": "record-1",
        "record_revision": 1,
        "content_hash": HASH,
    }]
    second_source = [{
        "record_kind": "commitment",
        "record_id": "record-1",
        "record_revision": 1,
        "content_hash": OTHER_HASH,
    }]
    documents[0]["source_references"] = first_source
    documents[1]["source_references"] = second_source
    entries[0]["source_references"] = first_source
    entries[1]["source_references"] = second_source
    with pytest.raises(ValidationError):
        PlayerWorkspaceSnapshot.model_validate(workspace)


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
    assert bundle.workspace.manifest_entries[0].content_hash == _document_hash(
        WorkspaceSection.PLAYER
    )
