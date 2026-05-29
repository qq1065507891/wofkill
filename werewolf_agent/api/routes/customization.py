"""Customization and marketplace routes."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from werewolf_agent.api.schemas import CallerRole
from werewolf_agent.customization.persona_adapter import adapt_persona_pack
from werewolf_agent.customization.preview import build_persona_preview
from werewolf_agent.customization.validators import (
    validate_persona_pack_yaml,
    validate_ruleset_yaml,
)
import yaml


def create_customization_router(
    *,
    authorized_callers: dict[str, CallerRole],
    customization_repo: Any,
    persist_fn: Any,
    project_root: Path,
) -> APIRouter:
    router = APIRouter()

    async def _read_upload_text(request: Request) -> str:
        body = await request.body()
        if len(body) > 256 * 1024:
            raise HTTPException(413, "Uploaded customization template is too large")
        return body.decode("utf-8")

    @router.get("/templates/ruleset", response_class=PlainTextResponse)
    def download_ruleset_template() -> PlainTextResponse:
        path = project_root / "config" / "rulesets" / "templates" / "custom_ruleset_template.yaml"
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    @router.get("/templates/persona-pack", response_class=PlainTextResponse)
    def download_persona_pack_template() -> PlainTextResponse:
        path = project_root / "config" / "personas" / "templates" / "player_profile_pack_template.yaml"
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    @router.post("/customization/rulesets/validate")
    async def validate_ruleset_upload(request: Request) -> dict:
        text = await _read_upload_text(request)
        return _validation_result_to_dict(validate_ruleset_yaml(text))

    @router.post("/customization/persona-packs/validate")
    async def validate_persona_pack_upload(request: Request) -> dict:
        text = await _read_upload_text(request)
        result = validate_persona_pack_yaml(text)
        data = _validation_result_to_dict(result)
        if result.normalized.get("players"):
            previews: dict[str, dict[str, str]] = {}
            for player in result.normalized["players"]:
                seat = int(player.get("seat", 0))
                previews[f"p{seat:02d}"] = build_persona_preview(player)
            data["persona_preview"] = previews
        return data

    @router.post("/customization/rulesets")
    async def save_ruleset_upload(
        request: Request,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
    ) -> dict:
        _require_customization_admin(authorized_callers, caller_id, caller_role)
        text = await _read_upload_text(request)
        result = validate_ruleset_yaml(text)
        data = _validation_result_to_dict(result)
        if not result.valid:
            raise HTTPException(400, data)
        record = customization_repo.save(
            config_type="ruleset",
            raw_yaml=text,
            normalized=result.normalized,
            validation_result=data,
            creator_id=caller_id,
        )
        persist_fn(record)
        return _record_to_public_dict(record)

    @router.post("/customization/persona-packs")
    async def save_persona_pack_upload(
        request: Request,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
    ) -> dict:
        _require_customization_admin(authorized_callers, caller_id, caller_role)
        text = await _read_upload_text(request)
        result = validate_persona_pack_yaml(text)
        data = _validation_result_to_dict(result)
        if not result.valid:
            raise HTTPException(400, data)
        adapted = adapt_persona_pack(result.normalized)
        normalized = dict(result.normalized)
        normalized["persona_profiles"] = adapted["persona_profiles"]
        normalized["player_assignments"] = adapted["player_assignments"]
        record = customization_repo.save(
            config_type="persona_pack",
            raw_yaml=text,
            normalized=normalized,
            validation_result=data,
            creator_id=caller_id,
        )
        persist_fn(record)
        return _record_to_public_dict(record)

    @router.get("/marketplace/rulesets")
    def list_ruleset_marketplace() -> dict:
        return _load_marketplace(str(project_root / "config" / "rulesets" / "marketplace.yaml"))

    @router.get("/marketplace/persona-packs")
    def list_persona_pack_marketplace() -> dict:
        return _load_marketplace(str(project_root / "config" / "personas" / "marketplace.yaml"))

    return router


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _require_customization_admin(
    authorized_callers: dict[str, CallerRole],
    caller_id: str,
    caller_role: CallerRole,
) -> None:
    if caller_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        raise HTTPException(403, "Customization save requires moderator or debugger access")
    if not caller_id or authorized_callers.get(caller_id) != caller_role:
        raise HTTPException(403, "Elevated caller role is not authorized")


def _validation_result_to_dict(result: Any) -> dict:
    return {
        "valid": result.valid,
        "summary": result.summary,
        "normalized": result.normalized,
        "errors": [asdict(issue) for issue in result.errors],
        "warnings": [asdict(issue) for issue in result.warnings],
        "diff_against_default": result.diff_against_default,
    }


def _record_to_public_dict(record: Any) -> dict:
    return {
        "config_id": record.config_id,
        "config_type": record.config_type,
        "content_hash": record.content_hash,
        "status": record.status,
        "version": record.version,
        "maturity": record.maturity,
        "compatibility_matrix": record.compatibility_matrix,
        "diff_against_default": record.diff_against_default,
        "creator_id": record.creator_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _record_to_storage_dict(record: Any) -> dict:
    return {
        "config_id": record.config_id,
        "config_type": record.config_type,
        "raw_yaml": record.raw_yaml,
        "normalized": record.normalized,
        "validation_result": record.validation_result,
        "content_hash": record.content_hash,
        "status": record.status,
        "version": record.version,
        "maturity": record.maturity,
        "compatibility_matrix": record.compatibility_matrix,
        "diff_against_default": record.diff_against_default,
        "creator_id": record.creator_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _load_marketplace(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"items": []}
    items = data.get("items", [])
    return {"items": items if isinstance(items, list) else []}
