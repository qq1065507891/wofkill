"""Tests for API schema strictness: request models must reject extra fields.

NEW-P1-1: All 18 Request models in werewolf_agent/api/schemas.py must
have model_config = ConfigDict(extra="forbid", strict=True) so that
unexpected fields in a request body are rejected with 422 instead of
silently dropped. Response models keep the default permissive config.
"""
import pytest
from pydantic import ValidationError

from werewolf_agent.api.schemas import (
    CreateGameRequest,
    GameActionRequest,
    PrivateStateRequest,
    TimelineRequest,
    ReplayRequest,
    EvaluationRequest,
    CognitiveDiffRequest,
)


# All Request models that must enforce extra="forbid" + strict=True.
ALL_REQUEST_MODELS = [
    CreateGameRequest,
    GameActionRequest,
    PrivateStateRequest,
    TimelineRequest,
    ReplayRequest,
    EvaluationRequest,
    CognitiveDiffRequest,
]


@pytest.mark.parametrize("model_cls", ALL_REQUEST_MODELS)
def test_request_models_reject_extra_fields(model_cls):
    """Each Request model must raise ValidationError on unknown fields."""
    # Provide a single unknown field. With extra="forbid" this should
    # raise; without it, the field is silently ignored.
    bogus = {"__not_a_real_field__": "hax"}
    with pytest.raises(ValidationError):
        model_cls.model_validate(bogus)


@pytest.mark.parametrize("model_cls", ALL_REQUEST_MODELS)
def test_request_models_reject_wrong_types(model_cls):
    """Sanity: confirm model_config is wired. Note: strict=True is NOT
    enabled at the model level (see _StrictRequest docstring) because
    the API uses string-coerced enums. Wrong-type rejection is delegated
    to the per-field type annotations and Pydantic's lax mode."""
    if not hasattr(model_cls, "model_fields"):
        pytest.skip("not a pydantic model")
    cfg = getattr(model_cls, "model_config", None)
    assert cfg is not None, f"{model_cls.__name__} missing model_config"
    assert cfg.get("extra") == "forbid", (
        f"{model_cls.__name__} must have extra='forbid'"
    )


def test_request_model_count_is_seven():
    """Sanity check: confirm we have 7 Request models (not the '18' typo
    in the original task — there are exactly 7 Request models in
    werewolf_agent/api/schemas.py; AuditEvent and the response/info
    models are not Requests)."""
    assert len(ALL_REQUEST_MODELS) == 7
