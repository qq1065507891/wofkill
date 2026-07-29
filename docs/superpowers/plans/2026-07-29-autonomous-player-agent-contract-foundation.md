# Autonomous Player Agent Contract Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the isolated, versioned, strict contract layer required by the first autonomous daytime-speech vertical slice without changing any live game path.

**Architecture:** Create a new `werewolf_agent.player_agents.contracts` package that has no dependency on legacy player-decision modules. Pydantic v2 models provide strict immutable versions, turn state, speech proposal, disclosure, public-record, and stable-error contracts; a checked-in canonical schema fixture makes contract drift reviewable before later provider-dialect transforms. Runtime scheduling, persistence transactions, model dispatch, rendering, and feature-gate integration are intentionally deferred to later plans.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, ruff, mypy, standard-library `ast`, `enum`, `hashlib`, and `json`; all Python commands use `conda run -n wofkill`.

---

## Scope and Follow-Up Plans

This is plan 1 of the stage-1 daytime-speech delivery. It produces useful,
testable contracts but does not route a live game through them.

The remaining stage-1 work is deliberately split into later plans:

1. repository revision, `CommitTurn`, audit, idempotency, and outbox transaction;
2. durable dispatch, serial public scheduler, and restart reconciliation;
3. player workspace projection, observation, context budget, and compaction;
4. minimal tool gateway and daytime speech agent loop;
5. public record commit, deterministic renderers, feature gate, and cutover gates.

Do not add compatibility adapters from the new package to `PlayerAgent`,
`SpeechAct`, `PlayerAction`, prompt directives, strategy handlers, or legacy
fallbacks in this plan.

## File Map

| File | Responsibility |
| --- | --- |
| `werewolf_agent/player_agents/__init__.py` | New subsystem namespace only |
| `werewolf_agent/player_agents/contracts/__init__.py` | Stable public contract exports |
| `werewolf_agent/player_agents/contracts/_base.py` | Shared strict-model and identifier/hash types |
| `werewolf_agent/player_agents/contracts/revisions.py` | Revision context and immutable read references |
| `werewolf_agent/player_agents/contracts/turns.py` | Legal windows, turn snapshots, budgets, and state transitions |
| `werewolf_agent/player_agents/contracts/errors.py` | Stable validation error codes and safe failures |
| `werewolf_agent/player_agents/contracts/speech.py` | Complete stage-1 speech move union and delivery plan |
| `werewolf_agent/player_agents/contracts/proposals.py` | Host-bound terminal speech envelope |
| `werewolf_agent/player_agents/contracts/disclosure.py` | One-time private disclosure grant contract |
| `werewolf_agent/player_agents/contracts/records.py` | Immutable public speech and rendered utterance records |
| `werewolf_agent/player_agents/contracts/schema_catalog.py` | Canonical schema export and content hash |
| `scripts/export_player_agent_schemas.py` | Deterministic schema fixture exporter |
| `tests/player_agents/` | Focused contract and package-boundary tests |
| `tests/fixtures/player_agents/speech_proposal_schema_v1.json` | Reviewed canonical JSON Schema snapshot for provider adapters |

### Task 1: Establish the Isolated Package Boundary

**Files:**
- Create: `tests/player_agents/__init__.py`
- Create: `tests/player_agents/test_import_boundary.py`
- Create: `werewolf_agent/player_agents/__init__.py`
- Create: `werewolf_agent/player_agents/contracts/__init__.py`

- [ ] **Step 1: Write the failing import-boundary test**

```python
# -*- coding: utf-8 -*-
"""
验证新玩家运行时包不依赖已废弃的玩家决策模块。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2] / "werewolf_agent" / "player_agents"
FORBIDDEN_PREFIXES = (
    "werewolf_agent.agents.player",
    "werewolf_agent.agents.action_schemas",
    "werewolf_agent.agents.schemas",
    "werewolf_agent.agents.speech_act_schemas",
    "werewolf_agent.runtime.agent_action_pipeline",
    "werewolf_agent.runtime.agent_adapter",
    "werewolf_agent.runtime.directives",
    "werewolf_agent.runtime.strategy",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_player_agents_package_exists_and_has_no_legacy_decision_imports() -> None:
    assert PACKAGE_ROOT.is_dir()
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for module in sorted(_imported_modules(path)):
            if module.startswith(FORBIDDEN_PREFIXES):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {module}")
    assert violations == []
```

Create `tests/player_agents/__init__.py` so pytest treats the directory
consistently with existing test packages:

```python
# -*- coding: utf-8 -*-
"""
自主玩家智能体契约测试包。

作者: Project contributors
创建日期: 2026-07-29
"""
```

- [ ] **Step 2: Run the boundary test and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_import_boundary.py -v
```

Expected: FAIL at `assert PACKAGE_ROOT.is_dir()` because the new package does
not exist.

- [ ] **Step 3: Add minimal namespace modules**

Create both package modules with the project-required header and no legacy
imports:

```python
# -*- coding: utf-8 -*-
"""
自主玩家智能体运行时命名空间；具体能力通过子包提供。

作者: Project contributors
创建日期: 2026-07-29
"""
```

Use this contract-package variant for `contracts/__init__.py`:

```python
# -*- coding: utf-8 -*-
"""
自主玩家智能体的严格、版本化且无副作用的公共契约。

作者: Project contributors
创建日期: 2026-07-29
"""
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_import_boundary.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the isolated namespace**

```bash
git add werewolf_agent/player_agents tests/player_agents
git commit -m "feat: establish autonomous player package boundary"
```

### Task 2: Define Revision and Read-Set Contracts

**Files:**
- Create: `werewolf_agent/player_agents/contracts/_base.py`
- Create: `werewolf_agent/player_agents/contracts/revisions.py`
- Create: `tests/player_agents/test_revision_contracts.py`
- Modify: `werewolf_agent/player_agents/contracts/__init__.py`

- [ ] **Step 1: Write failing revision-contract tests**

```python
# -*- coding: utf-8 -*-
"""
验证游戏修订版本、视图指纹和不可变读取引用契约。

作者: Project contributors
创建日期: 2026-07-29
"""

import json

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def test_revision_context_is_strict_and_immutable() -> None:
    context = RevisionContext(
        base_revision=7,
        window_id="window-1",
        window_version=2,
        view_fingerprint=HASH_A,
    )
    assert context.base_revision == 7
    with pytest.raises(ValidationError):
        RevisionContext.model_validate({
            "base_revision": "7",
            "window_id": "window-1",
            "window_version": 2,
            "view_fingerprint": HASH_A,
        })
    with pytest.raises(ValidationError):
        context.base_revision = 8


def test_read_reference_rejects_invalid_hash_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReadReference(record_id="record-1", revision=1, content_hash="short")
    with pytest.raises(ValidationError):
        ReadReference.model_validate({
            "record_id": "record-1",
            "revision": 1,
            "content_hash": HASH_B,
            "payload": "forbidden",
        })


def test_read_reference_accepts_revision_zero() -> None:
    reference = ReadReference(
        record_id="ruleset-snapshot",
        revision=0,
        content_hash=HASH_B,
    )
    assert reference.revision == 0
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_revision_contracts.py -v
```

Expected: collection ERROR because `contracts.revisions` does not exist.

- [ ] **Step 3: Implement the shared strict base types**

Create `_base.py`:

```python
# -*- coding: utf-8 -*-
"""
提供玩家智能体契约共享的严格不可变模型、标识符和哈希类型。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from typing import Any, Annotated, Iterable, Mapping, Self, TypeVar

from pydantic import BaseModel, ConfigDict, StringConstraints


NonEmptyId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
ContentHash = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class StrictFrozenModel(BaseModel):
    """拒绝额外字段、隐式类型转换和实例修改的基础模型。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """复制模型，并对所有更新重新执行完整契约校验。"""
        if not update:
            return super().model_copy(deep=deep)
        data = self.model_dump(round_trip=True)
        data.update(update)
        return type(self).model_validate(data)

    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """将旧复制 API 委托给受校验路径，并拒绝不完整副本。"""
        if include is not None or exclude is not None:
            raise TypeError("partial copies are not supported for strict contracts")
        return self.model_copy(update=update, deep=deep)


T = TypeVar("T")


def require_unique(values: Iterable[T], *, field_name: str) -> tuple[T, ...]:
    """返回稳定元组，并拒绝会使引用语义产生歧义的重复项。"""
    items = tuple(values)
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return items
```

- [ ] **Step 4: Implement revision models**

Create `revisions.py`:

```python
# -*- coding: utf-8 -*-
"""
定义终端提案与工具读取共用的修订版本和读取集合契约。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from pydantic import Field

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
)


class ReadReference(StrictFrozenModel):
    """把一次读取绑定到不可变记录、产生版本和内容哈希。"""

    record_id: NonEmptyId
    revision: int = Field(ge=0)
    content_hash: ContentHash


class RevisionContext(StrictFrozenModel):
    """绑定提案所依据的游戏、窗口和查看者视图版本。"""

    base_revision: int = Field(ge=0)
    window_id: NonEmptyId
    window_version: int = Field(ge=1)
    view_fingerprint: ContentHash
```

- [ ] **Step 5: Export and verify the revision contracts**

Add imports and `__all__` entries in `contracts/__init__.py`:

```python
from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)

__all__ = ["ReadReference", "RevisionContext"]
```

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_revision_contracts.py tests/player_agents/test_import_boundary.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit revision contracts**

```bash
git add werewolf_agent/player_agents/contracts tests/player_agents
git commit -m "feat: define autonomous player revision contracts"
```

### Task 3: Define Legal Windows and Durable Turn State

**Files:**
- Create: `werewolf_agent/player_agents/contracts/turns.py`
- Create: `tests/player_agents/test_turn_contracts.py`
- Modify: `werewolf_agent/player_agents/contracts/__init__.py`

- [ ] **Step 1: Write failing window and transition tests**

```python
# -*- coding: utf-8 -*-
"""
验证合法动作窗口、回合快照和显式状态转换。

作者: Project contributors
创建日期: 2026-07-29
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)
from werewolf_agent.player_agents.contracts.turns import (
    AgentTurn,
    AgentTurnStatus,
    ConflictClass,
    LegalActionWindow,
    TurnBudget,
    transition_turn,
)


HASH = "a" * 64


def _window() -> LegalActionWindow:
    return LegalActionWindow(
        window_id="speech-d1-p01",
        version=1,
        game_id="game-1",
        task_type="day_speech",
        conflict_class=ConflictClass.SERIAL_PUBLIC,
        participant_ids=("p01",),
        legal_actions=("speech",),
        legal_target_ids=("p02", "p03"),
        opened_revision=4,
        deadline=datetime(2026, 7, 29, 1, tzinfo=timezone.utc),
    )


def _turn() -> AgentTurn:
    return AgentTurn(
        turn_id="turn-1",
        game_id="game-1",
        player_id="p01",
        role_id="villager",
        phase="day_discussion",
        task_type="day_speech",
        revision=RevisionContext(
            base_revision=4,
            window_id="speech-d1-p01",
            window_version=1,
            view_fingerprint=HASH,
        ),
        window=_window(),
        read_set=(
            ReadReference(record_id="public-4", revision=4, content_hash=HASH),
        ),
        model_lease_hash=HASH,
        budget=TurnBudget(model_steps=8, tool_calls=12, repairs=1),
        status=AgentTurnStatus.OPEN,
        idempotency_key="turn-1:submit",
    )


def test_turn_requires_matching_window_and_participant() -> None:
    turn = _turn()
    assert turn.window.conflict_class is ConflictClass.SERIAL_PUBLIC
    with pytest.raises(ValidationError, match="player must be a window participant"):
        AgentTurn.model_validate({**turn.model_dump(), "player_id": "p09"})


def test_window_rejects_duplicate_participants_and_naive_deadline() -> None:
    data = _window().model_dump()
    with pytest.raises(ValidationError):
        LegalActionWindow.model_validate({
            **data,
            "participant_ids": ("p01", "p01"),
        })
    with pytest.raises(ValidationError, match="timezone-aware"):
        LegalActionWindow.model_validate({
            **data,
            "deadline": datetime(2026, 7, 29, 1),
        })


def test_transition_turn_allows_only_declared_edges() -> None:
    observing = transition_turn(_turn(), AgentTurnStatus.OBSERVING)
    thinking = transition_turn(observing, AgentTurnStatus.THINKING)
    assert thinking.status is AgentTurnStatus.THINKING
    with pytest.raises(ValueError, match="illegal agent turn transition"):
        transition_turn(thinking, AgentTurnStatus.COMMITTED)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_turn_contracts.py -v
```

Expected: collection ERROR because `contracts.turns` does not exist.

- [ ] **Step 3: Implement turn contracts and transition graph**

Create `turns.py` with the following public API and validation:

```python
# -*- coding: utf-8 -*-
"""
定义合法动作窗口、持久化玩家回合快照和允许的状态转换。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
    require_unique,
)
from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)


class ConflictClass(StrEnum):
    SERIAL_PUBLIC = "serial_public"
    SERIAL_PRIVATE = "serial_private"
    COMMUTATIVE_PRIVATE = "commutative_private"
    TEAM_COORDINATOR = "team_coordinator"


class AgentTurnStatus(StrEnum):
    OPEN = "open"
    OBSERVING = "observing"
    THINKING = "thinking"
    WAITING_TOOL = "waiting_tool"
    COMPACTING = "compacting"
    SUBMITTED = "submitted"
    VALIDATING = "validating"
    REPAIRING = "repairing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class LegalActionWindow(StrictFrozenModel):
    window_id: NonEmptyId
    version: int = Field(ge=1)
    game_id: NonEmptyId
    task_type: NonEmptyId
    conflict_class: ConflictClass
    participant_ids: tuple[NonEmptyId, ...] = Field(min_length=1)
    legal_actions: tuple[NonEmptyId, ...] = Field(min_length=1)
    legal_target_ids: tuple[NonEmptyId, ...] = ()
    opened_revision: int = Field(ge=0)
    deadline: datetime

    @field_validator("deadline")
    @classmethod
    def _aware_deadline(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("deadline must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _unique_sets(self) -> "LegalActionWindow":
        require_unique(self.participant_ids, field_name="participant_ids")
        require_unique(self.legal_actions, field_name="legal_actions")
        require_unique(self.legal_target_ids, field_name="legal_target_ids")
        return self


class TurnBudget(StrictFrozenModel):
    model_steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    repairs: int = Field(ge=0)


class AgentTurn(StrictFrozenModel):
    turn_id: NonEmptyId
    game_id: NonEmptyId
    player_id: NonEmptyId
    role_id: NonEmptyId
    phase: NonEmptyId
    task_type: NonEmptyId
    revision: RevisionContext
    window: LegalActionWindow
    read_set: tuple[ReadReference, ...] = ()
    model_lease_hash: ContentHash
    budget: TurnBudget
    status: AgentTurnStatus
    idempotency_key: NonEmptyId

    @model_validator(mode="after")
    def _consistent_context(self) -> "AgentTurn":
        if self.game_id != self.window.game_id:
            raise ValueError("turn game_id must match window game_id")
        if self.task_type != self.window.task_type:
            raise ValueError("turn task_type must match window task_type")
        if self.player_id not in self.window.participant_ids:
            raise ValueError("player must be a window participant")
        if self.revision.window_id != self.window.window_id:
            raise ValueError("revision window_id must match window")
        if self.revision.window_version != self.window.version:
            raise ValueError("revision window_version must match window")
        require_unique(
            (item.record_id for item in self.read_set),
            field_name="read_set record IDs",
        )
        return self


_TERMINAL = {
    AgentTurnStatus.COMMITTED,
    AgentTurnStatus.CANCELLED,
    AgentTurnStatus.EXPIRED,
}
_ALLOWED: dict[AgentTurnStatus, frozenset[AgentTurnStatus]] = {
    AgentTurnStatus.OPEN: frozenset({AgentTurnStatus.OBSERVING, AgentTurnStatus.CANCELLED, AgentTurnStatus.EXPIRED}),
    AgentTurnStatus.OBSERVING: frozenset({AgentTurnStatus.THINKING, AgentTurnStatus.CANCELLED, AgentTurnStatus.EXPIRED}),
    AgentTurnStatus.THINKING: frozenset({AgentTurnStatus.WAITING_TOOL, AgentTurnStatus.COMPACTING, AgentTurnStatus.SUBMITTED, AgentTurnStatus.CANCELLED, AgentTurnStatus.EXPIRED}),
    AgentTurnStatus.WAITING_TOOL: frozenset({AgentTurnStatus.THINKING, AgentTurnStatus.CANCELLED, AgentTurnStatus.EXPIRED}),
    AgentTurnStatus.COMPACTING: frozenset({AgentTurnStatus.THINKING, AgentTurnStatus.CANCELLED, AgentTurnStatus.EXPIRED}),
    AgentTurnStatus.SUBMITTED: frozenset({AgentTurnStatus.VALIDATING, AgentTurnStatus.CANCELLED, AgentTurnStatus.EXPIRED}),
    AgentTurnStatus.VALIDATING: frozenset({AgentTurnStatus.COMMITTED, AgentTurnStatus.REPAIRING, AgentTurnStatus.CANCELLED, AgentTurnStatus.EXPIRED}),
    AgentTurnStatus.REPAIRING: frozenset({AgentTurnStatus.SUBMITTED, AgentTurnStatus.CANCELLED, AgentTurnStatus.EXPIRED}),
    **{status: frozenset() for status in _TERMINAL},
}


def transition_turn(turn: AgentTurn, next_status: AgentTurnStatus) -> AgentTurn:
    if next_status not in _ALLOWED[turn.status]:
        raise ValueError(
            f"illegal agent turn transition: {turn.status.value} -> {next_status.value}"
        )
    return AgentTurn.model_validate({**turn.model_dump(), "status": next_status})
```

- [ ] **Step 4: Export and verify turn contracts**

Export `AgentTurn`, `AgentTurnStatus`, `ConflictClass`, `LegalActionWindow`,
`TurnBudget`, and `transition_turn` from `contracts/__init__.py`.

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_turn_contracts.py tests/player_agents/test_revision_contracts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit turn contracts**

```bash
git add werewolf_agent/player_agents/contracts tests/player_agents
git commit -m "feat: define autonomous player turn state"
```

### Task 4: Define Stable Validation Failures

**Files:**
- Create: `werewolf_agent/player_agents/contracts/errors.py`
- Create: `tests/player_agents/test_contract_errors.py`
- Modify: `werewolf_agent/player_agents/contracts/__init__.py`

- [ ] **Step 1: Write failing safe-error tests**

```python
# -*- coding: utf-8 -*-
"""
验证提案校验错误使用稳定代码、JSON Pointer 路径且不携带隐藏值。

作者: Project contributors
创建日期: 2026-07-29
"""

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.errors import (
    ProposalFailure,
    ValidationErrorCode,
)


def test_failure_serializes_stable_code_and_field_path() -> None:
    failure = ProposalFailure.for_code(
        code=ValidationErrorCode.TARGET_NOT_LEGAL,
        field_path="/body/moves/0/target_id",
        repairable=False,
    )
    assert failure.model_dump(mode="json")["code"] == "target_not_legal"
    assert failure.message == "target is not legal for this window"


def test_failure_rejects_non_json_pointer_and_extra_context() -> None:
    with pytest.raises(ValidationError):
        ProposalFailure(
            code=ValidationErrorCode.INVISIBLE_REFERENCE,
            field_path="body.secret",
            message="reference is not visible",
            repairable=False,
        )
    with pytest.raises(ValidationError):
        ProposalFailure.model_validate({
            "code": ValidationErrorCode.INVISIBLE_REFERENCE,
            "field_path": "/body/ref",
            "message": "reference is not visible",
            "repairable": False,
            "hidden_value": "seer:p03",
        })


def test_failure_rejects_unsafe_message_and_invalid_repair_scope() -> None:
    with pytest.raises(ValidationError, match="safe message catalog"):
        ProposalFailure(
            code=ValidationErrorCode.INVISIBLE_REFERENCE,
            field_path="/body/ref",
            message="hidden role is seer:p03",
            repairable=False,
        )
    with pytest.raises(ValidationError, match="not repairable"):
        ProposalFailure.for_code(
            code=ValidationErrorCode.STALE_READ_SET,
            field_path="/read_set",
            repairable=True,
        )
    with pytest.raises(ValidationError, match="field-local"):
        ProposalFailure.for_code(
            code=ValidationErrorCode.SEMANTIC_MISMATCH,
            field_path="",
            repairable=True,
        )


def test_failure_copy_updates_are_fully_revalidated() -> None:
    failure = ProposalFailure.for_code(
        code=ValidationErrorCode.INVISIBLE_REFERENCE,
        field_path="/body/ref",
        repairable=False,
    )
    invalid_updates = (
        {"message": "hidden role is seer:p03"},
        {"repairable": True},
        {"field_path": "body.ref"},
        {"hidden_value": "seer:p03"},
    )
    for update in invalid_updates:
        with pytest.raises(ValidationError):
            failure.model_copy(update=update)
        with pytest.raises(ValidationError):
            failure.copy(update=update)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_contract_errors.py -v
```

Expected: collection ERROR because `contracts.errors` does not exist.

- [ ] **Step 3: Implement stable failures**

```python
# -*- coding: utf-8 -*-
"""
定义玩家终端提案校验的稳定错误代码和安全失败载荷。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import StringConstraints, model_validator
from typing import Annotated

from werewolf_agent.player_agents.contracts._base import StrictFrozenModel


JsonPointer = Annotated[str, StringConstraints(pattern=r"^(|/(?:[^~/]|~0|~1)*)+$")]


class ValidationErrorCode(StrEnum):
    SCHEMA_INVALID = "schema_invalid"
    BOUND_CONTEXT_MISMATCH = "bound_context_mismatch"
    UNKNOWN_SCHEMA_VERSION = "unknown_schema_version"
    UNKNOWN_CAPABILITY = "unknown_capability"
    WRONG_ACTION_WINDOW = "wrong_action_window"
    STALE_READ_SET = "stale_read_set"
    TARGET_NOT_LEGAL = "target_not_legal"
    INVISIBLE_REFERENCE = "invisible_reference"
    GRANT_INACTIVE = "grant_inactive"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    RULE_ILLEGAL = "rule_illegal"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    SECURITY_VIOLATION = "security_violation"


_SAFE_MESSAGES: dict[ValidationErrorCode, str] = {
    ValidationErrorCode.SCHEMA_INVALID: "proposal does not match the required schema",
    ValidationErrorCode.BOUND_CONTEXT_MISMATCH: "proposal context does not match the bound turn",
    ValidationErrorCode.UNKNOWN_SCHEMA_VERSION: "proposal schema version is not supported",
    ValidationErrorCode.UNKNOWN_CAPABILITY: "proposal capability is not supported",
    ValidationErrorCode.WRONG_ACTION_WINDOW: "proposal is not legal for the current window",
    ValidationErrorCode.STALE_READ_SET: "proposal read set is stale",
    ValidationErrorCode.TARGET_NOT_LEGAL: "target is not legal for this window",
    ValidationErrorCode.INVISIBLE_REFERENCE: "reference is not visible",
    ValidationErrorCode.GRANT_INACTIVE: "disclosure grant is not active",
    ValidationErrorCode.SEMANTIC_MISMATCH: "proposal field does not match its semantic constraints",
    ValidationErrorCode.RULE_ILLEGAL: "proposal is not legal under the current rules",
    ValidationErrorCode.IDEMPOTENCY_CONFLICT: "idempotency key conflicts with an existing proposal",
    ValidationErrorCode.SECURITY_VIOLATION: "proposal violates a security constraint",
}


class ProposalFailure(StrictFrozenModel):
    code: ValidationErrorCode
    field_path: JsonPointer
    message: Annotated[str, StringConstraints(min_length=1, max_length=240)]
    repairable: bool

    @classmethod
    def for_code(
        cls,
        *,
        code: ValidationErrorCode,
        field_path: str,
        repairable: bool,
    ) -> "ProposalFailure":
        """使用封闭的安全消息目录构造失败载荷。"""
        return cls(
            code=code,
            field_path=field_path,
            message=_SAFE_MESSAGES[code],
            repairable=repairable,
        )

    @model_validator(mode="after")
    def _safe_repair_contract(self) -> "ProposalFailure":
        if self.message != _SAFE_MESSAGES[self.code]:
            raise ValueError("message must come from the safe message catalog")
        if self.repairable and self.code not in {
            ValidationErrorCode.SCHEMA_INVALID,
            ValidationErrorCode.SEMANTIC_MISMATCH,
        }:
            raise ValueError(f"{self.code.value} is not repairable")
        if (
            self.repairable
            and self.code is ValidationErrorCode.SEMANTIC_MISMATCH
            and self.field_path == ""
        ):
            raise ValueError("repairable semantic_mismatch must be field-local")
        return self
```

Validators that create a `ProposalFailure` must pass a `field_path` rendered
only from the canonical schema/model error location. Do not interpolate record
contents, player or role identifiers, observed values, or tool output into the
path. The contract model enforces the closed safe-message catalog and repair
eligibility; the validator boundary owns the provenance of the JSON Pointer.

- [ ] **Step 4: Export, run tests, and commit**

Export `ProposalFailure` and `ValidationErrorCode`. Then run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_contract_errors.py -v
git add werewolf_agent/player_agents/contracts tests/player_agents
git commit -m "feat: add stable player contract errors"
```

Expected: PASS, followed by a successful commit.

### Task 5: Implement the Strict Daytime Speech Proposal

**Files:**
- Create: `werewolf_agent/player_agents/contracts/speech.py`
- Create: `werewolf_agent/player_agents/contracts/proposals.py`
- Create: `tests/player_agents/test_speech_contracts.py`
- Modify: `werewolf_agent/player_agents/contracts/__init__.py`

- [ ] **Step 1: Write failing happy-path and strictness tests**

The first test fixture must use the public API the provider gateway will use:

```python
# -*- coding: utf-8 -*-
"""
验证昼间发言动作联合、引用关系、交付计划和终端信封。

作者: Project contributors
创建日期: 2026-07-29
"""

import json

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.proposals import SpeechProposalEnvelope


HASH = "a" * 64


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "turn_id": "turn-1",
        "player_id": "p01",
        "window_id": "speech-d1-p01",
        "window_version": 1,
        "base_revision": 4,
        "view_fingerprint": HASH,
        "body": {
            "kind": "speech",
            "objective": "declare_vote_position",
            "moves": [
                {
                    "move_id": "m1",
                    "move_type": "alignment_read",
                    "modality": "suspected",
                    "evidence_refs": ["public-3"],
                    "target_id": "p03",
                    "alignment": "wolf",
                    "strength": "leaning",
                },
                {
                    "move_id": "m2",
                    "move_type": "vote_position",
                    "modality": "asserted",
                    "evidence_refs": ["public-3"],
                    "target_id": "p03",
                    "commitment": "provisional",
                },
            ],
            "response_record_refs": [],
            "delivery_plan": {
                "tone": "firm",
                "length_class": "standard",
                "address_style": "room",
                "move_order": ["m1", "m2"],
                "emphasis_move_ids": ["m2"],
                "connector_ids": ["because"],
            },
        },
    }


def test_speech_envelope_parses_discriminated_moves() -> None:
    proposal = SpeechProposalEnvelope.model_validate_json(json.dumps(_payload()))
    assert proposal.body.moves[0].move_type == "alignment_read"
    assert proposal.body.delivery_plan.move_order == ("m1", "m2")


def test_speech_rejects_extra_fields_and_bad_move_order() -> None:
    payload = _payload()
    payload["body"]["moves"][0]["reasoning"] = "private thought"  # type: ignore[index]
    with pytest.raises(ValidationError):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))

    payload = _payload()
    payload["body"]["delivery_plan"]["move_order"] = ["m1"]  # type: ignore[index]
    with pytest.raises(ValidationError, match="move_order must contain every move ID"):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))


def test_speech_rejects_duplicate_and_cyclic_move_references() -> None:
    payload = _payload()
    payload["body"]["moves"][1]["move_id"] = "m1"  # type: ignore[index]
    with pytest.raises(ValidationError, match="move IDs must not contain duplicates"):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))

    payload = _payload()
    payload["body"]["moves"] = [  # type: ignore[index]
        {
            "move_id": "m1",
            "move_type": "public_evidence_citation",
            "modality": "asserted",
            "evidence_refs": ["public-3"],
            "relation": "supports",
            "subject_ids": ["p03"],
            "supports_move_ids": ["m2"],
        },
        {
            "move_id": "m2",
            "move_type": "public_evidence_citation",
            "modality": "asserted",
            "evidence_refs": ["public-4"],
            "relation": "supports",
            "subject_ids": ["p03"],
            "supports_move_ids": ["m1"],
        },
    ]
    with pytest.raises(ValidationError, match="move references must be acyclic"):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))


def test_speech_rejects_duplicate_response_record_refs() -> None:
    payload = _payload()
    payload["body"]["response_record_refs"] = [  # type: ignore[index]
        "public-3",
        "public-3",
    ]
    with pytest.raises(
        ValidationError,
        match="response_record_refs must not contain duplicates",
    ):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))


def test_speech_rejects_non_actor_role_claim() -> None:
    payload = _payload()
    payload["body"]["moves"][0] = {  # type: ignore[index]
        "move_id": "m1",
        "move_type": "role_claim",
        "modality": "asserted",
        "claimant_id": "p02",
        "role_id": "seer",
        "claim_mode": "claim",
    }
    with pytest.raises(ValidationError, match="role claim claimant must match player"):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))


def test_speech_rejects_external_refs_to_proposal_moves() -> None:
    payload = _payload()
    payload["body"]["moves"][0] = {  # type: ignore[index]
        "move_id": "m1",
        "move_type": "role_claim",
        "modality": "quoted",
        "claimant_id": "p01",
        "role_id": "seer",
        "claim_mode": "quote",
        "source_record_id": "m1",
    }
    payload["body"]["response_record_refs"] = ["m1"]  # type: ignore[index]
    with pytest.raises(
        ValidationError,
        match="external record refs must not reference proposal moves",
    ):
        SpeechProposalEnvelope.model_validate_json(json.dumps(payload))
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_speech_contracts.py -v
```

Expected: collection ERROR because the speech/proposal modules do not exist.

- [ ] **Step 3: Implement speech enums, common fields, and the full move union**

Create `speech.py` with this header, imports, enums, and move models:

```python
# -*- coding: utf-8 -*-
"""
定义昼间发言的严格语义动作联合、引用关系和交付计划。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from werewolf_agent.player_agents.contracts._base import (
    NonEmptyId,
    StrictFrozenModel,
    require_unique,
)


class Modality(StrEnum):
    ASSERTED = "asserted"
    SUSPECTED = "suspected"
    CONDITIONAL = "conditional"
    HYPOTHETICAL = "hypothetical"
    QUOTED = "quoted"


class SpeechObjective(StrEnum):
    STATE_CASE = "state_case"
    CHALLENGE_CLAIM = "challenge_claim"
    ANSWER_QUESTION = "answer_question"
    ASK_QUESTION = "ask_question"
    DEFEND_SELF = "defend_self"
    DECLARE_VOTE_POSITION = "declare_vote_position"
    RETRACT_OR_CORRECT = "retract_or_correct"
    EXPRESS_UNCERTAINTY = "express_uncertainty"
    NO_NEW_INFORMATION = "no_new_information"


class Alignment(StrEnum):
    GOOD = "good"
    WOLF = "wolf"
    UNCERTAIN = "uncertain"


class Strength(StrEnum):
    LEANING = "leaning"
    PROBABLE = "probable"
    COMMITTED = "committed"


class ClaimMode(StrEnum):
    CLAIM = "claim"
    DENY = "deny"
    QUOTE = "quote"


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"


class ComparisonDimension(StrEnum):
    CLAIM = "claim"
    VOTE = "vote"
    EVIDENCE = "evidence"
    COMMITMENT = "commitment"
    TIMELINE_CONSISTENCY = "timeline_consistency"


class QuestionTopic(StrEnum):
    ROLE_CLAIM = "role_claim"
    ALIGNMENT_READ = "alignment_read"
    VOTE_POSITION = "vote_position"
    EVIDENCE = "evidence"
    TIMELINE = "timeline"
    COMMITMENT = "commitment"


class RequestedField(StrEnum):
    CLAIM = "claim"
    EVIDENCE = "evidence"
    REASON = "reason"
    TIMELINE = "timeline"
    VOTE = "vote"
    CONFIDENCE = "confidence"


class ResponseKind(StrEnum):
    AGREE = "agree"
    DISAGREE = "disagree"
    CLARIFY = "clarify"
    CHALLENGE = "challenge"


class VoteCommitment(StrEnum):
    LEANING = "leaning"
    PROVISIONAL = "provisional"
    COMMITTED = "committed"


class ConsequenceKind(StrEnum):
    SUPPORT = "support"
    CHALLENGE = "challenge"
    VOTE_POSITION = "vote_position"
    SELF_DISCLOSURE = "self_disclosure"


class UncertaintyDimension(StrEnum):
    ROLE = "role"
    ALIGNMENT = "alignment"
    VOTE = "vote"
    CLAIM = "claim"
    TIMELINE = "timeline"


class ConfidenceBucket(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Tone(StrEnum):
    CALM = "calm"
    FIRM = "firm"
    SKEPTICAL = "skeptical"
    URGENT = "urgent"
    DEFENSIVE = "defensive"
    CONCILIATORY = "conciliatory"


class LengthClass(StrEnum):
    BRIEF = "brief"
    STANDARD = "standard"
    EXTENDED = "extended"


class AddressStyle(StrEnum):
    ROOM = "room"
    TARGETED = "targeted"
    MIXED = "mixed"


class PrivateFactKind(StrEnum):
    ALIGNMENT_CHECK = "alignment_check"
    ROLE_CHECK = "role_check"
    ATTACK = "attack"
    ABILITY_RESULT = "ability_result"


class BaseMove(StrictFrozenModel):
    move_id: NonEmptyId
    move_type: NonEmptyId
    modality: Modality
    evidence_refs: tuple[NonEmptyId, ...] = ()

    @field_validator("evidence_refs")
    @classmethod
    def _unique_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return require_unique(value, field_name="evidence_refs")


class AlignmentRead(BaseMove):
    move_type: Literal["alignment_read"]
    target_id: NonEmptyId
    alignment: Alignment
    strength: Strength


class RoleClaim(BaseMove):
    move_type: Literal["role_claim"]
    claimant_id: NonEmptyId
    role_id: NonEmptyId
    claim_mode: ClaimMode
    source_record_id: NonEmptyId | None = None

    @model_validator(mode="after")
    def _quoted_claim_has_source(self) -> "RoleClaim":
        if (self.claim_mode is ClaimMode.QUOTE) != (self.source_record_id is not None):
            raise ValueError("quoted role claim requires exactly one source record")
        return self


class PrivateResultDisclosure(BaseMove):
    move_type: Literal["private_result_disclosure"]
    fact_kind: PrivateFactKind
    fact_ref: NonEmptyId
    disclosure_grant_id: NonEmptyId
    timing_ref: NonEmptyId
    result_value_id: NonEmptyId
    target_id: NonEmptyId | None = None


class PublicEvidenceCitation(BaseMove):
    move_type: Literal["public_evidence_citation"]
    relation: EvidenceRelation
    subject_ids: tuple[NonEmptyId, ...] = Field(min_length=1, max_length=4)
    supports_move_ids: tuple[NonEmptyId, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _unique_citation_refs(self) -> "PublicEvidenceCitation":
        require_unique(self.subject_ids, field_name="subject_ids")
        require_unique(self.supports_move_ids, field_name="supports_move_ids")
        return self


class ComparisonAssessment(StrictFrozenModel):
    player_id: NonEmptyId
    value_id: NonEmptyId
    evidence_refs: tuple[NonEmptyId, ...] = Field(min_length=1)


class PlayerComparison(BaseMove):
    move_type: Literal["player_comparison"]
    dimension: ComparisonDimension
    assessments: tuple[ComparisonAssessment, ...] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def _unique_players(self) -> "PlayerComparison":
        require_unique(
            (item.player_id for item in self.assessments),
            field_name="comparison players",
        )
        return self


class QuestionMove(BaseMove):
    move_type: Literal["question"]
    target_id: NonEmptyId
    topic: QuestionTopic
    requested_fields: tuple[RequestedField, ...] = Field(min_length=1)

    @field_validator("requested_fields")
    @classmethod
    def _unique_requested_fields(
        cls,
        value: tuple[RequestedField, ...],
    ) -> tuple[RequestedField, ...]:
        return require_unique(value, field_name="requested_fields")


class ResponseMove(BaseMove):
    move_type: Literal["response"]
    source_record_id: NonEmptyId
    response_kind: ResponseKind


class VotePosition(BaseMove):
    move_type: Literal["vote_position"]
    target_id: NonEmptyId
    commitment: VoteCommitment


class CommitmentCondition(StrictFrozenModel):
    condition_id: NonEmptyId
    kind_id: NonEmptyId
    record_refs: tuple[NonEmptyId, ...] = ()

    @field_validator("record_refs")
    @classmethod
    def _unique_records(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return require_unique(value, field_name="record_refs")


class CommitmentConsequence(StrictFrozenModel):
    kind: ConsequenceKind
    target_id: NonEmptyId | None = None


class ConditionalCommitment(BaseMove):
    move_type: Literal["conditional_commitment"]
    condition: CommitmentCondition
    consequence: CommitmentConsequence
    expires_at_phase: NonEmptyId


class RetractionMove(BaseMove):
    move_type: Literal["retraction"]
    prior_public_move_ref: NonEmptyId
    replacement_move_id: NonEmptyId | None = None


class UncertaintyAlternative(StrictFrozenModel):
    value_id: NonEmptyId
    confidence: ConfidenceBucket
    support_refs: tuple[NonEmptyId, ...] = ()


class UncertaintyStatement(BaseMove):
    move_type: Literal["uncertainty"]
    subject_id: NonEmptyId
    dimension: UncertaintyDimension
    alternatives: tuple[UncertaintyAlternative, ...] = Field(min_length=2, max_length=4)


SpeechMove = Annotated[
    AlignmentRead
    | RoleClaim
    | PrivateResultDisclosure
    | PublicEvidenceCitation
    | PlayerComparison
    | QuestionMove
    | ResponseMove
    | VotePosition
    | ConditionalCommitment
    | RetractionMove
    | UncertaintyStatement,
    Field(discriminator="move_type"),
]
```

`kind_id` and `value_id` are catalog IDs, not free text. The later Host semantic
validator resolves them against the ruleset/schema snapshot pinned by the turn.

- [ ] **Step 4: Implement speech-body cross-field validation**

Add `DeliveryPlan` and `SpeechProposalBody`:

```python
def _require_acyclic(graph: dict[str, set[str]]) -> None:
    unseen, visiting, complete = 0, 1, 2
    colors = {node: unseen for node in graph}

    def visit(node: str) -> None:
        if colors[node] == visiting:
            raise ValueError("move references must be acyclic")
        if colors[node] == complete:
            return
        colors[node] = visiting
        for target in sorted(graph[node]):
            visit(target)
        colors[node] = complete

    for node in graph:
        visit(node)


class DeliveryPlan(StrictFrozenModel):
    tone: Tone
    length_class: LengthClass
    address_style: AddressStyle
    move_order: tuple[NonEmptyId, ...]
    emphasis_move_ids: tuple[NonEmptyId, ...] = ()
    connector_ids: tuple[NonEmptyId, ...] = ()


class SpeechProposalBody(StrictFrozenModel):
    kind: Literal["speech"]
    objective: SpeechObjective
    moves: tuple[SpeechMove, ...] = Field(min_length=1, max_length=8)
    response_record_refs: tuple[NonEmptyId, ...] = ()
    delivery_plan: DeliveryPlan

    @model_validator(mode="after")
    def _validate_move_graph(self) -> "SpeechProposalBody":
        move_ids = tuple(move.move_id for move in self.moves)
        require_unique(move_ids, field_name="move IDs")
        if set(self.delivery_plan.move_order) != set(move_ids) or len(
            self.delivery_plan.move_order
        ) != len(move_ids):
            raise ValueError("move_order must contain every move ID exactly once")
        require_unique(
            self.delivery_plan.emphasis_move_ids,
            field_name="emphasis_move_ids",
        )
        if not set(self.delivery_plan.emphasis_move_ids) <= set(move_ids):
            raise ValueError("emphasis_move_ids must reference proposal moves")

        require_unique(
            self.response_record_refs,
            field_name="response_record_refs",
        )
        external_refs = {
            move.source_record_id
            for move in self.moves
            if isinstance(move, ResponseMove)
        } | {
            move.source_record_id
            for move in self.moves
            if isinstance(move, RoleClaim) and move.source_record_id is not None
        }
        if set(self.response_record_refs) != external_refs:
            raise ValueError("response_record_refs must match referenced records")
        if set(self.response_record_refs) & set(move_ids):
            raise ValueError(
                "external record refs must not reference proposal moves"
            )

        graph: dict[str, set[str]] = {move_id: set() for move_id in move_ids}
        for move in self.moves:
            if isinstance(move, PublicEvidenceCitation):
                graph[move.move_id].update(move.supports_move_ids)
            if isinstance(move, RetractionMove) and move.replacement_move_id:
                graph[move.move_id].add(move.replacement_move_id)
        if any(target not in graph for targets in graph.values() for target in targets):
            raise ValueError("move reference must resolve inside the proposal")
        _require_acyclic(graph)
        return self
```

- [ ] **Step 5: Implement the host-bound speech envelope**

Create `proposals.py`:

```python
# -*- coding: utf-8 -*-
"""
定义由 Host 绑定身份与版本的玩家终端提案信封。

作者: Project contributors
创建日期: 2026-07-29
"""

from typing import Literal

from pydantic import Field, model_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
)
from werewolf_agent.player_agents.contracts.speech import (
    ClaimMode,
    RoleClaim,
    SpeechProposalBody,
)


class SpeechProposalEnvelope(StrictFrozenModel):
    schema_version: Literal["1.0.0"]
    turn_id: NonEmptyId
    player_id: NonEmptyId
    window_id: NonEmptyId
    window_version: int = Field(ge=1)
    base_revision: int = Field(ge=0)
    view_fingerprint: ContentHash
    body: SpeechProposalBody

    @model_validator(mode="after")
    def _actor_bound_role_claim(self) -> "SpeechProposalEnvelope":
        for move in self.body.moves:
            if (
                isinstance(move, RoleClaim)
                and move.claim_mode in {ClaimMode.CLAIM, ClaimMode.DENY}
                and move.claimant_id != self.player_id
            ):
                raise ValueError("role claim claimant must match player")
        return self
```

- [ ] **Step 6: Export contracts and verify GREEN**

Export `SpeechProposalEnvelope`, `SpeechProposalBody`, `SpeechMove`, all move
classes, and the bounded speech enums from `contracts/__init__.py`.

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_speech_contracts.py tests/player_agents/test_import_boundary.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit the speech contract**

```bash
git add werewolf_agent/player_agents/contracts tests/player_agents
git commit -m "feat: add strict autonomous speech proposal"
```

### Task 6: Define Disclosure and Public Record Contracts

**Files:**
- Create: `werewolf_agent/player_agents/contracts/disclosure.py`
- Create: `werewolf_agent/player_agents/contracts/records.py`
- Create: `tests/player_agents/test_public_record_contracts.py`
- Modify: `werewolf_agent/player_agents/contracts/__init__.py`

- [ ] **Step 1: Write failing disclosure and record tests**

```python
# -*- coding: utf-8 -*-
"""
验证一次性私密披露授权、公共语义记录和渲染记录契约。

作者: Project contributors
创建日期: 2026-07-29
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.disclosure import DisclosureGrant
from werewolf_agent.player_agents.contracts.records import (
    PublicSpeechRecord,
    RecordOrigin,
    RenderedUtterance,
)
from werewolf_agent.player_agents.contracts.speech import (
    Alignment,
    AlignmentRead,
    Modality,
    Strength,
    VoteCommitment,
    VotePosition,
)


HASH = "a" * 64


def test_disclosure_grant_requires_aware_expiry_and_exact_fact_hash() -> None:
    grant = DisclosureGrant(
        grant_id="grant-1",
        actor_id="p01",
        turn_id="turn-1",
        window_id="speech-d1-p01",
        game_revision=4,
        fact_kind="alignment_check",
        fact_record_id="seer-check-1",
        fact_hash=HASH,
        target_id="p03",
        timing_ref="night-1",
        expires_at=datetime(2026, 7, 29, 2, tzinfo=timezone.utc),
    )
    assert grant.fact_hash == HASH
    with pytest.raises(ValidationError, match="timezone-aware"):
        DisclosureGrant.model_validate({
            **grant.model_dump(),
            "expires_at": datetime(2026, 7, 29, 2),
        })


def _record() -> PublicSpeechRecord:
    moves = (
        AlignmentRead(
            move_id="m1",
            move_type="alignment_read",
            modality=Modality.SUSPECTED,
            evidence_refs=("public-3",),
            target_id="p03",
            alignment=Alignment.WOLF,
            strength=Strength.LEANING,
        ),
        VotePosition(
            move_id="m2",
            move_type="vote_position",
            modality=Modality.ASSERTED,
            evidence_refs=("public-3",),
            target_id="p03",
            commitment=VoteCommitment.PROVISIONAL,
        ),
    )
    return PublicSpeechRecord(
        record_id="speech-5",
        schema_version="1.0.0",
        game_id="game-1",
        turn_id="turn-1",
        actor_id="p01",
        day=1,
        phase="day_discussion",
        committed_revision=5,
        normalized_moves=moves,
        source_evidence_refs=("public-3",),
        disclosure_grant_refs=(),
        origin=RecordOrigin.MODEL_SUBMISSION,
        renderer_contract_version="speech-renderer-1",
        rendered_utterance_hash=HASH,
    )


def test_public_record_keeps_semantics_separate_from_rendered_text() -> None:
    record = _record()
    rendered = RenderedUtterance(
        record_id=record.record_id,
        sentence_plan_version="1.0.0",
        renderer_version="speech-renderer-1",
        text="我目前偏向投 p03。",
        content_hash=HASH,
        fallback_status="none",
    )
    assert "text" not in PublicSpeechRecord.model_fields
    assert rendered.record_id == record.record_id


def test_public_record_rejects_inconsistent_local_provenance() -> None:
    record = _record()
    record_data = record.model_dump()
    duplicate_move = record.normalized_moves[0].model_copy(
        update={"move_id": record.normalized_moves[1].move_id}
    )
    with pytest.raises(ValidationError, match="move IDs must not contain duplicates"):
        PublicSpeechRecord.model_validate({
            **record_data,
            "normalized_moves": (duplicate_move, record.normalized_moves[1]),
        })
    with pytest.raises(ValidationError, match="every move evidence ref"):
        PublicSpeechRecord.model_validate({
            **record_data,
            "source_evidence_refs": (),
        })


def test_rendered_utterance_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValidationError, match="non-whitespace"):
        RenderedUtterance(
            record_id="speech-5",
            sentence_plan_version="1.0.0",
            renderer_version="speech-renderer-1",
            text="   ",
            content_hash=HASH,
            fallback_status="none",
        )
```

The implementation test must also construct a `PrivateResultDisclosure` and
verify that its `disclosure_grant_id` is present exactly once in
`disclosure_grant_refs`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_public_record_contracts.py -v
```

Expected: collection ERROR because disclosure/record modules do not exist.

- [ ] **Step 3: Implement the disclosure grant**

Create `disclosure.py`:

```python
# -*- coding: utf-8 -*-
"""
定义由 Host 签发并在提交事务中一次性消费的私密披露授权。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
)


class DisclosureGrant(StrictFrozenModel):
    grant_id: NonEmptyId
    actor_id: NonEmptyId
    turn_id: NonEmptyId
    window_id: NonEmptyId
    game_revision: int = Field(ge=0)
    fact_kind: NonEmptyId
    fact_record_id: NonEmptyId
    fact_hash: ContentHash
    target_id: NonEmptyId | None = None
    timing_ref: NonEmptyId
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def _aware_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware")
        return value
```

- [ ] **Step 4: Implement public and rendered records**

Create `records.py`:

```python
# -*- coding: utf-8 -*-
"""
定义提交后的公共发言语义记录和无独立语义权威的渲染结果。

作者: Project contributors
创建日期: 2026-07-29
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
    require_unique,
)
from werewolf_agent.player_agents.contracts.speech import (
    PrivateResultDisclosure,
    SpeechMove,
)


class RecordOrigin(StrEnum):
    MODEL_SUBMISSION = "model_submission"
    REPAIRED_SUBMISSION = "repaired_submission"
    NEUTRAL_TERMINAL_FALLBACK = "neutral_terminal_fallback"


class PublicSpeechRecord(StrictFrozenModel):
    record_id: NonEmptyId
    schema_version: Literal["1.0.0"]
    game_id: NonEmptyId
    turn_id: NonEmptyId
    actor_id: NonEmptyId
    day: int = Field(ge=0)
    phase: NonEmptyId
    committed_revision: int = Field(ge=1)
    normalized_moves: tuple[SpeechMove, ...] = Field(min_length=1, max_length=8)
    source_evidence_refs: tuple[NonEmptyId, ...] = ()
    disclosure_grant_refs: tuple[NonEmptyId, ...] = ()
    origin: RecordOrigin
    renderer_contract_version: NonEmptyId
    rendered_utterance_hash: ContentHash

    @model_validator(mode="after")
    def _consistent_provenance(self) -> "PublicSpeechRecord":
        require_unique(
            (move.move_id for move in self.normalized_moves),
            field_name="move IDs",
        )
        require_unique(self.source_evidence_refs, field_name="source_evidence_refs")
        require_unique(self.disclosure_grant_refs, field_name="disclosure_grant_refs")
        move_evidence_refs = {
            evidence_ref
            for move in self.normalized_moves
            for evidence_ref in move.evidence_refs
        }
        if not move_evidence_refs <= set(self.source_evidence_refs):
            raise ValueError("source_evidence_refs must include every move evidence ref")
        used_grant_refs = {
            move.disclosure_grant_id
            for move in self.normalized_moves
            if isinstance(move, PrivateResultDisclosure)
        }
        if set(self.disclosure_grant_refs) != used_grant_refs:
            raise ValueError(
                "disclosure_grant_refs must match private disclosure moves"
            )
        return self


class RenderedUtterance(StrictFrozenModel):
    record_id: NonEmptyId
    sentence_plan_version: NonEmptyId
    renderer_version: NonEmptyId
    text: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    content_hash: ContentHash
    fallback_status: Literal["none", "template_fallback"]

    @field_validator("text")
    @classmethod
    def _non_whitespace_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain a non-whitespace character")
        return value
```

- [ ] **Step 5: Export, verify, and commit**

Export `DisclosureGrant`, `PublicSpeechRecord`, `RecordOrigin`, and
`RenderedUtterance`. Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_public_record_contracts.py tests/player_agents/test_speech_contracts.py -v
git add werewolf_agent/player_agents/contracts tests/player_agents
git commit -m "feat: add public speech record contracts"
```

Expected: PASS, followed by a successful commit.

### Task 7: Pin and Verify the Canonical Proposal JSON Schema

This task snapshots the canonical contract schema consumed by later provider
adapters. It does not claim direct compatibility with every provider's
restricted JSON Schema dialect; transformed provider fixtures and acceptance
tests belong to the later tool-gateway plan.

**Files:**
- Create: `werewolf_agent/player_agents/contracts/schema_catalog.py`
- Create: `scripts/export_player_agent_schemas.py`
- Create: `tests/player_agents/test_schema_catalog.py`
- Create: `tests/fixtures/player_agents/speech_proposal_schema_v1.json`
- Modify: `werewolf_agent/player_agents/contracts/__init__.py`

- [ ] **Step 1: Write a failing schema snapshot test**

```python
# -*- coding: utf-8 -*-
"""
验证规范发言 schema 与仓库快照及内容哈希完全一致。

作者: Project contributors
创建日期: 2026-07-29
"""

import json
from pathlib import Path

from werewolf_agent.player_agents.contracts.schema_catalog import (
    speech_proposal_schema,
    speech_proposal_schema_hash,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "player_agents" / "speech_proposal_schema_v1.json"


def test_speech_schema_matches_checked_in_fixture_and_hash() -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual = speech_proposal_schema()
    assert actual == expected
    assert speech_proposal_schema_hash() == expected["x-wofkill-content-hash"]
    assert len(expected["x-wofkill-content-hash"]) == 64
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents/test_schema_catalog.py -v
```

Expected: collection ERROR because `schema_catalog` does not exist.

- [ ] **Step 3: Implement canonical schema generation**

```python
# -*- coding: utf-8 -*-
"""
生成并哈希供 provider adapter 使用的规范自主玩家提案 JSON Schema。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from werewolf_agent.player_agents.contracts.proposals import SpeechProposalEnvelope


SCHEMA_VERSION = "1.0.0"


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _schema_without_hash() -> dict[str, Any]:
    schema = SpeechProposalEnvelope.model_json_schema()
    schema["$id"] = f"urn:wofkill:speech-proposal:{SCHEMA_VERSION}"
    schema["x-wofkill-schema-version"] = SCHEMA_VERSION
    return schema


def speech_proposal_schema_hash() -> str:
    return hashlib.sha256(_canonical_bytes(_schema_without_hash())).hexdigest()


def speech_proposal_schema() -> dict[str, Any]:
    schema = _schema_without_hash()
    schema["x-wofkill-content-hash"] = speech_proposal_schema_hash()
    return schema
```

- [ ] **Step 4: Implement the deterministic exporter**

```python
# -*- coding: utf-8 -*-
"""
导出仓库审查用的自主玩家提案 JSON Schema 快照。

作者: Project contributors
创建日期: 2026-07-29

使用示例:
    conda run -n wofkill python -m scripts.export_player_agent_schemas
"""

from __future__ import annotations

import json
from pathlib import Path

from werewolf_agent.player_agents.contracts.schema_catalog import (
    speech_proposal_schema,
)


OUTPUT = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "player_agents"
    / "speech_proposal_schema_v1.json"
)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(speech_proposal_schema(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Generate the fixture and verify GREEN**

Run:

```bash
conda run -n wofkill python -m scripts.export_player_agent_schemas
conda run -n wofkill python -m pytest tests/player_agents/test_schema_catalog.py -v
```

Expected: fixture is created and the test passes.

- [ ] **Step 6: Export schema helpers and run all contract checks**

Export `SCHEMA_VERSION`, `speech_proposal_schema`, and
`speech_proposal_schema_hash` from `contracts/__init__.py`.

Run:

```bash
conda run -n wofkill python -m pytest tests/player_agents -v
conda run -n wofkill python -m ruff check --ignore UP009 werewolf_agent/player_agents tests/player_agents scripts/export_player_agent_schemas.py
conda run -n wofkill python -m mypy werewolf_agent/player_agents
```

Expected: all tests pass; ruff and mypy report no errors.

- [ ] **Step 7: Run preserved contract-adjacent regression tests**

Run:

```bash
conda run -n wofkill python -m pytest tests/agents/test_schemas.py tests/agents/test_speech_intent_parser.py tests/runtime/test_event_metadata_v2.py tests/storage/test_repository.py -v
```

Expected: PASS. This confirms the isolated new contracts did not change legacy
runtime, event metadata, or repository behavior.

- [ ] **Step 8: Commit the pinned schema catalog**

```bash
git add werewolf_agent/player_agents/contracts scripts/export_player_agent_schemas.py tests/player_agents tests/fixtures/player_agents
git commit -m "feat: pin autonomous speech proposal schema"
```

## Completion Criteria

- The new package imports no rejected player-decision module.
- All new models are strict, immutable, and reject extra fields.
- Revision, legal-window, turn-state, and error contracts match the approved
  design terminology.
- All eleven public speech move types are present in one discriminated union.
- Cross-move references, move order, response references, and duplicate IDs are
  rejected deterministically.
- Private disclosure, public semantic records, and rendered text are separate
  contracts.
- The canonical proposal JSON Schema and its content hash are checked in and
  reproducible; provider-dialect transforms remain explicitly deferred.
- No live runtime, repository schema, RuleEngine behavior, or legacy fallback
  is modified by this plan.
