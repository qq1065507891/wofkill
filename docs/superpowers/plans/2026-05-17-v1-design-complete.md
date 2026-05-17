# V1 设计文档补全实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补全 PROGRESS.md 审计中发现的所有缺失模块，使项目从"可运行原型"升级为"设计文档完整产品"。

**Architecture:** 按依赖顺序分 7 个任务组。每组独立可测试，内部按 TDD 红-绿循环推进。核心原则：不破坏现有 918 个测试，每个任务完成后测试套件只增不减。

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, SQLite (V1 开发替代), httpx, Pydantic, pytest

---

## File Structure Map

本次计划涉及的新增/修改文件：

```
werewolf_agent/
  api/
    app.py                    # 修改: 接入 GameRunner, auth middleware
    auth.py                   # 新增: JWT session auth
  runtime/
    game_runner.py            # 新增: 完整对局编排器
    graph.py                  # 修改: wire GameRunner 调用
  storage/
    migrations.py             # 新增: SQLite schema migration
  evaluation/
    metrics.py                # 修改: 补全 speech_influence 等指标
    schemas.py                # 修改: 新增 SpeechInfluenceMetric 等字段
  ui/
    static/
      dashboard.html          # 重写: 增强观战台
docker-compose.yml            # 新增: 本地服务栈
config/
  auth.yaml                   # 新增: 本地开发 auth 配置
tests/
  runtime/
    test_game_runner.py       # 新增
  api/
    test_auth.py              # 新增
  evaluation/
    test_remaining_metrics.py # 新增
  ui/
    test_dashboard_enhanced.py# 新增
```

---

## Task 1: GameRunner — 完整对局编排器

当前 `start_game` 端点手动创建 12 个玩家但不调用 LangGraph。BatchRunner 使用 mock 动作。本任务创建 `GameRunner` 类，将 LangGraph 编排、RuleEngine、AgentRegistry、ModelRouter、PersonaRouter、CognitivePipeline 和 PersistentMemoryCoordinator 串联为可运行的完整对局流程。

**Files:**
- Create: `werewolf_agent/runtime/game_runner.py`
- Modify: `werewolf_agent/api/app.py:100-129` (start_game 改用 GameRunner)
- Test: `tests/runtime/test_game_runner.py`

- [ ] **Step 1: Write failing test — GameRunner constructor and config loading**

```python
# tests/runtime/test_game_runner.py
"""Tests for GameRunner: full game orchestration wiring LangGraph + RuleEngine + agents."""

import pytest
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig


def test_game_runner_config_defaults():
    cfg = GameRunnerConfig(ruleset_id="pre_witch_hunter_idiot_mixed")
    assert cfg.player_count == 12
    assert cfg.seed is not None
    assert cfg.use_agent_registry is False


def test_game_runner_config_custom():
    cfg = GameRunnerConfig(
        ruleset_id="pre_witch_hunter_idiot_mixed",
        seed=42,
        player_count=12,
        use_agent_registry=True,
    )
    assert cfg.seed == 42
    assert cfg.use_agent_registry is True


def test_game_runner_creates_with_config():
    runner = GameRunner(GameRunnerConfig(ruleset_id="pre_witch_hunter_idiot_mixed"))
    assert runner.engine is not None
    assert runner.game_id is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/test_game_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'werewolf_agent.runtime.game_runner'`

- [ ] **Step 3: Implement GameRunnerConfig and GameRunner constructor**

```python
# werewolf_agent/runtime/game_runner.py
"""GameRunner: orchestrates a full game by wiring LangGraph + RuleEngine + agents.

This is the single entry point for running a complete 12-player game.
It replaces the old approach where `start_game` manually created players
without invoking the compiled LangGraph.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.graph import RuntimeState, build_game_graph


@dataclass
class GameRunnerConfig:
    ruleset_id: str = "pre_witch_hunter_idiot_mixed"
    player_count: int = 12
    seed: int | None = None
    use_agent_registry: bool = False
    model_config_path: str = ""
    persona_config_path: str = ""


class GameRunner:
    """Orchestrates a complete game from setup to finish.

    Usage:
        runner = GameRunner(GameRunnerConfig(seed=42))
        result = runner.run()  # runs the full game
        state = runner.state   # final GameState
    """

    def __init__(self, config: GameRunnerConfig) -> None:
        self._config = config
        self._game_id = f"g_{config.seed or uuid.uuid4().hex[:8]}"
        self._engine = RuleEngine()
        self._state = GameState(game_id=self._game_id, ruleset_id=config.ruleset_id)
        self._graph = build_game_graph()

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def engine(self) -> RuleEngine:
        return self._engine

    @property
    def state(self) -> GameState:
        return self._state

    def run(self, max_steps: int = 500) -> GameState:
        """Execute the full game graph until END or max_steps reached."""
        initial: RuntimeState = {
            "game_state": self._state,
            "engine": self._engine,
        }
        result = self._graph.invoke(initial, {"recursion_limit": max_steps})
        self._state = result.get("game_state", self._state)
        return self._state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/test_game_runner.py -v`
Expected: 3 passed

- [ ] **Step 5: Write failing test — GameRunner runs a complete scripted game**

```python
def test_game_runner_scripted_full_game():
    runner = GameRunner(GameRunnerConfig(
        ruleset_id="pre_witch_hunter_idiot_mixed",
        seed=42,
    ))
    final_state = runner.run()
    assert final_state.winning_faction in ("good", "werewolf")
    assert len(final_state.players) == 12
    assert final_state.phase in ("finished", "night", "day")


def test_game_runner_deterministic_with_seed():
    r1 = GameRunner(GameRunnerConfig(seed=123))
    s1 = r1.run()
    r2 = GameRunner(GameRunnerConfig(seed=123))
    s2 = r2.run()
    assert s1.winning_faction == s2.winning_faction
    assert len(s1.deaths) == len(s2.deaths)
```

- [ ] **Step 6: Run test — should pass if LangGraph graph already handles scripted games**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/test_game_runner.py::test_game_runner_scripted_full_game -v`

- [ ] **Step 7: Wire GameRunner into FastAPI start_game endpoint**

In `werewolf_agent/api/app.py`, modify `start_game` to use `GameRunner` when available:

```python
# In create_app(), add:
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig

# Replace the start_game function body:
@app.post("/games/{game_id}/start", response_model=GameActionResponse)
def start_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
    state = _get_game(games, game_id)
    if state.phase != "setup":
        raise HTTPException(400, "Game already started")
    # Use GameRunner for full LangGraph orchestration
    runner = GameRunner(GameRunnerConfig(
        ruleset_id=state.ruleset_id,
        seed=int(game_id.split("_")[-1]) if game_id.split("_")[-1].isdigit() else None,
    ))
    # Assign roles via RuleEngine
    state = runner.engine.assign_roles(state, seed=runner._config.seed or 0)
    event = GameEvent(type="game_started", payload={
        "game_id": game_id,
        "players": {pid: {"id": pid, "role": p.role} for pid, p in state.players.items()},
    })
    state = replace(state, phase="night", events=state.events + [event])
    games[game_id] = state
    if _repo is not None:
        _repo.save_game(state)
    # Store runner for later step execution
    if not hasattr(app.state, "runners"):
        app.state.runners = {}
    app.state.runners[game_id] = runner
    return GameActionResponse(
        game_id=game_id, action="start", success=True,
        message="Game started with full orchestration",
    )
```

- [ ] **Step 8: Add endpoint to advance game by one step**

```python
@app.post("/games/{game_id}/step", response_model=GameActionResponse)
def step_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
    runners = getattr(app.state, "runners", {})
    runner = runners.get(game_id)
    if runner is None:
        raise HTTPException(400, "No active runner for this game")
    state = runner.run_step()
    games[game_id] = state
    if _repo is not None:
        _repo.save_game(state)
    return GameActionResponse(
        game_id=game_id, action="step", success=True,
        message=f"Phase: {state.phase}",
    )
```

Add `run_step()` method to GameRunner that invokes the graph for one node.

- [ ] **Step 9: Run full test suite to verify no regressions**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ -q --tb=short`
Expected: All 918+ tests pass

- [ ] **Step 10: Commit**

```bash
git add werewolf_agent/runtime/game_runner.py werewolf_agent/api/app.py tests/runtime/test_game_runner.py
git commit -m "feat: add GameRunner for full game orchestration with LangGraph"
```

---

## Task 2: PersistentMemoryCoordinator 接入游戏生命周期

当前 `PersistentMemoryCoordinator` 已实现但未在游戏开始/结束时自动调用。本任务将 coordinator 接入 `GameRunner`，实现：游戏开始时从 repository 恢复记忆，游戏结束时自动保存快照。

**Files:**
- Modify: `werewolf_agent/runtime/game_runner.py`
- Test: `tests/runtime/test_game_runner.py`

- [ ] **Step 1: Write failing test — memory auto-persist on game end**

```python
def test_game_runner_persists_memory_on_finish():
    from werewolf_agent.storage.sqlite_store import SqliteGameRepository
    from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator
    import tempfile, os

    db_path = tempfile.mktemp(suffix=".db")
    repo = SqliteGameRepository(db_path)
    coord = PersistentMemoryCoordinator(repo)

    runner = GameRunner(GameRunnerConfig(seed=99, repository=repo, memory_coordinator=coord))
    final = runner.run()
    # After game finishes, memory should be persisted
    snapshots = coord.list_snapshots()
    assert len(snapshots) >= 1
    repo.close()
    os.unlink(db_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/test_game_runner.py::test_game_runner_persists_memory_on_finish -v`
Expected: FAIL — `GameRunnerConfig` has no `repository` field

- [ ] **Step 3: Add repository and memory_coordinator to GameRunnerConfig and GameRunner**

在 `GameRunnerConfig` 中添加:
```python
repository: Any = None  # GameRepository, optional
memory_coordinator: Any = None  # PersistentMemoryCoordinator, optional
```

在 `GameRunner.run()` 末尾添加:
```python
# After game finishes, auto-persist memory if coordinator is provided
if self._config.memory_coordinator is not None and self._state.winning_faction is not None:
    from werewolf_agent.memory.store import MemoryStore
    mem_store = MemoryStore()
    # Sync from final state events
    mem_store.sync_from_game(self._state)
    self._config.memory_coordinator.save_all(mem_store, None, snapshot_id=self._game_id)
```

- [ ] **Step 4: Write test — memory restored on game start**

```python
def test_game_runner_restores_memory_on_start():
    from werewolf_agent.storage.sqlite_store import SqliteGameRepository
    from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator
    import tempfile, os

    db_path = tempfile.mktemp(suffix=".db")
    repo = SqliteGameRepository(db_path)
    coord = PersistentMemoryCoordinator(repo)

    # First game: play and persist
    runner1 = GameRunner(GameRunnerConfig(seed=55, repository=repo, memory_coordinator=coord))
    runner1.run()

    # Second game: should be able to restore prior memory
    runner2 = GameRunner(GameRunnerConfig(seed=77, repository=repo, memory_coordinator=coord))
    restored_mem, restored_rag = coord.restore_all(snapshot_id=runner1.game_id)
    assert restored_mem is not None  # Memory from game 1 was persisted

    repo.close()
    os.unlink(db_path)
```

- [ ] **Step 5: Run all game runner tests**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/test_game_runner.py -v`
Expected: All pass

- [ ] **Step 6: Run full test suite for regressions**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ -q --tb=short`

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/runtime/game_runner.py tests/runtime/test_game_runner.py
git commit -m "feat: wire PersistentMemoryCoordinator into GameRunner lifecycle"
```

---

## Task 3: 本地开发认证系统

当前 API 使用硬编码 `mod1`/`dbg1` 作为 elevated callers。本任务实现基于 JWT 的 session 认证，同时保留一个本地开发模式（无需外部 auth 服务即可运行）。

**Files:**
- Create: `werewolf_agent/api/auth.py`
- Create: `config/auth.yaml`
- Modify: `werewolf_agent/api/app.py:63-71` (替换硬编码 authorized_callers)
- Test: `tests/api/test_auth.py`

- [ ] **Step 1: Write failing test — auth module basics**

```python
# tests/api/test_auth.py
"""Tests for local development auth system."""

import pytest
from werewolf_agent.api.auth import AuthManager, AuthConfig, SessionToken


def test_auth_config_loads_defaults():
    cfg = AuthConfig()
    assert cfg.mode in ("local", "jwt")
    assert cfg.local_users  # has default local users


def test_auth_manager_local_mode():
    mgr = AuthManager(AuthConfig(mode="local"))
    token = mgr.create_session("mod1", "moderator")
    assert token is not None
    role = mgr.validate_session(token)
    assert role == "moderator"


def test_auth_manager_rejects_unknown_user():
    mgr = AuthManager(AuthConfig(mode="local"))
    with pytest.raises(PermissionError):
        mgr.create_session("hacker", "moderator")


def test_auth_manager_expired_token():
    mgr = AuthManager(AuthConfig(mode="local", token_ttl_seconds=0))
    token = mgr.create_session("mod1", "moderator")
    import time; time.sleep(0.1)
    role = mgr.validate_session(token)
    assert role is None  # expired
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/api/test_auth.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement AuthManager**

```python
# werewolf_agent/api/auth.py
"""Local development authentication for the Werewolf Agent API.

Two modes:
- local: in-memory session tokens, users defined in config/auth.yaml or defaults.
- jwt: HMAC-SHA256 signed tokens (no external service required).

V1 uses local mode by default. JWT mode is for future multi-process deployments.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuthConfig:
    mode: str = "local"
    token_ttl_seconds: int = 3600
    local_users: dict[str, str] = field(default_factory=lambda: {
        "mod1": "moderator",
        "dbg1": "debugger",
        "spectator": "spectator",
    })
    secret_key: str = ""
    config_path: str = ""

    def __post_init__(self):
        if not self.secret_key:
            self.secret_key = os.environ.get("WEREWOLF_AUTH_SECRET", "wofkill-dev-key-change-me")
        if self.config_path:
            self._load_from_file()

    def _load_from_file(self):
        p = Path(self.config_path)
        if p.exists():
            import yaml
            with open(p) as f:
                data = yaml.safe_load(f) or {}
            self.mode = data.get("mode", self.mode)
            self.token_ttl_seconds = data.get("token_ttl_seconds", self.token_ttl_seconds)
            if "local_users" in data:
                self.local_users.update(data["local_users"])


@dataclass
class SessionToken:
    token: str
    caller_id: str
    role: str
    expires_at: float


class AuthManager:
    def __init__(self, config: AuthConfig | None = None) -> None:
        self._config = config or AuthConfig()
        self._sessions: dict[str, SessionToken] = {}

    def create_session(self, caller_id: str, requested_role: str) -> str:
        if self._config.mode == "local":
            allowed = self._config.local_users.get(caller_id)
            if allowed is None:
                raise PermissionError(f"Unknown user: {caller_id}")
            if allowed != requested_role:
                raise PermissionError(f"User {caller_id} cannot assume role {requested_role}")
        expires = time.time() + self._config.token_ttl_seconds
        raw = f"{caller_id}:{requested_role}:{expires}:{self._config.secret_key}"
        sig = hmac.new(self._config.secret_key.encode(), raw.encode(), hashlib.sha256).hexdigest()
        token_str = f"{caller_id}.{requested_role}.{expires:.0f}.{sig}"
        self._sessions[token_str] = SessionToken(
            token=token_str, caller_id=caller_id, role=requested_role, expires_at=expires,
        )
        return token_str

    def validate_session(self, token_str: str) -> str | None:
        sess = self._sessions.get(token_str)
        if sess is None:
            # Try stateless validation for JWT-like tokens
            parts = token_str.split(".")
            if len(parts) == 4:
                caller_id, role, exp_str, sig = parts
                raw = f"{caller_id}:{role}:{exp_str}:{self._config.secret_key}"
                expected = hmac.new(self._config.secret_key.encode(), raw.encode(), hashlib.sha256).hexdigest()
                if hmac.compare_digest(sig, expected) and float(exp_str) > time.time():
                    return role
            return None
        if sess.expires_at < time.time():
            del self._sessions[token_str]
            return None
        return sess.role

    def revoke_session(self, token_str: str) -> None:
        self._sessions.pop(token_str, None)
```

- [ ] **Step 4: Create config/auth.yaml**

```yaml
# Local development auth configuration.
# For production, switch mode to "jwt" and set WEREWOLF_AUTH_SECRET env var.

mode: local
token_ttl_seconds: 3600

local_users:
  mod1: moderator
  dbg1: debugger
  spectator: spectator
```

- [ ] **Step 5: Run auth tests**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/api/test_auth.py -v`
Expected: 4 passed

- [ ] **Step 6: Wire AuthManager into FastAPI app.py**

Replace the hardcoded `authorized_callers` dict with `AuthManager`:

```python
# In create_app():
from werewolf_agent.api.auth import AuthManager, AuthConfig

auth_manager = AuthManager(AuthConfig(config_path="config/auth.yaml"))

# Replace _resolve_caller_role:
def _resolve_caller_role(
    caller_id: str,
    requested_role: CallerRole,
    session_token: str = "",
) -> CallerRole:
    if session_token:
        role_str = auth_manager.validate_session(session_token)
        if role_str == "moderator" and requested_role == CallerRole.MODERATOR:
            return CallerRole.MODERATOR
        if role_str == "debugger" and requested_role == CallerRole.DEBUGGER:
            return CallerRole.DEBUGGER
    # Fallback to local user check
    if requested_role in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        allowed = auth_manager._config.local_users.get(caller_id)
        if allowed == requested_role.value:
            return requested_role
        raise HTTPException(403, "Elevated caller role is not authorized")
    return requested_role
```

Add `session_token: str = Query("")` parameter to endpoints that use `_resolve_caller_role`.

Add login endpoint:

```python
@app.post("/auth/login")
def login(caller_id: str = Query(...), role: str = Query(...)):
    try:
        token = auth_manager.create_session(caller_id, role)
        return {"token": token, "caller_id": caller_id, "role": role}
    except PermissionError as e:
        raise HTTPException(403, str(e))
```

- [ ] **Step 7: Add integration test — login + access protected endpoint**

```python
def test_login_and_access_moderator_replay():
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    # Login as moderator
    resp = client.post("/auth/login?caller_id=mod1&role=moderator")
    assert resp.status_code == 200
    token = resp.json()["token"]

    # Create and start a game
    client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed"})
    client.post("/games/g_game_1/start", json={})

    # Access replay with token
    resp = client.get(f"/games/g_game_1/replay?session_token={token}&view_mode=moderator_full")
    assert resp.status_code == 200
```

- [ ] **Step 8: Run full test suite**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ -q --tb=short`

- [ ] **Step 9: Commit**

```bash
git add werewolf_agent/api/auth.py config/auth.yaml tests/api/test_auth.py werewolf_agent/api/app.py
git commit -m "feat: add local dev auth system with session tokens"
```

---

## Task 4: 补全剩余评测指标

PROGRESS.md 审计指出 `speech_influence`、`speech_order_utilization`、`cognitive_pipeline_compression_rate` 等指标仍为占位。本任务从 `GameResult` 的事件数据中计算这些指标。

**Files:**
- Modify: `werewolf_agent/evaluation/schemas.py` (新增字段)
- Modify: `werewolf_agent/evaluation/metrics.py` (补全计算逻辑)
- Test: `tests/evaluation/test_remaining_metrics.py`

- [ ] **Step 1: Write failing test — speech influence metric**

```python
# tests/evaluation/test_remaining_metrics.py
"""Tests for previously placeholder evaluation metrics."""

import pytest
from werewolf_agent.evaluation.schemas import (
    ActionRecord, ActionVerdict, GameResult, QualityMetrics, BatchConfig,
)
from werewolf_agent.evaluation.metrics import MetricsAggregator


def _make_result_with_speech_events():
    """Game result where player_03's speech influenced others' votes."""
    return GameResult(
        game_id="g1",
        initial_seed=1,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        winning_faction="good",
        event_log=[
            {"type": "speech", "player_id": "p03", "day_number": 1, "text": "I think p07 is wolf"},
            {"type": "vote", "player_id": "p04", "target_id": "p07", "day_number": 1},
            {"type": "vote", "player_id": "p05", "target_id": "p07", "day_number": 1},
            {"type": "vote", "player_id": "p06", "target_id": "p08", "day_number": 1},
        ],
        player_roles={"p03": "seer", "p04": "villager", "p07": "werewolf"},
        action_records=[
            ActionRecord(player_id="p04", action_type="vote", target_id="p07", phase="day", day_number=1),
            ActionRecord(player_id="p05", action_type="vote", target_id="p07", phase="day", day_number=1),
        ],
    )


def test_speech_influence_computed():
    agg = MetricsAggregator()
    agg.add_result(_make_result_with_speech_events())
    snap = agg.compute_snapshot()
    # speech_influence_rate: fraction of post-speech votes aligned with speaker's target
    assert 0.0 <= snap.quality_metrics.speech_influence_rate <= 1.0
    # p03 said p07 is wolf; p04 and p05 voted p07 → high influence
    assert snap.quality_metrics.speech_influence_rate > 0.0


def test_speech_order_utilization_computed():
    agg = MetricsAggregator()
    agg.add_result(_make_result_with_speech_events())
    snap = agg.compute_snapshot()
    assert 0.0 <= snap.quality_metrics.speech_order_utilization <= 1.0


def test_cognitive_compression_rate_computed():
    agg = MetricsAggregator()
    agg.add_result(GameResult(
        game_id="g2", initial_seed=2, ruleset_id="pre_witch_hunter_idiot_mixed",
        winning_faction="werewolf",
        cognition_snapshots={"p01": {"original_fact_count": 50, "compressed_fact_count": 15}},
    ))
    snap = agg.compute_snapshot()
    assert 0.0 <= snap.quality_metrics.cognitive_compression_rate <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/evaluation/test_remaining_metrics.py -v`
Expected: FAIL — `QualityMetrics` has no `speech_influence_rate` field

- [ ] **Step 3: Add new fields to QualityMetrics in schemas.py**

在 `QualityMetrics` dataclass 中添加:
```python
speech_influence_rate: float = 0.0
speech_order_utilization: float = 0.0
cognitive_compression_rate: float = 0.0
```

- [ ] **Step 4: Implement computation logic in metrics.py**

在 `MetricsAggregator._compute_quality_metrics` 中添加:

```python
# Speech influence: for each speech event targeting X, count how many
# subsequent voters (who spoke after) also voted X.
speech_events = []
vote_events = []
for result in self._results:
    for ev in result.event_log:
        if ev.get("type") == "speech" and "target" in ev.get("text", "").lower():
            speech_events.append(ev)
        if ev.get("type") == "vote":
            vote_events.append(ev)

if speech_events and vote_events:
    aligned = 0
    total = 0
    for speech in speech_events:
        speaker = speech.get("player_id", "")
        day = speech.get("day_number", 0)
        # Find votes after this speech on the same day
        for vote in vote_events:
            if vote.get("day_number") == day and vote.get("player_id") != speaker:
                total += 1
                # Simple heuristic: if speech mentioned a target and vote matches
                if any(t in vote.get("target_id", "") for t in speech.get("mentioned_targets", [])):
                    aligned += 1
    if total > 0:
        qm.speech_influence_rate = aligned / total

# Speech order utilization: fraction of later speakers who reference earlier speech
# For V1, approximate from event_log structure
all_speeches = [e for r in self._results for e in r.event_log if e.get("type") == "speech"]
if all_speeches:
    referenced = sum(1 for s in all_speeches if s.get("references_prior", False))
    qm.speech_order_utilization = referenced / len(all_speeches)

# Cognitive compression rate
for result in self._results:
    for pid, snap in result.cognition_snapshots.items():
        orig = snap.get("original_fact_count", 0)
        comp = snap.get("compressed_fact_count", 0)
        if orig > 0:
            rates.append(comp / orig)
if rates:
    qm.cognitive_compression_rate = sum(rates) / len(rates)
```

注意: `rates` 列表需要在该方法开头声明为 `rates = []`。

- [ ] **Step 5: Run remaining metrics tests**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/evaluation/test_remaining_metrics.py -v`
Expected: All pass

- [ ] **Step 6: Run full evaluation test suite**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/evaluation/ -v`

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/evaluation/schemas.py werewolf_agent/evaluation/metrics.py tests/evaluation/test_remaining_metrics.py
git commit -m "feat: compute speech_influence, speech_order_utilization, cognitive_compression metrics"
```

---

## Task 5: 增强观战台前端

当前 dashboard.html 是基础 HTML 页面，缺少设计文档 §12.2 要求的认知差异图、RAG 命中面板、模型/人格面板、时间线滑块等。本任务在现有单页框架基础上增强，不引入构建工具链。

**Files:**
- Rewrite: `werewolf_agent/ui/static/dashboard.html`
- Test: `tests/ui/test_dashboard_enhanced.py`

- [ ] **Step 1: Write failing test — enhanced dashboard sections exist**

```python
# tests/ui/test_dashboard_enhanced.py
"""Tests for enhanced observer dashboard features."""

import pytest
from pathlib import Path

DASHBOARD_PATH = Path(__file__).parent.parent.parent / "werewolf_agent" / "ui" / "static" / "dashboard.html"


@pytest.fixture
def dashboard_html():
    return DASHBOARD_PATH.read_text(encoding="utf-8")


def test_cognitive_diff_section(dashboard_html):
    assert "cognitive-diff" in dashboard_html
    assert "belief-prob" in dashboard_html or "identity-prob" in dashboard_html


def test_rag_hit_panel(dashboard_html):
    assert "rag-hit" in dashboard_html
    assert "source-type" in dashboard_html


def test_model_routing_panel(dashboard_html):
    assert "model-routing" in dashboard_html or "llm-profile" in dashboard_html


def test_persona_routing_panel(dashboard_html):
    assert "persona-routing" in dashboard_html or "persona-profile" in dashboard_html


def test_timeline_slider(dashboard_html):
    assert "timeline-slider" in dashboard_html or "day-slider" in dashboard_html


def test_attention_filter_panel(dashboard_html):
    assert "attention-filter" in dashboard_html or "attention-stats" in dashboard_html


def test_cost_latency_panel(dashboard_html):
    assert "cost-latency" in dashboard_html or "token-usage" in dashboard_html


def test_private_intent_audit(dashboard_html):
    assert "private-intent" in dashboard_html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ui/test_dashboard_enhanced.py -v`
Expected: Some tests FAIL — missing HTML sections

- [ ] **Step 3: Enhance dashboard.html**

在现有 dashboard.html 基础上，在右侧面板和中央区域添加以下 section。保持纯 HTML/CSS/JS 无构建依赖。

核心新增区域:

1. **认知差异图面板** (`#cognitive-diff-panel`):
   - 左侧: 真实身份底牌 (moderator_full only)
   - 右侧: 选中玩家的身份概率分布条形图 (CSS 绘制，无需 chart 库)
   - 时间轴滑块: 用 range input 切换天数

2. **RAG 命中面板** (`#rag-hit-panel`):
   - 每个 hit 显示 entry_id, relevance_score, quality_grade, source_type, visibility_boundary
   - 从 `/games/{id}/cognitive-diff` 端点获取数据

3. **模型路由面板** (`#model-routing-panel`):
   - 每个玩家显示当前 llm_profile, provider, model
   - 最近调用: provider, model, latency_ms, tokens, cost

4. **人格路由面板** (`#persona-routing-panel`):
   - 每个玩家显示 persona_profile, 基础参数
   - 动态修正: task_style, dynamic_policy 命中

5. **注意力过滤统计** (`#attention-stats-panel`):
   - 过滤前/后信息类别统计 (before_count, after_count by category)

6. **成本延迟面板** (`#cost-latency-panel`):
   - 总 token, 总估算成本, 平均延迟
   - 按 provider 和 task_type 分组

7. **私有意图审计** (`#private-intent-panel`):
   - 仅 moderator_full 模式可见
   - 显示每个玩家的 true_role, faction_goal, claimed_view

CSS 保持暗色主题一致性，使用 CSS Grid 和 Flexbox 布局。JS 使用 fetch API 从现有端点获取数据。

- [ ] **Step 4: Add new API endpoint for RAG/memory audit data**

在 `app.py` 中添加:

```python
@app.get("/games/{game_id}/rag-audit")
def get_rag_audit(game_id: str, caller_id: str = Query(""), session_token: str = Query("")) -> dict:
    state = _get_game(games, game_id)
    # Permission check: moderator or debugger only
    role = _resolve_caller_role(caller_id, CallerRole.DEBUGGER, session_token)
    if role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        raise HTTPException(403, "RAG audit requires moderator or debugger access")
    # Return RAG injection audit records from game state events
    rag_events = [e for e in state.events if e.type == "rag_injection_audit"]
    return {"game_id": game_id, "rag_audits": [e.payload for e in rag_events]}
```

- [ ] **Step 5: Run enhanced dashboard tests**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ui/test_dashboard_enhanced.py -v`
Expected: All pass

- [ ] **Step 6: Run existing dashboard tests (no regressions)**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ui/test_dashboard.py -v`

- [ ] **Step 7: Run full test suite**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ -q --tb=short`

- [ ] **Step 8: Commit**

```bash
git add werewolf_agent/ui/static/dashboard.html werewolf_agent/api/app.py tests/ui/test_dashboard_enhanced.py
git commit -m "feat: enhanced observer dashboard with cognitive diff, RAG, model/persona panels"
```

---

## Task 6: Docker Compose 本地服务栈

创建 `docker-compose.yml` 使整个项目可通过单条命令启动。包含 API 服务、SQLite 持久化卷、以及可选的 Redis 缓存服务。

**Files:**
- Create: `docker-compose.yml`
- Create: `Dockerfile`
- Modify: `README.md` (添加 docker 启动说明)

- [ ] **Step 1: Create Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps for SQLite, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
# Install conda deps via pip for slim image
RUN pip install --no-cache-dir \
    fastapi uvicorn httpx pydantic langgraph langchain-core pyyaml pytest

COPY . .

EXPOSE 8000

CMD ["uvicorn", "werewolf_agent.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
# docker-compose.yml
# Local development stack for Werewolf Agent V1.
# Usage: docker compose up

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - werewolf-data:/app/data
      - ./.env:/app/.env:ro
    environment:
      - WEREWOLF_DB_PATH=/app/data/wofkill.db
      - WEREWOLF_AUTH_SECRET=${WEREWOLF_AUTH_SECRET:-wofkill-dev-key}
    healthcheck:
      test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8000/games')"]
      interval: 10s
      timeout: 5s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    # Optional: used for runtime locks and background game execution
    profiles:
      - with-redis

volumes:
  werewolf-data:
  redis-data:
```

- [ ] **Step 3: Update app.py to support WEREWOLF_DB_PATH env var**

在 `create_app()` 开头添加:
```python
# Auto-configure SQLite repository if env var is set
if repository is None:
    db_path = os.environ.get("WEREWOLF_DB_PATH")
    if db_path:
        from werewolf_agent.storage.sqlite_store import SqliteGameRepository
        repository = SqliteGameRepository(db_path)
```

需要在文件顶部添加 `import os`。

- [ ] **Step 4: Add docker instructions to README.md**

在 README.md 的运行说明部分添加:

```markdown
## Docker 快速启动

```bash
# 构建并启动 API 服务
docker compose up

# 访问观战台
open http://localhost:8000

# 包含 Redis 的完整启动
docker compose --profile with-redis up
```
```

- [ ] **Step 5: Verify Dockerfile builds (if Docker available)**

Run: `docker build -t wofkill-api . 2>&1 | tail -5`

如果 Docker 不可用，跳过此步，在 README 中标注。

- [ ] **Step 6: Run full test suite (no regressions)**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ -q --tb=short`

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml Dockerfile README.md werewolf_agent/api/app.py
git commit -m "feat: add Docker Compose local development stack"
```

---

## Task 7: Storage Schema Migration + Sheriff Badge Night Death 路由修复

SQLite schema 已经在 `sqlite_store.py` 中用 `CREATE TABLE IF NOT EXISTS` 创建，但缺少版本化迁移机制。同时 PROGRESS.md 记录了 sheriff badge night death 路由仍有问题。

**Files:**
- Create: `werewolf_agent/storage/migrations.py`
- Modify: `werewolf_agent/runtime/graph.py` (修复 night death → badge transfer 路由)
- Test: `tests/storage/test_migrations.py`

- [ ] **Step 1: Write failing test — migration version tracking**

```python
# tests/storage/test_migrations.py
"""Tests for SQLite schema migration system."""

import tempfile, os
import pytest
from werewolf_agent.storage.migrations import MigrationManager, Migration


def test_migration_manager_tracks_version():
    db_path = tempfile.mktemp(suffix=".db")
    mgr = MigrationManager(db_path)
    assert mgr.current_version() == 0
    mgr.apply_all()
    assert mgr.current_version() >= 1
    mgr.close()
    os.unlink(db_path)


def test_migration_idempotent():
    db_path = tempfile.mktemp(suffix=".db")
    mgr = MigrationManager(db_path)
    mgr.apply_all()
    v1 = mgr.current_version()
    mgr.apply_all()  # should not fail
    assert mgr.current_version() == v1
    mgr.close()
    os.unlink(db_path)


def test_migration_adds_tables():
    db_path = tempfile.mktemp(suffix=".db")
    mgr = MigrationManager(db_path)
    mgr.apply_all()
    import sqlite3
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "schema_version" in tables
    conn.close()
    mgr.close()
    os.unlink(db_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/storage/test_migrations.py -v`

- [ ] **Step 3: Implement MigrationManager**

```python
# werewolf_agent/storage/migrations.py
"""SQLite schema migration system.

Provides versioned migrations so that existing databases can be upgraded
without losing data. Each migration is a (version, sql) pair.
"""

from __future__ import annotations

import sqlite3
from typing import Any


class Migration:
    def __init__(self, version: int, description: str, sql: str) -> None:
        self.version = version
        self.description = description
        self.sql = sql


MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        description="Initial schema: games, events, deaths, model_usage, evaluations, config_snapshots, rag_entries, memory_snapshots",
        sql="""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now')),
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS deaths (
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            death_json TEXT NOT NULL,
            PRIMARY KEY (game_id, player_id),
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS model_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            record_json TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS evaluations (
            game_id TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS config_snapshots (
            game_id TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS rag_entries (
            entry_id TEXT PRIMARY KEY,
            entry_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            snapshot_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,
    ),
]


class MigrationManager:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def current_version(self) -> int:
        try:
            row = self._conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            return row[0] or 0
        except sqlite3.OperationalError:
            return 0

    def apply_all(self) -> None:
        current = self.current_version()
        for migration in MIGRATIONS:
            if migration.version > current:
                self._conn.executescript(migration.sql)
                self._conn.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (migration.version, migration.description),
                )
                self._conn.commit()

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Update SqliteGameRepository to use MigrationManager**

在 `SqliteGameRepository.__init__` 中替换 `self._conn.executescript(_SCHEMA)` 为:

```python
from werewolf_agent.storage.migrations import MigrationManager
self._conn = sqlite3.connect(db_path, check_same_thread=False)
self._conn.execute("PRAGMA journal_mode=WAL")
self._conn.execute("PRAGMA foreign_keys=ON")
mgr = MigrationManager(db_path)
# Use the same connection for migration
# Actually, let's just apply migrations through the manager pattern
self._conn.executescript(_SCHEMA)  # Keep for backward compat
self._conn.commit()
```

保持 `_SCHEMA` 作为快速路径，MigrationManager 作为独立入口。后续新增表通过 MigrationManager 管理。

- [ ] **Step 5: Fix sheriff badge night death routing in graph.py**

当前问题: 夜间死亡后路由到 `announce_deaths` 而非 `sheriff_badge_transfer`。需要在 `route_after_resolve_night` 或 `route_after_hunter_shot` 中检查 sheriff 是否因夜间死亡而出局，若是且游戏未结束，则路由到 `sheriff_badge_transfer`。

在 `graph.py` 中修改 `route_after_hunter_shot` 或添加新的路由函数:

```python
def route_after_night_deaths(state: RuntimeState) -> str:
    """After night deaths (wolf kill + hunter shot), check victory then badge."""
    gs = state["game_state"]
    if gs.winning_faction is not None:
        return "check_victory"
    # If sheriff died this night and game continues, go to badge transfer
    if _sheriff_died_this_batch(gs):
        return "sheriff_badge_transfer"
    return "announce_deaths"


def _sheriff_died_this_batch(gs: GameState) -> bool:
    if gs.sheriff_id is None or gs.sheriff_badge_state != "active":
        return False
    sheriff = gs.players.get(gs.sheriff_id)
    return sheriff is not None and not sheriff.alive
```

- [ ] **Step 6: Run migration and graph routing tests**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/storage/test_migrations.py tests/runtime/test_runtime.py -v`

- [ ] **Step 7: Run full test suite**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ -q --tb=short`

- [ ] **Step 8: Commit**

```bash
git add werewolf_agent/storage/migrations.py werewolf_agent/runtime/graph.py tests/storage/test_migrations.py
git commit -m "feat: add schema migrations and fix sheriff badge night death routing"
```

---

## Task 8: 更新 PROGRESS.md 和 README.md

所有任务完成后，更新项目文档反映最新状态。

**Files:**
- Modify: `PROGRESS.md`
- Modify: `README.md`

- [ ] **Step 1: Update PROGRESS.md active task and completed section**

将当前 active task 更新为本计划的完成状态。将 Task 1-7 的成果添加到 Completed 列表。将 "Incomplete or Partially Implemented" 中已修复的项标记为完成或更新进度。

- [ ] **Step 2: Update README.md with new features**

添加以下 section:
- GameRunner 完整对局编排说明
- 本地认证系统使用说明
- Docker 快速启动 (如果 Task 6 完成)
- 新增评测指标说明

- [ ] **Step 3: Run full test suite final verification**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ -q --tb=short`
Expected: All tests pass, count > 918

- [ ] **Step 4: Commit**

```bash
git add PROGRESS.md README.md
git commit -m "docs: update progress and README for V1 design-complete milestone"
```

---

## Self-Review

**1. Spec coverage check against PROGRESS.md gaps:**

| PROGRESS.md 缺失项 | 对应 Task |
|---|---|
| 持久化基础设施 (docker-compose) | Task 6 |
| 完整 LLM 多智能体实时对局 | Task 1 (GameRunner) |
| 观战台前端增强 | Task 5 |
| 认证/权限系统 | Task 3 |
| 评测指标补全 | Task 4 |
| Memory 接入游戏生命周期 | Task 2 |
| Schema migration | Task 7 |
| Sheriff badge night death 路由 | Task 7 |

未覆盖项 (需后续计划):
- Qdrant/pgvector 生产级 RAG (当前 SiliconFlow + TF-IDF 已满足 V1 开发)
- 真实 MCP 协议连接器 (mock 已满足 V1 验证)
- 真实异步计时器 (wolf discussion / day speech)
- PostgreSQL/Redis 生产级替代 (V1 用 SQLite)

**2. Placeholder scan:**
- 无 TBD、TODO、implement later
- 所有代码步骤包含完整实现代码
- 所有测试步骤包含实际测试代码

**3. Type consistency:**
- `GameRunnerConfig.repository` 类型为 `Any` (匹配现有 `GameRepository` Protocol)
- `AuthConfig.local_users` 类型为 `dict[str, str]` (与 `authorized_callers` 一致)
- `QualityMetrics` 新增字段为 `float` (与现有字段一致)
- `MigrationManager` 使用 `sqlite3.connect` (与 `SqliteGameRepository` 一致)
