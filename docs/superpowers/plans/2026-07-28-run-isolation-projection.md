# Run Isolation and Projection Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让相同 seed 的并发或重复真实局拥有独立 run 身份、日志和应急产物，并使状态投影按事件隔离预算，避免单个异常 payload 污染整局评估。

**Architecture:** `game_id` 表示游戏实体，新增不可变 `run_id` 表示一次执行；未显式指定 game_id 时同时包含 seed 与 run 后缀。启动前通过 repository 预检冲突，事件、日志和产物统一携带 run_id。JSON 投影使用独立的每事件/每字段预算与稳定降级占位，关键事件投影不完整时明确使验收失败。

**Tech Stack:** Python 3.12、Pydantic v2、LangGraph runtime、repository abstraction、标准 logging、pytest。

## Global Constraints

- 设计依据：`docs/superpowers/specs/2026-07-28-run-isolation-projection-design.md`。
- 同 seed 默认运行不得复用 game_id、日志文件或应急快照路径。
- 显式 game_id 若已存在任何快照必须在模型调用前失败；不暗中覆盖，也不猜测 resume。
- `run_id` 只用于执行隔离，不能改变规则随机性或 seed 语义。
- 单个非关键异常事件不得耗尽整局投影预算；关键事件丢失必须显式阻断验收。
- 所有项目命令使用 `conda run -n wofkill`。
- 新 Python 文件遵守项目中文头部和注释规则，日期使用 `2026-07-28`。
- 每个任务独立执行红测、绿测和提交。

---

## File Structure

- `werewolf_agent/runtime/game_runner_config.py`：解析和校验 run 身份。
- `werewolf_agent/runtime/game_runner.py`：生成默认身份并执行启动预检。
- `werewolf_agent/core/models.py`：在状态和事件中携带 run_id。
- `werewolf_agent/storage/`：向后兼容序列化并提供冲突查询。
- `werewolf_agent/runtime/event_metadata.py`：统一为新事件盖 run_id。
- `werewolf_agent/runtime/game_termination.py`：隔离应急快照文件名。
- `scripts/run_logging.py`：每 run 的 logging handler/filter。
- `scripts/run_real_game.py`：CLI run_id 和日志初始化入口。
- `werewolf_agent/evaluation/game_projection.py`：独立预算的安全 JSON 投影。
- `werewolf_agent/evaluation/acceptance_audit.py`、`balance_audit.py`：关键事件完整性门。

### Task 1: 建立向后兼容的 run 身份模型

**Files:**
- Modify: `werewolf_agent/runtime/game_runner_config.py:1-95`
- Modify: `werewolf_agent/core/models.py:64-175`
- Modify: `werewolf_agent/runtime/game_runner.py`
- Create: `tests/runtime/test_game_runner_config.py`
- Create: `tests/core/test_models.py`
- Modify: `tests/runtime/test_game_runner.py`

**Interfaces:**
- Produces: `GameRunnerConfig.run_id`、`GameState.run_id`、`GameEvent.run_id`、`new_run_id() -> str`、`default_game_id(seed, run_id) -> str`。

- [ ] **Step 1: 写相同 seed 身份隔离和旧数据兼容红测**

```python
def test_same_seed_creates_distinct_default_game_and_run_ids() -> None:
    first = GameRunner(GameRunnerConfig(
        seed=42, enable_default_rag_service=False,
    ))
    second = GameRunner(GameRunnerConfig(
        seed=42, enable_default_rag_service=False,
    ))
    assert first.config.run_id != second.config.run_id
    assert first.config.game_id != second.config.game_id
    assert first.config.game_id.startswith("g_42_")
    assert second.config.game_id.startswith("g_42_")


def test_legacy_event_without_run_id_still_loads() -> None:
    event = GameEvent(**legacy_event_payload())
    assert event.run_id == ""
```

测试 helper `legacy_event_payload()` 返回当前必填字段但故意省略 `run_id`；runner 关闭默认 RAG，测试不访问网络。

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_game_runner_config.py tests/core/test_models.py tests/runtime/test_game_runner.py -k "run_id or same_seed or legacy_event" -q`

Expected: FAIL，因为当前默认 game_id 只有 seed，模型也没有 run_id。

- [ ] **Step 3: 实现安全身份生成和兼容字段**

```python
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")


def new_run_id() -> str:
    return f"r_{uuid.uuid4().hex[:12]}"


def default_game_id(seed: int, run_id: str) -> str:
    suffix = run_id.removeprefix("r_")
    return f"g_{seed}_{suffix}"
```

`GameRunnerConfig.__post_init__()` 先补 run_id，再补 game_id；run_id 使用新 `_SAFE_RUN_ID`，game_id 继续使用现有 `_SAFE_GAME_ID`，避免破坏原本允许点号的显式 ID。`GameState` 和 `GameEvent` 使用默认空串兼容旧快照；GameRunner 新建状态时必须写入非空 run_id。

- [ ] **Step 4: 运行身份模型测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_game_runner_config.py tests/core/test_models.py tests/runtime/test_game_runner.py -q`

Expected: PASS。

- [ ] **Step 5: 提交 run 身份模型**

```bash
git add werewolf_agent/runtime/game_runner_config.py werewolf_agent/core/models.py werewolf_agent/runtime/game_runner.py tests/runtime/test_game_runner_config.py tests/core/test_models.py tests/runtime/test_game_runner.py
git commit -m "feat: add isolated run identity"
```

### Task 2: 在任何模型调用前拒绝 game_id 冲突

**Files:**
- Modify: `werewolf_agent/storage/repository.py`
- Modify: `werewolf_agent/storage/memory_store.py`
- Modify: `werewolf_agent/storage/postgres_store.py`
- Modify: `werewolf_agent/storage/sqlite_store.py`
- Modify: `werewolf_agent/runtime/game_runner.py`
- Create: `tests/storage/test_repository_game_exists.py`
- Modify: `tests/storage/test_postgres_store.py`
- Modify: `tests/storage/test_storage.py`
- Modify: `tests/runtime/test_game_runner.py`

**Interfaces:**
- Produces: `GameIdConflictError`；repository 的 `game_exists(game_id: str) -> bool`；GameRunner 启动预检。

- [ ] **Step 1: 写冲突时零模型调用红测**

```python
def test_explicit_existing_game_id_fails_before_model_registry_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = in_memory_repository_with_game("g_fixed")
    registry_builds = 0

    def spy_registry_build(_runner: GameRunner) -> None:
        nonlocal registry_builds
        registry_builds += 1
        return None

    monkeypatch.setattr(GameRunner, "_build_agent_registry", spy_registry_build)
    with pytest.raises(GameIdConflictError, match="g_fixed"):
        GameRunner(GameRunnerConfig(
            seed=42,
            game_id="g_fixed",
            run_id="r_new",
            repository=repository,
            use_agent_registry=True,
            enable_default_rag_service=False,
        ))
    assert registry_builds == 0
```

`in_memory_repository_with_game()` 在同一测试文件内定义并保存指定 ID 的状态；参数化 fixture 同时覆盖已终局和未终局快照，二者都必须拒绝。

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_game_runner.py tests/storage/test_repository_game_exists.py -k "existing_game_id" -q`

Expected: FAIL，因为当前初始化没有统一冲突门。

- [ ] **Step 3: 增加窄查询和启动顺序**

```python
class GameIdConflictError(RuntimeError):
    pass


def _assert_game_id_available(repository: GameRepository, game_id: str) -> None:
    if repository.game_exists(game_id):
        raise GameIdConflictError(
            f"game_id already exists and resume is not enabled: {game_id}"
        )
```

GameRunner 初始化顺序固定为：规范化 config → repository 预检 → 构造 registry/graph → 创建初始状态。PostgreSQL 实现使用 `SELECT EXISTS`，不加载完整状态；任何 storage 异常都不得被解释为“不存在”。

- [ ] **Step 4: 运行 repository 和启动回归**

Run: `conda run -n wofkill python -m pytest -n 0 tests/storage/test_repository_game_exists.py tests/storage/test_postgres_store.py tests/storage/test_storage.py tests/runtime/test_game_runner.py -q`

Expected: PASS。

- [ ] **Step 5: 提交启动冲突门**

```bash
git add werewolf_agent/storage/repository.py werewolf_agent/storage/memory_store.py werewolf_agent/storage/postgres_store.py werewolf_agent/storage/sqlite_store.py werewolf_agent/runtime/game_runner.py tests/storage/test_repository_game_exists.py tests/storage/test_postgres_store.py tests/storage/test_storage.py tests/runtime/test_game_runner.py
git commit -m "fix: reject game id collisions before startup"
```

### Task 3: 为事件和应急产物统一盖 run_id

**Files:**
- Modify: `werewolf_agent/runtime/event_metadata.py`
- Modify: `werewolf_agent/runtime/game_termination.py`
- Modify: `werewolf_agent/runtime/game_runner.py`
- Modify: `tests/runtime/test_event_metadata_v2.py`
- Modify: `tests/runtime/test_game_termination.py`
- Modify: `tests/runtime/test_game_runner.py`

**Interfaces:**
- Consumes: state/config 的 game_id 与 run_id。
- Produces: `stamp_new_events(..., game_id, run_id)`；`emergency_artifact_path(output_dir, game_id, run_id)`。

- [ ] **Step 1: 写事件和文件路径隔离红测**

```python
def test_stamp_new_events_sets_run_id_without_changing_event_id_contract() -> None:
    event = GameEvent(type="speech", payload={"speaker": "p01", "text": "观察。"})
    stamped = stamp_new_events(
        "g_42_a", [], [event], run_id="r_a",
    )
    assert stamped[0].run_id == "r_a"
    assert stamped[0].event_id.startswith("g_42_a:")


def test_emergency_paths_differ_for_runs() -> None:
    first = emergency_artifact_path(Path("out"), "g_fixed", "r_a")
    second = emergency_artifact_path(Path("out"), "g_fixed", "r_b")
    assert first != second
    assert first.name == "emergency_abort_g_fixed_r_a.json"
```

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_event_metadata_v2.py tests/runtime/test_game_termination.py -k "run_id or emergency_paths" -q`

Expected: FAIL，因为事件和应急文件当前只使用 game_id。

- [ ] **Step 3: 实现统一盖章和安全文件名**

所有新事件从所属 state 取得 run_id；旧调用可省略并得到空串，但真实 GameRunner 路径不允许空串。文件名只接受已通过 config 校验的 ID，不再次清洗成可能冲突的名称。

- [ ] **Step 4: 运行事件、应急和 runner 回归**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_event_metadata_v2.py tests/runtime/test_game_termination.py tests/runtime/test_game_runner.py -q`

Expected: PASS。

- [ ] **Step 5: 提交事件与产物隔离**

```bash
git add werewolf_agent/runtime/event_metadata.py werewolf_agent/runtime/game_termination.py werewolf_agent/runtime/game_runner.py tests/runtime/test_event_metadata_v2.py tests/runtime/test_game_termination.py tests/runtime/test_game_runner.py
git commit -m "fix: stamp events and artifacts with run identity"
```

### Task 4: 将真实局日志改为每 run 独立 handler

**Files:**
- Create: `scripts/run_logging.py`
- Modify: `scripts/run_real_game.py:1-165`
- Create: `tests/scripts/test_run_logging.py`
- Modify: `tests/scripts/test_run_real_game.py`

**Interfaces:**
- Produces: `RunLogContext`、`configure_run_file_logging(context, *, log_path=None) -> Path`、`close_run_file_logging(context) -> None`；CLI `--run-id`、`--log-path`。

- [ ] **Step 1: 写并发日志隔离红测**

```python
def test_two_run_handlers_never_cross_write(tmp_path: Path) -> None:
    first = RunLogContext(game_id="g_42_a", run_id="r_a", output_dir=tmp_path)
    second = RunLogContext(game_id="g_42_b", run_id="r_b", output_dir=tmp_path)
    first_path = configure_run_file_logging(first)
    second_path = configure_run_file_logging(second)
    logger = logging.getLogger("werewolf_agent.test")
    with bind_run_log_context(first):
        logger.info("only-first")
    with bind_run_log_context(second):
        logger.info("only-second")
    close_run_file_logging(first)
    close_run_file_logging(second)
    assert "only-first" in first_path.read_text(encoding="utf-8")
    assert "only-second" not in first_path.read_text(encoding="utf-8")
    assert "only-second" in second_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/scripts/test_run_logging.py tests/scripts/test_run_real_game.py -k "cross_write or run_id" -q`

Expected: FAIL，因为当前 `_configure_file_logging()` 安装共享路径 handler。

- [ ] **Step 3: 实现 context filter 和确定性清理**

```python
@dataclass(frozen=True, slots=True)
class RunLogContext:
    game_id: str
    run_id: str
    output_dir: Path


class RunContextFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "run_id", "") == self._run_id
```

使用 `contextvars.ContextVar` 的 `bind_run_log_context()` 给 record factory 或 adapter 添加 game_id/run_id；默认路径为 `<output_dir>/<game_id>_<run_id>.log`。CLI 在 config 身份确定后配置 handler，并在 `finally` 中 flush、remove、close。

- [ ] **Step 4: 运行日志与 CLI 测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/scripts/test_run_logging.py tests/scripts/test_run_real_game.py -q`

Expected: PASS。

- [ ] **Step 5: 提交每 run 日志**

```bash
git add scripts/run_logging.py scripts/run_real_game.py tests/scripts/test_run_logging.py tests/scripts/test_run_real_game.py
git commit -m "fix: isolate real game logs per run"
```

### Task 5: 将状态投影预算改为每事件、每字段隔离

**Files:**
- Modify: `werewolf_agent/evaluation/game_projection.py`
- Modify: `werewolf_agent/evaluation/acceptance_audit.py`
- Modify: `werewolf_agent/evaluation/balance_audit.py`
- Modify: `tests/evaluation/test_acceptance_game_projection.py`
- Modify: `tests/evaluation/test_final_acceptance_integrity.py`

**Interfaces:**
- Produces: `_JsonLimits`、`ProjectionFailure`、投影结果中的 `event_projection_failures` 与 `critical_events_complete`。

- [ ] **Step 1: 写恶意单事件不污染后续事件的红测**

```python
def test_oversized_event_does_not_consume_following_event_budget() -> None:
    state = state_with_events(
        event("speech", {"nested": deeply_nested_payload(depth=80)}),
        event("vote_result", {"eliminated_player_id": "p07"}),
        event("game_over", {"winner": "werewolves"}),
    )
    projection = project_state_for_evaluation(state)
    assert projection.events[-2]["type"] == "vote_result"
    assert projection.events[-1]["type"] == "game_over"
    assert projection.event_projection_failures[0].source_index == 0
    assert projection.critical_events_complete is True


def test_failed_critical_event_blocks_acceptance() -> None:
    projection = projection_with_failed_event("game_over")
    metrics = compute_acceptance_audit_metrics([projection])
    assert metrics["acceptance_projection_supported"] is False
    assert metrics["acceptance_projection_unsupported_reason"] == (
        "critical_event_projection_failed"
    )


def test_sixteen_hundred_rich_events_keep_acceptance_metrics_available() -> None:
    projection = project_acceptance_game(
        state_with_events(*(rich_audit_event(index) for index in range(1_600)))
    )
    assert len(projection.events) == 1_600
    assert projection.events_supported is True
    assert projection.critical_events_complete is True
```

测试 helpers 在各自测试文件中定义：`deeply_nested_payload()` 迭代创建嵌套 dict，`projection_with_failed_event()` 直接构造合法投影 DTO，`rich_audit_event()` 生成含 action trace 白名单字段的富审计事件；均不依赖模型调用。

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/evaluation/test_acceptance_game_projection.py tests/evaluation/test_final_acceptance_integrity.py -k "event_budget or failed_critical" -q`

Expected: FAIL，因为当前共享预算可让单事件影响整批 events。

- [ ] **Step 3: 实现独立限制和稳定失败占位**

```python
@dataclass(frozen=True, slots=True)
class _JsonLimits:
    max_depth: int = 32
    max_items_per_value: int = 10_000
    max_events: int = 5_000
    max_total_bytes: int = 32_000_000


@dataclass(frozen=True, slots=True)
class ProjectionFailure:
    source_index: int
    source_type: str
    reason_code: str
    critical: bool
```

players、deaths、每个 event payload 和顶层 metadata 分别创建预算对象。事件失败时追加不含 payload 的 `projection_event_unsupported` 占位并继续处理下一个事件；超过 `max_events` 记录 `event_count_limit`。所有分区完成后只序列化计数一次，超过 `max_total_bytes` 时以 `projection_total_size_limit` fail closed。关键事件判定覆盖带 `action_trace` 的事件、`reflection_complete`、`reflection_persistence_audit` 和终局事件；任一关键事件失败则 `critical_events_complete=False`，验收流水线必须失败。

- [ ] **Step 4: 运行投影和验收测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/evaluation/test_acceptance_game_projection.py tests/evaluation/test_final_acceptance_integrity.py -q`

Expected: PASS。

- [ ] **Step 5: 提交投影隔离**

```bash
git add werewolf_agent/evaluation/game_projection.py werewolf_agent/evaluation/acceptance_audit.py werewolf_agent/evaluation/balance_audit.py tests/evaluation/test_acceptance_game_projection.py tests/evaluation/test_final_acceptance_integrity.py
git commit -m "fix: isolate evaluation projection budgets"
```

### Task 6: 执行重复 seed、并发和完整投影验收

**Files:**
- Modify: `scripts/run_real_game_reports.py`
- Create: `tests/scripts/test_run_real_game_reports.py`
- Create: `tests/integration/test_run_isolation.py`

**Interfaces:**
- Produces: 报告中的 game_id/run_id/log_path/emergency_artifact_path 和关键事件完整性区块。

- [ ] **Step 1: 写两次同 seed 的端到端隔离红测**

```python
def test_two_same_seed_runs_have_disjoint_outputs(tmp_path: Path) -> None:
    first = run_fake_game(seed=42, output_dir=tmp_path)
    second = run_fake_game(seed=42, output_dir=tmp_path)
    assert first.run_id != second.run_id
    assert first.game_id != second.game_id
    assert first.log_path != second.log_path
    assert first.event_run_ids == {first.run_id}
    assert second.event_run_ids == {second.run_id}
    assert first.role_assignment == second.role_assignment
    assert first.critical_events_complete
    assert second.critical_events_complete
```

`run_fake_game()` 在集成测试内使用现有 deterministic fake registry 和 in-memory repository，完整走 GameRunner、事件持久化、投影和报告，不访问真实网络。

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/integration/test_run_isolation.py tests/scripts/test_run_real_game_reports.py -q`

Expected: FAIL，直到报告字段和集成链路全部接通。

- [ ] **Step 3: 输出身份和投影完整性报告**

报告按固定字段输出 game_id、run_id、绝对或相对 log path、应急产物 path、投影失败计数、关键事件完整性。不得仅凭 `events_supported=True` 宣布完整；必须同时检查 `critical_events_complete=True`。

- [ ] **Step 4: 运行隔离相关测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/integration/test_run_isolation.py tests/runtime/test_game_runner.py tests/runtime/test_event_metadata_v2.py tests/runtime/test_game_termination.py tests/scripts/test_run_logging.py tests/scripts/test_run_real_game.py tests/scripts/test_run_real_game_reports.py tests/evaluation/test_acceptance_game_projection.py tests/evaluation/test_final_acceptance_integrity.py -q`

Expected: PASS。

- [ ] **Step 5: 提交端到端隔离验收**

```bash
git add scripts/run_real_game_reports.py tests/scripts/test_run_real_game_reports.py tests/integration/test_run_isolation.py
git commit -m "test: verify run and projection isolation"
```

### Task 7: 执行完整回归

**Files:**
- No code changes; verification only.

- [ ] **Step 1: 运行静态检查**

Run: `conda run -n wofkill python -m ruff check werewolf_agent/runtime/game_runner_config.py werewolf_agent/core/models.py werewolf_agent/runtime/game_runner.py werewolf_agent/storage werewolf_agent/runtime/event_metadata.py werewolf_agent/runtime/game_termination.py scripts/run_logging.py scripts/run_real_game.py werewolf_agent/evaluation/game_projection.py werewolf_agent/evaluation/acceptance_audit.py werewolf_agent/evaluation/balance_audit.py scripts/run_real_game_reports.py`

Expected: PASS，无输出。

- [ ] **Step 2: 运行完整测试套件**

Run: `conda run -n wofkill python -m pytest -n 0 -q`

Expected: PASS。

- [ ] **Step 3: 检查两次相同 seed 的真实局产物**

分别运行两次相同 seed 的真实模型验收命令，确认 game_id、run_id、日志、应急文件、事件 run_id 全部分离；两份报告的 `critical_events_complete` 都必须为 true。真实调用的凭据和 provider 响应不得写入报告。
