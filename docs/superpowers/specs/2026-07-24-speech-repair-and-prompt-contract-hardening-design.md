# 发言修复收敛与提示词合同校验加固设计

日期：2026-07-24
状态：已获用户方向确认，待规格审阅

## 1. 背景

2026-07-24 的真实对局暴露了两项相互独立的代码问题。

第一，玩家发言修复可能在不同校验门之间振荡。已观察到的失败路径是：首次输出触发 `speech_quality`，第二次输出因 `unsupported_public_claim` 被语义门拒绝，第三次又回到 `speech_quality`，最终进入安全 fallback。现有流程只把最近一次 `RetryInfo` 带入下一轮，无法保证后续修复继续满足更早的约束。

第二，MiniMax Anthropic-compatible provider 为 prompt cache 把 system prompt 包装为 text-block 列表后，最终提示词观察器对 canonical JSON 字节直接查找原始多行 persona 文本。JSON 会转义换行等字符，因此 persona 实际存在时也可能被误判为缺失，并在 HTTP 请求前抛出 `FinalPromptContractError`。

`safe_fallback` 记录本身不是一次 provider 请求。它由动作层追加为确定性终止边界，但当前记录继承最后一次模型尝试的 provider/model，容易在展示时被误解为又调用了一次模型。

## 2. 目标

1. 让动作修复约束单调累积，避免在质量门和语义门之间来回振荡。
2. 保持公开证据语义门 fail-closed，不降低任何安全约束。
3. 让 `safe_fallback` 明确表达“没有调用 provider”。
4. 让最终提示词合同检查逻辑 system 文本，同时继续保留真实 provider payload 的完整性证明。
5. 不改变现有 prompt cache 请求结构、模型路由、重试次数和公开 API。

## 3. 非目标

- 不处理系统 DNS、网络连通性、endpoint 熔断或启动预检。
- 不调整 provider 重试预算或退避时间。
- 不放宽 `unsupported_public_claim`、说话人归属或否定关系校验。
- 不引入新的模型调用、额外付费修复轮次或外部服务。
- 不在日志、trace 或持久化事件中保存被拒发言原文、完整 persona 或 provider 原始响应。
- 不重构整个 `player_action_flow.py` 或 model gateway。

## 4. 总体设计

改动分成两个可独立交付的责任边界：

1. `agents` 层增加一次动作生命周期内的累计修复状态，只负责聚合稳定错误码并生成下一轮约束；现有质量校验和语义校验仍由原模块负责。
2. `model_gateway` 层分离“逻辑 system 文本字节”和“真实 provider payload 字节”；合同校验使用前者，HMAC/送达证明使用后者。

两个改动之间没有运行时依赖，应拆成独立提交并分别回归。

## 5. 发言修复收敛设计

### 5.1 新增累计修复状态

新增 `werewolf_agent/agents/player_repair_state.py`，定义仅存在于单次 `PlayerAgent.act()` 调用内的 `RepairConstraintState`。该对象至少维护：

- 首次可修复的 `PlayerAction`，仅驻留内存；
- 按发生顺序去重的质量错误类别；
- 按稳定顺序去重的语义原因码；
- 当前事实策略：`normal` 或 `verified_claims_only`；
- 是否已经进入语义修复阶段。

该模块只负责状态累积和修复提示合成，不调用 provider、不读取游戏状态、不执行语义验证，也不持久化原文。

### 5.2 状态转移

`player_action_flow.py` 保持现有最多三次动作尝试和 structured-output mode 推进规则，只替换“单个最近错误决定下一轮提示”的行为。

状态转移如下：

1. 首次 `speech_quality` 失败：保存源动作和具体质量错误；下一轮要求定点修复源发言。
2. 后续 `semantic_claim_retention` 失败：累积全部稳定原因码；若包含 `unsupported_public_claim`，事实策略切换为 `verified_claims_only`。
3. 后续任何失败：新的 `RetryInfo` 必须包含全部仍适用的累计约束，而不是只包含最近一次错误。
4. `verified_claims_only` 一旦启用，在该动作生命周期内不得退回 `normal`。
5. 输出通过合法动作、语义修复、发言质量和投票质量检查后，按现有成功路径结束。
6. 尝试耗尽后继续调用现有 `build_task_terminal_fallback()` 和 `preserve_verified_claim_in_fallback()`，不新增第四次模型请求。

### 5.3 累计提示规则

累计提示必须同时表达：

- 原始 `speech_quality` 所缺少的必填成分；
- 已发生的语义原因码对应的固定修复动作；
- `verified_claims_only` 下禁止新增公开事实声明；
- 允许使用“我倾向”“我怀疑”“目前不能确定”等主观判断完成站队、目标和理由要求；
- 不把后台身份、私有事件、完整公开账本或不可见证据注入提示。

现有 `semantic_repair_correction_hint()` 继续提供按原因码生成的固定语义提示。累计状态负责组合提示，不复制语义原因码与中文说明的映射。

### 5.4 审计与隐私

`semantic_repair_audit` 必须增加只包含稳定码的 `repair_failure_history`，例如：

```json
["speech_quality", "semantic_claim_retention", "speech_quality"]
```

不得写入：

- 被拒发言原文；
- 模型 correction prompt；
- 解析后的未发布声明内容；
- 私有身份或夜间信息。

`RepairConstraintState` 暴露只读 `failure_history`。成功路径和 terminal fallback 路径都必须把它作为显式关键字参数传给 `build_semantic_repair_audit()`；该函数使用空元组默认值保持旧调用兼容，并在审计字典中序列化为列表。直接成功且从未进入修复的动作仍不生成 `semantic_repair_audit`。原有 `rejection_reason_codes`、公开证据安全计数和 fallback 分类保持含义不变。

### 5.5 `safe_fallback` 尝试语义

`GenerationAttemptContext.append_terminal_fallback()` 继续复制最后一次尝试以保留 request、模型配置和推理级别的归因信息，但必须显式设置：

```text
provider_attempted = false
route_kind = safe_fallback
attempt_outcome = attempt_failure
root_cause = invalid_output
reasoning_token_count = 0
```

推理状态必须继续满足 `AttemptExecutionRecord` 的既有不变量：

- `requested_reasoning_level != none`：使用 `ReasoningStatus.FALLBACK_DISABLED` 和 `EvidenceKind.FALLBACK_DISABLED`；
- `requested_reasoning_level == none`：使用 `ReasoningStatus.NOT_REQUESTED` 和 `EvidenceKind.NONE`。

稳定枚举不改名，以避免破坏已有报告和验收合同。`scripts/run_real_game_reports.py::_reasoning_evidence_summary()` 必须把 `provider_attempted` 投影到每条 attempt，并让文本/调用方依据 `provider_attempted=false` 把该条解释为“确定性安全兜底，未调用模型”，而不是新的 provider attempt。现有 `attempt_count` 和 `reasoning_fallback_disabled_count` 仍按终止路线记录安全退化事实，不因 `provider_attempted=false` 被删除或改义。

## 6. 最终提示词合同校验设计

### 6.1 明确两个字节边界

`FinalPromptAssembly` 中两个字段的责任明确为：

- `system_bytes`：provider 最终接收到的逻辑 system 文本，按文本块顺序还原后编码为 UTF-8；用于合同 marker 校验。
- `provider_payload_bytes`：实际请求体的 canonical JSON 字节；用于 HMAC、送达证明和 payload 一致性检查。

合同校验不得再从 canonical JSON 的容器表示中直接搜索原始多行文本。

### 6.2 text-block 还原

在 Anthropic provider 公共辅助代码中保留 `_wrap_system_prompt_for_cache()`，并把观察器辅助函数改为：

- `str`：直接 UTF-8 编码；
- text-block 列表：仅当每个元素都是字典、`type == "text"` 且 `text` 是字符串时有效；按原顺序无分隔符拼接全部 `text` 字段，再 UTF-8 编码；
- `None`：返回空字节；
- 任意一个 block 非法或为非 text 类型：整个逻辑 system 结果返回空字节，不得忽略坏块或只保留其他好块，使必需合同在 HTTP 前 fail-closed。

当前生产结构只有一个 system text block，因此还原结果必须与包装前 `system_prompt.encode("utf-8")` 完全相等。辅助函数由 Anthropic 与 MiniMax provider 共用，禁止各自实现一份。

### 6.3 合同失败行为

`validate_final_prompt_contract()` 的以下行为保持不变：

- 必需区块缺失时在 HTTP 前抛出 `FinalPromptContractError`；
- 非 persona 区块继续检查顺序；
- 重复 marker 继续视为未确认；
- 普通监控回调异常不得阻断 provider；合同异常仍必须阻断。

本次不修改 router 对 `FinalPromptContractError` 的分类或 fallback 策略，以保持行为边界稳定。

### 6.4 Prompt cache 兼容性

实际发送的 `payload["system"]` 仍是带 `cache_control: {"type": "ephemeral"}` 的 text-block 列表。修复只改变观察器读取逻辑，不改变：

- payload 字段；
- block 顺序；
- cache-control 标记；
- canonical payload 序列化；
- token/cache usage 统计。

## 7. 文件责任与预计改动

### 新增

- `werewolf_agent/agents/player_repair_state.py`：单次动作内累计修复约束与组合提示。

### 修改

- `werewolf_agent/agents/player_action_flow.py`：接入累计状态，不再让最近错误覆盖早期约束。
- `werewolf_agent/agents/player_quality_retries.py`：提供可被累计状态复用的质量提示片段。
- `werewolf_agent/agents/semantic_repair_audit.py`：记录稳定的失败路径，保持公开证据语义门不变。
- `werewolf_agent/model_gateway/generation_attempt_context.py`：终止 fallback 设置 `provider_attempted=False`。
- `werewolf_agent/model_gateway/providers/anthropic.py`：从 text-block 还原逻辑 system 文本字节。
- `werewolf_agent/model_gateway/providers/minimax.py`：继续复用 Anthropic 辅助函数，并验证观察器边界。
- `werewolf_agent/model_gateway/final_prompt_observer.py`：同步字段文档，合同算法保持逻辑不变。
- `scripts/run_real_game_reports.py`：在 attempt 快照中投影 `provider_attempted`，明确终止 fallback 未调用模型。

### 测试

- `tests/agents/test_player_retry.py`
- `tests/model_gateway/test_execution_records.py`
- `tests/model_gateway/test_final_prompt_observer.py`
- `tests/model_gateway/test_anthropic_provider.py`
- `tests/model_gateway/test_providers.py`
- `tests/scripts/test_run_real_game.py`
- `tests/evaluation/test_report_acceptance_metrics.py`：验证新增稳定审计字段不会造成 V1/V2 投影分歧，并锁定既有 reasoning 计数语义。

## 8. 测试设计

### 8.1 发言修复

必须覆盖：

1. `speech_quality -> unsupported_public_claim -> success`，第三次提示仍包含第一次质量约束。
2. `verified_claims_only` 启用后不能退回普通事实生成模式。
3. 多个语义原因码按既有稳定顺序去重。
4. 三次失败后只发布经过既有 fallback 安全处理的内容。
5. `repair_failure_history` 只有稳定码，不含原文或私密信息。
6. 直接成功、不涉及 semantic repair 的路径行为不变。
7. `safe_fallback.provider_attempted` 为 `False`，前面的真实尝试仍为 `True`。
8. 有推理请求时保持 `fallback_disabled`/`EvidenceKind.FALLBACK_DISABLED`，无推理请求时保持 `not_requested`/`EvidenceKind.NONE`；两种情况均为 `tokens=0` 和 `root=invalid_output`。
9. reasoning 报告的 attempt 快照包含 `provider_attempted=False`；既有 `attempt_count` 和 `reasoning_fallback_disabled_count` 语义不变。

### 8.2 Persona 合同

必须覆盖：

1. 多行中文 persona 经 cache text-block 包装后仍能通过合同。
2. 包含引号、反斜杠和换行的 persona 可精确还原。
3. 逻辑 `system_bytes` 与包装前 system prompt 的 UTF-8 字节完全相等。
4. `provider_payload_bytes` 与实际 canonical payload 完全相等。
5. persona 真正缺失时仍在 HTTP 前失败，fake HTTP client 调用次数为零。
6. persona 存在时 fake HTTP client 精确调用一次。
7. Anthropic 和 MiniMax 使用同一辅助函数得到一致结论。
8. prompt cache block 和 usage 字段保持原样。
9. 有效 block 与非法/非 text block 混合时整体 fail-closed，不能静默忽略坏块。
10. 对实际发送 payload 做类型敏感的深度相等断言，并单独断言 canonical payload 字节一致。

## 9. 验收标准

自动化验收：

- 新增聚焦测试全部通过；
- 现有 agent retry、semantic repair、final prompt observer、Anthropic/MiniMax provider、execution record 测试全部通过；
- `semantic_repair_public_evidence_safety_rate == 1.0`；
- `semantic_repair_speaker_attribution_preservation_rate == 1.0`；
- `semantic_repair_negation_preservation_rate == 1.0`；
- `semantic_repair_success_rate >= 0.95`；
- `semantic_repair_generic_template_count == 0`；
- `critical_task_reasoning_fallback_disabled_count == 0`；
- `python -m compileall -q werewolf_agent tests scripts` 通过；
- `git diff --check` 通过。

真实运行验收仅观察本设计范围：

- 不再出现 persona 实际存在却被判为 `missing required sections: persona`；
- 发言修复不再因丢失早期约束而出现已知的质量/语义振荡；
- terminal safe fallback 明确显示未调用 provider；
- DNS 或其他系统网络错误不计入本设计的成败。

## 10. 实施与回滚顺序

建议两个独立实现提交：

1. 修复逻辑 system 文本观察边界及其 provider 测试。
2. 增加累计修复状态、safe fallback 审计修正及其 agent/acceptance 测试。

若第一项回归失败，只回滚 provider 观察器提交，不影响动作生成。若第二项回归失败，只回滚累计状态提交，恢复现有三次重试行为。两个提交均不需要配置迁移或数据迁移。

## 11. 风险与控制

- **风险：累计提示过长。** 控制：只累计稳定错误码和短提示片段，不重复原文和完整公开账本。
- **风险：模型把“不得新增事实”理解为不允许表达观点。** 控制：提示中明确允许主观倾向、怀疑和不确定性表达。
- **风险：text-block 还原与实际 provider 语义不一致。** 控制：单 block 必须字节等于原 system prompt；多 block 按顺序提取，并对非法形状 fail-closed。
- **风险：新增审计字段破坏旧报告。** 控制：字段只做可选增量，旧 trace 缺失时保持现有归一化结果，不改变既有字段含义。
