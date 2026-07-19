# 语义修复公开证据门禁设计

## 1. 背景

当前发言重试链会把第一次通过解析、但未通过 `speech_quality` 的输出保存为
`semantic_repair_source`。后续候选不仅要重新通过发言质量校验，还必须满足：

- 目标与源输出一致；
- 全部已有公开来源支撑的事实声明都被保留；
- 说话人归属与否定关系不变；
- 不新增任何事实声明。

这套复合门禁保护了公开信息边界，但也把尚未被接受的失败稿当成权威业务决策。
质量重试提示要求模型补充公开证据，而门禁又拒绝原稿中没有的事实声明；即使新增声明已由
`public_claim_ledger` 证实也会失败。历史冻结样本中 110 个语义修复行动只有 47 个修复成功，
同时出现大量 `semantic_claim_retention` 终态 fallback，说明当前契约会系统性误伤正常修订。

## 2. 目标

1. 保留狼人杀不完全信息边界：不得捏造公开事实、篡改说话人或翻转原声明的否定关系。
2. 允许修复稿删除旧论点、调整未执行的普通发言目标，以及补充公开账本已经支持的新事实。
3. 每次语义门禁拒绝必须产生稳定、可聚合且可读的具体理由。
4. 拒绝理由进入重试提示、终态 action trace 和运行日志，但不得记录原始私密发言或隐藏角色真值。
5. 保持现有 `semantic_repair_audit` 指标字段兼容；新增字段必须有安全默认值。

## 3. 非目标

- 不重写整个发言质量系统。
- 不改变普通动作合法性、投票合法性或 provider 重试策略。
- 不用 LLM 再判断一次 LLM 输出是否真实。
- 不扩大公共事件的可见范围，不把 moderator-only 审计内容写入玩家视图。
- 不为未来可能存在的“已提交发言目标”预建配置系统。

## 4. 方案比较

### 方案 A：删除整个语义修复门禁

实现最简单，也能立即减少 fallback，但说话人归属和否定关系可能在重试中被静默篡改；
发布前清洗器只验证事实是否有来源，不能完整保护这两种语义关系。风险不可接受。

### 方案 B：保留当前门禁，只把错误信息写详细

能改善可观测性，但继续禁止有公开支撑的新证据、删除旧论点和合法目标调整，无法解决
重试提示与验收条件互相冲突的问题。

### 方案 C：缩窄为公开证据真实性门禁（采用）

每个最终候选独立对照权威 `public_claim_ledger`；只对事实真实性和被保留声明的语义关系
做硬拒绝。源输出仍用于发现“同一内容被换说话人或翻转否定关系”，不再作为必须完整复制的模板。
这个方案复用现有分类器和审计结构，改动边界最小。

## 5. 验证契约

新增一个不可变验证结果，例如：

```python
@dataclass(frozen=True)
class SemanticRepairValidationResult:
    accepted: bool
    reason_codes: tuple[str, ...]
    audit: dict[str, object]
```

验证器必须一次返回全部失败原因，不能只返回第一个布尔失败。原因码固定为：

| 原因码 | 含义 | 是否硬拒绝 |
|---|---|---|
| `unsupported_public_claim` | 最终候选包含权威公开账本不能支持的事实声明 | 是 |
| `speaker_attribution_changed` | 最终候选保留同一声明内容，但把已有非空说话人换成其他人 | 是 |
| `negation_changed` | 最终候选保留同一声明内容，但翻转肯定/否定关系 | 是 |

以下变化继续写入审计指标，但不再拒绝：

- 删除源输出中的部分或全部已验证论点；
- 新增 `public_claim_ledger` 已支持的事实声明；
- 修改普通发言中的怀疑对象或倾向目标；
- 对缺失说话人归属的声明，依据公开账本补齐正确归属。

`target_preserved`、`verified_claim_count`、`retained_verified_claim_count`、
`introduced_claim_count` 等现有字段继续保留，便于观察策略漂移。新增：

- `unsupported_public_claim_count`；
- `rejection_reason_codes`。

兼容函数 `semantic_repair_retains_verified_claim()` 暂时保留为布尔包装器，避免已有导入路径失效；
主运行流改用结构化结果。

## 6. 数据流与错误处理

1. 模型输出先完成协议、schema 和动作合法性校验。
2. 若本行动存在 `semantic_repair_source`，必须先调用结构化语义验证器，再调用
   `speech_quality`。这样无公开支撑事实由本门禁给出稳定原因码，不会先被通用
   `public_record_grounding` 错误吞掉。
3. 若尚无 `semantic_repair_source`，首次输出仍按现有 `speech_quality` 路径处理；首次质量失败
   只负责建立修复源。后续候选的公开事实真实性一律由结构化语义验证器先判定。
4. 语义验证和发言质量都通过时正常生成 action trace；审计仍记录保留率、目标变化和新增声明数。
5. 语义验证失败时：
   - `RetryInfo.error_code` 保持 `semantic_claim_retention`，兼容现有聚合；
   - `RetryInfo.reason_codes` 写入全部稳定原因码；
   - `error_message` 写入对应中文原因，不含原始发言；
   - `correction_hint` 只针对实际失败项，不再笼统要求保留全部论点或禁止所有新事实；
   - 运行日志写 `agent/task/attempt/reason_codes`，不写原始响应；
   - 被拒输出继续进入现有 attempt 拒绝链。
6. 重试耗尽时，最终 warning 除 `last_error` 外必须包含 `last_reason_codes`；终态 action trace 的
   `retry` 对象保留相同原因码。
7. 发布前仍执行 `sanitize_public_text()` 作为纵深防御；公开写入后的文本不能包含无来源事实。

语义验证器以 `AgentContext.public_claim_ledger` 为权威输入，但该账本只能由当前行动之前、对该玩家
可见的公开发言事件构造。发布前清洗器使用 `public_speech_history(gs.events)` 重建同一事件边界的
公开历史。两条路径必须继续共享 `classify_public_claims()` 和 `_claim_is_supported()` 的判定语义，
并通过契约测试证明同一组公开事件在两条路径上的 support 结论一致。不得用 `recent_transcript`、
摘要文本、moderator-only 事件或角色真值作为任一路径的权威事实来源。

`RetryInfo.reason_codes` 使用空列表默认值，其他失败类型无需改动即可保持兼容。

## 7. 日志示例

```text
Agent p05 semantic repair rejected
(task=TaskType.SHERIFF_SPEECH, attempt=2,
reason_codes=unsupported_public_claim,speaker_attribution_changed)
```

终退日志：

```text
Agent p05 exhausted retries
(task=TaskType.SHERIFF_SPEECH, attempts=3,
last_error=semantic_claim_retention,
last_reason_codes=unsupported_public_claim) -> fallback
```

日志只给出稳定原因码；中文解释保存在 `RetryInfo.error_message`，供提示和审计使用。

## 8. 测试策略

### 8.1 验证器单元测试

- 新增有公开账本支撑的声明可以通过。
- 删除一个或全部已验证旧声明可以通过。
- 合法目标改变不因语义 retention 被拒绝。
- 新增或保留无公开支撑的事实被拒绝，理由为 `unsupported_public_claim`。
- 换说话人被拒绝，理由为 `speaker_attribution_changed`。
- 翻转否定关系被拒绝，理由为 `negation_changed`。
- 同一候选同时违反多项规则时返回全部原因，顺序稳定且无重复。

### 8.2 运行流测试

- 语义拒绝生成包含 `reason_codes` 的 `RetryInfo`。
- 重试提示包含对应中文原因，不再声称“不得新增任何事实”。
- 重试耗尽日志包含 `last_reason_codes`。
- terminal fallback 的 action trace 中 `retry.reason_codes` 与日志一致。
- 修复成功仍写兼容的 `semantic_repair_audit`，且新增字段可被序列化。
- 当候选同时违反公开证据门禁和普通发言质量时，先得到公开证据原因码，不被
  `public_record_grounding` 覆盖。

### 8.3 回归测试

- 公开文本清洗与 public ledger 测试保持通过。
- 给定同一事件边界，`public_claim_ledger` 验证与 `public_speech_history()` 清洗必须对事实支持得出
  相同结论；私有事件不得进入任一权威证据快照。
- 日志、retry metadata 和 action trace 只包含稳定原因码与固定安全文案，不包含被拒原文、私有角色
  或隐藏真值。
- player retry、action trace、semantic acceptance metrics 和相关 runtime action audit 测试保持通过。
- Ruff 与 `git diff --check` 通过。

## 9. 验收标准

1. 有公开证据支持的新事实不会再触发 `semantic_claim_retention`。
2. 删除旧论点或调整未执行的普通发言目标不会触发该门禁。
3. 无公开支持事实、说话人篡改和否定翻转仍会稳定失败。
4. 每次失败可从 `RetryInfo` 和日志看到至少一个具体原因码。
5. 终态 fallback action trace 保留最后一次拒绝的相同原因码。
6. 所有相关自动化测试与静态检查通过。
