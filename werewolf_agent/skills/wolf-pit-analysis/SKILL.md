---
name: wolf_pit
display_name: 盘狼坑
description: 系统性分析场上可能的狼人分布，缩小嫌疑范围
applicable_roles:
  - villager
  - seer
  - witch
  - hunter
  - idiot
  - hybrid
applicable_phases:
  - speech
  - sheriff_speech
  - pk_speech
  - hunter_shot
faction: good
tags:
  - analysis
  - logic
is_tool_skill: true
tool_name: skill_analyze_wolf_pit
tool_description: >
  分析当前狼坑（嫌疑区和排除区）。基于验人信息、投票模式、发言矛盾等
  给出完整分析。调用后会返回详细分析结果供你参考。
---

# 盘狼坑

系统性分析场上可能的狼人身份分布，基于可观测行为构建嫌疑人区和排除区。

## 何时使用

- 白天发言阶段，需要输出有逻辑支撑的判断时
- 票型异常、多人跟风、发言自相矛盾时
- 猎人开枪前需要确认目标身份时
- 警上环节需要给好人阵营提供分析框架时

## 何时不使用

- 首夜无公开信息时（分析数据不足）
- 夜间行动阶段（不需要公开发言时）

## 如何使用

调用此技能后，系统将基于当前游戏状态自动分析：
1. 从 belief_state 提取行为偏向狼人的玩家 → 嫌疑人区
2. 从 world_state 提取被查杀、被金水的玩家 → 交叉验证
3. 结合投票模式、发言矛盾构建证据链
4. 输出嫌疑人列表 + 排除列表 + 行为证据

## 注意事项

- 排除区的玩家不代表绝对好人，只代表当前证据下嫌疑较低
- 嫌疑人按证据强度排序，需要结合后续发言动态调整
- 对跳预言家的情况下，优先分析两边查验结果的逻辑自洽性
