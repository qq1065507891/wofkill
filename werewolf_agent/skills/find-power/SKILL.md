---
name: find_power
display_name: 找神
description: 通过发言和行为模式分析找出神职玩家
applicable_roles:
  - werewolf
  - villager
  - seer
  - witch
  - idiot
  - hybrid
applicable_phases:
  - speech
  - night_action
  - wolf_discussion
  - hunter_shot
faction: common
tags:
  - analysis
  - information
---

# 找神

通过分析发言行为、信息量和投票模式，识别场上可能的强神玩家。

## 何时使用

- 狼人夜间讨论，需要精确定位刀口目标时
- 白天需要区分真预言家和悍跳狼时
- 猎人需要判断开枪目标是否为强神时

## 何时不使用

- 首轮信息极度匮乏时（分析可靠性不足）

## 如何使用

调用此技能后，系统将基于当前游戏状态分析：
1. 信息量异常的玩家（掌握私下信息的可能是神职）
2. 发言中无意暴露角色特征的玩家
3. 投票模式与已知身份矛盾的目标
4. 被刀/被救/被查验后的行为异常

## 注意事项

- 找神目标是动态的，需要结合后续信息持续更新
- 狼人找神是为了刀口精准度；好人找神是为了保护强神和识别悍跳
- 好人侧的找神结论仅用于私下防守分析，不得在公开发言中点明疑似神职或替狼人缩小目标范围
