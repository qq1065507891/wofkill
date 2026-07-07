# -*- coding: utf-8 -*-
"""
离线对比 belief 改动对狼人与好人区分度的影响。

作者: Project contributors
修改日期: 2026-07-07

使用示例:
    >>> import scripts.offline_belief_worlds_compare
    >>> scripts.offline_belief_worlds_compare.ALL_PLAYERS
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from werewolf_agent.cognition.belief import BeliefUpdater
from werewolf_agent.cognition.world_state import StructuredFact

ALL_PLAYERS = [f"p{i:02d}" for i in range(1, 13)]

_TRUTH = {
    "p01": "werewolf", "p02": "werewolf", "p03": "werewolf", "p04": "werewolf",
    "p05": "villager", "p06": "villager", "p07": "villager",
    "p08": "seer", "p09": "witch", "p10": "hunter", "p11": "idiot", "p12": "hybrid",
}

SCENARIOS: list[tuple] = []


def _scene(name, facts, viewer, viewer_role="villager"):
    SCENARIOS.append((name, _TRUTH, facts, viewer, viewer_role))


# 场景1 (P0 投票信号): 真 seer p08 查杀真狼 p01 + 金水 p07;
#   好人 p05/p06 投被查杀的 p01; 狼 p02 投金水 p07. P0 应让投票流向更新 trust.
_scene("vote_signal", [
    StructuredFact(fact_type="claimed_role", source_player="p08", value="seer", day=1),
    StructuredFact(fact_type="seer_check_claim", source_player="p08", target_player="p01", value="wolf", day=1),
    StructuredFact(fact_type="seer_check_claim", source_player="p08", target_player="p07", value="good", day=1),
    StructuredFact(fact_type="vote", source_player="p05", target_player="p01", day=1),
    StructuredFact(fact_type="vote", source_player="p06", target_player="p01", day=1),
    StructuredFact(fact_type="vote", source_player="p02", target_player="p07", day=1),
], "p05")

# 场景2 (P1 对跳): 真 seer p08 + 悍跳狼 p01 都跳 seer;
#   p01 假查杀好人 p05, p08 真查杀狼 p03. 对跳应削弱假查杀对 p05 的污染.
_scene("counterclaim", [
    StructuredFact(fact_type="claimed_role", source_player="p08", value="seer", day=1),
    StructuredFact(fact_type="claimed_role", source_player="p01", value="seer", day=1),
    StructuredFact(fact_type="seer_check_claim", source_player="p01", target_player="p05", value="wolf", day=1),
    StructuredFact(fact_type="seer_check_claim", source_player="p08", target_player="p03", value="wolf", day=1),
], "p06")

# 场景3 (回归): 单预言家查杀真狼, 无投票无对跳. P0/P1 不应让它变差.
_scene("single_seer_regression", [
    StructuredFact(fact_type="claimed_role", source_player="p08", value="seer", day=1),
    StructuredFact(fact_type="seer_check_claim", source_player="p08", target_player="p01", value="wolf", day=1),
], "p05")


def belief_metrics(state, truth, viewer):
    wolf_trust, good_trust, wolf_prob, good_prob = [], [], [], []
    wolf_suspect = good_suspect = wolf_n = good_n = 0
    for pid, role in truth.items():
        if pid == viewer:
            continue
        b = state.beliefs.get(pid)
        if b is None:
            continue
        is_suspect = b.faction_lean == "wolf_lean" or b.trust < 0.35
        wp = b.role_probabilities.get("werewolf", 0.0)
        if role == "werewolf":
            wolf_n += 1
            wolf_trust.append(b.trust)
            wolf_prob.append(wp)
            wolf_suspect += int(is_suspect)
        else:
            good_n += 1
            good_trust.append(b.trust)
            good_prob.append(wp)
            good_suspect += int(is_suspect)
    awt = sum(wolf_trust) / len(wolf_trust) if wolf_trust else 0.0
    agt = sum(good_trust) / len(good_trust) if good_trust else 0.0
    awp = sum(wolf_prob) / len(wolf_prob) if wolf_prob else 0.0
    agp = sum(good_prob) / len(good_prob) if good_prob else 0.0
    return {
        "trust_sep": round(agt - awt, 3),
        "wolf_prob_sep": round(awp - agp, 3),
        "wolf_sus_rec": round(wolf_suspect / wolf_n, 3) if wolf_n else 0.0,
        "good_false_sus": round(good_suspect / good_n, 3) if good_n else 0.0,
    }


def run() -> None:
    print(f"{'scenario':<24}{'trust_sep':>11}{'wolf_prob_sep':>15}{'wolf_sus_rec':>13}{'good_false_sus':>16}")
    for name, truth, facts, viewer, vrole in SCENARIOS:
        updater = BeliefUpdater()
        state = updater.initialize(ALL_PLAYERS, viewer)
        state = updater.update(state, facts, 1)
        m = belief_metrics(state, truth, viewer)
        print(f"{name:<24}{m['trust_sep']:>11}{m['wolf_prob_sep']:>15}{m['wolf_sus_rec']:>13}{m['good_false_sus']:>16}")


if __name__ == "__main__":
    run()
