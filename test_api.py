"""Quick API connectivity test."""
import os, sys
sys.path.insert(0, ".")
from werewolf_agent.model_gateway.providers import load_local_dotenv
load_local_dotenv(".env")
from werewolf_agent.model_gateway.router import ModelRouter
router = ModelRouter.from_yaml("config/models.yaml", register_env_providers=True)

# Test 1: basic
result = router.generate(
    agent_id="p01",
    task_type="wolf_discussion",
    prompt='Visible state: {"phase": "night", "alive_players": ["p01", "p02", "p03"]}\nAvailable actions: ["wolf_kill", "wolf_no_kill"]\nLegal targets: ["p02", "p03"]\nOutput your action JSON:',
    system_prompt='You are a werewolf. Output ONLY valid JSON: {"action_type": "wolf_kill", "target_id": "p02", "speech": "", "reason": "", "confidence": 0.8, "private_intent": {"true_role": "werewolf", "faction_goal": "kill", "claimed_view": "good", "pressure_target": null, "risk_flags": []}}',
)
print(f"Test 1 (wolf_kill): {'OK' if result.text else 'EMPTY'}")
if result.text:
    print(f"  Response: {result.text[:200]}")
    print(f"  Tokens: in={getattr(result, 'prompt_tokens', '?')} out={getattr(result, 'completion_tokens', '?')}")
else:
    print("  ERROR: No text returned")

# Test 2: speech
result2 = router.generate(
    agent_id="p05",
    task_type="speech",
    prompt='Day 2 discussion. You are a villager. Speak in Chinese.',
    system_prompt='You are a villager in a werewolf game. Output ONLY valid JSON: {"action_type": "speech", "target_id": null, "speech": "Chinese text here", "reason": "reason", "confidence": 0.7, "private_intent": {"true_role": "villager", "faction_goal": "find_wolves", "claimed_view": "good", "pressure_target": null, "risk_flags": []}}',
)
print(f"\nTest 2 (speech): {'OK' if result2.text else 'EMPTY'}")
if result2.text:
    print(f"  Response: {result2.text[:200]}")
