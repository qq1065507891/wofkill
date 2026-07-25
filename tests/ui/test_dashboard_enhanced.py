# -*- coding: utf-8 -*-
"""
验证增强观战面板及版本化投票展示辅助函数。

作者: Project contributors
修改日期: 2026-07-25
"""

import json
from pathlib import Path
import re
import subprocess

import pytest

DASHBOARD_PATH = Path(__file__).parent.parent.parent / "werewolf_agent" / "ui" / "static" / "dashboard.html"
DASHBOARD_JS_PATH = Path(__file__).parent.parent.parent / "werewolf_agent" / "ui" / "static" / "dashboard.js"

@pytest.fixture
def dashboard_html():
    return DASHBOARD_PATH.read_text(encoding="utf-8")

@pytest.fixture
def dashboard_js():
    return DASHBOARD_JS_PATH.read_text(encoding="utf-8")

def test_cognitive_diff_section(dashboard_html):
    assert "cognitive-diff" in dashboard_html or "identity-prob" in dashboard_html

def test_rag_hit_panel(dashboard_html):
    assert "rag-hit" in dashboard_html or "rag-audit" in dashboard_html

def test_world_model_panel(dashboard_html, dashboard_js):
    assert "world-model-panel" in dashboard_html
    assert "worldModelBody" in dashboard_html
    assert "loadWorldModelAudit" in dashboard_js
    assert "/world-model-audit" in dashboard_js

def test_world_model_panel_reads_nested_audit_cards_and_escapes_json(dashboard_js):
    assert "a.possible_worlds?.top_worlds" in dashboard_js
    assert "a.simulation_predictions?.predictions" in dashboard_js
    assert "escapeHtml(JSON.stringify" in dashboard_js

def test_model_routing_panel(dashboard_html):
    assert "model-routing" in dashboard_html or "llm-profile" in dashboard_html

def test_persona_routing_panel(dashboard_html):
    assert "persona-routing" in dashboard_html or "persona-profile" in dashboard_html

def test_timeline_slider(dashboard_html):
    assert "timeline-slider" in dashboard_html or "day-slider" in dashboard_html or "day-select" in dashboard_html

def test_attention_filter_panel(dashboard_html):
    assert "attention-filter" in dashboard_html or "attention-stats" in dashboard_html

def test_cost_latency_panel(dashboard_html):
    assert "cost-latency" in dashboard_html or "token-usage" in dashboard_html

def test_private_intent_audit(dashboard_html):
    assert "private-intent" in dashboard_html

def test_dashboard_is_localized_to_chinese(dashboard_html, dashboard_js):
    assert "创建新游戏" in dashboard_html
    assert "开始游戏" in dashboard_html
    assert "当前阶段" in dashboard_html
    assert "阶段" in dashboard_html
    assert "天数" in dashboard_html
    assert "胜利方" in dashboard_js

def test_create_game_selects_created_game_before_start(dashboard_js):
    assert "const created = await readJsonOrThrow(r);" in dashboard_js
    assert "currentGame = created.game.game_id;" in dashboard_js

def test_start_without_selected_game_shows_status_message(dashboard_js):
    assert "请先创建或选择一局游戏" in dashboard_js

def test_dashboard_has_growth_oriented_werewolf_table_design(dashboard_html):
    assert "game-shell" in dashboard_html
    assert "table-board" in dashboard_html
    assert "recruit-panel" in dashboard_html
    assert "智能体广场" in dashboard_html
    assert "邀请好友" in dashboard_html
    assert "公开观战" in dashboard_html
    assert "房间分享" in dashboard_html
def test_dashboard_references_split_static_assets(dashboard_html):
    assert "dashboard.css" in dashboard_html
    assert "dashboard.js" in dashboard_html

def test_dashboard_has_launch_wizard_customization_controls(dashboard_html):
    assert "开局向导" in dashboard_html or "å¼€å±€å‘å¯¼" in dashboard_html
    assert "下载规则模板" in dashboard_html or "ä¸‹è½½è§„åˆ™æ¨¡æ¿" in dashboard_html
    assert "上传规则" in dashboard_html or "ä¸Šä¼ è§„åˆ™" in dashboard_html
    assert "下载玩家模板" in dashboard_html or "ä¸‹è½½çŽ©å®¶æ¨¡æ¿" in dashboard_html
    assert "上传玩家配置" in dashboard_html or "ä¸Šä¼ çŽ©å®¶é…ç½®" in dashboard_html
    assert "规则校验结果" in dashboard_html or "è§„åˆ™æ ¡éªŒç»“æžœ" in dashboard_html
    assert "人格预览" in dashboard_html or "äººæ ¼é¢„è§ˆ" in dashboard_html
    assert "公开观战" in dashboard_html or "å…¬å¼€è§‚æˆ˜" in dashboard_html
    assert "我参与一席" in dashboard_html or "æˆ‘å‚ä¸Žä¸€å¸­" in dashboard_html

def test_dashboard_upload_js_keeps_validated_config_state(dashboard_js):
    assert "let selectedRulesetConfig" in dashboard_js
    assert "let selectedPersonaPackConfig" in dashboard_js
    assert "selectedRulesetConfig = data.normalized" in dashboard_js
    assert "selectedPersonaPackConfig = data.normalized" in dashboard_js
    assert "getSelectedRulesetId" in dashboard_js

def test_create_game_uses_selected_ruleset(dashboard_js):
    assert "buildCreateGamePayload()" in dashboard_js
    assert "ruleset_id: getSelectedRulesetId()" in dashboard_js

def test_dashboard_has_static_marketplace_cards(dashboard_html):
    assert "规则市场" in dashboard_html or "è§„åˆ™å¸‚åœº" in dashboard_html
    assert "经典 12 人狼王守卫" in dashboard_html or "ç»å…¸ 12 äººç‹¼çŽ‹å®ˆå«" in dashboard_html
    assert "display_only" in dashboard_html
    assert "人格市场" in dashboard_html or "äººæ ¼å¸‚åœº" in dashboard_html

def test_marketplace_cards_wire_to_selector_helpers(dashboard_html, dashboard_js):
    assert "selectMarketplaceRuleset" in dashboard_html
    assert "selectMarketplacePersonaPack" in dashboard_html
    assert "function selectMarketplaceRuleset" in dashboard_js
    assert "function selectMarketplacePersonaPack" in dashboard_js
    assert "rulesetSelector" in dashboard_js
    assert "personaPackSelector" in dashboard_js

def test_dashboard_has_share_summary_button(dashboard_html, dashboard_js):
    assert "生成复盘分享" in dashboard_html or "ç”Ÿæˆå¤ç›˜åˆ†äº«" in dashboard_html
    assert "generateShareSummary" in dashboard_js
    assert "/share-summary" in dashboard_js

def test_dashboard_has_human_seat_planning_controls(dashboard_html, dashboard_js):
    assert "humanSeatSelector" in dashboard_html
    assert "human_seat" in dashboard_js
    assert "experience_mode" in dashboard_js


def test_vote_display_tally_supports_v1_v2_and_unknown_legacy_base(
    dashboard_js,
) -> None:
    match = re.search(
        r"function voteDisplayTally\(data, rulesetBaseVoteWeight\) \{.*?^\}",
        dashboard_js,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None
    cases = [
        [
            {
                "weighted_tally": {"p03": 3},
                "vote_weights": {"p01": 3},
            },
            2,
        ],
        [
            {
                "vote_weight_format_version": 2,
                "weighted_tally": {"p03": 9},
                "weighted_tally_units": {"p03": 3},
                "weighted_tally_display": {"p03": 1.5},
            },
            None,
        ],
        [
            {
                "weighted_tally": {"p03": 3},
                "vote_weights": {"p01": 3},
            },
            None,
        ],
    ]
    script = (
        f"{match.group(0)}\n"
        f"const cases = {json.dumps(cases)};\n"
        "console.log(JSON.stringify(cases.map(([data, base]) => "
        "voteDisplayTally(data, base))));"
    )

    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        {"p03": 1.5},
        {"p03": 1.5},
        None,
    ]


def test_dashboard_labels_unknown_v1_vote_units_as_unsupported(
    dashboard_js,
) -> None:
    assert "不支持的旧版票权" in dashboard_js
    assert "voteDisplayTally(d, data.ruleset_base_vote_weight)" in dashboard_js


def _run_vote_renderer(dashboard_js: str, vote_data_expression: str) -> dict:
    """在最小可执行 DOM 中调用真实 renderVotes 并返回节点快照。"""
    start = dashboard_js.index("function voteDisplayTally")
    end = dashboard_js.index("// -- Moderator Data --", start)
    functions = dashboard_js[start:end]
    script = f"""
class FakeElement {{
  constructor(tagName) {{
    this.tagName = tagName;
    this.className = '';
    this.children = [];
    this.innerHTML = '';
    this._textContent = '';
    this.style = {{}};
  }}
  set textContent(value) {{
    this._textContent = String(value);
    this.children = [];
  }}
  get textContent() {{
    return this._textContent + this.children.map(child => child.textContent).join('');
  }}
  append(...children) {{
    this.children.push(...children);
  }}
  replaceChildren(...children) {{
    this.innerHTML = '';
    this._textContent = '';
    this.children = [...children];
  }}
}}
const votePanel = new FakeElement('div');
const document = {{
  createElement: tagName => new FakeElement(tagName),
  getElementById: id => {{
    if (id !== 'votePanel') throw new Error(`unexpected element: ${{id}}`);
    return votePanel;
  }},
}};
{functions}
renderVotes({{
  events: [{{
    event_type: 'vote_resolved',
    data: {vote_data_expression},
  }}],
}});
function snapshot(node) {{
  return {{
    tagName: node.tagName,
    className: node.className,
    textContent: node._textContent,
    children: node.children.map(snapshot),
  }};
}}
console.log(JSON.stringify({{
  innerHTML: votePanel.innerHTML,
  textContent: votePanel.textContent,
  children: votePanel.children.map(snapshot),
}}));
"""
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_vote_rows_render_malicious_player_id_as_text(dashboard_js) -> None:
    malicious_id = '<img src=x onerror="globalThis.pwned=true">'
    rendered = _run_vote_renderer(
        dashboard_js,
        json.dumps({
            "vote_weight_format_version": 2,
            "weighted_tally_display": {malicious_id: 1.5},
        }),
    )

    assert rendered["innerHTML"] == ""
    assert rendered["children"] == [{
        "tagName": "div",
        "className": "vote-row",
        "textContent": "",
        "children": [
            {
                "tagName": "span",
                "className": "",
                "textContent": malicious_id,
                "children": [],
            },
            {
                "tagName": "span",
                "className": "vote-count",
                "textContent": "1.5票",
                "children": [],
            },
        ],
    }]


@pytest.mark.parametrize(
    "value_expression",
    ["Infinity", "-1", "{amount: 1.5}"],
    ids=["non-finite", "negative", "object"],
)
def test_vote_rows_fail_closed_for_unsafe_display_values(
    dashboard_js,
    value_expression: str,
) -> None:
    rendered = _run_vote_renderer(
        dashboard_js,
        (
            "{vote_weight_format_version: 2, "
            f"weighted_tally_display: {{p03: {value_expression}}}}}"
        ),
    )

    assert rendered["innerHTML"] == ""
    assert rendered["textContent"] == "不支持的投票载荷"
