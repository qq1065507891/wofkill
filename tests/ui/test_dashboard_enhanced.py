"""Tests for enhanced observer dashboard features."""
import pytest
from pathlib import Path

DASHBOARD_PATH = Path(__file__).parent.parent.parent / "werewolf_agent" / "ui" / "static" / "dashboard.html"

@pytest.fixture
def dashboard_html():
    return DASHBOARD_PATH.read_text(encoding="utf-8")

def test_cognitive_diff_section(dashboard_html):
    assert "cognitive-diff" in dashboard_html or "identity-prob" in dashboard_html

def test_rag_hit_panel(dashboard_html):
    assert "rag-hit" in dashboard_html or "rag-audit" in dashboard_html

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
