"""Tests for --prompt_templates: overriding the agent prompt templates from a
JSON file. Run from the repo root:

    python test_prompt_overrides.py

No LLM calls.
"""

import json
import tempfile
from pathlib import Path

from core.agents import llm_agent, prompt_templates
from core.experiment.config import build_config

STOCK_SYS = prompt_templates.SYS_PROMPT
STOCK_AGENT = prompt_templates.AGENT_PROMPT


def test_config_field():
    cfg = build_config({"prompt_templates": "x.json"})
    assert cfg.agent.prompt_templates == "x.json"
    assert build_config({}).agent.prompt_templates is None
    print("PASS: config carries prompt_templates")


def test_override_and_agent_pickup():
    sys_src = (
        "You are {{ agent_name }}, a villager of the fever valley. "
        "{% if persona %}{{ persona }}{% endif %} "
        "Obs: {{ short_obs_descr }} / {{ detailed_obs_descr }} ({{ obs_style }}). "
        "{% if use_internal_memory %}Memory {{ internal_memory_size }} tok.{% endif %} "
        "{% if use_inventory %}Inventory on.{% endif %} "
        "{% if artifact_creation %}Artifacts on.{% endif %} "
        "{% if food_mechanism %}Food on.{% endif %} {{ exogenous_motivation }}"
    )
    agent_src = "VALLEY {{ history }} {{ observation }} {{ actions }} {{ action_keys }}"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "overrides.json"
        path.write_text(json.dumps({"sys_prompt": sys_src, "agent_prompt": agent_src}))
        try:
            prompt_templates.load_prompt_overrides(path)
            assert prompt_templates.SYS_PROMPT is not STOCK_SYS
            assert prompt_templates.AGENT_PROMPT is not STOCK_AGENT

            # llm_agent must read through the module, not a frozen import
            assert not hasattr(llm_agent, "SYS_PROMPT")
            assert not hasattr(llm_agent, "AGENT_PROMPT")

            agent = llm_agent.LLMAgent(
                agent_name="Tester",
                agent_tag="being0",
                log_dir=tmp,
                persona="You are cautious.",
            )
            assert "villager of the fever valley" in agent.system_prompt
            assert "You are cautious." in agent.system_prompt
            print("PASS: override reaches a fresh LLMAgent's system prompt")

            rendered = prompt_templates.AGENT_PROMPT.render(
                history="H", observation="O", actions="{}", action_keys="stay"
            )
            assert rendered.startswith("VALLEY")
            print("PASS: per-step template overridden")
        finally:
            prompt_templates.SYS_PROMPT = STOCK_SYS
            prompt_templates.AGENT_PROMPT = STOCK_AGENT


def test_partial_and_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "partial.json"
        path.write_text(json.dumps({"sys_prompt": "only sys {{ agent_name }}"}))
        try:
            prompt_templates.load_prompt_overrides(path)
            assert prompt_templates.AGENT_PROMPT is STOCK_AGENT, "agent untouched"
        finally:
            prompt_templates.SYS_PROMPT = STOCK_SYS
            prompt_templates.AGENT_PROMPT = STOCK_AGENT

        bad = Path(tmp) / "bad.json"
        bad.write_text(json.dumps({"system": "wrong key"}))
        try:
            prompt_templates.load_prompt_overrides(bad)
            raise AssertionError("unknown key accepted")
        except ValueError:
            pass
    print("PASS: partial overrides and unknown keys")


if __name__ == "__main__":
    test_config_field()
    test_override_and_agent_pickup()
    test_partial_and_invalid()
    print("ALL PROMPT OVERRIDE TESTS PASSED")
