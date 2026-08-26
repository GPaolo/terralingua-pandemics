"""Tests for persona support: file loading, prompt injection, checkpointing."""

import json
import tempfile
from pathlib import Path

from core.agents.llm_agent import LLMAgent
from core.experiment.runner import load_personas


def test_load_personas():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "personas.json"
        path.write_text(
            json.dumps(
                [
                    {"persona": "You are a cautious doctor.", "count": 2},
                    "You are a reckless explorer.",
                    {"persona": "You are a skeptic."},
                ]
            )
        )
        personas = load_personas(str(path))
        assert personas == [
            "You are a cautious doctor.",
            "You are a cautious doctor.",
            "You are a reckless explorer.",
            "You are a skeptic.",
        ], personas
        print("PASS: personas file expands counts in order")

    assert load_personas(None) == []
    assert load_personas("") == []
    print("PASS: no personas file yields empty list")


def test_persona_in_system_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        persona = "You are a cautious doctor who avoids crowds."
        agent = LLMAgent(
            agent_name="doc0",
            agent_tag="doc0",
            log_dir=tmp,
            persona=persona,
        )
        assert persona in agent.system_prompt, agent.system_prompt
        print("PASS: persona is rendered into the system prompt")

        plain = LLMAgent(agent_name="plain0", agent_tag="plain0", log_dir=tmp)
        assert plain.persona == ""
        assert "persona" not in plain.system_prompt.lower()
        assert "{%" not in plain.system_prompt and "{{" not in plain.system_prompt
        print("PASS: default persona is empty and leaves no template residue")


def test_persona_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        persona = "You are a skeptic who distrusts strangers."
        agent = LLMAgent(
            agent_name="skeptic0",
            agent_tag="skeptic0",
            log_dir=tmp,
            persona=persona,
        )
        ckpt = agent.get_state_ckpt()
        assert ckpt["persona"] == persona

        restored = LLMAgent(agent_name="tmp", agent_tag="tmp", log_dir=tmp)
        restored.set_state_ckpt(ckpt)
        assert restored.persona == persona
        assert persona in restored.system_prompt
        print("PASS: persona survives checkpoint save/load")

        # Checkpoints written before personas existed have no "persona" key.
        old_ckpt = {k: v for k, v in ckpt.items() if k != "persona"}
        legacy = LLMAgent(agent_name="tmp2", agent_tag="tmp2", log_dir=tmp)
        legacy.set_state_ckpt(old_ckpt)
        assert legacy.persona == ""
        print("PASS: pre-persona checkpoints load with empty persona")


if __name__ == "__main__":
    test_load_personas()
    test_persona_in_system_prompt()
    test_persona_checkpoint_roundtrip()
    print("All persona tests passed.")
