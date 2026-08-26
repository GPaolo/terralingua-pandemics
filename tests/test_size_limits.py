"""Checks for the configurable message and text-artifact size limits.

Run from the repo root with: python tests/test_size_limits.py
No LLM calls, no API credits.
"""

import tempfile
from pathlib import Path

import numpy as np
import tiktoken

from core.environment.artifact import TextArtifact
from core.environment.env import OpenGridWorld

ENC = tiktoken.get_encoding("cl100k_base")


def make_env(tmp, **kwargs):
    env = OpenGridWorld(
        grid_size=20,
        init_food=1,
        food_mechanism=False,
        use_inventory=True,
        log_path=tmp,
        verbose=0,
        **kwargs,
    )
    env.rng = np.random.default_rng(0)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})
    return env


def test_artifact_limit_configurable():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, max_text_artifact_size=10)
    long_payload = " ".join(["word"] * 50)
    status = env.add_artifact(
        pose=(5, 5), art_type="text", art_name="big", payload=long_payload,
        creator="a", lifespan=-1,
    )
    assert "exceeds maximum token limit of 10" in status, status
    status = env.add_artifact(
        pose=(5, 5), art_type="text", art_name="small", payload="short note",
        creator="a", lifespan=-1,
    )
    assert status.startswith("Created"), status

    actions = env._get_avail_actions("a")
    assert "Maximum size is 10 tokens" in actions["create_artifact"]["params"]["payload"]
    print("PASS: text artifact limit is configurable and shown in the prompt")


def test_message_cap():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, max_message_size=5)
    long_msg = " ".join(["hello"] * 40)
    _, _, _, _, infos = env.step(
        {"a": {"action": "move", "params": {"direction": "stay"}, "message": long_msg}}
    )
    sent = env.msg_raw["a"]
    assert len(ENC.encode(sent)) <= 5, sent
    outcomes = str(infos["a"])
    assert "cut off at 5 tokens" in outcomes, infos["a"]
    print("PASS: messages are cut at max_message_size and the agent is told")


def test_message_unlimited_by_default():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp)
    long_msg = " ".join(["hello"] * 40)
    env.step(
        {"a": {"action": "move", "params": {"direction": "stay"}, "message": long_msg}}
    )
    assert env.msg_raw["a"] == long_msg
    print("PASS: default (-1) leaves messages untouched")


def test_oversized_artifact_survives_checkpoint():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, max_text_artifact_size=2000)
    payload = " ".join(["word"] * 800)  # > default 500 tokens
    status = env.add_artifact(
        pose=(5, 5), art_type="text", art_name="tome", payload=payload,
        creator="a", lifespan=-1,
    )
    assert status.startswith("Created"), status

    env2 = make_env(Path(tempfile.mkdtemp()))
    env2.set_state_ckpt(env.get_state_ckpt())
    restored = env2.artifacts["tome"]
    assert restored.payload == payload
    assert restored.max_size == 2000
    print("PASS: artifact above the default limit survives a checkpoint roundtrip")


if __name__ == "__main__":
    test_artifact_limit_configurable()
    test_message_cap()
    test_message_unlimited_by_default()
    test_oversized_artifact_survives_checkpoint()
    print("\nAll size limit checks passed ✅")
