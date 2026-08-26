"""Sanity checks for environment-seeded (init) artifacts.

Run from the repo root with: python test_init_artifacts.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from core.environment.artifact import TextArtifact
from core.environment.env import OpenGridWorld


def make_env(tmp, init_artifacts=None):
    env = OpenGridWorld(
        grid_size=20,
        vision_radius=3,
        init_agent_energy=50,
        lifespan=200,
        init_food=1,
        food_mechanism=False,
        use_inventory=True,
        log_path=tmp,
        verbose=0,
        init_artifacts=init_artifacts,
    )
    env.rng = np.random.default_rng(0)
    return env


def test_seeding_at_start_and_mid_run():
    tmp = Path(tempfile.mkdtemp())
    entries = [
        {"name": "stone", "payload": "hello", "pose": [3, 3]},
        {"name": "sign", "payload": "later", "pose": [7, 7], "lifespan": 2, "step": 2},
        {"name": "wanderer", "payload": "anywhere", "step": 1},
    ]
    env = make_env(tmp, init_artifacts=entries)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})
    env.agent_energy["a"] = 1000.0

    # step-0 artifact is on the map before the first step, visible in obs
    assert "stone" in env.artifacts
    assert "stone" in env.artifacts_map[(3, 3)]
    assert isinstance(env.artifacts["stone"], TextArtifact)
    assert env.artifacts["stone"].creator == "environment"
    assert env.artifacts["stone"].remaining_time == np.inf
    assert "sign" not in env.artifacts and "wanderer" not in env.artifacts

    env.step({})  # step_count 0 -> 1
    assert "wanderer" not in env.artifacts  # seeds during the step with count 1
    env.step({})  # runs with step_count 1: wanderer appears
    assert "wanderer" in env.artifacts
    wanderer_cells = [p for p, arts in env.artifacts_map.items() if "wanderer" in arts]
    assert len(wanderer_cells) == 1

    env.step({})  # runs with step_count 2: sign appears with lifespan 2
    assert "sign" in env.artifacts
    assert env.artifacts["sign"].remaining_time == 2
    env.step({})
    env.step({})
    assert "sign" not in env.artifacts, "sign should have expired after 2 steps"
    assert any(a.name == "sign" for a in env.expired_artifacts)
    # permanent artifacts are still around
    assert "stone" in env.artifacts and "wanderer" in env.artifacts

    # agents can interact with seeded artifacts like any text artifact
    env._update_agent_pos(agent="a", new_pos=(3, 3))
    avail = env._get_avail_actions("a")
    assert "pickup_artifact" in avail
    assert "modify_artifact_stone" in avail
    print("PASS: seeding at start, mid-run, lifespan expiry, interactability")


def test_json_file_and_name_collision():
    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "arts.json"
    with open(cfg, "w") as f:
        json.dump([{"name": "twin", "pose": [1, 1]}, {"name": "twin", "pose": [2, 2]}], f)
    env = make_env(tmp, init_artifacts=str(cfg))
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})
    assert "twin" in env.artifacts and "twin_1" in env.artifacts
    print("PASS: loading from JSON file and name collision handling")


def test_checkpoint_no_double_seeding():
    tmp = Path(tempfile.mkdtemp())
    entries = [{"name": "early", "pose": [2, 2]}, {"name": "late", "pose": [9, 9], "step": 4}]
    env = make_env(tmp, init_artifacts=entries)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})
    env.agent_energy["a"] = 1000.0
    env.step({})
    ckpt = env.get_state_ckpt()

    # Restore into a fresh env configured with the same entries
    env2 = make_env(Path(tempfile.mkdtemp()), init_artifacts=entries)
    env2.set_state_ckpt(ckpt)
    assert "early" in env2.artifacts and "early_1" not in env2.artifacts
    assert [e["name"] for e in env2._pending_init_artifacts] == ["late"]
    for _ in range(4):
        env2.step({})
        env2.agent_energy["a"] = 1000.0
    assert "late" in env2.artifacts and "late_1" not in env2.artifacts
    print("PASS: checkpoint keeps pending seeds without double-seeding")


def test_inventory_seeding():
    tmp = Path(tempfile.mkdtemp())
    entries = [
        {"name": "gift", "payload": "for a", "agent": "a"},
        {"name": "late_gift", "payload": "for b", "agent": "bee", "step": 2},
        {"name": "ghost_gift", "payload": "for nobody", "agent": "zz", "step": 1},
    ]
    env = make_env(tmp, init_artifacts=entries)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.add_agent(agent_tag="b", agent_name="bee", agent_type="text")  # name != tag
    env.restart_env(agent_poses={"a": (5, 5), "b": (10, 10)})
    for tag in env.agent_registry:
        env.agent_energy[tag] = 1000.0

    # step-0 inventory seed present before the first step, not on the map
    assert "gift" in env.agent_inventories["a"]
    assert all("gift" not in arts for arts in env.artifacts_map.values())
    assert env.artifacts["gift"].creator == "environment"
    obs = env._build_obs("a")
    assert any("gift" in item for item in obs["inventory"])

    env.step({})  # step_count 0 -> 1; ghost_gift targets a missing agent: warns, dropped
    env.step({})  # step_count 1 -> 2
    env.step({})  # runs with step_count 2: late_gift lands in b's inventory (by name)
    assert "late_gift" in env.agent_inventories["b"]
    assert "ghost_gift" not in env.artifacts
    assert env._pending_init_artifacts == []

    # seeded inventory artifacts are normal text artifacts: droppable/giveable
    avail = env._get_avail_actions("a")
    assert "drop_artifact" in avail and "modify_artifact_gift" in avail
    print("PASS: inventory seeding by tag and name, missing-agent warning")


def test_validation_errors():
    tmp = Path(tempfile.mkdtemp())
    bad_entries = [
        [{"payload": "no name"}],
        [{"name": "x", "type": "viral"}],
        [{"name": "x", "pose": [50, 50]}],
        [{"name": "x", "lifespan": 0}],
        [{"name": "x", "step": -1}],
        [{"name": "x", "pose": [1, 1], "agent": "a"}],
    ]
    for entries in bad_entries:
        try:
            make_env(tmp, init_artifacts=entries)
        except ValueError:
            continue
        raise AssertionError(f"entries {entries} should have been rejected")
    print("PASS: invalid entries are rejected at construction")


def test_environment_creator_direct_call():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})
    status = env.add_artifact(
        pose=(8, 8), art_type="text", art_name="manual", payload="by hand",
        creator="environment", lifespan=np.inf,
    )
    assert status.startswith("Created"), status
    assert "manual" in env.artifacts_map[(8, 8)]
    print("PASS: add_artifact accepts a non-agent creator")


if __name__ == "__main__":
    test_seeding_at_start_and_mid_run()
    test_json_file_and_name_collision()
    test_checkpoint_no_double_seeding()
    test_inventory_seeding()
    test_validation_errors()
    test_environment_creator_direct_call()
    print("\nAll init artifact checks passed ✅")
