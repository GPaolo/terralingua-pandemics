"""Sanity checks for the PPE (personal protective equipment) artifact.

Run from the repo root with: python test_ppe_artifacts.py
No LLM calls, no API credits.
"""

import tempfile
from pathlib import Path

import numpy as np

from core.environment.artifact import PPEArtifact, ViralArtifact
from core.environment.env import OpenGridWorld


def make_env(tmp, **kwargs):
    kwargs.setdefault("viral_incubation_min", 0)
    kwargs.setdefault("viral_incubation_max", 0)
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
        **kwargs,
    )
    env.rng = np.random.default_rng(0)
    return env


def get_viral(env, agent_tag):
    return [
        art_name
        for art_name in env.agent_inventories[agent_tag]
        if isinstance(env.artifacts.get(art_name), ViralArtifact)
    ]


def test_seeding():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        init_artifacts=[
            {"name": "mask_b", "type": "ppe", "agent": "b"},
            {"name": "mask_map", "type": "ppe", "pose": [3, 3]},
        ],
    )
    for tag in ("a", "b"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 6)})

    assert "mask_b" in env.agent_inventories["b"]
    assert isinstance(env.artifacts["mask_b"], PPEArtifact)
    assert "mask_map" in env.artifacts_map[(3, 3)]
    assert env.artifacts["mask_b"].infection_protection == env.ppe_protection
    assert env.artifacts["mask_b"].payload, "PPE should carry a description"
    print("PASS: PPE seeds into an inventory and onto the map via init_artifacts")


def test_protection():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_infection_radius=1,
        viral_infection_probability=1.0,
        viral_lifespan=-1,
        ppe_protection=0.0,
        init_artifacts=[{"name": "mask_b", "type": "ppe", "agent": "b"}],
    )
    for tag in ("a", "b", "c"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 6), "c": (5, 4)})

    env.infect_agent(agent_tag="a")
    for _ in range(5):
        env.step({})
    assert get_viral(env, "c"), "unprotected neighbour should be infected"
    assert not get_viral(env, "b"), "PPE holder should never be infected at 0.0"
    print("PASS: PPE scales contraction probability (0.0 -> immune, bare -> infected)")


def test_protection_does_not_stack():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        ppe_protection=0.2,
        init_artifacts=[
            {"name": f"mask_{i}", "type": "ppe", "agent": "a"} for i in range(5)
        ],
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})

    assert len(env.agent_inventories["a"]) == 5
    assert env._infection_protection("a") == 0.2, "five PPEs must not stack"
    assert env._infection_protection("a") != 0.2**5
    print("PASS: protection is min over the inventory, not a product")


def test_carrier_perceives_ppe():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        init_artifacts=[{"name": "mask", "type": "ppe", "agent": "a"}],
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})

    obs, _, _, _, infos = env.step(
        {"a": {"action": "move", "params": {"direction": "stay"}}}
    )
    effects = infos["a"].get("Passive interaction result - Artifacts in inventory")
    assert effects and any("mask" in e for e in effects), effects
    assert any("protective" in e.lower() for e in effects), (
        "the carrier should be told what PPE does"
    )
    assert any("A(ppe): mask" in item for item in obs["a"]["inventory"]), obs["a"][
        "inventory"
    ]
    print("PASS: a carrier sees the PPE and its description in its prompt")


def test_world_log_emits_n_ppe():
    import json

    tmp = Path(tempfile.mkdtemp())
    env = OpenGridWorld(
        grid_size=20,
        init_food=1,
        food_mechanism=False,
        use_inventory=True,
        log_path=tmp,
        verbose=0,
        init_artifacts=[{"name": "mask", "type": "ppe", "agent": "a"}],
    )
    env.rng = np.random.default_rng(0)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})

    lines = [json.loads(line) for line in open(tmp / "world_state.jsonl")]
    meta, frame = lines[0], lines[1]
    assert "n_ppe" in meta["agent_fields"], meta["agent_fields"]
    idx = meta["agent_fields"].index("n_ppe")
    assert frame["agents"]["a"][idx] == 1, frame["agents"]["a"]
    print("PASS: world_state.jsonl carries n_ppe (schema 3)")


def test_seeding_by_role():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        init_artifacts=[{"name": "mask", "type": "ppe", "role": "health_worker"}],
    )
    env.add_agent(agent_tag="a", agent_name="Miriam", agent_type="text",
                  agent_role="health_worker")
    env.add_agent(agent_tag="b", agent_name="Sam", agent_type="text",
                  agent_role="health_worker")
    env.add_agent(agent_tag="c", agent_name="Eve", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 8), "c": (5, 11)})

    assert env._count_ppe("a") == 1 and env._count_ppe("b") == 1
    assert env._count_ppe("c") == 0, "roleless beings get nothing"
    print("PASS: role-targeted seeding reaches every being with the role")


def test_agents_cannot_create_ppe():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})

    status = env.add_artifact(
        pose=(5, 5),
        art_type="ppe",
        art_name="diy_mask",
        payload="",
        creator="a",
        lifespan=-1,
    )
    assert "not a valid type" in status, status
    assert "diy_mask" not in env.artifacts
    print("PASS: agents cannot create PPE (environment-seeded only)")


def test_take_affordance_and_death_drop():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        init_artifacts=[{"name": "mask", "type": "ppe", "pose": [5, 5]}],
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})

    actions = env._get_avail_actions("a")
    assert "pickup_artifact" in actions, "PPE on the agent's cell should be pickupable"

    env.agent_inventories["a"].add("mask")
    env.artifacts_map[(5, 5)].discard("mask")
    env.agent_energy["a"] = 0
    env._kill("a")
    assert "mask" in env.artifacts_map[(5, 5)], "PPE should drop where its host died"
    assert env.artifacts["mask"].remaining_time == np.inf
    print("PASS: PPE is takeable and drops on the map at its host's death")


def test_checkpoint_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        ppe_protection=0.25,
        init_artifacts=[{"name": "mask", "type": "ppe", "agent": "a"}],
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})

    env2 = make_env(Path(tempfile.mkdtemp()))
    env2.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env2.set_state_ckpt(env.get_state_ckpt())
    restored = env2.artifacts["mask"]
    assert isinstance(restored, PPEArtifact)
    assert restored.infection_protection == 0.25
    print("PASS: PPE protection survives a checkpoint roundtrip")


if __name__ == "__main__":
    test_seeding()
    test_protection()
    test_protection_does_not_stack()
    test_carrier_perceives_ppe()
    test_world_log_emits_n_ppe()
    test_seeding_by_role()
    test_agents_cannot_create_ppe()
    test_take_affordance_and_death_drop()
    test_checkpoint_roundtrip()
    print("\nAll PPE artifact checks passed ✅")
