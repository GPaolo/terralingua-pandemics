"""Sanity checks for the health-spot artifact.

Run from the repo root with: python tests/test_health_spot.py
No LLM calls, no API credits.
"""

import tempfile
from pathlib import Path

import numpy as np

from core.environment.artifact import HealthSpotArtifact, ViralArtifact
from core.environment.env import OpenGridWorld

SPOT = {
    "name": "clinic",
    "type": "health_spot",
    "pose": [5, 5],
    "radius": 2,
    "heal_probability": 1.0,
    "operators": ["Miriam"],
}


def make_env(tmp, init_artifacts=None, **kwargs):
    kwargs.setdefault("viral_incubation_min", 0)
    kwargs.setdefault("viral_incubation_max", 0)
    kwargs.setdefault("viral_lifespan", -1)
    env = OpenGridWorld(
        grid_size=20,
        init_food=1,
        food_mechanism=False,
        use_inventory=True,
        log_path=tmp,
        verbose=0,
        init_artifacts=init_artifacts,
        **kwargs,
    )
    env.rng = np.random.default_rng(0)
    return env


def step_all(env):
    return env.step(
        {t: {"action": "move", "params": {"direction": "stay"}} for t in list(env.agent_registry)}
    )


def infected(env, tag):
    return [
        n
        for n in env.agent_inventories[tag]
        if isinstance(env.artifacts.get(n), ViralArtifact)
    ]


def test_staffed_spot_heals():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, init_artifacts=[SPOT])
    for tag, name in (("w", "Miriam"), ("sick_in", "a"), ("sick_out", "b")):
        env.add_agent(agent_tag=tag, agent_name=name, agent_type="text")
    env.restart_env(agent_poses={"w": (5, 6), "sick_in": (5, 4), "sick_out": (15, 15)})
    env.infect_agent(agent_tag="sick_in")
    env.infect_agent(agent_tag="sick_out")

    _, _, _, _, infos = step_all(env)
    assert not infected(env, "sick_in"), "patient in a staffed radius should be cured"
    assert infected(env, "sick_out"), "patient out of radius should stay infected"
    assert "treated at clinic" in infos["sick_in"].get("Health", ""), infos["sick_in"]
    print("PASS: staffed spot cures a sick patient in radius (and tells it)")


def test_zero_probability_never_heals():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, init_artifacts=[{**SPOT, "heal_probability": 0.0}])
    for tag, name in (("w", "Miriam"), ("s", "a")):
        env.add_agent(agent_tag=tag, agent_name=name, agent_type="text")
    env.restart_env(agent_poses={"w": (5, 6), "s": (5, 4)})
    env.infect_agent(agent_tag="s")
    for _ in range(5):
        step_all(env)
    assert infected(env, "s"), "heal_probability 0 must never cure"
    print("PASS: heal_probability 0 never cures")


def test_unstaffed_or_wrong_operator_does_nothing():
    for worker_name, poses in (
        ("Miriam", {"w": (15, 15), "s": (5, 4)}),  # operator too far
        ("NotMiriam", {"w": (5, 6), "s": (5, 4)}),  # attendant not an operator
    ):
        tmp = Path(tempfile.mkdtemp())
        env = make_env(tmp, init_artifacts=[SPOT])
        for tag, name in (("w", worker_name), ("s", "a")):
            env.add_agent(agent_tag=tag, agent_name=name, agent_type="text")
        env.restart_env(agent_poses=poses)
        env.infect_agent(agent_tag="s")
        step_all(env)
        assert infected(env, "s"), (worker_name, poses)
    print("PASS: an unstaffed spot (absent or non-operator attendant) heals nobody")


def test_incubating_cure_stays_silent():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        init_artifacts=[SPOT],
        viral_incubation_min=10,
        viral_incubation_max=10,
    )
    for tag, name in (("w", "Miriam"), ("s", "a")):
        env.add_agent(agent_tag=tag, agent_name=name, agent_type="text")
    env.restart_env(agent_poses={"w": (5, 6), "s": (5, 4)})
    env.infect_agent(agent_tag="s")
    _, _, _, _, infos = step_all(env)
    assert not infected(env, "s"), "latent infection should be cured too"
    assert "Health" not in infos.get("s", {}), (
        "curing a silent carrier must not reveal it was infected"
    )
    print("PASS: curing an incubating infection stays silent")


def test_fixed_and_unmovable():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, init_artifacts=[SPOT])
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})

    assert isinstance(env.artifacts["clinic"], HealthSpotArtifact)
    assert not env.artifacts["clinic"].interactable
    actions = env._get_avail_actions("a")
    assert "pickup_artifact" not in actions, "health spot must not be pickupable"

    status = env.add_artifact(
        pose=None, art_type="health_spot", art_name="carried", payload="",
        creator="environment", lifespan=-1, to_inventory="a",
    )
    assert status.startswith("Failed"), status

    try:
        make_env(Path(tempfile.mkdtemp()),
                 init_artifacts=[{**SPOT, "pose": None, "agent": "a"}])
        raise AssertionError("health spot in an inventory should be rejected")
    except ValueError:
        pass
    print("PASS: health spots are fixed to the map and unmovable")


def test_checkpoint_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, init_artifacts=[{**SPOT, "heal_probability": 0.4, "radius": 4}])
    env.add_agent(agent_tag="a", agent_name="Miriam", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})

    env2 = make_env(Path(tempfile.mkdtemp()))
    env2.add_agent(agent_tag="a", agent_name="Miriam", agent_type="text")
    env2.set_state_ckpt(env.get_state_ckpt())
    restored = env2.artifacts["clinic"]
    assert isinstance(restored, HealthSpotArtifact)
    assert restored.heal_probability == 0.4 and restored.radius == 4
    assert restored.operators == ["Miriam"]
    print("PASS: health spot survives a checkpoint roundtrip")


if __name__ == "__main__":
    test_staffed_spot_heals()
    test_zero_probability_never_heals()
    test_unstaffed_or_wrong_operator_does_nothing()
    test_incubating_cure_stays_silent()
    test_fixed_and_unmovable()
    test_checkpoint_roundtrip()
    print("\nAll health spot checks passed ✅")
