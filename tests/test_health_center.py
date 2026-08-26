"""Sanity checks for the health-center artifact.

Run from the repo root with: python tests/test_health_center.py
No LLM calls, no API credits.
"""

import tempfile
from pathlib import Path

import numpy as np

from core.environment.artifact import HealthCenterArtifact, ViralArtifact
from core.environment.env import OpenGridWorld

CENTER = {
    "name": "health_center",
    "type": "health_center",
    "pose": [5, 5],
    "heal_probability": 1.0,
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


def test_always_active():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, init_artifacts=[CENTER])
    env.add_agent(agent_tag="s", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"s": (5, 4)})  # inside the 3x3, all alone
    env.infect_agent(agent_tag="s")
    _, _, _, _, infos = step_all(env)
    assert not infected(env, "s"), "the center needs no attendant"
    assert "treated at health_center" in infos["s"].get("Health", ""), infos["s"]
    assert env.agent_recoveries["s"] == 1
    print("PASS: the center works with nobody staffing it")


def test_area_is_nine_cells():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, init_artifacts=[CENTER], viral_infection_probability=0.0)
    for tag in ("edge", "out"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"edge": (6, 6), "out": (5, 7)})
    env.infect_agent(agent_tag="edge")
    env.infect_agent(agent_tag="out")
    step_all(env)
    assert not infected(env, "edge"), "a diagonal neighbour is inside the area"
    assert infected(env, "out"), "two cells away is outside the area"
    print("PASS: the area is exactly the center's cell plus the 8 around it")


def test_incubating_cure_stays_silent():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp, init_artifacts=[CENTER], viral_incubation_min=10, viral_incubation_max=10
    )
    env.add_agent(agent_tag="s", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"s": (5, 5)})
    env.infect_agent(agent_tag="s")
    _, _, _, _, infos = step_all(env)
    assert not infected(env, "s")
    assert "Health" not in infos.get("s", {}), (
        "curing a silent carrier must not reveal it was infected"
    )
    print("PASS: curing an incubating infection stays silent")


def test_fixed_and_unmovable():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, init_artifacts=[CENTER])
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})

    center = env.artifacts["health_center"]
    assert isinstance(center, HealthCenterArtifact)
    assert not center.interactable
    assert "health center" in center.payload.lower(), "payload should describe it"
    actions = env._get_avail_actions("a")
    assert "pickup_artifact" not in actions, "a health center must not be pickupable"

    status = env.add_artifact(
        pose=None, art_type="health_center", art_name="carried", payload="",
        creator="environment", lifespan=-1, to_inventory="a",
    )
    assert status.startswith("Failed"), status

    try:
        make_env(Path(tempfile.mkdtemp()),
                 init_artifacts=[{**CENTER, "pose": None, "agent": "a"}])
        raise AssertionError("a health center in an inventory should be rejected")
    except ValueError:
        pass
    print("PASS: health centers are fixed to the map and unmovable")


def test_everyone_in_radius_reads_the_payload():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, init_artifacts=[CENTER], viral_infection_probability=0.0)
    for tag in ("near", "far"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"near": (6, 6), "far": (5, 9)})
    _, _, _, _, infos = step_all(env)
    nearby = infos["near"].get("Nearby facilities", [])
    assert any("health center" in s.lower() for s in nearby), infos["near"]
    assert "Nearby facilities" not in infos.get("far", {}), infos["far"]
    print("PASS: every being in the radius is told what the place is")


def test_radius_is_configurable():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        init_artifacts=[{**CENTER, "radius": 2}],
        viral_infection_probability=0.0,
    )
    env.add_agent(agent_tag="s", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"s": (5, 7)})  # distance 2: healed only at radius 2
    env.infect_agent(agent_tag="s")
    step_all(env)
    assert not infected(env, "s"), "radius 2 must reach two cells out"
    print("PASS: the healing radius is configurable (default 1 = 9 cells)")


def test_checkpoint_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp, init_artifacts=[{**CENTER, "heal_probability": 0.4, "radius": 3}]
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})

    env2 = make_env(Path(tempfile.mkdtemp()))
    env2.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env2.set_state_ckpt(env.get_state_ckpt())
    restored = env2.artifacts["health_center"]
    assert isinstance(restored, HealthCenterArtifact)
    assert restored.heal_probability == 0.4
    assert restored.radius == 3
    print("PASS: health center survives a checkpoint roundtrip")


if __name__ == "__main__":
    test_always_active()
    test_area_is_nine_cells()
    test_incubating_cure_stays_silent()
    test_fixed_and_unmovable()
    test_everyone_in_radius_reads_the_payload()
    test_radius_is_configurable()
    test_checkpoint_roundtrip()
    print("\nAll health center checks passed ✅")
