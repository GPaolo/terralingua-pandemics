"""Sanity checks for the burial mechanic.

Run from the repo root with: python tests/test_burials.py
No LLM calls, no API credits.
"""

import tempfile
from pathlib import Path

import numpy as np

from core.environment.artifact import ViralArtifact
from core.environment.env import OpenGridWorld


def make_env(tmp, **kwargs):
    kwargs.setdefault("burials", True)
    kwargs.setdefault("viral_incubation_min", 0)
    kwargs.setdefault("viral_incubation_max", 0)
    kwargs.setdefault("viral_lifespan", -1)
    kwargs.setdefault("viral_dropped_lifespan", -1)
    kwargs.setdefault("viral_infection_probability", 0.0)
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
    return env


def drop_remains(env, pos):
    """A corpse's viral artifact, placed on the map like _kill does."""
    art = ViralArtifact(
        name="virus_corpse",
        lifespan=np.inf,
        pose=pos,
        creator="environment",
        creation_time=env.step_count,
    )
    env.artifacts[art.name] = art
    env.artifacts_map[pos].add(art.name)
    return art.name


def bury(env, tag, name):
    # Remains were planted after reset; refresh the cached available actions
    # the way the runner does between steps.
    env.agent_avail_actions[tag] = env._get_avail_actions(tag)
    return env.step({tag: {"action": "bury", "params": {"name": name}}})


def test_action_gating():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp)
    for tag in ("near", "far"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"near": (5, 6), "far": (15, 15)})
    drop_remains(env, (5, 5))

    assert "bury" in env._get_avail_actions("near")
    assert "bury" not in env._get_avail_actions("far")

    env.infect_agent(agent_tag="near")  # incubation 0 -> sick at once
    assert "bury" not in env._get_avail_actions("near"), "the sick cannot bury"
    print("PASS: bury offered only near remains, and only to the well")


def test_flag_off():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, burials=False)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 6)})
    drop_remains(env, (5, 5))
    assert "bury" not in env._get_avail_actions("a")
    print("PASS: without --burials the action never appears")


def test_burial_removes_remains():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 6)})
    name = drop_remains(env, (5, 5))

    _, _, _, _, infos = bury(env, "a", name)
    assert name not in env.artifacts
    assert name not in env.artifacts_map[(5, 5)]
    assert not env.agent_inventories["a"], "risk 0 must not infect the burier"
    assert any("You buried" in str(v) for v in infos["a"].values()), infos["a"]
    print("PASS: burying removes the remains from the world")


def test_burial_risk_and_multiplier():
    # probability 0.5 x multiplier 2 -> certain infection for a bare burier
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp, viral_infection_probability=0.5, burial_infection_multiplier=2.0
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 6)})
    name = drop_remains(env, (5, 5))
    _, _, _, _, infos = bury(env, "a", name)
    assert name not in env.artifacts, "remains are buried even when it infects"
    assert env._count_viral("a") == 1, "0.5 x 2.0 = certain infection"
    assert "Infection" in infos["a"], infos["a"]
    print("PASS: burial risk = probability x multiplier, and it still buries")

    # multiplier 0 -> burying is safe
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp, viral_infection_probability=1.0, burial_infection_multiplier=0.0
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 6)})
    bury(env, "a", drop_remains(env, (5, 5)))
    assert env._count_viral("a") == 0
    print("PASS: multiplier 0 makes burial safe")


def test_ppe_protects_the_burier():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_infection_probability=1.0,
        burial_infection_multiplier=2.0,
        ppe_protection=0.0,
        init_artifacts=[{"name": "mask", "type": "ppe", "agent": "a"}],
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 6)})
    bury(env, "a", drop_remains(env, (5, 5)))
    assert env._count_viral("a") == 0, "full PPE protection must cover burials"
    print("PASS: PPE protects the burier")


def test_incubating_catch_stays_silent():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_infection_probability=0.5,
        burial_infection_multiplier=2.0,
        viral_incubation_min=5,
        viral_incubation_max=5,
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 6)})
    _, _, _, _, infos = bury(env, "a", drop_remains(env, (5, 5)))
    assert env._count_viral("a") == 1
    assert "Infection" not in infos["a"], (
        "an infection caught while burying must stay silent through incubation"
    )
    print("PASS: an infection caught at a burial incubates silently")


def test_bogus_target():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 6)})
    drop_remains(env, (5, 5))
    _, _, _, _, infos = bury(env, "a", "no_such_thing")
    assert "virus_corpse" in env.artifacts, "wrong name must bury nothing"
    assert any("no remains" in str(v) for v in infos["a"].values()), infos["a"]
    print("PASS: naming the wrong remains buries nothing")


if __name__ == "__main__":
    test_action_gating()
    test_flag_off()
    test_burial_removes_remains()
    test_burial_risk_and_multiplier()
    test_ppe_protects_the_burier()
    test_incubating_catch_stays_silent()
    test_bogus_target()
    print("\nAll burial checks passed ✅")
