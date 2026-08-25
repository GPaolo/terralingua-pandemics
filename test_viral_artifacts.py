"""Sanity checks for the viral artifact mechanics.

Run from the repo root with: python test_viral_artifacts.py
"""

import tempfile
from pathlib import Path

import numpy as np

from core.environment.artifact import TextArtifact, ViralArtifact
from core.environment.env import OpenGridWorld


def make_env(tmp, **viral_kwargs):
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
        **viral_kwargs,
    )
    env.rng = np.random.default_rng(0)
    return env


def get_viral(env, agent_tag):
    return [
        art_name
        for art_name in env.agent_inventories[agent_tag]
        if isinstance(env.artifacts.get(art_name), ViralArtifact)
    ]


def test_spread_and_energy():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_infection_radius=2,
        viral_infection_probability=1.0,
        viral_energy_multiplier=3.0,
        viral_lifespan=-1,
        viral_dropped_lifespan=5,
    )
    for tag in ("a", "b", "c"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 7), "c": (15, 15)})
    for tag in env.agent_registry:
        env.agent_energy[tag] = 1000.0

    # Manual seeding
    name = env.infect_agent(agent_tag="a")
    assert name == "virus_i1", name
    assert get_viral(env, "a") == ["virus_i1"]
    assert env.infect_agent(agent_tag="a") is None, "same strain hosted twice"

    # Spread: b is at distance 2 (within radius), c is far away
    e_before = dict(env.agent_energy)
    _, _, _, _, infos = env.step({})
    assert len(get_viral(env, "b")) == 1, "b should be infected"
    assert len(get_viral(env, "c")) == 0, "c should not be infected"
    assert "Infection" in infos["b"]
    b_art = env.artifacts[get_viral(env, "b")[0]]
    assert b_art.strain == "virus"
    assert not b_art.interactable

    # Energy: a pays 3x, b was infected this step so pays 3x too, c pays 1x
    assert e_before["a"] - env.agent_energy["a"] == 3.0
    assert e_before["b"] - env.agent_energy["b"] == 3.0
    assert e_before["c"] - env.agent_energy["c"] == 1.0

    # No ping-pong: another step must not add more copies of the same strain
    env.step({})
    assert len(get_viral(env, "a")) == 1
    assert len(get_viral(env, "b")) == 1

    # Affordances: viral-only inventory offers no drop/give actions
    avail = env._get_avail_actions("a")
    assert "drop_artifact" not in avail
    assert "give_artifact" not in avail
    assert not any("virus" in act for act in avail), avail

    # With a text artifact too, drop/give appear but refuse to act on the virus
    env.viral_infection_probability = 0.0  # freeze the dynamics
    note = TextArtifact(
        name="note", payload="hi", lifespan=np.inf, pose=(5, 5), creator="a",
        creation_time=env.step_count,
    )
    env.artifacts["note"] = note
    env.agent_inventories["a"].add("note")
    avail = env._get_avail_actions("a")
    assert "drop_artifact" in avail and "give_artifact" in avail

    _, _, _, _, infos = env.step(
        {"a": {"action": "drop_artifact", "params": {"name": "virus_i1"}}}
    )
    assert infos["a"]["Artifact drop status"] == "Failed. Artifact virus_i1 cannot be dropped"
    assert get_viral(env, "a") == ["virus_i1"]

    env._get_avail_actions("a")
    _, _, _, _, infos = env.step(
        {"a": {"action": "give_artifact",
               "params": {"artifact_name": "virus_i1", "target_agent": "b"}}}
    )
    assert infos["a"]["Artifact give status"] == "Failed. Artifact virus_i1 cannot be given away"
    assert get_viral(env, "a") == ["virus_i1"]

    # Death: virus dropped on the map at death position with set lifespan
    env.agent_inventories["a"].discard("note")
    env.artifacts.pop("note")
    death_pos = env.agent_pos["a"]
    env.agent_energy["a"] = 0.5
    env.step({})
    assert "a" not in env.agent_registry
    assert "virus_i1" in env.artifacts_map[death_pos], "corpse virus not on map"
    assert env.artifacts["virus_i1"].remaining_time == 5
    assert env.artifacts["virus_i1"].pose == death_pos

    # Corpse still spreads: fresh agent next to the corpse gets infected
    env.viral_infection_probability = 1.0
    env.add_agent(agent_tag="d", agent_name="d", agent_type="text", position=(5, 4))
    env.agent_energy["d"] = 1000.0
    env.step({})
    assert len(get_viral(env, "d")) == 1, "corpse should infect d"

    # Corpse cannot be picked up (and pickup is not even offered)
    env.viral_infection_probability = 0.0
    env._update_agent_pos(agent="d", new_pos=death_pos)
    avail = env._get_avail_actions("d")
    assert "pickup_artifact" not in avail
    _, _, _, _, infos = env.step(
        {"d": {"action": "pickup_artifact", "params": {"name": "virus_i1"}}}
    )
    # not advertised -> action rejected and replaced with a stay move
    assert "virus_i1" in env.artifacts_map[death_pos]

    # Corpse expires after viral_dropped_lifespan steps on the map
    for _ in range(6):
        env.step({})
        for tag in env.agent_registry:
            env.agent_energy[tag] = 1000.0
    assert "virus_i1" not in env.artifacts
    assert any(a.name == "virus_i1" for a in env.expired_artifacts)

    # Checkpoint roundtrip keeps viral state
    # (get_state_ckpt needs a food distribution even with food_mechanism off)
    env._get_food_distribution()
    ckpt = env.get_state_ckpt()
    tmp2 = Path(tempfile.mkdtemp())
    env2 = make_env(tmp2)
    env2.set_state_ckpt(ckpt)
    assert env2.viral_infection_count == env.viral_infection_count
    restored = [a for a in env2.artifacts.values() if isinstance(a, ViralArtifact)]
    assert restored, "no viral artifacts restored"
    assert all(a.strain == "virus" and not a.interactable for a in restored)
    assert any(
        isinstance(a, ViralArtifact) for a in env2.expired_artifacts
    ), "expired viral artifacts not restored as ViralArtifact"
    print("PASS: spread, energy drain, affordances, death drop, corpse spread, ckpt")


def test_contact_only_spread():
    """The default radius transmits by contact: the 8 adjacent cells, no further."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, viral_infection_probability=1.0, viral_lifespan=-1)
    assert env.viral_infection_radius == 1, "default radius should be contact-only"

    # h hosts; ortho and diag are adjacent; far is at Chebyshev distance 2.
    poses = {"h": (5, 5), "ortho": (5, 6), "diag": (6, 6), "far": (5, 7)}
    for tag in poses:
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses=poses)
    for tag in env.agent_registry:
        env.agent_energy[tag] = 1000.0

    env.infect_agent(agent_tag="h")
    env.step({})
    assert get_viral(env, "ortho"), "orthogonal neighbour should be infected"
    assert get_viral(env, "diag"), "diagonal neighbour should be infected"
    assert not get_viral(env, "far"), "distance 2 is out of contact range"

    # Contact wraps around the torus: rows 0 and grid_size - 1 are adjacent.
    tmp2 = Path(tempfile.mkdtemp())
    env2 = make_env(tmp2, viral_infection_probability=1.0, viral_lifespan=-1)
    seam = {"north": (0, 0), "south": (env2.grid_size - 1, 0)}
    for tag in seam:
        env2.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env2.restart_env(agent_poses=seam)
    for tag in env2.agent_registry:
        env2.agent_energy[tag] = 1000.0

    env2.infect_agent(agent_tag="north")
    env2.step({})
    assert get_viral(env2, "south"), "contact should wrap across the grid seam"
    print("PASS: contact-only spread (8 adjacent cells, wrapping, nothing further)")


def test_recovery():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_infection_probability=0.0,
        viral_energy_multiplier=4.0,
        viral_lifespan=3,
    )
    env.add_agent(agent_tag="x", agent_name="x", agent_type="text")
    env.restart_env(agent_poses={"x": (5, 5)})
    env.agent_energy["x"] = 1000.0

    name = env.infect_agent(agent_tag="x")
    assert env.artifacts[name].remaining_time == 3
    for _ in range(3):
        assert len(get_viral(env, "x")) == 1
        env.step({})
    assert len(get_viral(env, "x")) == 0, "infection should have expired"
    assert any(a.name == name for a in env.expired_artifacts)
    e_before = env.agent_energy["x"]
    env.step({})
    assert e_before - env.agent_energy["x"] == 1.0, "drain should be back to 1"

    # Reinfection is possible after recovery
    assert env.infect_agent(agent_tag="x") is not None
    print("PASS: recovery after viral_lifespan and reinfection")


def test_outbreak():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_init_infected=2,
        viral_outbreak_step=1,
        viral_infection_probability=0.0,
    )
    for tag in ("e", "f", "g"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"e": (2, 2), "f": (10, 10), "g": (17, 17)})
    for tag in env.agent_registry:
        env.agent_energy[tag] = 1000.0

    env.step({})  # step_count 0: no outbreak yet
    assert sum(len(get_viral(env, t)) for t in env.agent_registry) == 0
    env.step({})  # step_count 1: outbreak
    infected = [t for t in env.agent_registry if get_viral(env, t)]
    assert len(infected) == 2, infected
    env.step({})  # no second outbreak
    assert sum(len(get_viral(env, t)) for t in env.agent_registry) == 2
    print("PASS: outbreak at configured step infects the configured count")


def test_multiplier_stacks_per_strain():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp, viral_infection_probability=0.0, viral_energy_multiplier=3.0
    )
    env.add_agent(agent_tag="z", agent_name="z", agent_type="text")
    env.restart_env(agent_poses={"z": (5, 5)})
    env.agent_energy["z"] = 1000.0
    env.infect_agent(agent_tag="z", strain="virus")
    env.infect_agent(agent_tag="z", strain="virus2")
    e_before = env.agent_energy["z"]
    env.step({})
    assert e_before - env.agent_energy["z"] == 9.0, "two strains -> K*K drain"
    print("PASS: energy multiplier compounds per hosted strain")


if __name__ == "__main__":
    test_spread_and_energy()
    test_contact_only_spread()
    test_recovery()
    test_outbreak()
    test_multiplier_stacks_per_strain()
    print("\nAll viral artifact checks passed ✅")
