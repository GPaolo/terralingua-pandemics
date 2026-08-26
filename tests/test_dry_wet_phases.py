"""Sanity checks for the two-phase symptomatic illness (dry -> wet).

Run from the repo root with: python tests/test_dry_wet_phases.py
"""

import tempfile
from pathlib import Path

import numpy as np

from core.environment.artifact import ViralArtifact
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
        food_mechanism=True,
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


def test_dry_phase_keeps_affordances():
    """A feverish host still moves, eats and is told a milder notice."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_infection_probability=0.0,
        viral_lifespan=10,
        viral_mobile_days=2,
        viral_energy_multiplier=2.0,
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})
    env.agent_energy["a"] = 1000.0
    env._food_decay_rate = 0.0
    env._food_spawn_rate = 0.0
    env.food.clear()

    env.infect_agent(agent_tag="a")

    # Full action set, and it moves and eats while dry
    avail = env._get_avail_actions("a")
    assert "sick" not in avail["move"]["description"]
    pos = env.agent_pos["a"]
    target = (pos[0] + 1, pos[1])
    env.food[target] = 5.0
    e_before = env.agent_energy["a"]
    _, _, _, _, infos = env.step(
        {"a": {"action": "move", "params": {"direction": "down"}}}
    )
    assert env.agent_pos["a"] == target, "a dry host must still move"
    assert target not in env.food, "a dry host must still eat"
    assert env.agent_energy["a"] == e_before + 5.0 - 2.0, "but pays the drain"
    assert "feverish" in infos["a"]["Health"], "told it is ill, mildly"

    # Second dry day: it still moves, but the end-of-step notice already
    # announces the wet phase it will wake into
    pos = env.agent_pos["a"]
    _, _, _, _, infos = env.step(
        {"a": {"action": "move", "params": {"direction": "down"}}}
    )
    assert env.agent_pos["a"] != pos, "the second dry day still moves"
    assert "too weak to move" in infos["a"]["Health"]
    pos = env.agent_pos["a"]
    _, _, _, _, infos = env.step(
        {"a": {"action": "move", "params": {"direction": "down"}}}
    )
    assert env.agent_pos["a"] == pos, "a wet host must not move"
    assert "too sick to move" in infos["a"]["Action outcome"]
    print("PASS: dry days keep every affordance, wet days freeze the host")


def test_dry_phase_transmits_reduced():
    """Dry hosts transmit at the reduced rate; wet hosts at the full one."""
    tmp = Path(tempfile.mkdtemp())
    # Reduced rate 0.0: while dry, nothing at all transmits
    env = make_env(
        tmp,
        viral_infection_probability=1.0,
        viral_lifespan=10,
        viral_mobile_days=2,
        viral_mobile_infectiousness=0.0,
    )
    for tag in ("a", "b"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 6)})
    for tag in env.agent_registry:
        env.agent_energy[tag] = 1000.0

    env.infect_agent(agent_tag="a")

    # Two dry days: adjacency at probability 1.0 still passes nothing,
    # and neither does a touch
    env.step({"a": {"action": "give", "params": {"target": "b", "amount": 1}}})
    env.step({"a": {"action": "give", "params": {"target": "b", "amount": 1}}})
    assert get_viral(env, "b") == [], "a dry host at 0.0 must transmit nothing"

    # Wet from here: the very next step infects the neighbour
    env.step({"a": {"action": "move", "params": {"direction": "stay"}}})
    assert get_viral(env, "b"), "a wet host transmits at the full rate"
    print("PASS: dry hosts transmit at the reduced rate, wet at the full one")


def test_remains_ignore_dry_reduction():
    """A corpse is fully infectious even if its host died in the dry days."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_infection_probability=1.0,
        viral_lifespan=10,
        viral_dropped_lifespan=5,
        viral_mobile_days=8,
        viral_mobile_infectiousness=0.0,
        dead_agent_food="none",
    )
    for tag in ("a", "b"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 6)})
    env.agent_energy["b"] = 1000.0

    env.infect_agent(agent_tag="a")
    env.agent_energy["a"] = 0.5  # dies of the drain this step, still dry
    env.step({})
    assert "a" not in env.agent_registry, "the host must be dead"
    assert get_viral(env, "b") == [], "alive and dry at 0.0: no transmission"
    env.step({})  # the remains lie on the ground through this spread pass
    assert get_viral(env, "b"), "remains must infect at the full rate"
    print("PASS: remains are fully infectious regardless of the host's phase")


def test_health_state_and_world_log():
    """health_state walks the four states and the world log records them."""
    import json

    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_infection_probability=0.0,
        viral_lifespan=6,
        viral_mobile_days=2,
        viral_incubation_min=1,
        viral_incubation_max=1,
    )
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})
    env.agent_energy["a"] = 1000.0

    assert env.health_state("a") == "healthy"
    env.infect_agent(agent_tag="a")
    assert env.health_state("a") == "incubating"
    env.step({})  # incubation 1 -> 0
    assert env.health_state("a") == "feverish"
    env.step({})
    env.step({})
    assert env.health_state("a") == "bedridden"

    lines = [json.loads(x) for x in open(tmp / "world_state.jsonl")]
    meta, frames = lines[0], lines[1:]
    assert meta["schema_version"] == 6
    assert meta["agent_fields"][9] == "n_bedridden"
    bed = [f["agents"]["a"][9] for f in frames if "a" in f["agents"]]
    assert bed[0] == 0 and bed[-1] == 1, bed
    assert frames[-1]["n_bedridden"] == 1
    assert frames[-1]["n_sick"] == 1, "bedridden still counts as sick"
    print("PASS: health_state is canonical and the world log carries n_bedridden")


if __name__ == "__main__":
    test_dry_phase_keeps_affordances()
    test_dry_phase_transmits_reduced()
    test_remains_ignore_dry_reduction()
    test_health_state_and_world_log()
    print("\nAll dry/wet phase checks passed ✅")
