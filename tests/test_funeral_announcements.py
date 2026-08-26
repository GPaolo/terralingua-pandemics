"""Sanity checks for funeral announcements.

Run from the repo root with: python tests/test_funeral_announcements.py
"""

import tempfile
from pathlib import Path

import numpy as np

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
        dead_agent_food="none",
        **kwargs,
    )
    env.rng = np.random.default_rng(0)
    return env


def test_death_with_remains_is_announced():
    """Every living being hears of the death, with directions it can walk."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        funeral_announcements=True,
        viral_infection_probability=0.0,
        viral_lifespan=10,
        viral_dropped_lifespan=5,
    )
    for tag in ("a", "b", "c"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    # b is 2 up from a; c is across the seam, 3 down from a's cell
    env.restart_env(agent_poses={"a": (5, 5), "b": (3, 5), "c": (8, 5)})
    env.agent_energy["b"] = 1000.0
    env.agent_energy["c"] = 1000.0

    env.infect_agent(agent_tag="a")
    env.agent_energy["a"] = 0.5  # dies of the drain this step
    _, _, _, _, infos = env.step({})

    assert "a" not in env.agent_registry
    assert "Deaths" in infos["b"], "the death must be announced"
    assert "a has died" in infos["b"]["Deaths"]
    assert "2 cells down" in infos["b"]["Deaths"], infos["b"]["Deaths"]
    assert "3 cells up" in infos["c"]["Deaths"], infos["c"]["Deaths"]

    # One announcement, not one per step: the next step is quiet
    _, _, _, _, infos = env.step({})
    assert "Deaths" not in infos["b"]
    print("PASS: a death that leaves remains is announced once, with directions")


def test_announcements_wrap_and_gate():
    """Directions take the short way around the torus; flag off = silence."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        funeral_announcements=True,
        viral_infection_probability=0.0,
        viral_lifespan=10,
        viral_dropped_lifespan=5,
    )
    for tag in ("a", "b"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (0, 0), "b": (19, 19)})
    env.agent_energy["b"] = 1000.0
    env.infect_agent(agent_tag="a")
    env.agent_energy["a"] = 0.5
    _, _, _, _, infos = env.step({})
    assert "1 cell down, 1 cell right" in infos["b"]["Deaths"], infos["b"]["Deaths"]

    # Uninfected deaths leave no remains, so there is nothing to gather at
    tmp2 = Path(tempfile.mkdtemp())
    env2 = make_env(tmp2, funeral_announcements=True)
    for tag in ("a", "b"):
        env2.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env2.restart_env(agent_poses={"a": (5, 5), "b": (3, 5)})
    env2.agent_energy["b"] = 1000.0
    env2.agent_energy["a"] = 0.5
    _, _, _, _, infos = env2.step({})
    assert "a" not in env2.agent_registry
    assert "Deaths" not in infos["b"], "no remains, no funeral"

    # Flag off: remains drop silently, as before
    tmp3 = Path(tempfile.mkdtemp())
    env3 = make_env(
        tmp3,
        viral_infection_probability=0.0,
        viral_lifespan=10,
        viral_dropped_lifespan=5,
    )
    for tag in ("a", "b"):
        env3.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env3.restart_env(agent_poses={"a": (5, 5), "b": (3, 5)})
    env3.agent_energy["b"] = 1000.0
    env3.infect_agent(agent_tag="a")
    env3.agent_energy["a"] = 0.5
    _, _, _, _, infos = env3.step({})
    assert "Deaths" not in infos["b"], "announcements must be opt-in"
    print("PASS: directions wrap the seam; no remains or no flag, no announcement")


if __name__ == "__main__":
    test_death_with_remains_is_announced()
    test_announcements_wrap_and_gate()
    print("\nAll funeral announcement checks passed ✅")
