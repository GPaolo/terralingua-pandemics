"""Sanity checks for funeral announcements.

Run from the repo root with: python tests/test_funeral_announcements.py
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
    # No cause: working out what kills is the beings' problem
    assert "sickness" not in infos["b"]["Deaths"]
    # 0 mourning days reads as "may be buried", never "in 0 days"
    assert "The remains may be buried." in infos["b"]["Deaths"]
    assert "0 day" not in infos["b"]["Deaths"]

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


def test_announcement_radius():
    """News only travels funeral_announcement_radius on its own."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        funeral_announcements=True,
        funeral_announcement_radius=3,
        viral_infection_probability=0.0,
        viral_lifespan=10,
        viral_dropped_lifespan=5,
    )
    for tag in ("a", "b", "c"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 8), "c": (5, 12)})
    env.agent_energy["b"] = 1000.0
    env.agent_energy["c"] = 1000.0
    env.infect_agent(agent_tag="a")
    env.agent_energy["a"] = 0.5
    _, _, _, _, infos = env.step({})
    assert "Deaths" in infos["b"], "3 cells away is within earshot"
    assert "Deaths" not in infos["c"], "7 cells away must hear nothing"
    print("PASS: funeral news only travels the announcement radius")


def test_mourning_gates_burial_and_reminds():
    """Remains refuse burial for the mourning days, then a reminder fires."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        funeral_announcements=True,
        funeral_mourning_days=2,
        burials=True,
        viral_infection_probability=0.0,
        viral_lifespan=10,
        viral_dropped_lifespan=8,
    )
    for tag in ("a", "b"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 6)})
    env.agent_energy["b"] = 1000.0
    env.infect_agent(agent_tag="a")
    env.agent_energy["a"] = 0.5
    _, _, _, _, infos = env.step({})
    assert "The mourning lasts 2 days" in infos["b"]["Deaths"]
    assert "Word spreads" not in infos["b"]["Deaths"]

    # The action stays offered during the mourning: trying it is how a
    # being learns how long to wait
    assert "bury" in infos["b"]["available_actions"]
    _, _, _, _, infos = env.step(
        {"b": {"action": "bury", "params": {"name": "remains_of_a"}}}
    )
    assert "may be buried in 2 days" in infos["b"]["Action outcome"], infos["b"]

    # Last mourning day: refused with the countdown, and the reminder goes out
    _, _, _, _, infos = env.step(
        {"b": {"action": "bury", "params": {"name": "remains_of_a"}}}
    )
    assert "may be buried in 1 day." in infos["b"]["Action outcome"], infos["b"]
    assert any(env.artifacts_map.values()), "the remains must still lie there"
    assert "may now be buried" in infos["b"]["Deaths"], infos["b"]

    # Next step the burial goes through
    _, _, _, _, infos = env.step(
        {"b": {"action": "bury", "params": {"name": "remains_of_a"}}}
    )
    assert "You buried remains_of_a" in infos["b"]["Action outcome"]
    print("PASS: mourning refuses burial with a countdown, then reminds the living")


def test_attendees_roll_at_the_burial():
    """Everyone beside the grave is exposed when the remains are buried."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        funeral_announcements=True,
        burials=True,
        viral_infection_probability=1.0,
        viral_infection_radius=0,  # no passive spread: any catch is the funeral
        funeral_attendance_multiplier=1.0,
        burial_infection_multiplier=1.0,
        viral_lifespan=10,
        viral_dropped_lifespan=8,
    )
    for tag in ("a", "b", "c", "d"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    # b will dig, c stands at the grave, d keeps its distance
    env.restart_env(
        agent_poses={"a": (5, 5), "b": (5, 6), "c": (4, 5), "d": (5, 9)}
    )
    for tag in ("b", "c", "d"):
        env.agent_energy[tag] = 1000.0
    env.infect_agent(agent_tag="a")
    env.agent_energy["a"] = 0.5
    env.step({})
    _, _, _, _, infos = env.step(
        {"b": {"action": "bury", "params": {"name": "remains_of_a"}}}
    )
    assert "You buried" in infos["b"]["Action outcome"]

    def infected(tag):
        return any(
            isinstance(env.artifacts.get(n), ViralArtifact)
            for n in env.agent_inventories[tag]
        )

    assert infected("b"), "the digger rolls at the burial multiplier"
    assert infected("c"), "a mourner beside the grave rolls too"
    assert not infected("d"), "staying away from the funeral is safe"
    print("PASS: burying is the funeral — attendees beside the grave roll")


if __name__ == "__main__":
    test_death_with_remains_is_announced()
    test_announcements_wrap_and_gate()
    test_announcement_radius()
    test_mourning_gates_burial_and_reminds()
    test_attendees_roll_at_the_burial()
    print("\nAll funeral announcement checks passed ✅")
