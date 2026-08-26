"""Sanity checks for touch-based transmission and adjacency-gated give/take.

Run from the repo root with: python tests/test_contact_transmission.py
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


def test_give_take_need_adjacency():
    """Energy moves by touch: a being merely in view is out of reach."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, viral_infection_probability=0.0)
    for tag in ("a", "b"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 8)})

    # b is within vision (3 cells) but not adjacent: no give/take offered
    avail = env._get_avail_actions("a")
    assert "give" not in avail, "give must need an adjacent being"
    assert "take" not in avail, "take must need an adjacent being"

    # Forcing the action through is refused, with the reason
    env.agent_avail_actions["a"]["give"] = {"params": {"target": "", "amount": ""}}
    e_b = env.agent_energy["b"]
    _, _, _, _, infos = env.step(
        {"a": {"action": "give", "params": {"target": "b", "amount": 5}}}
    )
    assert env.agent_energy["b"] == e_b - 1, "no energy must arrive"
    assert "adjacent" in infos["a"]["Action outcome"]

    # A made-up target is refused too, not crashed on
    env.agent_avail_actions["a"]["give"] = {"params": {"target": "", "amount": ""}}
    _, _, _, _, infos = env.step(
        {"a": {"action": "give", "params": {"target": "nobody", "amount": 5}}}
    )
    assert "no being named nobody" in infos["a"]["Action outcome"]

    # Naming yourself is refused: self-distance is 0 and must not pass
    env.agent_avail_actions["a"]["give"] = {"params": {"target": "", "amount": ""}}
    e_a = env.agent_energy["a"]
    _, rew, _, _, infos = env.step(
        {"a": {"action": "give", "params": {"target": "a", "amount": 5}}}
    )
    assert "yourself" in infos["a"]["Action outcome"]
    assert env.agent_energy["a"] == e_a - 1 and rew["a"] == -1

    # Adjacent (diagonal counts): both actions appear and the transfer lands
    env.agent_pos["b"] = (6, 6)
    env.pos_to_agent.pop((5, 8), None)
    env.pos_to_agent[(6, 6)] = "b"
    avail = env._get_avail_actions("a")
    assert "give" in avail and "take" in avail
    e_b = env.agent_energy["b"]
    env.step({"a": {"action": "give", "params": {"target": "b", "amount": 5}}})
    assert env.agent_energy["b"] == e_b + 5 - 1
    print("PASS: give/take need an adjacent being, not one merely in view")


def test_give_wraps_across_seam():
    """Adjacency is toroidal: beings on opposite edges can touch."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, viral_infection_probability=0.0)
    for tag in ("a", "b"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (0, 5), "b": (19, 5)})

    assert "give" in env._get_avail_actions("a")
    e_b = env.agent_energy["b"]
    env.step({"a": {"action": "give", "params": {"target": "b", "amount": 7}}})
    assert env.agent_energy["b"] == e_b + 7 - 1, "the seam must not block a gift"
    print("PASS: give works across the torus seam")


def test_touch_transmits():
    """A transfer with a symptomatic being is a contact exposure, both ways."""
    tmp = Path(tempfile.mkdtemp())
    # Radius 0 silences proximity spread: any infection here came by touch.
    env = make_env(
        tmp,
        viral_infection_probability=1.0,
        viral_infection_radius=0,
        viral_contact_multiplier=1.0,
        viral_lifespan=-1,
    )
    for tag in ("a", "b", "c"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 6), "c": (5, 4)})
    for tag in env.agent_registry:
        env.agent_energy[tag] = 100.0

    env.infect_agent(agent_tag="a")
    assert get_viral(env, "b") == []

    # The sick being gives energy: the receiver catches it through the touch
    env.step({"a": {"action": "give", "params": {"target": "b", "amount": 5}}})
    assert get_viral(env, "b"), "touching a sick being must transmit at risk 1.0"
    assert get_viral(env, "c") == [], "no touch, no exposure at radius 0"

    # The other direction: a healthy being taking from a sick one catches too
    env.step({"c": {"action": "take", "params": {"target": "a", "amount": 5}}})
    assert get_viral(env, "c"), "taking from a sick being is contact too"
    print("PASS: give/take with a symptomatic being transmits by touch")


def test_touch_respects_incubation_and_ppe():
    """Incubating carriers transmit nothing; PPE protects the toucher."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_infection_probability=1.0,
        viral_infection_radius=0,
        viral_contact_multiplier=1.0,
        viral_lifespan=-1,
        viral_incubation_min=5,
        viral_incubation_max=5,
        ppe_protection=0.0,
        init_artifacts=[{"name": "mask", "type": "ppe", "agent": "d"}],
    )
    for tag in ("a", "b", "d"):
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5), "b": (5, 6), "d": (5, 4)})
    for tag in env.agent_registry:
        env.agent_energy[tag] = 100.0

    env.infect_agent(agent_tag="a")
    assert not env.artifacts[get_viral(env, "a")[0]].symptomatic

    # A silent carrier hands energy over: nothing passes with it
    env.step({"a": {"action": "give", "params": {"target": "b", "amount": 5}}})
    assert get_viral(env, "b") == [], "an incubating infection must not transmit"

    # Symptoms on, but the toucher wears full protection
    env.artifacts[get_viral(env, "a")[0]].incubation = 0
    env.step({"d": {"action": "take", "params": {"target": "a", "amount": 5}}})
    assert get_viral(env, "d") == [], "PPE at 0.0 must block a touch exposure"
    print("PASS: touch exposure respects incubation silence and PPE")


if __name__ == "__main__":
    test_give_take_need_adjacency()
    test_give_wraps_across_seam()
    test_touch_transmits()
    test_touch_respects_incubation_and_ppe()
    print("\nAll contact transmission checks passed ✅")
