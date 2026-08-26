"""Checks for viral lethality and the recovered metric.

Run from the repo root with: python tests/test_lethality_recovery.py
No LLM calls, no API credits.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from core.environment.env import OpenGridWorld


def make_env(tmp, **kwargs):
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
        **kwargs,
    )
    env.rng = np.random.default_rng(0)
    env.add_agent(agent_tag="a", agent_name="a", agent_type="text")
    env.restart_env(agent_poses={"a": (5, 5)})
    return env


def step_one(env):
    return env.step({t: {"action": "move", "params": {"direction": "stay"}} for t in list(env.agent_registry)})


def test_lethality():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, viral_death_probability=1.0)
    env.infect_agent(agent_tag="a")
    step_one(env)
    assert "a" not in env.agent_registry, "certain lethality must kill the sick"
    events = [json.loads(line) for line in open(tmp / "open_gridworld.log")]
    died = [e for e in events if e.get("event") == "AGENT_DIED"]
    assert died and died[-1]["reason"] == "sickness", died
    print("PASS: a symptomatic agent dies of the sickness at probability 1")

    env = make_env(Path(tempfile.mkdtemp()), viral_death_probability=0.0)
    env.infect_agent(agent_tag="a")
    for _ in range(5):
        step_one(env)
    assert "a" in env.agent_registry
    print("PASS: probability 0 (the default) never kills")


def test_deaths_record_the_infection():
    """Starving while sick is a disease death for the accounting."""
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, viral_death_probability=0.0, viral_energy_multiplier=2.0)
    env.infect_agent(agent_tag="a")
    env.agent_energy["a"] = 1.0  # the doubled drain starves it this step
    step_one(env)
    assert "a" not in env.agent_registry
    events = [json.loads(line) for line in open(tmp / "open_gridworld.log")]
    died = [e for e in events if e.get("event") == "AGENT_DIED"][-1]
    assert died["reason"] == "hunger" and died["infected"] is True, died

    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp)
    env.agent_energy["a"] = 0.5
    step_one(env)
    events = [json.loads(line) for line in open(tmp / "open_gridworld.log")]
    died = [e for e in events if e.get("event") == "AGENT_DIED"][-1]
    assert died["reason"] == "hunger" and died["infected"] is False, died
    print("PASS: every death records whether the being was infected")


def test_hazard_ramps_with_sickness_age():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, viral_death_probability=0.5, viral_lifespan=10)
    env.infect_agent(agent_tag="a")
    assert env._death_hazard("a") == 0.0, "hazard must start at zero"
    hazards = []
    for _ in range(4):
        step_one(env)
        if "a" in env.agent_registry and env._count_viral("a"):
            hazards.append(env._death_hazard("a"))
    assert hazards == sorted(hazards) and hazards[-1] > 0, hazards
    assert all(h <= 0.5 for h in hazards)
    print(f"PASS: hazard ramps up with sickness age {[round(h, 2) for h in hazards]}")


def test_recovered_are_immune():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, viral_lifespan=2)
    env.infect_agent(agent_tag="a")
    for _ in range(3):
        step_one(env)
    assert env.agent_recoveries["a"] == 1
    assert env.infect_agent(agent_tag="a") is None, "recovered must be immune"
    assert env._count_viral("a") == 0
    print("PASS: a recovered agent can never catch the virus again")


def test_incubating_do_not_roll():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(
        tmp,
        viral_death_probability=1.0,
        viral_incubation_min=8,
        viral_incubation_max=8,
    )
    env.infect_agent(agent_tag="a")
    for _ in range(3):
        step_one(env)
    assert "a" in env.agent_registry, "only the symptomatic face the lethality roll"
    print("PASS: an incubating carrier faces no lethality roll")


def test_natural_recovery_is_counted():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, viral_lifespan=2)
    env.infect_agent(agent_tag="a")
    infos = None
    for _ in range(3):
        _, _, _, _, infos = step_one(env)
        if env._count_viral("a") == 0:
            break
    assert env.agent_recoveries["a"] == 1
    assert "recovered" in infos["a"].get("Health", ""), (
        "the agent must be told it recovered, or it keeps believing it is frozen"
    )
    print("PASS: an infection expiring in an inventory counts as a recovery")

    # And the recovered agent moves again on the next step.
    pos = env.agent_pos["a"]
    env.step({"a": {"action": "move", "params": {"direction": "up"}}})
    assert env.agent_pos["a"] != pos, "a recovered agent must be able to move"
    print("PASS: a recovered agent moves again")


def test_world_log_and_checkpoint():
    tmp = Path(tempfile.mkdtemp())
    env = make_env(tmp, viral_lifespan=2)
    env.infect_agent(agent_tag="a")
    for _ in range(3):
        step_one(env)

    lines = [json.loads(line) for line in open(tmp / "world_state.jsonl")]
    meta, last = lines[0], lines[-1]
    assert meta["agent_fields"][-1] == "n_recovered", meta["agent_fields"]
    idx = meta["agent_fields"].index("n_recovered")
    assert last["agents"]["a"][idx] == 1, last["agents"]["a"]

    from viz.reader import RunReader
    with open(tmp / "params.json", "w") as f:
        json.dump({"run": {"exp_name": "x"}, "env": {"grid_size": 20}}, f)
    r = RunReader(tmp)
    r.refresh()
    s = r.series()
    assert s["n_recovered"][-1] == 1, s["n_recovered"]
    print("PASS: n_recovered flows into world_state.jsonl (schema 4) and the series")

    env2 = make_env(Path(tempfile.mkdtemp()))
    env2.set_state_ckpt(env.get_state_ckpt())
    assert env2.agent_recoveries["a"] == 1
    print("PASS: recoveries survive a checkpoint roundtrip")


if __name__ == "__main__":
    test_lethality()
    test_deaths_record_the_infection()
    test_hazard_ramps_with_sickness_age()
    test_recovered_are_immune()
    test_incubating_do_not_roll()
    test_natural_recovery_is_counted()
    test_world_log_and_checkpoint()
    print("\nAll lethality and recovery checks passed ✅")
