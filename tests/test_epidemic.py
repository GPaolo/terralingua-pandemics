"""The Ebola mechanic on a grid world, driven with scripted actions and no model calls."""

import json
import os
import types
from pathlib import Path

import pytest
from terralingua.agents.personas import load_personas
from terralingua.config.compose import compose
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.experiment import runner as runner_module
from terralingua.experiment.runner import SimulationRunner
from terralingua.experiment.scenario_loader import load_scenario

from pandemics.artifacts import HealthCenterArtifact, PPEArtifact, RemainsArtifact
from pandemics.epidemic import (
    BEDRIDDEN_NOTICE,
    FEVERISH_NOTICE,
    NO_APPETITE_NOTICE,
    RECOVERED_NOTICE,
    Epidemic,
    EpidemicOptions,
)

STAY = {"action": "move", "params": {"direction": "stay"}}

# Deterministic settings: one silent step, everything certain unless a test says otherwise.
# An infection caught at step S incubates during S+1, spreads from S+1 on, and counts its
# first sick step at the end of S+1. With mobile_days=2 it is feverish at the end of S+1
# and bedridden at the end of S+2.
FAST = dict(
    init_infected=0, incubation_min=1, incubation_max=1, infection_duration=-1, mobile_days=2,
    feverish_multiplier=1.0, infection_probability=1.0, contact_multiplier=1.0,
    energy_multiplier=1.0, case_fatality=0.0, ppe_protection=0.0, ppe_per_worker=0,
    funeral_announcement_radius=-1, funeral_mourning_days=0, remains_lifespan=-1,
    health_centers_path=None,
)


def make_env(tmp_path, agents, options=None, **env_kwargs):
    """A 10x10 grid with no food, the mechanic attached, and agents at fixed cells."""
    mechanic = Epidemic(EpidemicOptions(**{**FAST, **(options or {})}))
    defaults = dict(
        grid_size=10, vision_radius=2, init_agent_energy=100, lifespan=500, init_food=0,
        food_spawn_rate=0, food_mechanism=False, log_path=tmp_path, drop_food_on_death=False,
        use_inventory=True, use_colors=False, reproduction_cost=-1, artifact_creation_cost=0,
        headless=True,
    )
    defaults.update(env_kwargs)
    env = OpenGridWorld(**defaults)  # type: ignore
    env.attach(mechanic)
    for i, pos in enumerate(agents):
        env.add_agent(f"a{i}", f"Name{i}", "text", position=pos)
    env.restart_env(seed=3, agent_poses={f"a{i}": pos for i, pos in enumerate(agents)})
    return env, mechanic


def step(env, **actions):
    """One step where every agent stays unless an action is given for it."""
    acts = {tag: STAY for tag in env.agent_registry}
    acts.update(actions)
    return env.step(acts)[-1]


def events(env, name):
    env.logger.fp.flush()
    lines = [json.loads(line) for line in env.logger.save_path.read_text().splitlines()]
    return [line for line in lines if line["event"] == name]


def test_outbreak_infects_silently_and_symptoms_follow_the_incubation(tmp_path):
    env, mechanic = make_env(
        tmp_path, [(2, 2), (7, 7)], {"init_infected": 1, "incubation_min": 1, "incubation_max": 1},
    )
    infos = step(env)  # step 0: outbreak
    host = next(iter(mechanic.state["infections"]))
    assert mechanic.health(host) == "incubating"
    assert "Health" not in infos[host]
    assert events(env, "VIRAL_INFECTION")[0]["source_kind"] == "outbreak"
    infos = step(env)  # incubation ticks to 0
    assert mechanic.health(host) == "feverish"
    assert infos[host]["Health"] == FEVERISH_NOTICE


def test_bedridden_hosts_lose_take_and_cannot_move(tmp_path):
    env, mechanic = make_env(tmp_path, [(2, 2), (2, 3)], {"mobile_days": 0})
    mechanic.infect(env, "a0", None, "test")
    step(env)  # the fresh infection is not ticked this step
    step(env)  # first symptomatic step: bedridden right away
    assert mechanic.health("a0") == "bedridden"
    assert "take" not in env.agent_avail_actions["a0"]
    assert env.agent_avail_actions["a0"]["move"]["params"]["direction"].startswith("Must be 'stay'")
    infos = step(env, a0={"action": "move", "params": {"direction": "up"}})
    assert infos["a0"]["Action outcome"] == "You are too sick to move. You stayed where you are."
    assert env.agent_pos["a0"] == (2, 2)
    assert infos["a0"]["Health"] == BEDRIDDEN_NOTICE


def test_sick_hosts_infect_neighbours_unless_they_carry_protection(tmp_path):
    env, mechanic = make_env(tmp_path, [(2, 2), (2, 3), (7, 7)])
    env.seed_artifact((2, 3), "ppe", "mask", "", -1, to_inventory="a1", protection=0.0)
    mechanic.infect(env, "a0", None, "test")
    step(env)
    step(env)  # a0 turns feverish and spreads at once: a1 is protected, a2 is far away
    assert mechanic.health("a0") == "feverish"
    assert mechanic.health("a1") == "healthy"
    assert mechanic.health("a2") == "healthy"
    env.agent_inventories["a1"].discard("mask")
    step(env)
    assert mechanic.health("a1") == "incubating"
    assert events(env, "VIRAL_INFECTION")[-1]["source_kind"] == "proximity"


def test_energy_changes_hands_only_between_adjacent_beings_and_is_a_contact(tmp_path):
    env, mechanic = make_env(
        tmp_path, [(2, 2), (2, 3), (2, 5), (2, 6), (8, 8)], {"infection_radius": 0}, vision_radius=4,
    )
    mechanic.infect(env, "a0", None, "test")
    step(env)
    step(env)
    assert mechanic.health("a1") == "healthy"  # no proximity spread at radius 0
    assert "give" not in env.agent_avail_actions["a4"]  # nobody adjacent
    menu = env.agent_avail_actions["a2"]
    assert menu["give"]["params"]["target"]["choices"] == ["Name3"]  # a0 is in view but not adjacent
    assert menu["give"]["description"].endswith("This is physical contact.")
    infos = step(env, a2={"action": "give", "params": {"target": "Name0", "amount": 5}})
    assert infos["a2"]["Action outcome"] == "Cannot give energy: Name0 is not on a cell adjacent to yours."
    assert mechanic.health("a2") == "healthy"
    infos = step(env, a1={"action": "give", "params": {"target": "Name0", "amount": 5}})
    assert "gave you" in infos["a0"]["Energy received"]
    assert mechanic.health("a1") == "incubating"
    assert events(env, "VIRAL_INFECTION")[-1]["source_kind"] == "touch"


def test_recovery_is_immunity(tmp_path):
    env, mechanic = make_env(tmp_path, [(2, 2)], {"infection_duration": 2, "mobile_days": 5})
    mechanic.infect(env, "a0", None, "test")
    for _ in range(4):
        step(env)
    assert mechanic.health("a0") == "healthy"
    assert mechanic.state["recovered"] == {"a0": True}
    assert events(env, "VIRAL_HEALED")[-1]["cause"] == "recovery"
    assert mechanic.infect(env, "a0", None, "test") is None
    assert step(env)["a0"]["Health"] == RECOVERED_NOTICE


def test_sick_hosts_lose_extra_energy_and_may_die_leaving_remains(tmp_path):
    env, mechanic = make_env(
        tmp_path, [(2, 2), (2, 4), (8, 8)],
        {"mobile_days": 0, "energy_multiplier": 3.0, "case_fatality": 1.0, "funeral_announcement_radius": 3},
    )
    mechanic.infect(env, "a0", None, "test")
    step(env)
    before = env.agent_energy["a0"]
    step(env)  # first sick step: extra drain, then the certain death roll
    assert "a0" not in env.agent_registry
    assert env.logger.data["a0"]["death_reason"] == "sickness"
    assert env.logger.data["a0"]["energy"] == before - 2
    infos = step(env)  # the death record is seen now: remains appear and the funeral is announced
    remains = [a for a in env.artifacts.values() if isinstance(a, RemainsArtifact)]
    assert len(remains) == 1 and env.artifact_location[remains[0].name] == ("map", (2, 2))
    assert infos["a1"]["Deaths"].startswith("Name0 has died. Their remains lie unburied 2 cells left.")
    assert "Deaths" not in infos["a2"]


def test_burial_waits_for_the_mourning_then_removes_the_remains(tmp_path):
    env, mechanic = make_env(
        tmp_path, [(2, 2), (2, 3), (8, 8)],
        {"mobile_days": 0, "case_fatality": 1.0, "funeral_mourning_days": 2, "infection_probability": 0.0},
    )
    mechanic.infect(env, "a0", None, "test")
    step(env)
    step(env)  # a0 dies
    infos = step(env)  # remains appear, mourning lasts two steps
    assert "The mourning lasts 2 days" in infos["a1"]["Deaths"]
    name = next(n for n, a in env.artifacts.items() if isinstance(a, RemainsArtifact))
    assert "bury" in env.agent_avail_actions["a1"]
    assert "bury" not in env.agent_avail_actions["a2"]
    infos = step(env, a1={"action": "bury", "params": {"name": name}})
    assert infos["a1"]["Action outcome"].startswith("The mourning for Name0 has not ended")
    assert "may now be buried" in infos["a1"]["Deaths"]  # the reminder arrives with the first allowed step
    infos = step(env, a1={"action": "bury", "params": {"name": name}})
    assert infos["a1"]["Action outcome"] == f"You buried {name}."
    assert name not in env.artifacts
    assert events(env, "BURIAL")[-1]["remains"] == name


def test_burial_is_an_exposure_for_the_digger(tmp_path):
    env, mechanic = make_env(
        tmp_path, [(2, 2), (2, 3)],
        {"mobile_days": 0, "case_fatality": 1.0, "infection_probability": 0.0, "burial_infection_multiplier": 1.0},
    )
    mechanic.infect(env, "a0", None, "test")
    step(env)
    step(env)
    step(env)
    name = next(n for n, a in env.artifacts.items() if isinstance(a, RemainsArtifact))
    mechanic.options.infection_probability = 1.0
    step(env, a1={"action": "bury", "params": {"name": name}})
    assert mechanic.health("a1") == "incubating"
    assert events(env, "VIRAL_INFECTION")[-1]["source_kind"] == f"burial:{name}"


def centers_file(tmp_path, centers):
    path = tmp_path / "health_centers.json"
    path.write_text(json.dumps(centers))
    return str(path)


def test_health_center_heals_and_announces_itself(tmp_path):
    center = {"pose": [2, 3], "radius": 1, "heal_probability": 1.0, "hazard_multiplier": 0.5}
    path = centers_file(tmp_path, [center])
    env, mechanic = make_env(tmp_path, [(2, 2), (7, 7)], {"health_centers_path": path})
    mechanic.infect(env, "a0", None, "test")
    infos = step(env)  # seeds the center, then care heals a0 at once
    assert isinstance(env.artifacts["Health Center"], HealthCenterArtifact)
    assert infos["a0"]["Nearby facilities"].startswith("A health center.")
    assert "Nearby facilities" not in infos["a1"]
    assert mechanic.health("a0") == "healthy"
    assert events(env, "VIRAL_HEALED")[-1]["cause"] == "care"


def test_each_health_center_keeps_its_own_parameters(tmp_path):
    path = centers_file(tmp_path, [
        {"pose": [2, 3], "radius": 1, "heal_probability": 1.0, "name": "Clinic"},
        {"pose": [7, 7], "radius": 1, "heal_probability": 0.0, "hazard_multiplier": 0.25, "name": "Shrine"},
    ])
    env, mechanic = make_env(tmp_path, [(2, 2), (7, 6)], {"health_centers_path": path})
    mechanic.infect(env, "a0", None, "test")
    mechanic.infect(env, "a1", None, "test")
    step(env)
    assert isinstance(env.artifacts["Clinic"], HealthCenterArtifact)
    assert isinstance(env.artifacts["Shrine"], HealthCenterArtifact)
    assert mechanic.health("a0") == "healthy"  # the Clinic heals at once
    assert mechanic.health("a1") != "healthy"  # the Shrine never heals
    assert mechanic.care(env)["a1"] == 0.25  # but it lowers the hazard


def test_health_centers_file_must_exist_and_validate(tmp_path):
    env, _ = make_env(tmp_path, [(2, 2)], {"health_centers_path": str(tmp_path / "missing.json")})
    with pytest.raises(FileNotFoundError):
        step(env)
    path = centers_file(tmp_path, [{"pose": [2, 3], "heal_chance": 1.0}])
    env, _ = make_env(tmp_path / "second", [(2, 2)], {"health_centers_path": path})
    with pytest.raises(ValueError, match="heal_chance"):
        step(env)


def test_infection_state_survives_a_checkpoint(tmp_path):
    env, mechanic = make_env(tmp_path, [(2, 2), (2, 3)])
    mechanic.infect(env, "a0", None, "test")
    step(env)
    ckpt = env.get_state_ckpt()
    restored, fresh = make_env(tmp_path / "second", [(2, 2), (2, 3)])
    restored.set_state_ckpt(ckpt)
    assert fresh.state["infections"] == mechanic.state["infections"]
    assert fresh.health("a0") == mechanic.health("a0")


def test_scenario_module_loads_and_the_preset_composes():
    mechanics = load_scenario("pandemics", {"init_infected": 2, "infection_duration": 5})
    assert [type(m).__name__ for m in mechanics] == ["Epidemic"]
    assert mechanics[0].options.init_infected == 2
    with pytest.raises(ValueError):
        load_scenario("pandemics", {"incubation_min": 5, "incubation_max": 2})
    cfg = compose("ebola")
    assert cfg.run.scenario == "pandemics"
    assert cfg.env.world_type == "grid"
    assert Path(cfg.agent.scenario_specific_instructions).read_text() == ""
    options = EpidemicOptions.model_validate(cfg.run.scenario_options)
    assert len(Epidemic(options).health_centers()) == 1  # ebola/health_centers.json, from the working directory


def test_a_sick_host_that_starves_still_leaves_infectious_remains(tmp_path):
    env, mechanic = make_env(
        tmp_path, [(2, 2), (2, 3), (7, 7)],
        {"mobile_days": 5, "feverish_multiplier": 0.0, "energy_multiplier": 2.0},
        init_agent_energy=3, energy_death=True,
    )
    mechanic.infect(env, "a0", None, "test")
    for _ in range(4):
        step(env)  # feverish a0 spreads nothing at factor 0 but loses one extra energy per sick step
    assert mechanic.health("a1") == "healthy"
    assert "a0" not in env.agent_registry
    assert env.logger.data["a0"]["death_reason"] == "energy"
    step(env)  # remains appear and spread at the full rate
    assert any(isinstance(a, RemainsArtifact) for a in env.artifacts.values())
    assert mechanic.health("a1") == "incubating"
    assert events(env, "VIRAL_INFECTION")[-1]["source_kind"].startswith("remains:")
    assert mechanic.health("a2") == "healthy"


def test_beings_beside_the_grave_are_exposed_at_a_burial(tmp_path):
    env, mechanic = make_env(
        tmp_path, [(2, 2), (2, 3), (2, 1), (7, 7)],
        {"mobile_days": 0, "case_fatality": 1.0, "infection_probability": 0.0,
         "burial_infection_multiplier": 1.0, "burial_bystander_multiplier": 1.0},
    )
    mechanic.infect(env, "a0", None, "test")
    for _ in range(3):
        step(env)
    name = next(n for n, a in env.artifacts.items() if isinstance(a, RemainsArtifact))
    mechanic.options.infection_probability = 1.0  # the option is mutable on purpose
    step(env, a1={"action": "bury", "params": {"name": name}})
    assert mechanic.health("a1") == "incubating"
    assert mechanic.health("a2") == "incubating"
    assert mechanic.health("a3") == "healthy"
    kinds = {e["agent_tag"]: e["source_kind"] for e in events(env, "VIRAL_INFECTION")}
    assert kinds["a1"] == f"burial:{name}" and kinds["a2"] == f"funeral:{name}"
    assert events(env, "BURIAL")[-1]["attendees"] == ["a2"]
    assert name not in env.artifacts


def test_personas_follow_the_file_order_and_counts(tmp_path):
    personas = tmp_path / "personas.json"
    personas.write_text(json.dumps([
        {"persona": "You brew tea.", "name": "Amara", "count": 2, "role": "healer"},
        {"persona": "You preach.", "name": "Ezekiel", "role": "leader"},
    ]))
    env, mechanic = make_env(tmp_path, [(2, 2), (2, 3), (2, 4), (2, 5)], {"ppe_role": "healer", "ppe_per_worker": 1})
    entries = load_personas(personas)  # TerraLingua reads the file and keeps the roles
    idents = [env.agent_identity_settled(f"a{i}", dict(entries[i]) if i < len(entries) else {}) for i in range(4)]
    # a name applies only to an entry with count 1 (TerraLingua's rule), so both tea brewers get human names
    assert idents[0]["persona"] == "You brew tea." and idents[0]["role"] == "healer" and idents[0]["name"] not in ("Amara", "a0")
    assert idents[1]["persona"] == "You brew tea." and idents[1]["name"] not in ("Amara", "a1")
    assert idents[2] == {"name": "Ezekiel", "persona": "You preach.", "role": "leader"}
    assert idents[3] == {"name": idents[3]["name"]} and idents[3]["name"]
    assert len({i["name"] for i in idents}) == 4
    assert mechanic.state["identities"]["a1"]["role"] == "healer"
    assert env.agent_identity_settled("a1", {}) == {"name": idents[1]["name"]}  # stable on a second call
    step(env)  # fixtures: protective equipment goes to the two healers
    for tag in ("a0", "a1"):
        assert any(isinstance(env.artifacts[n], PPEArtifact) for n in env.agent_inventories[tag])
    assert not env.agent_inventories["a2"]


def test_bedridden_hosts_have_no_appetite(tmp_path):
    env, mechanic = make_env(tmp_path, [(2, 2), (5, 5)], {"mobile_days": 0})
    mechanic.infect(env, "a0", None, "test")
    step(env)
    step(env)  # a0 is bedridden now
    env.food = {(2, 2): 7.0, (5, 5): 7.0}
    before = dict(env.agent_energy)
    infos = step(env)
    assert env.food == {(2, 2): 7.0}
    assert env.agent_energy["a0"] == before["a0"]
    assert env.agent_energy["a1"] == before["a1"] + 7
    assert infos["a0"]["Action outcome"] == NO_APPETITE_NOTICE
    assert infos["a0"]["Health"] == BEDRIDDEN_NOTICE


def test_runner_gives_the_personas_and_human_names_from_the_preset(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(runner_module, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(runner_module, "LLMRouter", lambda **kw: types.SimpleNamespace(**kw))
    cfg = compose("ebola", {"init_agents": 6, "min_agents": 0, "grid_size": 12, "exp_name": "ebola_runner_test",
                            "scenario_options": {"health_centers_path": None}})
    runner = SimulationRunner(cfg)
    names = [runner.env.agent_names[f"being{i}"] for i in range(6)]
    assert names[:4] == ["Ezekiel", "Amara", "Miriam", "Tendai"]
    assert names[4] not in ("being4", "") and names[5] not in ("being5", "")
    assert "You are a religious leader." in runner.agents["being0"].system_prompt
    assert runner.agents["being4"].persona == ""
    assert runner.agents["being4"].system_prompt.count("You are a ") == runner.agents["being5"].system_prompt.count("You are a ")


def test_case_fatality_is_the_share_that_dies_over_the_sick_period(tmp_path):
    env, mechanic = make_env(tmp_path, [(2, 2)], {"infection_duration": 13, "mobile_days": 4, "case_fatality": 0.5})
    mechanic.infect(env, "a0", None, "test")
    infection = mechanic.state["infections"]["a0"]
    infection["incubation"] = 0

    def hazards(fatality):
        mechanic.options.case_fatality = fatality
        out = []
        for day in range(1, 13):  # the 13th sick step recovers before the roll
            infection["days_symptomatic"] = day
            out.append(mechanic.death_hazard("a0"))
        return out

    alive = 1.0
    for hazard in hazards(0.5):
        alive *= 1 - hazard
    assert alive == pytest.approx(0.5)
    half = hazards(0.5)
    assert half == sorted(half) and half[0] < 0.01
    assert hazards(1.0)[-1] == 1.0
    assert hazards(0.0) == [0.0] * 12
    mechanic.options.infection_duration = -1  # never recovers: the share is the chance per bedridden step
    mechanic.options.case_fatality = 0.5
    infection["days_symptomatic"] = 1
    assert mechanic.death_hazard("a0") == 0.0
    infection["days_symptomatic"] = 4
    assert mechanic.death_hazard("a0") == 0.5
