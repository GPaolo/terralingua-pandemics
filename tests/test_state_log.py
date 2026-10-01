"""The mechanic writes one world state line per step, and the file replays to the live world."""

import json

from terralingua.config.models import GraphConfig
from terralingua.environment.graph_env import OpenGraphWorld

from pandemics.anthropologist.epidemic_utils import load_frames
from pandemics.epidemic import Epidemic, EpidemicOptions
from pandemics.state_log import AGENT_FIELDS, KEYFRAME_INTERVAL, WorldStateLogger
from tests.test_epidemic import FAST, make_env, step


def lines(env):
    path = env.log_path / "world_state.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def replay(rows):
    """Food and artifacts at the last step, rebuilt from the key and delta lines."""
    food, artifacts = {}, set()
    for row in rows:
        if row["kind"] == "key":
            food = {(x, y): v for x, y, v in row["food"]["set"]}
            artifacts = {tuple(a) for a in row["artifacts"]["set"]}
        elif row["kind"] == "delta":
            for x, y, v in row["food"]["add"]:
                food[(x, y)] = v
            for x, y in row["food"]["del"]:
                food.pop((x, y), None)
            artifacts |= {tuple(a) for a in row["artifacts"]["add"]}
            artifacts -= {tuple(a) for a in row["artifacts"]["del"]}
    return food, artifacts


def test_each_step_writes_the_world_and_the_header_names_the_fields(tmp_path):
    env, mechanic = make_env(
        tmp_path, [(2, 2), (7, 7)], {"init_infected": 1, "incubation_min": 1, "incubation_max": 1},
        init_food=20, food_mechanism=True,
    )
    for _ in range(4):
        step(env)
    rows = lines(env)
    assert rows[0]["kind"] == "meta"
    assert rows[0]["agent_fields"] == AGENT_FIELDS
    assert rows[0]["grid_size"] == 10
    assert [row["t"] for row in rows[1:]] == [0, 1, 2, 3]
    assert rows[1]["kind"] == "key" and rows[2]["kind"] == "delta"
    last = rows[-1]
    assert set(last["agents"]) == {"a0", "a1"}
    assert all(len(values) == len(AGENT_FIELDS) for values in last["agents"].values())
    for tag, values in last["agents"].items():
        assert tuple(values[:2]) == tuple(env.agent_pos[tag])
        assert values[2] >= env.agent_energy[tag]  # written before the step's energy drain
    assert last["n_agents"] == 2
    assert last["n_infected"] == 1 and last["n_sick"] == 1
    sick = [tag for tag, values in last["agents"].items() if values[6] == 1]
    assert sick == [tag for tag in env.agent_registry if mechanic.symptomatic(tag)]
    food, artifacts = replay(rows)
    assert food == {tuple(k): v for k, v in env.food.items()}
    assert artifacts == {
        (pos[0], pos[1], name, env.artifacts[name].art_type)
        for pos, names in env.pos_artifacts.items()
        for name in names
    }
    assert last["food_total"] == sum(env.food.values())


def test_a_resumed_run_appends_and_starts_with_a_full_line(tmp_path):
    env, mechanic = make_env(tmp_path, [(2, 2)])
    step(env)
    step(env)
    mechanic.world_log.close()
    mechanic.world_log = None  # a new process picks the file up with the restored state
    step(env)
    rows = lines(env)
    assert [row["kind"] for row in rows] == ["meta", "key", "delta", "key"]
    assert [row["t"] for row in rows[1:]] == [0, 1, 2]


def test_keyframes_repeat_at_the_interval():
    assert KEYFRAME_INTERVAL == 50


def test_a_resume_into_a_folder_without_the_file_writes_the_header(tmp_path):
    log = WorldStateLogger(tmp_path / "world_state.jsonl", grid_size=10, max_food_value=10.0, append=True)
    log.log_step(t=5, agents={}, food={}, artifacts=[], food_total=0.0, n_infected=0)
    log.close()
    rows = [json.loads(line) for line in (tmp_path / "world_state.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["meta", "key"]
    assert rows[0]["agent_fields"] == AGENT_FIELDS


def test_the_metrics_loader_keeps_the_latest_line_of_a_repeated_step(tmp_path):
    lines = [
        {"kind": "meta", "schema_version": 6, "grid_size": 10, "max_food_value": 10.0, "agent_fields": AGENT_FIELDS},
        {"kind": "key", "t": 0, "agents": {"a0": [1, 1, 100.0, 50.0, 0, 0, 0, 0, 0, 0]}, "food": {"set": [[3, 3, 10.0]]}, "artifacts": {"set": []}, "food_total": 10.0, "n_agents": 1, "n_infected": 0, "n_sick": 0, "n_bedridden": 0},
        {"kind": "delta", "t": 1, "agents": {"a0": [1, 2, 99.0, 49.0, 0, 0, 0, 0, 0, 0]}, "food": {"add": [], "del": []}, "artifacts": {"add": [[2, 2, "Note", "text"]], "del": []}, "food_total": 10.0, "n_agents": 1, "n_infected": 0, "n_sick": 0, "n_bedridden": 0},
        {"kind": "key", "t": 1, "agents": {"a0": [1, 2, 99.0, 49.0, 0, 1, 0, 0, 0, 0]}, "food": {"set": [[3, 3, 10.0]]}, "artifacts": {"set": []}, "food_total": 10.0, "n_agents": 1, "n_infected": 1, "n_sick": 0, "n_bedridden": 0},
        {"kind": "delta", "t": 2, "agents": {"a0": [1, 3, 98.0, 48.0, 0, 1, 0, 0, 0, 0]}, "food": {"add": [], "del": [[3, 3]]}, "artifacts": {"add": [], "del": []}, "food_total": 0.0, "n_agents": 1, "n_infected": 1, "n_sick": 0, "n_bedridden": 0},
    ]
    (tmp_path / "world_state.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    meta, frames = load_frames(tmp_path)
    assert [frame["t"] for frame in frames] == [0, 1, 2]
    assert frames[1]["agents"]["a0"]["n_viral"] == 1  # the resumed line replaced the first one
    assert frames[1]["artifacts"] == set()
    assert frames[2]["food_total"] == 0.0


def test_the_mechanic_runs_on_a_graph_world_without_a_state_file(tmp_path):
    env = OpenGraphWorld(
        graph_cfg=GraphConfig(topology="ring", n_nodes=6, hop_radius=1), init_agent_energy=100, lifespan=200,
        init_food=0, food_spawn_rate=0, log_path=tmp_path, drop_food_on_death=False, use_inventory=True,
        use_colors=False, reproduction_cost=-1, artifact_creation_cost=0, headless=True,
    )
    mechanic = Epidemic(EpidemicOptions(**{**FAST, "init_infected": 1, "health_center": None}))
    env.attach(mechanic)
    nodes = env.world_graph.all_nodes()
    env.add_agent("a0", "Ada", "text", position=nodes[0])
    env.add_agent("a1", "Bo", "text", position=nodes[1])
    env.restart_env(seed=3, agent_poses={"a0": nodes[0], "a1": nodes[1]})
    for _ in range(3):
        env.step({tag: {"action": "move", "params": {"direction": "stay"}} for tag in env.agent_registry})
    assert mechanic.world_log is None
    assert not (tmp_path / "world_state.jsonl").exists()
    assert len(mechanic.state["infections"]) + len(mechanic.state["recovered"]) >= 1
