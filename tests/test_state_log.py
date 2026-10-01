"""The mechanic writes one world state line per step, and the file replays to the live world."""

import json

from pandemics.state_log import AGENT_FIELDS, KEYFRAME_INTERVAL
from tests.test_epidemic import make_env, step


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
