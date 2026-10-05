"""The run viewer over a scripted run of the mechanic, with no model calls."""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient
from terralingua.agents.agent_logger import AgentLogger
from terralingua.environment.env_logger import Event

from pandemics.viewer.reader import LIVE_GRACE_SECONDS, RunReader
from pandemics.viewer.server import create_app
from tests.test_epidemic import make_env, step

PARAMS = {
    "agent": {"model": "test"},
    "env": {"grid_size": 10, "vision_radius": 2, "agent_lifespan": 500, "init_agent_energy": 100},
    "run": {"exp_description": "test", "max_ts": 5},
}
STATUS_COMPLETE = {"simulation": {"status": "complete"}, "process": {"status": "complete"}}


def scripted_run(tmp_path, options=None):
    """Five steps of a three-being outbreak, plus the files the runner writes."""
    logs_root = tmp_path / "logs"
    run = logs_root / "run1"
    env, mechanic = make_env(
        run, [(2, 2), (2, 3), (7, 7)],
        {"init_infected": 1, "incubation_min": 1, "incubation_max": 1, **(options or {})},
        init_food=20, food_mechanism=True,
    )
    env.agent_identity("a0")  # the first persona of the file, with its role
    for _ in range(5):
        step(env)
    env.logger.fp.flush()
    mechanic.world_log.fp.flush()
    (run / "params.json").write_text(json.dumps(PARAMS, indent=2))
    (run / "costs.csv").write_text(
        "timestep,cost_usd,input_tokens,output_tokens\n"
        + "".join(f"{t},0.01,100,20\n" for t in range(5))
    )
    (run / "run_status.json").write_text(json.dumps(STATUS_COMPLETE, indent=2))
    return logs_root, run, env


def test_the_viewer_serves_a_scripted_run(tmp_path):
    logs_root, run, env = scripted_run(tmp_path)
    client = TestClient(create_app(logs_root))

    runs = client.get("/api/runs").json()["runs"]
    assert [r["name"] for r in runs] == ["run1"]
    assert runs[0]["status"] == "finished" and runs[0]["has_viral"] is True

    meta = client.get("/api/runs/run1/meta").json()
    assert meta["agent_names"] == {"a0": "Name0", "a1": "Name1", "a2": "Name2"}
    assert meta["agents"] == ["a0", "a1", "a2"]
    assert meta["has_viral"] is True
    assert meta["last_step"] == 4 and meta["last_decision_step"] == 4
    assert meta["agent_roles"] == {"a0": "religious_leader"}
    assert meta["personas"]["a0"].startswith("You are a religious leader.")

    world = client.get("/api/runs/run1/step/4").json()
    assert set(world["agents"]) == {"a0", "a1", "a2"}
    assert all(len(values) == 10 for values in world["agents"].values())
    assert world["food"] and world["artifacts"] == []
    assert world["chat"] == [] and world["ticks"] == {"a0": None, "a1": None, "a2": None}
    assert world["n_infected"] == 2 and world["n_sick"] == 2

    series = client.get("/api/runs/run1/series").json()
    assert series["t"] == [0, 1, 2, 3, 4]
    assert series["n_infected"] == [1, 2, 2, 2, 2]
    assert [row["cum_cost"] for row in series["tokens"]] == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])
    assert series["tokens"][-1] == {"t": 4, "input": 100, "output": 20, "cum_input": 500, "cum_output": 100, "cum_cost": pytest.approx(0.05)}

    viral = client.get("/api/runs/run1/viral").json()
    roots = [node for node in viral["chain"] if node["generation"] == 0]
    assert len(roots) == 1 and roots[0]["source"] is None and roots[0]["host"] == "a0"
    children = [node for node in viral["chain"] if node["generation"] == 1]
    assert children and all(node["source"] == roots[0]["infection"] for node in children)
    assert roots[0]["secondary"] == len(children)
    assert viral["censored"] == 2 and viral["generations"] == [] and viral["r0"] is None


def test_ended_infections_count_toward_r0(tmp_path):
    logs_root, run, env = scripted_run(tmp_path, {"infection_duration": 2, "mobile_days": 5})
    viral = TestClient(create_app(logs_root)).get("/api/runs/run1/viral").json()
    by_host = {node["host"]: node for node in viral["chain"]}
    assert by_host["a0"]["ended_at"] == 2 and by_host["a1"]["ended_at"] == 3
    assert viral["censored"] == 0
    assert viral["generations"] == [
        {"generation": 0, "cases": 1, "mean_secondary": 1.0},
        {"generation": 1, "cases": 1, "mean_secondary": 0.0},
    ]
    assert viral["r0"] == 0.5


def test_agent_logs_feed_the_chat_the_thought_and_the_genome(tmp_path):
    logs_root, run, env = scripted_run(tmp_path)
    logger = AgentLogger(run / "agent_logs", "a0")
    logger.save_genome("a0", {"openness": 0.5})
    logger.log(
        agent_name="Name0",
        agent_tag="a0",
        observation={
            "observation": {(0, 0): ["yourself"]},
            "observation_text": "",
            "incoming_broadcasts": {"Name1": "hi"},
            "energy": 96.0,
            "time": 496.0,
            "inventory": [],
            "vision_radius": 2,
        },
        available_actions={},
        action={"action": "move", "message": "hello", "params": {"direction": "stay"}},
        time="3",
        internal_memory="stay put",
        input_prompt="",
    )
    client = TestClient(create_app(logs_root))
    world = client.get("/api/runs/run1/step/3").json()
    assert world["chat"] == [{"t": 3, "agent_tag": "a0", "agent_name": "Name0", "message": "hello"}]
    tick = world["ticks"]["a0"]
    assert tick["action"] == "move" and tick["internal_memory"] == "stay put"
    assert tick["heard"] == {"Name1": "hi"} and tick["energy"] == 96.0
    meta = client.get("/api/runs/run1/meta").json()
    assert meta["genomes"] == {"a0": {"openness": 0.5}}
    assert meta["last_decision_step"] == 3


def test_a_token_counts_file_in_agent_logs_is_not_a_being(tmp_path):
    logs_root, run, env = scripted_run(tmp_path)
    (run / "agent_logs").mkdir(exist_ok=True)
    (run / "agent_logs" / "token_counts.jsonl").write_text(
        json.dumps({"timestep": 0, "agent_tag": "a0", "total_input_tokens": 1}) + "\n"
    )
    client = TestClient(create_app(logs_root))
    assert [r["name"] for r in client.get("/api/runs").json()["runs"]] == ["run1"]
    assert "token_counts" not in client.get("/api/runs/run1/meta").json()["agents"]


def test_the_status_follows_the_status_file_then_the_end_event(tmp_path):
    logs_root, run, env = scripted_run(tmp_path)
    reader = RunReader(run)
    reader.refresh()
    assert reader.status() == "finished"
    (run / "run_status.json").write_text(json.dumps({"simulation": {"status": "running"}}))
    assert reader.status() == "live"
    old = time.time() - 2 * LIVE_GRACE_SECONDS
    for name in ("world_state.jsonl", "open_gridworld.log"):
        os.utime(run / name, (old, old))
    assert reader.status() == "stalled"
    (run / "run_status.json").unlink()
    assert reader.status() == "stalled"
    env.logger.log(time=5, event_type=Event.END_RUN)
    env.logger.fp.flush()
    reader.refresh()
    assert reader.status() == "finished"
