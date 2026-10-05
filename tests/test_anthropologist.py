"""The report, the sandbox and the dashboard over a scripted run, with no model call."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from pandemics.anthropologist import compare, report, sandbox
from pandemics.anthropologist import epidemic_utils as eu
from pandemics.anthropologist.dashboard import create_app
from tests.test_epidemic import FAST, make_env, step

OPTIONS = {"init_infected": 1, "incubation_min": 1, "incubation_max": 1}


def scripted_run(logs_root, name, steps=6, options=None):
    """A run folder written by the mechanic: two beings side by side, one outbreak."""
    run_dir = logs_root / name
    run_dir.mkdir(parents=True)
    env, mechanic = make_env(run_dir, [(2, 2), (2, 3), (7, 7)], {**OPTIONS, **(options or {})},
                             init_food=10, food_mechanism=True)
    for _ in range(steps):
        step(env)
    env.logger.fp.flush()
    mechanic.world_log.fp.flush()
    params = {
        "agent": {"model": "test"},
        "env": {"grid_size": 10, "vision_radius": 2, "agent_lifespan": 500, "init_agent_energy": 100},
        "run": {"exp_description": "test", "max_ts": steps, "scenario_options": {**FAST, **OPTIONS}},
    }
    (run_dir / "params.json").write_text(json.dumps(params, indent=2))
    return run_dir


def test_metrics_follow_the_infection_chain(tmp_path):
    run_dir = scripted_run(tmp_path, "run1")
    metrics, series, infections, exposures = eu.compute_all(run_dir)
    assert metrics["outbreak"]["index_cases"] == 1
    assert metrics["outbreak"]["infections"] == 2
    assert metrics["outbreak"]["attack_rate"] == 2 / 3
    index = next(r for r in infections if r["source_kind"] == "outbreak")
    child = next(r for r in infections if r["source_kind"] == "proximity")
    assert child["parent"] == index["infection"]
    assert index["generation"] == 0 and child["generation"] == 1
    assert index["secondary"] == 1
    assert exposures and all(e["ppe"] is False for e in exposures)
    assert any(e["infected"] for e in exposures)
    assert series[-1]["alive"] == 3
    assert series[-1]["cum_infections"] == 2
    assert metrics["ppe"]["efficiency"]["protection_configured"] == 0.0


def test_report_writes_metrics_and_plots(tmp_path):
    run_dir = scripted_run(tmp_path, "run1")
    report.generate(run_dir)
    out = run_dir / "epidemic_analysis"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["outbreak"]["infections"] == 2
    for name in ("epidemic_curves.png", "infections.png", "transmission_tree.png", "secondary_cases.png", "ppe.png"):
        assert (out / name).exists(), name
    assert (out / "timeseries.csv").read_text().count("\n") >= 6


def test_report_plots_the_health_center(tmp_path):
    centers = tmp_path / "health_centers.json"
    centers.write_text(json.dumps([{"pose": [2, 3], "radius": 1, "hazard_multiplier": 0.25}]))
    run_dir = scripted_run(tmp_path, "run1", options={"health_centers_path": str(centers)})
    report.generate(run_dir)
    assert (run_dir / "epidemic_analysis" / "health_center.png").exists()
    found = eu.health_centers(eu.load_events(run_dir))
    assert found[0]["pose"] == (2, 3) and found[0]["hazard_multiplier"] == 0.25
    assert eu.care_coverage(found, 10) == 9
    care = eu.care_series(eu.load_frames(run_dir)[1], found, 10)
    assert care and care[-1]["in_care"] == 2  # both beings next to the center


def test_compare_two_runs(tmp_path):
    a = scripted_run(tmp_path, "run_a")
    b = scripted_run(tmp_path, "run_b")
    out = tmp_path / "_comparisons"
    compare.compare([a, b], "average", out)
    summary = json.loads((out / "summary.json").read_text())
    assert [row["run"] for row in summary["table"]] == ["run_a", "run_b"]
    assert (out / "avg_epidemic_curves.png").exists()


def test_sandbox_screen_and_worker(tmp_path):
    run_dir = scripted_run(tmp_path, "run1")
    assert sandbox.check("import os\nprint(os.listdir('.'))")
    assert sandbox.check("print(().__class__)")
    assert not sandbox.check("import epidemic_utils\nprint(1)")
    box = sandbox.Sandbox(run_dir)
    try:
        assert box.run("x = len(eu.load_events(RUN))\nprint(x)").strip().isdigit()
        assert box.run("print(x > 0)").strip() == "True"
        assert box.run("import epidemic_utils\nprint(epidemic_utils.compute_all(RUN)[0]['outbreak']['infections'])").strip() == "2"
        outside = Path.home() / "tl_sandbox_probe.txt"  # the temp dir itself allows writes
        assert "PermissionError" in box.run(f"open({str(outside)!r}, 'w').write('no')")
        assert not outside.exists()
        assert box.run("import os\nprint(1)").startswith("Blocked by the sandbox")
    finally:
        box.close()


def test_dashboard_lists_runs_and_serves_the_report(tmp_path):
    run_dir = scripted_run(tmp_path, "run1")
    client = TestClient(create_app(tmp_path, "claude-opus-5-5"))
    runs = client.get("/api/runs").json()
    assert runs["runs"] == ["run1"] and runs["model"] == "claude-opus-5-5"
    assert client.post("/api/report", json={"run": "run1"}).json() == {"ok": True}
    state = client.get("/api/state", params={"run": "run1"}).json()
    assert state["metrics"]["outbreak"]["infections"] == 2
    assert any(url.endswith("epidemic_curves.png") for url in state["plots"])
    files = client.get("/api/files", params={"run": "run1"}).json()["files"]
    assert "world_state.jsonl" in files and "open_gridworld.log" in files
    assert (run_dir / "epidemic_analysis" / "metrics.json").exists()
