"""Sanity checks for the anthropologist's sandbox and dashboard
(scenarios/ebola_simulation/anthropologist/).

Run from the repo root with: python tests/test_anthropologist_sandbox.py

No LLM calls: the sandbox runs handwritten snippets, the dashboard is hit
through FastAPI's TestClient with the chat left untouched.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scenarios/ebola_simulation/anthropologist"))
sys.path.insert(0, str(ROOT / "tests"))
import sandbox as sb  # noqa: E402
from test_epidemic_analysis import build_epidemic_run  # noqa: E402


def test_ast_screen():
    blocked = [
        "import os",
        "from subprocess import run",
        "import socket",
        "import urllib.request",
        "x = ().__class__",
        "getattr(json, 'loads')",
        "eval('1+1')",
        "exec('pass')",
        "__import__('os')",
    ]
    for code in blocked:
        assert sb.check(code), f"should be blocked: {code}"
    allowed = [
        "import numpy as np\nprint(np.mean([1, 2, 3]))",
        "from collections import Counter\nprint(Counter('aa'))",
        "import matplotlib.pyplot as plt",
        "print(json.dumps({'a': 1}))",
    ]
    for code in allowed:
        assert not sb.check(code), f"should be allowed: {code} -> {sb.check(code)}"
    print("PASS: AST screen blocks os/subprocess/socket/dunder/eval, "
          "allows analysis code")


def test_worker(run_dir: Path):
    box = sb.Sandbox(run_dir)
    try:
        assert "Blocked by the sandbox" in box.run("import os\nprint(os.getcwd())")
        print("PASS: blocked code never reaches the worker")

        box.run("x = 41")
        assert box.run("print(x + 1)").strip() == "42"
        print("PASS: namespace persists between calls")

        out = box.run("print(open('/etc/hosts').read())")
        assert "PermissionError" in out and "denied" in out, out
        out = box.run("open('/tmp/../etc/thing', 'w')")
        assert "PermissionError" in out, out
        out = box.run(
            "p = RUN / 'epidemic_analysis' / 'chat' / 'note.txt'\n"
            "p.write_text('ok')\nprint(p.read_text())"
        )
        assert out.strip() == "ok", out
        print("PASS: reads/writes confined (run+repo read, chat dir write)")

        out = box.run(
            "p = RUN / 'epidemic_analysis' / 'chat' / 'note.txt'\np.unlink()"
        )
        assert "PermissionError" in out and "disabled" in out, out
        print("PASS: deletion is disabled even inside the chat dir")

        out = box.run("i = 0\nwhile True: i += 1", timeout=2)
        assert "TimeoutError" in out or "Killed" in out, out
        assert box.run("print('alive')").strip() == "alive"
        print("PASS: runaway code times out and the session survives")

        out = box.run(
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\nax.plot([1, 2], [3, 4])\n"
            "out = RUN / 'epidemic_analysis' / 'chat' / 'smoke.png'\n"
            "fig.savefig(out)\nprint(out.exists())"
        )
        assert out.strip().endswith("True"), out
        print("PASS: matplotlib renders and saves under the chat dir")

        out = box.run("m, s, i, e = eu.compute_all(RUN)\n"
                      "print(m['outbreak']['infections'])")
        assert int(out.strip()) >= 3, out
        print("PASS: epidemic_utils works inside the worker")
    finally:
        box.close()


def test_dashboard(run_dir: Path):
    import dashboard
    from fastapi.testclient import TestClient

    run = run_dir.name
    client = TestClient(
        dashboard.create_app(run_dir.parent, "claude-opus-5", run))
    assert client.get("/").status_code == 200
    runs = client.get("/api/runs").json()
    assert run in runs["runs"] and runs["initial"] == run
    state = client.get(f"/api/state?run={run}").json()
    assert state["run"] == run
    assert client.post("/api/report", json={"run": run}).status_code == 200
    state = client.get(f"/api/state?run={run}").json()
    assert state["metrics"]["outbreak"]["infections"] >= 3
    assert any("epidemic_curves" in u for u in state["plots"])
    for url in state["plots"]:
        assert client.get(url).status_code == 200
    assert client.get(f"/plots/{run}/report/../params.json").status_code == 404
    assert client.get(f"/plots/{run}/report/nope.png").status_code == 404
    assert client.get("/api/state?run=..").status_code in (400, 404)
    ev = client.get(f"/api/events?run={run}").json()
    assert ev == {"events": [], "busy": False, "auto_run": False,
                  "model": "claude-opus-5"}
    assert client.post("/api/autorun",
                       json={"run": run, "enabled": True}).json()["auto_run"]
    assert "claude-sonnet-5" in runs["models"]
    r = client.post("/api/model", json={"run": run, "model": "claude-sonnet-5"})
    assert r.json()["model"] == "claude-sonnet-5"
    assert client.get(f"/api/events?run={run}").json()["model"] == "claude-sonnet-5"
    assert client.post("/api/model",
                       json={"run": run, "model": "gpt-9"}).status_code == 400
    print("PASS: dashboard serves runs, state, report, plots, model switch; "
          "rejects traversal")


def test_approval_gate(run_dir: Path):
    import threading
    import time

    import dashboard

    st = dashboard.State(run_dir, "claude-opus-5")
    try:
        results = {}
        execute = dashboard._executor(st)

        def ask(key, code):
            results[key] = execute(code)

        t = threading.Thread(target=ask, args=("denied", "print('hi')"))
        t.start()
        while not st.pending:
            time.sleep(0.01)
        (gate, decision), = st.pending.values()
        decision[0] = False
        gate.set()
        t.join(5)
        assert results["denied"] == dashboard.DENIED
        assert st.events[-1]["status"] == "denied"

        t = threading.Thread(target=ask, args=("approved", "print('hi')"))
        t.start()
        while not st.pending:
            time.sleep(0.01)
        (gate, decision), = st.pending.values()
        decision[0] = True
        gate.set()
        t.join(30)
        assert results["approved"].strip() == "hi"

        st.auto_run = True
        assert execute("print(2 * 2)").strip() == "4"
        print("PASS: approval gate — denied code never runs, approve and auto-run do")
    finally:
        st.sandbox.close()


def test_file_tools(run_dir: Path):
    import json

    import filetools as ft

    scope = ft.Scope(run_dir, ROOT)
    assert "outside the run and the repo" in ft.read_file(scope, "/etc/hosts")
    assert "not found" in ft.read_file(scope, "no_such_file.txt")
    assert "TerraLingua" in ft.read_file(scope, "README.md"), "repo fallback"
    assert "world_state.jsonl" in ft.list_files(scope, ".")

    out = ft.read_file(scope, "world_state.jsonl", offset=2, limit=1)
    assert out.startswith("2\t"), out

    log = run_dir / "agent_logs"
    log.mkdir(exist_ok=True)
    (log / "x.jsonl").write_text(json.dumps(
        {"action": "move", "input_prompt": "SECRETWORD " + "x" * 15000}) + "\n")
    out = ft.read_file(scope, "agent_logs/x.jsonl")
    assert "…stripped…" in out and "SECRETWORD" not in out
    assert ft.grep_files(scope, "SECRETWORD", "agent_logs").startswith("no matches")
    assert "action" in ft.grep_files(scope, "move", "agent_logs", "*.jsonl")
    assert "VIRAL_INFECTION" in ft.grep_files(
        scope, "VIRAL_INFECTION", "open_gridworld.log")

    (run_dir / "blob.png").write_bytes(b"\x89PNG\x00\x00")
    assert "binary" in ft.read_file(scope, "blob.png")
    print("PASS: file tools — confinement, repo fallback, input_prompt "
          "stripping, binary refusal")


def test_field_notes(run_dir: Path):
    import agent

    system = agent.build_system(run_dir)
    assert "(none yet)" in system
    reply = agent.write_note.func(note="- The index case never moved after day 3.")
    assert reply.startswith("noted")
    system = agent.build_system(run_dir)
    assert "index case never moved" in system
    print("PASS: field notes persist and reload into the system prompt")


def test_interrupt(run_dir: Path):
    import threading
    import time

    box = sb.Sandbox(run_dir)
    try:
        result = {}

        def slow():
            result["out"] = box.run("i = 0\nwhile True: i += 1", timeout=25)

        t = threading.Thread(target=slow)
        t.start()
        time.sleep(0.5)
        box.interrupt()
        t.join(15)
        assert not t.is_alive() and "Killed" in result["out"], result
        assert box.run("print('back')").strip() == "back"
        print("PASS: interrupt kills an in-flight call, session respawns")
    finally:
        box.close()


def test_session_persistence(run_dir: Path):
    import dashboard
    from fastapi.testclient import TestClient

    run = run_dir.name
    st = dashboard.State(run_dir, "claude-opus-5")
    st.add(type="user", text="who was patient zero?")
    st.add(type="assistant", text="Amara.")
    st.messages = [{"role": "user", "content": "who was patient zero?"}]
    st.save()

    st2 = dashboard.State(run_dir, "claude-opus-5")
    assert st2.messages == st.messages
    assert [e["type"] for e in st2.events] == ["user", "assistant", "notice"]
    assert "Restored previous session (1 questions)" in st2.events[-1]["text"]

    client = TestClient(
        dashboard.create_app(run_dir.parent, "claude-opus-5", run))
    assert len(client.get(f"/api/events?run={run}").json()["events"]) == 3
    files = client.get(f"/api/files?run={run}").json()["files"]
    assert "world_state.jsonl" in files
    assert client.post("/api/stop", json={"run": run}).status_code == 200
    app_state = client.app.state.states[run]
    app_state.stop_requested = False
    assert client.post("/api/reset", json={"run": run}).status_code == 200
    assert client.get(f"/api/events?run={run}").json()["events"] == []
    assert not app_state.session_path.exists()
    print("PASS: session persists, restores, resets; files endpoint works")


def test_compare(run_a: Path, run_b: Path):
    import compare as cm
    import dashboard
    from fastapi.testclient import TestClient

    out = run_a.parent / "_comparisons"
    for mode, expected in [("average", "avg_epidemic_curves.png"),
                           ("sidebyside", "cmp_curves.png")]:
        result = cm.compare([run_a, run_b], mode, out)
        assert len(result["table"]) == 2
        assert all(row["infections"] >= 1 for row in result["table"])
        for name in result["plots"]:
            assert (out / name).stat().st_size > 0
        assert expected in result["plots"]
    try:
        cm.compare([run_a] * 7, "sidebyside", out)
        raise AssertionError("7 side-by-side runs should be rejected")
    except ValueError:
        pass

    client = TestClient(
        dashboard.create_app(run_a.parent, "claude-opus-5", run_a.name))
    r = client.post("/api/compare",
                    json={"runs": [run_a.name, run_b.name], "mode": "average"})
    assert r.status_code == 200
    for url in r.json()["plots"]:
        assert client.get(url).status_code == 200
    assert client.post("/api/compare",
                       json={"runs": [run_a.name]}).status_code == 400
    print("PASS: compare — averaging and side-by-side plots, table, endpoint")


def main():
    test_ast_screen()
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "epidemic_test"
        build_epidemic_run(run, steps=40)
        run_b = Path(tmp) / "epidemic_test_b"
        build_epidemic_run(run_b, steps=30, seed=7)
        test_worker(run)
        test_dashboard(run)
        test_approval_gate(run)
        test_file_tools(run)
        test_field_notes(run)
        test_interrupt(run)
        test_session_persistence(run)
        test_compare(run, run_b)


if __name__ == "__main__":
    main()
