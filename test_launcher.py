"""Tests for the terralingua_launcher package. Run from the repo root:

    python test_launcher.py

No LLM calls; the designer is tested at the parsing layer only.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent

from terralingua_launcher import args as argsmod  # noqa: E402
from terralingua_launcher import designer, prompts, store  # noqa: E402
from terralingua_launcher.procs import ProcRegistry  # noqa: E402


def test_introspect():
    out = subprocess.run(
        [sys.executable, str(ROOT / "terralingua_launcher" / "introspect.py")],
        cwd=ROOT,
        capture_output=True,
        timeout=180,
    )
    schema = json.loads(out.stdout.decode())
    assert not schema["errors"], schema["errors"]
    keys = [g["key"] for g in schema["groups"]]
    assert keys == ["agent", "env", "run"], keys
    index = {p["name"]: p for g in schema["groups"] for p in g["params"]}
    assert index["genome"]["choices"], "genome should carry choices"
    assert index["use_colors"]["type"] == "bool"
    assert index["food_zones"]["autocoerce"] == "food_zones"
    assert index["ports"]["nargs"] is True
    assert index["prompt_templates"]["optional"] is True
    assert "empty_countdown" not in index, "excluded params must not surface"
    assert schema["extras"]["model_suggestions"]
    print("PASS: introspection")
    return schema


def test_args(schema):
    build = lambda values, **kw: argsmod.build_argv(schema, values, **kw)  # noqa: E731
    assert build({}) == ["main.py"]
    # value equal to default is not emitted
    assert build({"grid_size": 50}) == ["main.py"]
    assert build({"grid_size": 80}) == ["main.py", "--grid_size", "80"]
    # int/float equivalence does not emit
    assert build({"food_decay_rate": 0.05}) == ["main.py"]
    # booleans two-sided
    assert build({"use_colors": True}) == ["main.py", "--use_colors"]
    assert build({"burials": True}) == ["main.py", "--burials"]
    assert build({"save_video": False}) == ["main.py", "--no-save_video"]
    # autocoerce: int and pair-list forms
    zones = ["main.py", "--food_zones", "10,10", "12,5"]
    assert build({"food_zones": 4}) == ["main.py", "--food_zones", "4"]
    assert build({"food_zones": "10,10 12,5"}) == zones
    assert build({"food_zones": [[10, 10], [12, 5]]}) == zones
    # nargs list
    assert build({"ports": [8000, 8001]}) == ["main.py", "--ports", "8000", "8001"]
    assert build({"ports": "8000 8001"}) == ["main.py", "--ports", "8000", "8001"]
    # default-equal list not emitted
    assert build({"ports": [9000, 9001, 9002, 9003, 9010, 9011, 9012]}) == ["main.py"]
    # empty / None dropped
    assert build({"personas": "", "exp_description": None}) == ["main.py"]
    assert build({"personas": "p.json"}) == ["main.py", "--personas", "p.json"]
    assert build({}, resume=True) == ["main.py", "--resume"]
    cmd = argsmod.command_string("/usr/bin/python", ["main.py", "--exp_name", "a b"])
    assert cmd == "/usr/bin/python main.py --exp_name 'a b'"
    print("PASS: argv building")


def test_prompts():
    info = prompts.extract_templates(ROOT)
    assert info["supports_override"] is True
    assert "an autonomous living being" in info["sys_prompt"]
    assert "Reply Format" in info["agent_prompt"]
    ph = prompts.placeholders(info["sys_prompt"])
    assert {"agent_name", "persona", "use_internal_memory"} <= ph, ph
    # a rewrite that drops a placeholder is flagged
    issues = prompts.validate_rewrite(
        info["sys_prompt"], "You are {{ agent_name }}.", "sys"
    )
    assert any("dropped placeholders" in i for i in issues)
    assert prompts.validate_rewrite(info["sys_prompt"], info["sys_prompt"], "sys") == []
    bad_jinja = prompts.validate_rewrite(None, "{% if x %}", "t")
    assert any("not valid jinja" in i for i in bad_jinja)
    # unknown filters fail at compile, not at launch inside the sim
    bad_filter = prompts.validate_rewrite(None, "{{ x|nosuchfilter }}", "t")
    assert any("not valid jinja" in i for i in bad_filter), bad_filter
    # artifact rules mirror the env loader
    bad = [
        {"name": "a", "pose": [1, 1], "agent": "being0"},
        {"name": "b", "type": "wrong"},
        {"name": "c", "pose": [99, 0]},
        {"name": "d", "lifespan": 0},
        {"name": "e", "type": "health_center", "agent": "being0"},
    ]
    issues = prompts.validate_artifacts(bad, grid_size=50)
    assert len(issues) == 5, issues
    assert prompts.validate_artifacts([{"name": "ok", "pose": [1, 1]}], 50) == []
    assert prompts.validate_personas([{"persona": ""}, "fine"]) != []
    # LLM-shaped garbage must produce issues, never exceptions
    garbage = [
        {"name": "s", "step": "not_a_number", "lifespan": "x"},
        {"name": "p", "pose": ["a", 2]},
        {"name": "q", "pose": [None, None]},
        {"name": "hc", "type": "health_center", "heal_probability": 1.4,
         "hazard_multiplier": "high", "radius": -1},
    ]
    issues = prompts.validate_artifacts(garbage, 50)
    assert len(issues) == 7, issues
    # a placeholder neutralized by a comment or kept inside an expression
    assert "genome" in prompts.placeholders("{{ genome }}")
    assert "genome" not in prompts.placeholders("{# {{ genome }} #}")
    assert "energy" in prompts.placeholders("{{ 'low' if energy < 5 else energy }}")
    print("PASS: prompt extraction & validation")


def test_designer_parsing():
    fenced = 'notes\n```json\n{"sys_prompt": "s"}\n```'
    assert designer._extract_json(fenced) == {"sys_prompt": "s"}
    assert designer._extract_json('{"a": 1}') == {"a": 1}
    # literal newline inside a JSON string (json mode not enforced)
    assert designer._extract_json('{"a": "line1\nline2"}') == {"a": "line1\nline2"}
    # prose (with braces) around the object
    wrapped = 'Use {x} wisely. {"a": 1} Hope this helps!'
    assert designer._extract_json(wrapped) == {"a": 1}
    # an example object before the real design must not win
    two = 'Example: {"foo": 1}\nFinal:\n```json\n{"sys_prompt": "real"}\n```'
    assert designer._extract_json(two) == {"sys_prompt": "real"}
    cleaned = designer._clean(
        {
            "sys_prompt": " s ",
            "agent_prompt": "a",
            "personas": ["plain string", {"persona": "p", "name": None, "count": 3}],
            "init_artifacts": [
                {"name": "x", "type": "text", "payload": None, "pose": [1, 2]},
                {"no_name": True},
            ],
            "suggested_params": None,
            "design_notes": None,
        }
    )
    assert cleaned["sys_prompt"] == "s"
    assert cleaned["personas"][0] == {"persona": "plain string"}
    assert cleaned["personas"][1] == {"persona": "p", "count": 3}
    assert cleaned["init_artifacts"] == [{"name": "x", "type": "text", "pose": [1, 2]}]
    assert cleaned["suggested_params"] == []
    # LLM-shaped garbage: non-numeric count, non-dict suggested_params
    messy = designer._clean(
        {
            "personas": [{"persona": "p", "count": "two"}],
            "suggested_params": [
                "not a dict", {"no_name": 1}, {"name": "grid_size", "value": 80},
            ],
        }
    )
    assert messy["personas"] == [{"persona": "p"}]
    assert messy["suggested_params"] == [{"name": "grid_size", "value": 80, "why": ""}]
    print("PASS: designer parsing")


def test_store():
    assert store.looks_like_tl_repo(ROOT)
    p = store.safe_path(ROOT, "launcher_configs/x.json")
    assert p.is_relative_to(ROOT)
    try:
        store.safe_path(ROOT, "../outside.json")
        raise AssertionError("escape not rejected")
    except ValueError:
        pass
    files = store.find_json_files(ROOT, ["persona"])
    assert any("personas_example.json" in f for f in files), files
    print("PASS: store & path safety")


def test_procs():
    reg = ProcRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        child_src = "print('hello from child'); import time; time.sleep(30)"
        proc = reg.spawn(
            "sim", "hello world!", [sys.executable, "-u", "-c", child_src], Path(tmp)
        )
        assert proc.status() == "running"
        deadline = time.time() + 5
        text = ""
        offset = 0
        while time.time() < deadline and "hello from child" not in text:
            r = reg.read_log(proc.id, offset)
            offset = r["offset"]
            text += r["text"]
            time.sleep(0.1)
        assert "hello from child" in text
        assert reg.stop(proc.id)
        deadline = time.time() + 5
        while time.time() < deadline and proc.popen.poll() is None:
            time.sleep(0.1)
        assert proc.popen.poll() is not None
        assert reg.list()[0]["status"] in ("stopped", "exited (-15)")
    print("PASS: process registry")


def test_server():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("SKIP: server API (no httpx/testclient)")
        return
    from terralingua_launcher.server import create_app

    client = TestClient(create_app(repo=ROOT, python=sys.executable))
    s = client.get("/api/settings").json()
    assert s["repo_ok"] and s["python_ok"]
    schema = client.get("/api/schema").json()
    assert [g["key"] for g in schema["groups"]] == ["agent", "env", "run"]
    prev = client.post(
        "/api/preview", json={"values": {"grid_size": 80, "burials": True}}
    ).json()
    assert "--grid_size 80" in prev["cmd"] and "--burials" in prev["cmd"]
    # path guards
    assert client.get("/api/file", params={"path": "../secrets"}).status_code == 400
    assert client.get("/api/file", params={"path": ".env"}).status_code == 400
    assert client.get("/api/file", params={"path": "main.py"}).status_code == 400
    r = client.get("/api/file", params={"path": "personas_example.json"})
    assert r.status_code == 200
    r = client.post("/api/file", json={"path": "core/hack.py", "content": "{}"})
    assert r.status_code == 400, "must refuse non-json writes"
    r = client.post(
        "/api/file", json={"path": "launcher_configs/t.json", "content": "not json"}
    )
    assert r.status_code == 400
    # launch preflight refuses a missing personas file
    r = client.post("/api/launch", json={"values": {"personas": "nope_missing.json"}})
    assert r.status_code == 400
    p = client.get("/api/prompts").json()
    assert p["supports_override"] and p["placeholders"]["sys_prompt"]
    # path completion: names only, dirs get a trailing slash
    fs = client.get("/api/fs", params={"prefix": str(ROOT / "co")}).json()
    assert str(ROOT / "core") + "/" in fs["paths"], fs
    fs = client.get(
        "/api/fs", params={"prefix": str(ROOT) + "/", "dirs_only": True}
    ).json()
    assert all(x.endswith("/") for x in fs["paths"]), fs
    print("PASS: server API")


if __name__ == "__main__":
    schema = test_introspect()
    test_args(schema)
    test_prompts()
    test_designer_parsing()
    test_store()
    test_procs()
    test_server()
    print("ALL LAUNCHER TESTS PASSED")
