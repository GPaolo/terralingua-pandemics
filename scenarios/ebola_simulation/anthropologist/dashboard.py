"""Web dashboard for viral runs: report metrics + plots + the anthropologist chat.

    python scenarios/ebola_simulation/anthropologist/dashboard.py [logs_root|run_dir]
    (serves http://127.0.0.1:8010; --port/--host/--model to change)

Serves every run under the logs root; pick the active run in the UI, or
select several and compare them (seed averaging or side-by-side). Chat needs
ANTHROPIC_API_KEY or a .env; the metrics/plots panes work without.
Model-written code runs in the sandbox and, unless auto-run is toggled on in
the UI, waits for an Approve click first.
"""

import argparse
import json
import re
import sys
import threading
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent  # noqa: E402
import compare as compare_mod  # noqa: E402
import report  # noqa: E402
from sandbox import Sandbox  # noqa: E402

DENIED = ("The user declined to run this code. Explain what it would have "
          "done, or try a different approach.")
INTERRUPTED = "The user interrupted the turn — this code did not run."
APPROVAL_TIMEOUT = 600
SAFE_NAME = re.compile(r"^[\w.-]+\.png$")
SAFE_RUN = re.compile(r"^[\w.-]+$")


def _jsonable(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else str(obj)


class State:
    """One run's chat session; the sandbox worker spawns on first use."""

    def __init__(self, run_dir, model):
        self.run_dir = run_dir
        self.model = model
        self._sandbox = None
        self.scope = agent.make_scope(run_dir)
        self.system = agent.build_system(run_dir)
        try:
            self.client = anthropic.Anthropic()
        except anthropic.AnthropicError:
            self.client = None  # metrics/plots still work; chat reports it
        self.messages = []
        self.events = []
        self.lock = threading.Lock()
        self.pending = {}  # event id -> (threading.Event, [decision])
        self.busy = False
        self.auto_run = False
        self.stop_requested = False
        self.session_path = run_dir / "epidemic_analysis" / "chat_session.json"
        self._restore()

    @property
    def sandbox(self):
        if self._sandbox is None:
            self._sandbox = Sandbox(self.run_dir)
        return self._sandbox

    def shutdown(self):
        if self._sandbox is not None:
            self._sandbox.close()

    def _restore(self):
        if not self.session_path.exists():
            return
        try:
            data = json.loads(self.session_path.read_text())
            self.events = data["events"]
            self.messages = data["messages"]
        except (json.JSONDecodeError, KeyError):
            return
        asked = sum(1 for e in self.events if e.get("type") == "user")
        if asked:
            self.add(type="notice",
                     text=f"Restored previous session ({asked} questions). "
                          "Sandbox variables were not restored.")

    def save(self):
        with self.lock:
            payload = json.dumps(
                {"events": self.events, "messages": self.messages},
                default=_jsonable,
            )
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_path.write_text(payload)

    def add(self, **event):
        with self.lock:
            event["id"] = len(self.events)
            self.events.append(event)
            return event

    def chat_plots(self):
        d = self.run_dir / "epidemic_analysis" / "chat"
        return set(d.glob("*.png")) if d.is_dir() else set()


def _executor(st: State):
    def run(code):
        if st.stop_requested:
            st.add(type="code", code=code, status="interrupted", output="")
            return INTERRUPTED
        event = st.add(type="code", code=code, status="pending", output="")
        if st.auto_run:
            approved = True
        else:
            gate = threading.Event()
            decision = [False]
            st.pending[event["id"]] = (gate, decision)
            gate.wait(APPROVAL_TIMEOUT)
            st.pending.pop(event["id"], None)
            approved = decision[0]
        if st.stop_requested:
            event["status"] = "interrupted"
            return INTERRUPTED
        if not approved:
            event["status"] = "denied"
            return DENIED
        event["status"] = "running"
        before = st.chat_plots()
        out = st.sandbox.run(code)
        event["status"] = "done"
        event["output"] = out
        new = sorted(p.name for p in st.chat_plots() - before)
        if new:
            st.add(type="plots",
                   urls=[f"/plots/{st.run_dir.name}/chat/{n}" for n in new])
        return out
    return run


def _turn(st: State, question):
    st.stop_requested = False
    st.add(type="user", text=question)
    try:
        last = agent.run_turn(st.client, st.model, st.system, st.messages,
                              question, _executor(st),
                              on_text=lambda t: st.add(type="assistant", text=t),
                              on_tool=lambda t: st.add(type="tool", text=t),
                              should_stop=lambda: st.stop_requested,
                              scope=st.scope)
        if st.stop_requested:
            st.add(type="notice", text="Turn interrupted.")
        elif last is not None and last.stop_reason == "refusal":
            st.add(type="error", text="The model declined this question.")
    except anthropic.APIError as e:
        st.add(type="error", text=f"API error: {e}. The turn was dropped — re-ask.")
    finally:
        st.busy = False
        st.add(type="done")
        st.save()


def create_app(logs_root: Path, model: str, initial_run: str = None) -> FastAPI:
    app = FastAPI(title="Ebola anthropologist")
    states = {}
    states_lock = threading.Lock()
    app.state.states = states

    def run_dir_of(name: str) -> Path:
        if not name or not SAFE_RUN.match(name):
            raise HTTPException(400, "bad run name")
        run_dir = (logs_root / name).resolve()
        if not run_dir.is_relative_to(logs_root.resolve()) \
                or not (run_dir / "world_state.jsonl").exists():
            raise HTTPException(404, f"no such run: {name}")
        return run_dir

    def state_of(name: str) -> State:
        run_dir = run_dir_of(name)
        with states_lock:
            if name not in states:
                states[name] = State(run_dir, model)
            return states[name]

    @app.get("/")
    def index():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/api/runs")
    def runs():
        found = sorted(
            (p.parent for p in logs_root.glob("*/world_state.jsonl")),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        names = [p.name for p in found]
        return {"runs": names, "model": model,
                "initial": initial_run if initial_run in names
                else (names[0] if names else None)}

    @app.get("/api/state")
    def state(run: str):
        import epidemic_utils as eu

        run_dir = run_dir_of(run)
        metrics_path = run_dir / "epidemic_analysis" / "metrics.json"
        metrics = (json.loads(metrics_path.read_text())
                   if metrics_path.exists() else None)
        plots = sorted(
            p.name for p in (run_dir / "epidemic_analysis").glob("*.png")
        ) if (run_dir / "epidemic_analysis").is_dir() else []
        return {
            "run": run,
            "params": eu.load_params(run_dir).get("env", {}),
            "metrics": metrics,
            "plots": [f"/plots/{run}/report/{n}" for n in plots],
        }

    @app.post("/api/report")
    def regenerate(body: dict):
        report.generate(run_dir_of(body.get("run")))
        return {"ok": True}

    @app.post("/api/compare")
    def compare(body: dict):
        names = body.get("runs") or []
        if len(names) < 2:
            raise HTTPException(400, "pick at least two runs")
        dirs = [run_dir_of(n) for n in names]
        out = logs_root / "_comparisons"
        try:
            result = compare_mod.compare(dirs, body.get("mode", "average"), out)
        except ValueError as e:
            raise HTTPException(400, str(e))
        result["plots"] = [f"/compare_plots/{n}" for n in result["plots"]]
        return result

    @app.post("/api/chat")
    def chat(body: dict):
        st = state_of(body.get("run"))
        question = (body.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "empty question")
        if st.client is None:
            raise HTTPException(500, "no API credentials — set ANTHROPIC_API_KEY")
        with st.lock:
            if st.busy:
                raise HTTPException(409, "a turn is already running")
            st.busy = True
        threading.Thread(target=_turn, args=(st, question), daemon=True).start()
        return {"ok": True}

    @app.get("/api/events")
    def events(run: str):
        st = state_of(run)
        with st.lock:
            return {"events": list(st.events), "busy": st.busy,
                    "auto_run": st.auto_run}

    @app.post("/api/approve")
    def approve(body: dict):
        st = state_of(body.get("run"))
        entry = st.pending.get(body.get("id"))
        if entry is None:
            raise HTTPException(404, "nothing pending under that id")
        gate, decision = entry
        decision[0] = bool(body.get("approved"))
        gate.set()
        return {"ok": True}

    @app.post("/api/autorun")
    def autorun(body: dict):
        st = state_of(body.get("run"))
        st.auto_run = bool(body.get("enabled"))
        return {"auto_run": st.auto_run}

    @app.post("/api/stop")
    def stop(body: dict):
        st = state_of(body.get("run"))
        st.stop_requested = True
        for gate, decision in list(st.pending.values()):
            decision[0] = False
            gate.set()
        if st._sandbox is not None:
            st._sandbox.interrupt()
        return {"ok": True}

    @app.post("/api/reset")
    def reset(body: dict):
        st = state_of(body.get("run"))
        with st.lock:
            if st.busy:
                raise HTTPException(409, "stop the running turn first")
        st.events.clear()
        st.messages.clear()
        st.session_path.unlink(missing_ok=True)
        if st._sandbox is not None:
            st._sandbox.reset()
        return {"ok": True}

    @app.get("/api/files")
    def files(run: str):
        run_dir = run_dir_of(run)
        out = []
        for p in sorted(run_dir.rglob("*")):
            if len(out) >= 500:
                break
            if p.is_file() and not p.name.startswith(".") \
                    and "frames" not in p.parts and "__pycache__" not in p.parts:
                out.append(str(p.relative_to(run_dir)))
        return {"files": out}

    @app.get("/plots/{run}/{kind}/{name}")
    def plot(run: str, kind: str, name: str):
        run_dir = run_dir_of(run)
        if kind not in ("report", "chat") or not SAFE_NAME.match(name):
            raise HTTPException(404)
        base = run_dir / "epidemic_analysis" / ("chat" if kind == "chat" else "")
        path = (base / name).resolve()
        if not path.is_relative_to(run_dir) or not path.exists():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/compare_plots/{name}")
    def compare_plot(name: str):
        if not SAFE_NAME.match(name):
            raise HTTPException(404)
        path = logs_root / "_comparisons" / name
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path)

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?",
                        default=HERE.parents[2] / "logs",
                        help="logs root, or one run dir to open first")
    parser.add_argument("--model", default=agent.DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    target = args.path.resolve()
    if (target / "world_state.jsonl").exists():
        logs_root, initial = target.parent, target.name
    elif target.is_dir():
        logs_root, initial = target, None
    else:
        sys.exit(f"{target} is neither a logs root nor a run directory")

    print(f"🩺 Ebola anthropologist → http://{args.host}:{args.port}  ({logs_root})")
    uvicorn.run(create_app(logs_root, args.model, initial),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
