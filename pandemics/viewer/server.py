"""FastAPI app that serves the run viewer.

Run it next to a simulation, not inside it::

    python -m pandemics.viewer                      # serves ./logs on http://127.0.0.1:8000
    python -m pandemics.viewer --logs /data/logs --port 9999

The server stays out of the simulation process on purpose. A run costs real
money in model calls and can last hours. A web server in the same interpreter
is one more thing that can stop it.
"""

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pandemics.viewer.reader import RunReader

STATIC_DIR = Path(__file__).parent / "static"

#: How often the SSE stream re-reads the log directory while a run is live.
POLL_SECONDS = 1.0


def transmission(reader: RunReader) -> dict:
    """The chain of infections and R0 per generation.

    One node per infection, keyed by its id. A node ends when its host heals
    (VIRAL_HEALED with that id) or dies (AGENT_DIED of the host at or after the
    infection). A node without an end is active and is not counted: its host
    can still spread. R0 is the mean number of children over the ended nodes
    of generations 0 and 1.
    """
    nodes: Dict[str, dict] = {}
    for e in reader.events(["VIRAL_INFECTION"]):
        if not e.get("infection_id"):
            continue
        nodes[e["infection_id"]] = {
            "infection": e["infection_id"],
            "source": e.get("source_infection_id"),
            "host": e.get("agent_tag"),
            "t": e.get("timestamp"),
        }
    healed_at = {
        e["infection_id"]: e.get("timestamp")
        for e in reader.events(["VIRAL_HEALED"])
        if e.get("infection_id")
    }
    died_at: Dict[str, list] = {}
    for e in reader.events(["AGENT_DIED"]):
        if e.get("agent_tag") and e.get("timestamp") is not None:
            died_at.setdefault(e["agent_tag"], []).append(e["timestamp"])
    children = Counter(n["source"] for n in nodes.values() if n["source"])

    def generation(infection: str) -> int:
        gen = 0
        seen = set()
        while infection not in seen:
            seen.add(infection)
            parent = nodes[infection]["source"]
            if parent not in nodes:
                return gen
            infection = parent
            gen += 1
        return gen

    def ended_at(node: dict) -> Optional[int]:
        if node["infection"] in healed_at:
            return healed_at[node["infection"]]
        if node["t"] is None:
            return None
        deaths = [t for t in died_at.get(node["host"], []) if t >= node["t"]]
        return min(deaths) if deaths else None

    by_gen: Dict[int, list] = {}
    censored = 0
    for infection, node in nodes.items():
        node["generation"] = generation(infection)
        node["secondary"] = children.get(infection, 0)
        node["ended_at"] = ended_at(node)
        if node["ended_at"] is None:
            censored += 1
        else:
            by_gen.setdefault(node["generation"], []).append(node["secondary"])

    generations = [
        {"generation": g, "cases": len(v), "mean_secondary": sum(v) / len(v)}
        for g, v in sorted(by_gen.items())
    ]
    early = by_gen.get(0, []) + by_gen.get(1, [])
    return {
        "chain": list(nodes.values()),
        "generations": generations,
        "censored": censored,
        "r0": (sum(early) / len(early)) if early else None,
    }


def create_app(logs_root: Path) -> FastAPI:
    app = FastAPI(title="Pandemics run viewer")
    app.state.logs_root = Path(logs_root)
    app.state.readers: Dict[str, RunReader] = {}

    def get_reader(name: str, refresh: bool = True) -> RunReader:
        run_dir = app.state.logs_root / name
        # Guard against ../ escaping the logs root.
        if not run_dir.resolve().is_relative_to(app.state.logs_root.resolve()):
            raise HTTPException(status_code=400, detail="Invalid run name")
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"No such run: {name}")

        reader = app.state.readers.get(name)
        if reader is None:
            if not (run_dir / "world_state.jsonl").exists():
                raise HTTPException(status_code=404, detail=f"{name} has no world_state.jsonl")
            reader = RunReader(run_dir)
            app.state.readers[name] = reader
            refresh = True
        if refresh:
            reader.refresh()
        return reader

    @app.get("/api/runs")
    def list_runs():
        root = app.state.logs_root
        if not root.is_dir():
            return {"runs": [], "logs_root": str(root)}
        runs = []
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "params.json").exists():
                continue
            try:
                reader = get_reader(run_dir.name)
            except HTTPException:
                continue
            params = reader.params
            runs.append(
                {
                    "name": run_dir.name,
                    "description": params.get("run", {}).get("exp_description", ""),
                    "model": params.get("agent", {}).get("model", "unknown"),
                    "grid_size": params.get("env", {}).get("grid_size"),
                    "max_ts": params.get("run", {}).get("max_ts"),
                    "last_step": reader.last_step,
                    "status": reader.status(),
                    "has_viral": reader.meta()["has_viral"],
                }
            )
        # Live runs first, then by name.
        runs.sort(key=lambda r: (r["status"] != "live", r["name"]), reverse=False)
        return {"runs": runs, "logs_root": str(root)}

    @app.get("/api/runs/{name}/meta")
    def run_meta(name: str):
        return get_reader(name).meta()

    @app.get("/api/runs/{name}/step/{t}")
    def run_step(name: str, t: int):
        reader = get_reader(name)
        world = reader.world_at(t)
        if world is None:
            raise HTTPException(status_code=404, detail=f"No state at step {t}")
        world["chat"] = reader.chat(t, t)
        world["ticks"] = {
            tag: reader.agent_tick(tag, t) for tag in world["agents"]
        }
        return world

    @app.get("/api/runs/{name}/trail/{tag}")
    def agent_trail(name: str, tag: str, start: int = Query(0, ge=0), end: int = 0):
        """Just the positions, so the map can draw a trail without pulling
        a full world state for every step behind the current one."""
        reader = get_reader(name, refresh=False)
        points = []
        for t in range(max(0, start), end + 1):
            world = reader._steps.get(t)
            a = world and world.get("agents", {}).get(tag)
            if a:
                points.append([t, a[0], a[1]])
        return {"agent_tag": tag, "points": points}

    @app.get("/api/runs/{name}/agent/{tag}")
    def agent_history(
        name: str,
        tag: str,
        start: int = Query(0, ge=0),
        end: Optional[int] = None,
    ):
        reader = get_reader(name)
        end = reader.last_step if end is None else end
        ticks = [reader.agent_tick(tag, t) for t in range(start, end + 1)]
        return {"agent_tag": tag, "ticks": [t for t in ticks if t]}

    @app.get("/api/runs/{name}/chat")
    def run_chat(name: str, start: int = Query(0, ge=0), end: Optional[int] = None):
        reader = get_reader(name)
        end = reader.last_step if end is None else end
        return {"messages": reader.chat(start, end)}

    @app.get("/api/runs/{name}/artifacts")
    def run_artifacts(name: str):
        return {"artifacts": get_reader(name).artifacts()}

    @app.get("/api/runs/{name}/events")
    def run_events(name: str, types: Optional[str] = None):
        reader = get_reader(name)
        wanted = types.split(",") if types else None
        return {"events": reader.events(wanted)}

    @app.get("/api/runs/{name}/series")
    def run_series(name: str):
        reader = get_reader(name)
        return {**reader.series(), "tokens": reader.token_series()}

    @app.get("/api/runs/{name}/viral")
    def run_viral(name: str):
        return transmission(get_reader(name))

    @app.get("/api/runs/{name}/stream")
    async def stream(name: str, since: int = -1):
        """Server-sent events: one message per newly written step."""

        async def gen():
            last = since
            idle = 0
            while True:
                reader = get_reader(name)
                latest = reader.last_step
                if latest > last:
                    idle = 0
                    meta = reader.meta()
                    payload = {
                        "last_step": latest,
                        # The world log is written after the step, so it runs one
                        # ahead of the decisions that produced it. A live viewer
                        # wants the newest frame where the map, the chat and the
                        # beings' reasoning all describe the same instant.
                        "last_decision_step": meta["last_decision_step"],
                        # An outbreak can start after the viewer opened the run;
                        # without this the client's has_viral stays stale and the
                        # transmission panel never appears.
                        "has_viral": meta["has_viral"],
                        "status": reader.status(),
                        # so the spend chart moves during a live run
                        "series": {**reader.series(), "tokens": reader.token_series()},
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    last = latest
                else:
                    idle += 1
                    if reader.status() == "finished":
                        done = {"last_step": latest, "status": "finished"}
                        yield f"data: {json.dumps(done)}\n\n"
                        return
                    if idle % 15 == 0:  # keep proxies from closing the connection
                        yield ": keepalive\n\n"
                await asyncio.sleep(POLL_SECONDS)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        type=Path,
        default=Path(os.environ.get("TL_LOGS_DIR") or Path.cwd() / "logs"),
        help="Folder holding the run folders (default: TL_LOGS_DIR, else logs/ under the working directory)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Pandemics run viewer: http://{args.host}:{args.port}")
    print(f"reading runs from {args.logs}")
    uvicorn.run(
        create_app(args.logs), host=args.host, port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
