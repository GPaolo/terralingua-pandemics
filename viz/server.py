"""FastAPI app serving the experiment dashboard.

Run it alongside an experiment rather than inside it::

    python -m viz                    # serves ./logs on http://127.0.0.1:8000
    python -m viz --logs /data/logs --port 9999

Keeping the server out of the simulation process is deliberate: a run costs real
money in LLM calls and can last hours, and a web server sharing its interpreter is
one more thing that can wedge it.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.utils.generic import ROOT
from viz.backfill import backfill
from viz.reader import RunReader

STATIC_DIR = Path(__file__).parent / "static"

#: How often the SSE stream re-reads the log directory while a run is live.
POLL_SECONDS = 1.0


def create_app(logs_root: Path) -> FastAPI:
    app = FastAPI(title="TerraLingua Dashboard")
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
            # Runs predating world_state.jsonl are reconstructed on first open.
            if not (run_dir / "world_state.jsonl").exists():
                try:
                    backfill(run_dir)
                except Exception as e:
                    print(f"Backfill failed for {name}: {e}")
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
                    "provenance": reader.meta()["provenance"],
                    "has_viral": reader.meta()["has_viral"],
                }
            )
        # Live runs first, then most recently active.
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
        return {
            **reader.series(),
            "tokens": reader.token_totals(),
            "artifacts": [
                {"t": a.get("created_at", 0), "name": a.get("name")}
                for a in reader.artifacts()
            ],
        }

    @app.get("/api/runs/{name}/viral")
    def run_viral(name: str):
        """Transmission chain and R0 per generation.

        Mirrors ``analysis_scripts/compute_r0.py``: each infection is one host
        episode, and its secondary cases are the events naming its artifact as
        the source.
        """
        reader = get_reader(name)
        infections = reader.events(["VIRAL_INFECTION"])
        removed = {
            e["artifact"]["name"]
            for e in reader.events(["ARTIFACT_REMOVED"])
            if isinstance(e.get("artifact"), dict)
            and e["artifact"].get("art_type") == "viral"
        }

        source_of, infected_at, host_of = {}, {}, {}
        secondary: Dict[str, int] = {}
        for e in infections:
            art = e["artifact"]["name"]
            source_of[art] = e.get("source_artifact")
            infected_at[art] = e.get("timestamp")
            host_of[art] = e.get("agent_tag")
            if e.get("source_artifact"):
                secondary[e["source_artifact"]] = (
                    secondary.get(e["source_artifact"], 0) + 1
                )

        def generation(art):
            gen = 0
            seen = set()
            while source_of.get(art) is not None and art not in seen:
                seen.add(art)
                art = source_of[art]
                gen += 1
            return gen

        by_gen: Dict[int, list] = {}
        censored = 0
        for art in source_of:
            if art in removed:
                by_gen.setdefault(generation(art), []).append(secondary.get(art, 0))
            else:
                censored += 1

        generations = [
            {"generation": g, "cases": len(v), "mean_secondary": sum(v) / len(v)}
            for g, v in sorted(by_gen.items())
        ]
        early = by_gen.get(0, []) + by_gen.get(1, [])
        return {
            "chain": [
                {
                    "artifact": art,
                    "source": source_of[art],
                    "host": host_of[art],
                    "t": infected_at[art],
                    "generation": generation(art),
                    "secondary": secondary.get(art, 0),
                    # False = the episode was still running when the log ends,
                    # so its secondary count is a lower bound.
                    "ended": art in removed,
                }
                for art in source_of
            ],
            "generations": generations,
            "censored": censored,
            "r0": (sum(early) / len(early)) if early else None,
        }

    @app.get("/api/runs/{name}/stream")
    async def stream(name: str, since: int = -1):
        """Server-sent events: one message per newly written timestep."""

        async def gen():
            last = since
            idle = 0
            while True:
                reader = get_reader(name)
                latest = reader.last_step
                if latest > last:
                    idle = 0
                    payload = {
                        "last_step": latest,
                        # The world log is written after the step, so it runs one
                        # ahead of the decisions that produced it. A live viewer
                        # wants the newest frame where the map, the chat and the
                        # agents' reasoning all describe the same instant.
                        "last_decision_step": reader.meta()["last_decision_step"],
                        "status": reader.status(),
                        "series": reader.series(),
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
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        type=Path,
        default=Path(ROOT) / "logs",
        help="Directory containing experiment folders (default: <repo>/logs)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"📊 TerraLingua dashboard → http://{args.host}:{args.port}")
    print(f"   reading runs from {args.logs}")
    uvicorn.run(
        create_app(args.logs), host=args.host, port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
