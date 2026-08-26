"""Incremental readers for an experiment's log directory.

A run is followed while it is still being written, so every reader here tails its
file from a stored byte offset instead of re-parsing from the top. The only
sources used are the ones the simulation flushes as it goes:

===========================  =========================================
``world_state.jsonl``        per-step world snapshot (see world_logger)
``open_gridworld.log``       sparse world events, incl. VIRAL_INFECTION
``agent_logs/<tag>.jsonl``   per-tick action, message, internal memory
``agent_logs/<tag>_genome.json``   personality traits
``agent_logs/token_counts.jsonl``  LLM spend
``params.json``              config snapshot
===========================  =========================================

Deliberately unused: ``messages.json``, ``artifacts.json``, ``food_counts.json``
and ``agent_names.json`` are only written by ``env.close()``, so they do not
exist mid-run. Everything they contain is derived from the tailable files above,
which keeps live and replay on one code path.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

#: A run with no END_RUN whose newest log was touched within this many seconds is
#: considered live. Long LLM steps mean this has to tolerate slow ticks.
LIVE_GRACE_SECONDS = 180

#: Dropped on ingest: ~12 KB per agent per tick and fully reconstructible from
#: the other fields.
_HEAVY_AGENT_FIELDS = ("input_prompt", "available_actions")


def _iter_json_lines(path: Path, offset: int) -> tuple[Iterator[dict], int]:
    """Parse whole JSON lines from ``offset``, returning them and the new offset.

    A line-buffered writer can leave a partial final line; that line is left
    unconsumed so the next refresh picks it up once complete.
    """
    if not path.exists():
        return iter(()), offset

    records = []
    with open(path, "r") as f:
        f.seek(offset)
        for line in f:
            if not line.endswith("\n"):
                break  # partial write, retry next refresh
            offset += len(line.encode("utf-8"))
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return iter(records), offset


class RunReader:
    """Incrementally materializes one experiment directory."""

    def __init__(self, run_dir: Path | str):
        self.dir = Path(run_dir)
        self.name = self.dir.name

        self._offsets: Dict[str, int] = {}
        self._world_meta: dict = {}
        # Raw per-step records, indexed by timestep. Food/artifacts stay in delta
        # form; a full map is materialized on demand from the nearest keyframe.
        self._steps: Dict[int, dict] = {}
        self._keyframes: List[int] = []
        self._events: List[dict] = []
        self._agent_ticks: Dict[str, Dict[int, dict]] = {}
        self._genomes: Dict[str, dict] = {}
        self._personas: Dict[str, str] = {}
        self._tokens: Dict[int, Dict[str, dict]] = {}
        self._params: Optional[dict] = None

    # ---------- ingestion ----------
    def refresh(self):
        """Pull in everything appended since the last call."""
        self._reset_if_truncated()
        self._read_world_state()
        self._read_events()
        self._read_agent_logs()
        self._read_tokens()

    def _reset_if_truncated(self):
        """Start over if a file got shorter than what we already consumed.

        Re-running an experiment under an existing ``exp_name`` rewrites
        ``world_state.jsonl`` from scratch (and deleting the directory shrinks
        everything). Reading from a stale byte offset would then splice the tail
        of a new run onto the head of an old one.
        """
        world = self.dir / "world_state.jsonl"
        offset = self._offsets.get("world", 0)
        if offset and (not world.exists() or world.stat().st_size < offset):
            self.__init__(self.dir)

    def _read_world_state(self):
        path = self.dir / "world_state.jsonl"
        records, offset = _iter_json_lines(path, self._offsets.get("world", 0))
        self._offsets["world"] = offset
        for r in records:
            kind = r.get("kind")
            if kind == "meta":
                self._world_meta = r
                continue
            t = r.get("t")
            if t is None:
                continue
            # A re-run of the same exp_name appends and rewinds the clock; the
            # latest record for a timestep wins.
            if kind == "key":
                if t not in self._steps:
                    self._keyframes.append(t)
                    self._keyframes.sort()
            self._steps[t] = r

    def _read_events(self):
        records, offset = _iter_json_lines(
            self.dir / "open_gridworld.log", self._offsets.get("events", 0)
        )
        self._offsets["events"] = offset
        self._events.extend(records)

    def _read_agent_logs(self):
        log_dir = self.dir / "agent_logs"
        if not log_dir.is_dir():
            return
        for path in sorted(log_dir.glob("*.jsonl")):
            if path.name == "token_counts.jsonl":
                continue
            tag = path.stem
            key = f"agent:{tag}"
            records, offset = _iter_json_lines(path, self._offsets.get(key, 0))
            self._offsets[key] = offset
            ticks = self._agent_ticks.setdefault(tag, {})
            for r in records:
                for field in _HEAVY_AGENT_FIELDS:
                    r.pop(field, None)
                ticks[int(r["timestamp"])] = r

        for path in sorted(log_dir.glob("*_genome.json")):
            tag = path.name[: -len("_genome.json")]
            if tag in self._genomes:
                continue
            try:
                self._genomes[tag] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                pass

        for path in sorted(log_dir.glob("*_persona.txt")):
            tag = path.name[: -len("_persona.txt")]
            if tag in self._personas:
                continue
            try:
                self._personas[tag] = path.read_text().strip()
            except OSError:
                pass

    def _read_tokens(self):
        records, offset = _iter_json_lines(
            self.dir / "agent_logs" / "token_counts.jsonl",
            self._offsets.get("tokens", 0),
        )
        self._offsets["tokens"] = offset
        for r in records:
            self._tokens.setdefault(int(r["timestep"]), {})[r["agent_tag"]] = r

    # ---------- accessors ----------
    @property
    def params(self) -> dict:
        if self._params is None:
            path = self.dir / "params.json"
            try:
                self._params = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                self._params = {}
        return self._params

    @property
    def last_step(self) -> int:
        return max(self._steps) if self._steps else -1

    def status(self) -> str:
        """``finished`` | ``live`` | ``stalled``.

        Keyed off END_RUN rather than ``video.mp4`` (what ``get_exp_folders`` in
        ``core/utils/analysis_utils.py`` uses), because the video only appears
        after ffmpeg has run — far too late to drive a live view.
        """
        import time

        # Current-run events only: a re-run under the same name would otherwise
        # inherit the previous run's END_RUN and report "finished" while live.
        if any(e.get("event") == "END_RUN" for e in self._current_run_events()):
            return "finished"

        newest = 0.0
        for path in (
            self.dir / "world_state.jsonl",
            self.dir / "open_gridworld.log",
        ):
            if path.exists():
                newest = max(newest, path.stat().st_mtime)
        for path in (self.dir / "agent_logs").glob("*.jsonl"):
            newest = max(newest, path.stat().st_mtime)

        if newest and time.time() - newest < LIVE_GRACE_SECONDS:
            return "live"
        return "stalled"

    def _agent_names(self) -> Dict[str, str]:
        """tag -> display name, from the live event log (personas rename agents)."""
        names: Dict[str, str] = {}
        for e in self._current_run_events():
            if e.get("event") == "ENV_RESET":
                names.update(e.get("agent_names") or {})
            elif e.get("event") == "AGENT_ADDED" and e.get("agent_tag"):
                names[e["agent_tag"]] = e.get("agent_name") or e["agent_tag"]
        return names

    def meta(self) -> dict:
        params = self.params
        env = params.get("env", {})
        run = params.get("run", {})
        agent = params.get("agent", {})
        # Every roster the run has ever shown, so beings that died mid-run still
        # appear (greyed out) and the map never has an agent the UI cannot name.
        # Falls back cleanly when agent_logs is absent or lags the world log.
        tags = set(self._agent_ticks) | set(self._genomes)
        for record in self._steps.values():
            tags.update(record.get("agents", {}))
        tags = sorted(tags)
        return {
            "name": self.name,
            "description": run.get("exp_description", ""),
            "model": agent.get("model", "unknown"),
            "status": self.status(),
            "grid_size": self._world_meta.get("grid_size", env.get("grid_size")),
            "max_food_value": self._world_meta.get("max_food_value", 10.0),
            "vision_radius": env.get("vision_radius"),
            "max_ts": run.get("max_ts"),
            "last_step": self.last_step,
            # The world outlives the last decision by one step, so the final
            # frame has positions but no actions. The UI lands here instead.
            # Bounded by last_step because the agent logs are opened in append
            # mode: re-running an exp_name leaves the previous run's later
            # timestamps in the file, and an unbounded max would point the UI at
            # a step this run never reached.
            "last_decision_step": max(
                (
                    t
                    for ticks in self._agent_ticks.values()
                    for t in ticks
                    if t <= self.last_step
                ),
                default=self.last_step,
            ),
            "provenance": self._world_meta.get("provenance", "recorded"),
            "food_source": self._world_meta.get("food_source", "actual"),
            "sighting_agreement": self._world_meta.get("sighting_agreement"),
            "agent_fields": self._world_meta.get("agent_fields", []),
            "agents": tags,
            "agent_names": self._agent_names(),
            "genomes": self._genomes,
            "personas": self._personas,
            "has_viral": any(
                e.get("event") == "VIRAL_INFECTION"
                for e in self._current_run_events()
            ),
            "params": params,
        }

    def world_at(self, t: int) -> Optional[dict]:
        """Materialize the full world at timestep ``t`` from the nearest keyframe."""
        if t not in self._steps:
            return None

        base = 0
        for k in self._keyframes:
            if k <= t:
                base = k
            else:
                break

        food: Dict[tuple, float] = {}
        artifacts: Dict[tuple, List[str]] = {}
        for ts in range(base, t + 1):
            r = self._steps.get(ts)
            if r is None:
                continue
            if r["kind"] == "key":
                food = {(x, y): v for x, y, v in r["food"].get("set", [])}
                artifacts = {}
                for x, y, name in r["artifacts"].get("set", []):
                    artifacts.setdefault((x, y), []).append(name)
            else:
                for x, y, v in r["food"].get("add", []):
                    food[(x, y)] = v
                for x, y in r["food"].get("del", []):
                    food.pop((x, y), None)
                for x, y, name in r["artifacts"].get("add", []):
                    artifacts.setdefault((x, y), []).append(name)
                for x, y, name in r["artifacts"].get("del", []):
                    cell = artifacts.get((x, y))
                    if cell and name in cell:
                        cell.remove(name)
                        if not cell:
                            del artifacts[(x, y)]

        r = self._steps[t]
        return {
            "t": t,
            "agents": r["agents"],
            "food": [[x, y, v] for (x, y), v in food.items()],
            "artifacts": [
                [x, y, n] for (x, y), names in artifacts.items() for n in names
            ],
            "food_total": r.get("food_total", 0.0),
            "n_agents": r.get("n_agents", len(r["agents"])),
            "n_infected": r.get("n_infected", 0),
            # Schema 1 predates the incubation phase: every infection there was
            # symptomatic from the start. Same fallback as series().
            "n_sick": r.get("n_sick", r.get("n_infected", 0)),
        }

    def agent_tick(self, tag: str, t: int) -> Optional[dict]:
        """One agent's decision at a timestep: action, message, internal memory."""
        rec = self._agent_ticks.get(tag, {}).get(t)
        if rec is None:
            return None
        action = rec.get("action", {}) or {}
        obs = rec.get("observation", {}) or {}
        return {
            "t": t,
            "agent_tag": tag,
            "agent_name": rec.get("agent", tag),
            "action": action.get("action"),
            "params": action.get("params", {}),
            "message": action.get("message", ""),
            "internal_memory": rec.get("internal_memory", ""),
            "energy": obs.get("energy"),
            "time": obs.get("time"),
            "inventory": obs.get("inventory", []),
            "heard": obs.get("message", {}),
        }

    def chat(self, lo: int, hi: int) -> List[dict]:
        """Broadcast messages in ``[lo, hi]``, in timestep then agent order.

        Derived from the agent logs rather than ``messages.json`` so it works
        while the run is still going.
        """
        out = []
        for tag, ticks in self._agent_ticks.items():
            for t, rec in ticks.items():
                if not lo <= t <= hi:
                    continue
                msg = (rec.get("action", {}) or {}).get("message", "")
                if msg:
                    out.append(
                        {
                            "t": t,
                            "agent_tag": tag,
                            "agent_name": rec.get("agent", tag),
                            "message": msg,
                        }
                    )
        out.sort(key=lambda m: (m["t"], m["agent_tag"]))
        return out

    def _current_run_events(self) -> List[dict]:
        """Events belonging to the current run only.

        Every logger appends, so re-running an ``exp_name`` leaves the previous
        run's events in front of the current ones — stale infections, artifacts
        and even an old END_RUN. A fresh start is an ENV_RESET *not* immediately
        followed by SET_STATE_CKPT (that pair is a ``--resume``, which continues
        the same run and must keep its history). The run's preamble — agents and
        init artifacts are logged at step 0 *before* the reset — is kept by
        walking back over the timestamp-0 events in front of it.
        """
        start = 0
        for i, e in enumerate(self._events):
            if e.get("event") != "ENV_RESET":
                continue
            nxt = self._events[i + 1] if i + 1 < len(self._events) else None
            if nxt is not None and nxt.get("event") == "SET_STATE_CKPT":
                continue
            j = i
            while (
                j > 0
                and self._events[j - 1].get("timestamp") == 0
                and self._events[j - 1].get("event") != "ENV_RESET"
            ):
                j -= 1
            start = j
        return self._events[start:]

    def events(self, types: Optional[List[str]] = None) -> List[dict]:
        current = self._current_run_events()
        if types is None:
            return list(current)
        wanted = set(types)
        return [e for e in current if e.get("event") in wanted]

    def artifacts(self) -> List[dict]:
        """Artifacts with their edit history, rebuilt from the event stream."""
        by_name: Dict[str, dict] = {}
        for e in self._current_run_events():
            art = e.get("artifact")
            if not isinstance(art, dict) or "name" not in art:
                continue
            # Viral, PPE and health-spot artifacts are simulation state, not
            # authored content; the map and the being badges cover them.
            if art.get("art_type") in ("viral", "ppe", "health_spot"):
                continue
            event = e.get("event")
            if event == "ARTIFACT_ADDED":
                by_name[art["name"]] = {
                    **art,
                    "created_by": e.get("agent_tag"),
                    "created_at": e.get("timestamp"),
                    "readers": [],
                    "editors": [],
                }
            elif event in ("ARTIFACT_INTERACTION", "ARTIFACT_PASSIVE_INTERACTION"):
                entry = by_name.get(art["name"])
                if entry is None:
                    continue
                who = {"agent_tag": e.get("agent_tag"), "t": e.get("timestamp")}
                if event == "ARTIFACT_INTERACTION":
                    # The action string tells an edit from a destroy; a destroy
                    # leaves the payload alone, so copying it is still correct.
                    who["action"] = e.get("action")
                    entry["editors"].append(who)
                    entry["payload"] = art.get("payload", entry.get("payload"))
                    entry["version"] = art.get("version", entry.get("version"))
                    entry["version_creation_time"] = art.get(
                        "version_creation_time", entry.get("version_creation_time")
                    )
                    entry["past_versions"] = art.get("past_versions", [])
                else:
                    entry["readers"].append(who)
            elif event == "ARTIFACT_REMOVED":
                entry = by_name.get(art["name"])
                if entry is not None:
                    entry["removed_at"] = e.get("timestamp")
        return sorted(by_name.values(), key=lambda a: a.get("created_at", 0))

    def token_totals(self) -> List[dict]:
        """Cumulative LLM spend per timestep, split by model in ``by_model``
        (records that predate the ``model`` field land under ``""``)."""
        out = []
        cum_in = cum_out = 0
        for t in sorted(self._tokens):
            step_in = step_out = 0
            by_model: Dict[str, dict] = {}
            for r in self._tokens[t].values():
                step_in += r["total_input_tokens"]
                step_out += r["total_output_tokens"]
                bucket = by_model.setdefault(
                    r.get("model") or "", {"input": 0, "output": 0}
                )
                bucket["input"] += r["total_input_tokens"]
                bucket["output"] += r["total_output_tokens"]
            cum_in += step_in
            cum_out += step_out
            out.append(
                {
                    "t": t,
                    "input": step_in,
                    "output": step_out,
                    "cum_input": cum_in,
                    "cum_output": cum_out,
                    "by_model": by_model,
                }
            )
        return out

    def series(self) -> dict:
        """Per-timestep aggregates for the footer charts."""
        ts = sorted(self._steps)
        return {
            "t": ts,
            "food_total": [self._steps[t].get("food_total", 0.0) for t in ts],
            "n_agents": [self._steps[t].get("n_agents", 0) for t in ts],
            "n_infected": [self._steps[t].get("n_infected", 0) for t in ts],
            # Schema 1 runs have no n_sick: every infection there was
            # symptomatic from the start, so it matches n_infected.
            "n_sick": [
                self._steps[t].get("n_sick", self._steps[t].get("n_infected", 0))
                for t in ts
            ],
            # Carriers counted from the agent rows (n_ppe is index 7, schema 3);
            # rows from older runs are shorter and count as 0.
            "n_ppe": [
                sum(
                    1
                    for a in self._steps[t].get("agents", {}).values()
                    if len(a) > 7 and a[7]
                )
                for t in ts
            ],
        }
