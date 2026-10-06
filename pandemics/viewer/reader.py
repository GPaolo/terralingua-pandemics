"""Readers for one run folder of the pandemics scenario.

The viewer follows a run while it is still being written. Every reader here
tails its file from a stored byte offset. It never parses a file again from
the top. A run folder holds these files.

``world_state.jsonl``
    Written by the mechanic, one line per step. Line 1 is a ``meta`` header.
    It gives ``grid_size``, ``max_food_value`` and ``agent_fields``, the order
    of the values in each being's row. Every later line has ``kind`` (``key``
    or ``delta``), ``t``, ``agents`` (tag to row), ``food``, ``artifacts`` and
    ``air`` (older files have no ``air``). A ``key`` line gives them in full
    under ``set``. A ``delta`` line gives the changes under ``add`` and
    ``del``. An artifact entry is ``[row, col, name, kind]``; an air entry is
    ``[row, col, load]``. Each line also carries ``food_total``,
    ``n_agents``, ``n_infected``, ``n_sick`` and ``n_bedridden``. The line for
    step T is written before that step's energy drain and deaths.

``open_gridworld.log``
    JSON lines ``{"timestamp": T, "event": NAME, ...}``. The world logs
    ``ENV_RESET``, ``AGENT_ADDED``, ``AGENT_DIED``, the ``ARTIFACT_*`` events,
    ``SET_STATE_CKPT`` on a resume and ``END_RUN`` when the run closes. The
    mechanic logs ``IDENTITY`` (name, role and persona of a new being),
    ``VIRAL_INFECTION``, ``VIRAL_HEALED``, ``VIRAL_EXPOSURE`` and ``BURIAL``.
    The file is only appended to. A re-run under the same name starts with a
    new ``ENV_RESET``.

``agent_logs/<tag>.jsonl``
    One line per decision of a being. ``timestamp`` is a string. ``action``
    holds ``action``, ``message`` and ``params``. ``observation`` holds
    ``incoming_broadcasts``, ``energy``, ``time`` and ``inventory``.
    ``internal_memory`` is the being's note to itself.
    ``agent_logs/<tag>_genome.json`` maps a trait to a value. A scripted run
    has no ``agent_logs`` folder.

``costs.csv``
    Header ``timestep,cost_usd,input_tokens,output_tokens``. One row per step
    with the totals over all beings.

``run_status.json``
    ``simulation.status`` is ``running``, ``complete``, ``stopped_early`` or
    ``failed``.

``params.json``
    The run configuration, with ``agent``, ``env`` and ``run`` sections.
"""

import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from pandemics.state_log import AGENT_FIELDS

#: A running simulation whose newest log was touched within this many seconds is
#: live. A model call can take long, so this tolerates slow steps.
LIVE_GRACE_SECONDS = 180

#: Dropped on ingest. They are large and the viewer does not show them.
_HEAVY_AGENT_FIELDS = ("input_prompt", "available_actions")
# Older cores wrote this usage file into agent_logs/ next to the beings' logs.
_NON_AGENT_LOG_FILES = {"token_counts.jsonl"}

#: Artifact types that are simulation state, not texts written by beings.
STATE_ARTIFACT_TYPES = ("viral", "ppe", "health_center", "remains")


def _iter_lines(path: Path, offset: int) -> tuple[List[str], int]:
    """Complete lines from ``offset`` on, and the new offset.

    A line-buffered writer can leave a partial final line. That line stays
    unread, so the next refresh picks it up once it is complete.
    """
    if not path.exists():
        return [], offset

    lines = []
    with open(path, "rb") as f:
        f.seek(offset)
        for raw in f:
            if not raw.endswith(b"\n"):
                break
            offset += len(raw)
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                lines.append(line)
    return lines, offset


def _iter_json_lines(path: Path, offset: int) -> tuple[Iterator[dict], int]:
    """Parse the complete JSON lines from ``offset`` on. Damaged lines are skipped."""
    lines, offset = _iter_lines(path, offset)
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return iter(records), offset


class RunReader:
    """Reads one run folder and keeps what it has read so far."""

    def __init__(self, run_dir: Path | str):
        self.dir = Path(run_dir)
        self.name = self.dir.name

        self._offsets: Dict[str, int] = {}
        self._world_meta: dict = {}
        # Raw per-step records, keyed by step. Food and artifacts stay in delta
        # form; a full map is built on demand from the nearest keyframe.
        self._steps: Dict[int, dict] = {}
        self._keyframes: List[int] = []
        self._events: List[dict] = []
        self._agent_ticks: Dict[str, Dict[int, dict]] = {}
        self._genomes: Dict[str, dict] = {}
        self._costs: Dict[int, dict] = {}
        self._params: Optional[dict] = None

    # ---------- ingestion ----------
    def refresh(self):
        """Pull in everything appended since the last call."""
        self._reset_if_truncated()
        self._read_world_state()
        self._read_events()
        self._read_agent_logs()
        self._read_costs()

    def _reset_if_truncated(self):
        """Start over if the world file got shorter than what was already read.

        A fresh run under an existing name rewrites ``world_state.jsonl`` from
        the start, and deleting the folder shrinks everything. Reading from a
        stale offset would then join the tail of a new run to the head of an
        old one.
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
            # A resumed run appends and repeats the steps after its checkpoint.
            # The latest record for a step wins.
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
            if path.name in _NON_AGENT_LOG_FILES:
                continue
            tag = path.stem
            key = f"agent:{tag}"
            records, offset = _iter_json_lines(path, self._offsets.get(key, 0))
            self._offsets[key] = offset
            ticks = self._agent_ticks.setdefault(tag, {})
            for r in records:
                if "timestamp" not in r:
                    continue
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

    def _read_costs(self):
        lines, offset = _iter_lines(
            self.dir / "costs.csv", self._offsets.get("costs", 0)
        )
        self._offsets["costs"] = offset
        for line in lines:
            try:
                t, cost, tokens_in, tokens_out = line.split(",")[:4]
                self._costs[int(t)] = {
                    "cost": float(cost),
                    "input": int(tokens_in),
                    "output": int(tokens_out),
                }
            except ValueError:
                continue  # the header line

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
        """``live``, ``stalled`` or ``finished``.

        ``run_status.json`` decides when it exists: any simulation status other
        than ``running`` means finished. Without the file, an ``END_RUN`` event
        of the current run means finished. A run that is not finished is live
        while its newest log was touched within :data:`LIVE_GRACE_SECONDS`.
        """
        simulation = self._simulation_status()
        if simulation is None:
            if any(e.get("event") == "END_RUN" for e in self._current_run_events()):
                return "finished"
        elif simulation != "running":
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

    def _simulation_status(self) -> Optional[str]:
        """``simulation.status`` of ``run_status.json``. None without a readable file."""
        try:
            data = json.loads((self.dir / "run_status.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        return (data.get("simulation") or {}).get("status")

    def _agent_names(self) -> Dict[str, str]:
        """tag -> display name, from the reset and the additions of the current run."""
        names: Dict[str, str] = {}
        for e in self._current_run_events():
            if e.get("event") == "ENV_RESET":
                names.update(e.get("agent_names") or {})
            elif e.get("event") == "AGENT_ADDED" and e.get("agent_tag"):
                names[e["agent_tag"]] = e.get("agent_name") or e["agent_tag"]
        return names

    def _identities(self) -> Dict[str, dict]:
        """tag -> the IDENTITY event of the being, with its name, role and persona."""
        return {
            e["agent_tag"]: e
            for e in self._current_run_events()
            if e.get("event") == "IDENTITY" and e.get("agent_tag")
        }

    def _agent_deaths(self) -> Dict[str, list]:
        """tag -> [{t, reason}, ...]; a list because a tag can be used again."""
        deaths: Dict[str, list] = {}
        for e in self._current_run_events():
            if e.get("event") == "AGENT_DIED" and e.get("agent_tag"):
                deaths.setdefault(e["agent_tag"], []).append(
                    {"t": e.get("timestamp"), "reason": e.get("reason")}
                )
        return deaths

    def meta(self) -> dict:
        params = self.params
        env = params.get("env", {})
        run = params.get("run", {})
        agent = params.get("agent", {})
        # Every being the run has ever shown, so beings that died mid-run still
        # appear (greyed out) and the map never has a being the UI cannot name.
        # Works when agent_logs is absent or lags behind the world log.
        tags = set(self._agent_ticks) | set(self._genomes)
        for record in self._steps.values():
            tags.update(record.get("agents", {}))
        tags = sorted(tags)
        identities = self._identities()
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
            # Bounded by last_step because the agent logs are only appended to:
            # a re-run under the same name leaves the previous run's later
            # steps in the file, and an unbounded max would point the UI at a
            # step this run never reached.
            "last_decision_step": max(
                (
                    t
                    for ticks in self._agent_ticks.values()
                    for t in ticks
                    if t <= self.last_step
                ),
                default=self.last_step,
            ),
            "agent_fields": self._world_meta.get("agent_fields", []),
            "agents": tags,
            "agent_names": self._agent_names(),
            "agent_roles": {
                tag: e["role"] for tag, e in identities.items() if e.get("role")
            },
            "agent_deaths": self._agent_deaths(),
            "genomes": self._genomes,
            "personas": {
                tag: e["persona"] for tag, e in identities.items() if e.get("persona")
            },
            "has_viral": any(
                e.get("event") == "VIRAL_INFECTION"
                for e in self._current_run_events()
            ),
            "params": params,
        }

    def world_at(self, t: int) -> Optional[dict]:
        """The full world at step ``t``, built from the nearest keyframe."""
        if t not in self._steps:
            return None

        base = 0
        for k in self._keyframes:
            if k <= t:
                base = k
            else:
                break

        food: Dict[tuple, float] = {}
        air: Dict[tuple, float] = {}
        artifacts: Dict[tuple, List[tuple]] = {}
        for ts in range(base, t + 1):
            r = self._steps.get(ts)
            if r is None:
                continue
            if r["kind"] == "key":
                food = {(x, y): v for x, y, v in r["food"].get("set", [])}
                air = {(x, y): v for x, y, v in r.get("air", {}).get("set", [])}
                artifacts = {}
                for x, y, name, kind in r["artifacts"].get("set", []):
                    artifacts.setdefault((x, y), []).append((name, kind))
            else:
                for cells, part in ((food, r["food"]), (air, r.get("air", {}))):
                    for x, y, v in part.get("add", []):
                        cells[(x, y)] = v
                    for x, y in part.get("del", []):
                        cells.pop((x, y), None)
                for x, y, name, kind in r["artifacts"].get("add", []):
                    artifacts.setdefault((x, y), []).append((name, kind))
                for x, y, name, kind in r["artifacts"].get("del", []):
                    cell = artifacts.get((x, y))
                    if cell and (name, kind) in cell:
                        cell.remove((name, kind))
                        if not cell:
                            del artifacts[(x, y)]

        r = self._steps[t]
        return {
            "t": t,
            "agents": r["agents"],
            "food": [[x, y, v] for (x, y), v in food.items()],
            "air": [[x, y, v] for (x, y), v in air.items()],
            "artifacts": [
                [x, y, *entry]
                for (x, y), entries in artifacts.items()
                for entry in entries
            ],
            "food_total": r.get("food_total", 0.0),
            "n_agents": r.get("n_agents", len(r["agents"])),
            "n_infected": r.get("n_infected", 0),
            "n_sick": r.get("n_sick", 0),
            "n_bedridden": r.get("n_bedridden", 0),
        }

    def agent_tick(self, tag: str, t: int) -> Optional[dict]:
        """One being's decision at a step: action, message, internal memory."""
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
            "heard": obs.get("incoming_broadcasts", {}),
        }

    def chat(self, lo: int, hi: int) -> List[dict]:
        """Broadcast messages in ``[lo, hi]``, in step then being order.

        Read from the agent logs, so it works while the run is still going.
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
        """Events of the current run only.

        The event log is only appended to. A re-run under the same name leaves
        the previous run's events in front of the current ones: old infections,
        old artifacts and an old END_RUN. A fresh start is an ENV_RESET that is
        not followed at once by SET_STATE_CKPT. That pair is a resume, which
        continues the same run and keeps its history. The identities, beings
        and seeded artifacts of a run are logged at step 0 before its reset.
        They are kept by walking back over the step-0 events in front of it.
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
        """Artifacts with their edit history, built from the event stream."""
        by_name: Dict[str, dict] = {}
        for e in self._current_run_events():
            art = e.get("artifact")
            if not isinstance(art, dict) or "name" not in art:
                continue
            # State artifacts are not authored texts. The map and the being
            # badges show them.
            if art.get("art_type") in STATE_ARTIFACT_TYPES:
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
                    # The action string tells an edit from a destroy. A destroy
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

    def token_series(self) -> List[dict]:
        """Token and cost totals per step, with running sums.

        Bounded by the last recorded step. The cost file is only appended to,
        so a re-run under the same name can leave later rows of a previous run.
        """
        out = []
        cum_in = cum_out = 0
        cum_cost = 0.0
        for t in sorted(self._costs):
            if t > self.last_step:
                break
            row = self._costs[t]
            cum_in += row["input"]
            cum_out += row["output"]
            cum_cost += row["cost"]
            out.append(
                {
                    "t": t,
                    "input": row["input"],
                    "output": row["output"],
                    "cum_input": cum_in,
                    "cum_output": cum_out,
                    "cum_cost": cum_cost,
                }
            )
        return out

    def series(self) -> dict:
        """Per-step counts for the footer charts."""
        ts = sorted(self._steps)
        rows = [self._steps[t] for t in ts]
        fields = self._world_meta.get("agent_fields") or AGENT_FIELDS
        viral, ppe, recovered = (
            fields.index(name) for name in ("n_viral", "n_ppe", "n_recovered")
        )
        return {
            "t": ts,
            "food_total": [r.get("food_total", 0.0) for r in rows],
            "n_agents": [r.get("n_agents", 0) for r in rows],
            "n_infected": [r.get("n_infected", 0) for r in rows],
            "n_sick": [r.get("n_sick", 0) for r in rows],
            "n_bedridden": [r.get("n_bedridden", 0) for r in rows],
            "n_ppe": [
                sum(1 for a in r.get("agents", {}).values() if a[ppe]) for r in rows
            ],
            # Recovered: cleared at least one infection and hosts none now.
            "n_recovered": [
                sum(
                    1
                    for a in r.get("agents", {}).values()
                    if a[recovered] and not a[viral]
                )
                for r in rows
            ],
        }
