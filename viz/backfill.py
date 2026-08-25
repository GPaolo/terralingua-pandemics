"""Reconstruct ``world_state.jsonl`` for runs recorded before it existed.

Runs like ``logs/oliv_3`` predate the per-step world log, so the dashboard would
otherwise have nothing to draw. Everything needed is still recoverable from the
JSON logs — with one exception, noted below — so this module rebuilds a file in
exactly the schema ``core/environment/world_logger.py`` writes, tagged
``"provenance": "reconstructed"`` so the UI can say so.

Positions
---------
``ENV_RESET`` in ``open_gridworld.log`` gives every agent's true starting cell
(``AGENT_ADDED.position`` is the pre-reset random placement and is *not* it), and
each ``agent_logs/<tag>.jsonl`` line gives the direction the agent tried to move.
The missing piece is which moves the environment *blocked* — it refuses a move
into a cell another agent already occupies, and logs nothing when it does.

Replaying moves naively therefore drifts. But the observations are a strong
cross-check: every agent reports the other agents it can see, as offsets relative
to itself. So each agent is decoded with Viterbi over "did this move land or was
it blocked", scored by agreement with what everybody saw, plus the environment's
own rule that a cell holds at most one agent. The agents are then
coordinate-descended against each other until stable.

Accuracy, measured against the ``agent_trajectories`` the environment itself
recorded: ``oliv_3`` 788/788 sightings and all 10 agents exact; ``oliv_2``
652/654 sightings and 8/10 exact.

**The remaining gap is inherent, not a bug.** When nobody can see an agent,
"moved right, then left" and "was blocked, then moved left" leave identical
traces in the logs, and no amount of inference separates them — in ``oliv_2`` the
two imperfect agents are the two least-observed ones. This is precisely why
``core/environment/world_logger.py`` exists: runs recorded with it need none of
this.

Food
----
The per-cell map is genuinely unrecoverable. What *is* per-tick is each agent's
view of the food around it, so the map is rebuilt as the union of everything
anybody has seen — a fog of war. Cells nobody has visited are absent rather than
empty, and the header carries ``"food_source": "observed"`` so the UI can shade
them as unknown instead of implying they hold nothing.

The per-step *total* was kept in ``food_counts.json``, so that is used verbatim
for ``food_total``. Summing the fog instead would produce a chart that climbs as
beings explore, which reads as food appearing when it is only being discovered.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.environment.world_logger import (
    AGENT_FIELDS,
    KEYFRAME_INTERVAL,
    SCHEMA_VERSION,
)

#: Matches an observation key like ``"(-3, 4)"``.
_OFFSET_RE = re.compile(r"^\((-?\d+), (-?\d+)\)$")

#: Must match ``MOVE_DICT`` in ``core/environment/env.py``.
_MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
}


def _load_jsonl(path: Path) -> List[dict]:
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


class _Reconstruction:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.params = json.loads((run_dir / "params.json").read_text())
        self.grid = self.params["env"]["grid_size"]
        self.max_food = 10.0

        # The per-cell food map is unrecoverable, but the per-step *total* was
        # kept, so the charts can still be exact even though the map is a fog.
        self.food_totals: List[float] = []
        totals_path = run_dir / "food_counts.json"
        if totals_path.exists():
            try:
                self.food_totals = json.loads(totals_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass

        self.events = _load_jsonl(run_dir / "open_gridworld.log")
        self.logs: Dict[str, List[dict]] = {}
        for path in sorted((run_dir / "agent_logs").glob("*.jsonl")):
            if path.name == "token_counts.jsonl":
                continue
            self.logs[path.stem] = _load_jsonl(path)

        self.tags = sorted(self.logs)
        self.n_ticks = max((len(v) for v in self.logs.values()), default=0)

        self.start: Dict[str, Tuple[int, int]] = {}
        for e in self.events:
            if e.get("event") == "ENV_RESET":
                for tag, pose in e.get("agent_poses", {}).items():
                    self.start[tag] = (int(pose[0]), int(pose[1]))
                break

        self.attempts: Dict[str, List[Tuple[int, int]]] = {}
        for tag, recs in self.logs.items():
            moves = []
            for r in recs:
                action = r.get("action", {}) or {}
                if action.get("action") == "move":
                    d = (action.get("params") or {}).get("direction", "stay")
                    moves.append(_MOVES.get(d, (0, 0)))
                else:
                    moves.append((0, 0))
            self.attempts[tag] = moves

        # sightings[observer][tick] -> [(target_tag, (dr, dc)), ...]
        self.sightings: Dict[str, List[list]] = {}
        self.observed_food: Dict[str, List[list]] = {}
        for tag, recs in self.logs.items():
            per_tick_agents: List[list] = []
            per_tick_food: List[list] = []
            for r in recs:
                agents, food = [], []
                obs = (r.get("observation") or {}).get("observation", {}) or {}
                for key, values in obs.items():
                    m = _OFFSET_RE.match(key)
                    if not m:
                        continue
                    off = (int(m.group(1)), int(m.group(2)))
                    for v in values:
                        if v in self.logs:
                            agents.append((v, off))
                        else:
                            try:
                                food.append((off, float(v)))
                            except ValueError:
                                pass  # artifact token "A(text): Name"
                per_tick_agents.append(agents)
                per_tick_food.append(food)
            self.sightings[tag] = per_tick_agents
            self.observed_food[tag] = per_tick_food

        # positions[tag][tick] -> (row, col); tick 0 is the pose at ENV_RESET
        self.positions: Dict[str, List[Tuple[int, int]]] = {}

    # ---------- position decoding ----------
    def _wrap(self, r: int, c: int) -> Tuple[int, int]:
        return r % self.grid, c % self.grid

    def _naive(self):
        for tag in self.tags:
            pos = self.start.get(tag, (0, 0))
            path = [pos]
            for dr, dc in self.attempts[tag]:
                pos = self._wrap(pos[0] + dr, pos[1] + dc)
                path.append(pos)
            self.positions[tag] = path

    def _score(self, tag: str, tick: int, pos: Tuple[int, int]) -> int:
        """Sightings satisfied at ``tick`` if ``tag`` were standing at ``pos``."""
        score = 0
        for target, (dr, dc) in self.sightings[tag][tick]:
            other = self.positions.get(target)
            if other is None or tick >= len(other):
                continue
            if other[tick] == self._wrap(pos[0] + dr, pos[1] + dc):
                score += 1
        for observer in self.tags:
            if observer == tag or tick >= len(self.sightings[observer]):
                continue
            seen_from = self.positions.get(observer)
            if seen_from is None or tick >= len(seen_from):
                continue
            base = seen_from[tick]
            for target, (dr, dc) in self.sightings[observer][tick]:
                if target == tag and pos == self._wrap(base[0] + dr, base[1] + dc):
                    score += 1
        return score

    #: Landing where another agent already stands is something the environment
    #: refuses. Kept comparable to a couple of sightings rather than absolute:
    #: early in the coordinate descent the other agent's position is itself a
    #: guess, and a dominating weight locks in whatever it guessed first.
    _COLLISION = 2.0
    #: Mild, uniform bias against explaining a move away as "blocked". Blocking
    #: needs somebody standing in the target cell; without this bias, an agent
    #: nobody can see stalls, because blocked and landed score identically.
    _BLOCKED = 0.2

    def _occupied_by_other(self, tag: str, tick: int, cell: Tuple[int, int]) -> bool:
        """Is another agent believed to be standing on ``cell`` at ``tick``?"""
        for other in self.tags:
            if other == tag:
                continue
            path = self.positions.get(other)
            if path is not None and tick < len(path) and path[tick] == cell:
                return True
        return False

    def _viterbi(self, tag: str) -> Optional[List[Tuple[int, int]]]:
        """Decode one agent's path, choosing per tick whether its move landed.

        State is "number of moves that have landed so far", which pins the
        position exactly given the attempted directions.
        """
        attempts = self.attempts[tag]
        n_ticks = len(attempts)
        if n_ticks == 0:
            return None

        # cell[k] = position after k landed moves, following the attempt order.
        # A blocked move is skipped, so k indexes into the successful subsequence
        # and depends on which ticks landed. Track (tick, landed_count) instead,
        # rebuilding the position by replaying the chosen mask.
        # best[tick][k] = (score, position, backpointer_k)
        start = self.start.get(tag, (0, 0))
        best: List[Dict[int, tuple]] = [dict() for _ in range(n_ticks + 1)]
        best[0][0] = (self._score(tag, 0, start) if n_ticks else 0, start, None)

        for tick in range(n_ticks):
            for k, (score, pos, _) in best[tick].items():
                dr, dc = attempts[tick]
                options = [(k, pos, 0.0)]  # a no-op action costs nothing
                if (dr, dc) != (0, 0):
                    target = self._wrap(pos[0] + dr, pos[1] + dc)
                    # Two readings of "the agent did not move": the environment
                    # blocked it, or it landed. Blocking needs an occupant, and
                    # landing on an occupied cell is impossible -- so score the
                    # two against where everyone else is believed to be.
                    blocked_cost = self._BLOCKED
                    landed_cost = (
                        self._COLLISION
                        if self._occupied_by_other(tag, tick + 1, target)
                        else 0.0
                    )
                    options[0] = (k, pos, -blocked_cost)
                    options.append((k + 1, target, -landed_cost))
                for nk, npos, penalty in options:
                    step_score = (
                        self._score(tag, tick + 1, npos) if tick + 1 < n_ticks else 0
                    )
                    total = score + step_score + penalty
                    cur = best[tick + 1].get(nk)
                    if cur is None or total > cur[0]:
                        best[tick + 1][nk] = (total, npos, k)

        final = best[n_ticks]
        if not final:
            return None
        end_k = max(final, key=lambda k: final[k][0])

        path = [None] * (n_ticks + 1)
        k = end_k
        for tick in range(n_ticks, -1, -1):
            score, pos, prev = best[tick][k]
            path[tick] = pos
            if prev is None:
                break
            k = prev
        return path

    #: Coordinate-descent passes. Measured over random runs, accuracy plateaus
    #: by 8 and the loop exits early once nothing moves, so this is a ceiling
    #: rather than a cost.
    def decode_positions(self, rounds: int = 8):
        self._naive()
        for _ in range(rounds):
            changed = False
            for tag in self.tags:
                path = self._viterbi(tag)
                if path and path != self.positions[tag]:
                    self.positions[tag] = path
                    changed = True
            if not changed:
                break

    def agreement(self) -> Tuple[int, int]:
        """(satisfied, total) sighting constraints — the quality of the decode."""
        ok = total = 0
        for observer in self.tags:
            path = self.positions[observer]
            for tick, seen in enumerate(self.sightings[observer]):
                if tick >= len(path):
                    continue
                base = path[tick]
                for target, (dr, dc) in seen:
                    other = self.positions.get(target)
                    if other is None or tick >= len(other):
                        continue
                    total += 1
                    if other[tick] == self._wrap(base[0] + dr, base[1] + dc):
                        ok += 1
        return ok, total

    # ---------- emission ----------
    def _artifact_cells(self) -> Dict[int, List[Tuple[int, int, str]]]:
        """Artifacts present on the map at each tick, from ARTIFACT_* events."""
        timeline: Dict[int, List[Tuple[str, Tuple[int, int], bool]]] = {}
        for e in self.events:
            art = e.get("artifact")
            if not isinstance(art, dict) or "pose" not in art:
                continue
            t = int(e.get("timestamp", 0))
            pose = (int(art["pose"][0]), int(art["pose"][1]))
            if e.get("event") == "ARTIFACT_ADDED":
                timeline.setdefault(t, []).append((art["name"], pose, True))
            elif e.get("event") == "ARTIFACT_REMOVED":
                timeline.setdefault(t, []).append((art["name"], pose, False))

        cells: Dict[int, List[Tuple[int, int, str]]] = {}
        present: Dict[str, Tuple[int, int]] = {}
        for t in range(self.n_ticks + 1):
            for name, pose, added in timeline.get(t, []):
                if added:
                    present[name] = pose
                else:
                    present.pop(name, None)
            cells[t] = [(p[0], p[1], n) for n, p in sorted(present.items())]
        return cells

    def _fog_food(self):
        """Fog-of-war food map per tick: last seen value for every observed cell."""
        known: Dict[Tuple[int, int], float] = {}
        per_tick = []
        for tick in range(self.n_ticks):
            for tag in self.tags:
                path = self.positions[tag]
                if tick >= len(path) or tick >= len(self.observed_food[tag]):
                    continue
                base = path[tick]
                for (dr, dc), value in self.observed_food[tag][tick]:
                    known[self._wrap(base[0] + dr, base[1] + dc)] = value
            per_tick.append(dict(known))
        # The world log has one more entry than agent decisions (the final state).
        per_tick.append(dict(known))
        return per_tick

    def write(self, out_path: Path) -> dict:
        self.decode_positions()
        ok, total = self.agreement()
        artifacts = self._artifact_cells()
        food_by_tick = self._fog_food()
        def _vitals(record: dict) -> Tuple[Optional[float], Optional[int]]:
            obs = record.get("observation") or {}
            return obs.get("energy"), obs.get("time")

        energies = {
            tag: [_vitals(r) for r in recs] for tag, recs in self.logs.items()
        }

        prev_food: Dict[Tuple[int, int], float] = {}
        prev_arts: set = set()
        with open(out_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "kind": "meta",
                        "schema_version": SCHEMA_VERSION,
                        "grid_size": self.grid,
                        "max_food_value": self.max_food,
                        "provenance": "reconstructed",
                        "food_source": "observed",
                        "agent_fields": AGENT_FIELDS,
                        "sighting_agreement": [ok, total],
                    }
                )
                + "\n"
            )

            for t in range(self.n_ticks + 1):
                agents = {}
                for tag in self.tags:
                    path = self.positions[tag]
                    if t >= len(path):
                        continue
                    row, col = path[t]
                    # Energy is logged with the observation the agent acted on,
                    # i.e. its state at the start of that tick. There is no
                    # observation for the final frame (the world outlives the
                    # last decision by one step), so hold the last known values
                    # rather than reporting the agent as having none.
                    series = energies[tag]
                    e_t = (
                        series[t]
                        if t < len(series)
                        else (series[-1] if series else (None, None))
                    )
                    agents[tag] = [row, col, e_t[0], e_t[1], 0, 0]

                food = food_by_tick[t] if t < len(food_by_tick) else {}
                arts = set(artifacts.get(t, []))
                keyframe = t % KEYFRAME_INTERVAL == 0

                if keyframe:
                    food_part = {"set": [[x, y, v] for (x, y), v in food.items()]}
                    art_part = {"set": [[x, y, n] for x, y, n in sorted(arts)]}
                else:
                    food_part = {
                        "add": [
                            [x, y, v]
                            for (x, y), v in food.items()
                            if prev_food.get((x, y)) != v
                        ],
                        "del": [[x, y] for (x, y) in prev_food if (x, y) not in food],
                    }
                    art_part = {
                        "add": [[x, y, n] for x, y, n in sorted(arts - prev_arts)],
                        "del": [[x, y, n] for x, y, n in sorted(prev_arts - arts)],
                    }

                f.write(
                    json.dumps(
                        {
                            "kind": "key" if keyframe else "delta",
                            "t": t,
                            "agents": agents,
                            "food": food_part,
                            "artifacts": art_part,
                            "food_total": (
                                self.food_totals[t]
                                if t < len(self.food_totals)
                                else sum(food.values())
                            ),
                            "n_agents": len(agents),
                            "n_infected": 0,
                        }
                    )
                    + "\n"
                )
                prev_food, prev_arts = food, arts

        return {"sighting_agreement": [ok, total], "ticks": self.n_ticks + 1}


def backfill(run_dir: Path | str, force: bool = False) -> Optional[dict]:
    """Write ``world_state.jsonl`` for a legacy run. No-op if one already exists."""
    run_dir = Path(run_dir)
    out = run_dir / "world_state.jsonl"
    if out.exists() and not force:
        return None
    if not (run_dir / "params.json").exists():
        return None
    if not (run_dir / "agent_logs").is_dir():
        return None

    recon = _Reconstruction(run_dir)
    if not recon.tags or not recon.start:
        return None
    return recon.write(out)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Path to logs/<exp_name>")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing file"
    )
    args = parser.parse_args()

    result = backfill(args.run_dir, force=args.force)
    if result is None:
        print(f"Nothing to do for {args.run_dir}")
        return
    ok, total = result["sighting_agreement"]
    pct = 100 * ok / total if total else 100.0
    print(
        f"Reconstructed {result['ticks']} ticks for {args.run_dir.name}; "
        f"sighting agreement {ok}/{total} ({pct:.1f}%)"
    )


if __name__ == "__main__":
    main()
