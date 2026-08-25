"""Per-timestep world-state log, written as JSON Lines to ``world_state.jsonl``.

The other loggers in this package record *events* (an agent died, an artifact was
created). Nothing recorded the world itself as it changed, so the only per-step
picture of a run was the rendered PNG frames, and ``env_state.pkl`` — written once,
at ``close()`` — held just the final state. That made it impossible to follow a run
while it was happening, or to scrub back through one afterwards.

This logger closes that gap: one line per timestep, flushed as it is written, so a
reader can tail the file live.

Why a separate file rather than a new ``Event`` on ``open_gridworld.log``: the
analysis scripts iterate that log end-to-end (``core/utils/analysis_utils.py``
``load_worldlog`` / ``get_last_ts``), and a high-volume per-step record would slow
all of them down and change what "the last timestamp" means.

Format
------
Line 1 is a ``meta`` header. Every later line is one timestep::

    {"kind":"meta","schema_version":1,"grid_size":25,"max_food_value":10.0,
     "provenance":"recorded",
     "agent_fields":["row","col","energy","time","n_inv","n_viral"]}

    {"kind":"key","t":0,"agents":{"being0":[10,10,100.0,100,0,0]},
     "food":{"set":[[3,4,10.0]]},"artifacts":{"set":[[10,21,"Marker_1"]]},
     "food_total":5000.0,"n_agents":10,"n_infected":0}

    {"kind":"delta","t":1,"agents":{"being0":[10,11,109.0,99,0,0]},
     "food":{"add":[[7,2,10.0]],"del":[[10,11]]},
     "artifacts":{"add":[],"del":[]},
     "food_total":4936.0,"n_agents":10,"n_infected":0}

``agents`` is written in full every step — the roster changes as beings die and
reproduce, so a delta would cost more code than it saves. ``food`` and
``artifacts`` are deltas against the previous line, with a full keyframe
(``kind: "key"``) at ``t=0`` and every ``KEYFRAME_INTERVAL`` steps, so seeking to
an arbitrary step replays a bounded number of deltas.

Positions are ``[row, col]``, matching ``MOVE_DICT`` in ``env.py`` where
``up = (-1, 0)`` — component 0 is the row.

Note this logger truncates on a fresh start and appends only on ``--resume``. The
event loggers always append, so re-running an ``exp_name`` silently concatenates
runs into one file; here that would produce a timestep sequence that rewinds, so
we start clean instead.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

SCHEMA_VERSION = 1

#: Full world snapshot every this many steps, bounding how many deltas a reader
#: must replay to seek to an arbitrary timestep.
KEYFRAME_INTERVAL = 50

#: Order of the per-agent value arrays. Mirrored in the ``meta`` header so a
#: reader never has to hardcode it.
AGENT_FIELDS = ["row", "col", "energy", "time", "n_inv", "n_viral"]


class WorldStateLogger:
    """Writes one JSON line per timestep to ``world_state.jsonl``."""

    def __init__(
        self,
        filepath: Path | str,
        grid_size: int,
        max_food_value: float,
        append: bool = False,
    ):
        self.save_path = Path(filepath)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.save_path, "a" if append else "w", buffering=1)

        self._prev_food: Dict[Tuple[int, int], float] = {}
        self._prev_artifacts: Set[Tuple[int, int, str]] = set()
        # Force a keyframe on the first line written, including after a resume.
        self._need_keyframe = True

        if not append:
            self._write(
                {
                    "kind": "meta",
                    "schema_version": SCHEMA_VERSION,
                    "grid_size": grid_size,
                    "max_food_value": max_food_value,
                    "provenance": "recorded",
                    "agent_fields": AGENT_FIELDS,
                }
            )

    def log_step(
        self,
        t: int,
        agents: Dict[str, list],
        food: Dict[Tuple[int, int], float],
        artifacts: Iterable[Tuple[int, int, str]],
        food_total: float,
        n_infected: int,
    ):
        """Record the world at timestep ``t``.

        ``agents`` maps agent_tag to a list ordered as :data:`AGENT_FIELDS`.
        ``artifacts`` yields ``(row, col, artifact_name)`` for every artifact
        lying on the map.
        """
        artifacts = set(artifacts)
        keyframe = self._need_keyframe or t % KEYFRAME_INTERVAL == 0

        if keyframe:
            food_part = {"set": [[x, y, v] for (x, y), v in food.items()]}
            art_part = {"set": [[x, y, n] for x, y, n in sorted(artifacts)]}
        else:
            food_part = {
                "add": [
                    [x, y, v]
                    for (x, y), v in food.items()
                    if self._prev_food.get((x, y)) != v
                ],
                "del": [
                    [x, y] for (x, y) in self._prev_food if (x, y) not in food
                ],
            }
            art_part = {
                "add": [
                    [x, y, n] for x, y, n in sorted(artifacts - self._prev_artifacts)
                ],
                "del": [
                    [x, y, n] for x, y, n in sorted(self._prev_artifacts - artifacts)
                ],
            }

        self._write(
            {
                "kind": "key" if keyframe else "delta",
                "t": t,
                "agents": agents,
                "food": food_part,
                "artifacts": art_part,
                "food_total": food_total,
                "n_agents": len(agents),
                "n_infected": n_infected,
            }
        )

        self._prev_food = dict(food)
        self._prev_artifacts = artifacts
        self._need_keyframe = False

    def close(self):
        if not self.fp.closed:
            self.fp.close()

    def _write(self, entry: dict):
        try:
            self.fp.write(json.dumps(entry, default=_jsonable) + "\n")
        except Exception as e:  # never let logging kill a run
            print(f"Failed logging world state at t={entry.get('t')}: {e}")


def _jsonable(obj):
    """Fallback for numpy scalars, which appear in energy/food values."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
