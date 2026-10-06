"""Per-step world state of a run, written as JSON Lines to ``world_state.jsonl``.

The core logs events. This file records the world itself, one line per step,
flushed as it is written, so the viewer and the anthropologist can follow a
run while it is going and scrub through it afterwards.

Line 1 is a ``meta`` header. Every later line is one step::

    {"kind":"meta","schema_version":7,"grid_size":25,"max_food_value":10.0,
     "provenance":"recorded",
     "agent_fields":["row","col","energy","time","n_inv","n_viral","n_sick",
                     "n_ppe","n_recovered","n_bedridden"]}

    {"kind":"key","t":0,"agents":{"being0":[10,10,100.0,100,0,0,0,0,0,0]},
     "food":{"set":[[3,4,10.0]]},"artifacts":{"set":[[10,21,"Marker_1","text"]]},
     "air":{"set":[[10,10,1.0]]},
     "food_total":5000.0,"n_agents":10,"n_infected":0,"n_sick":0,"n_bedridden":0}

    {"kind":"delta","t":1,"agents":{"being0":[10,11,109.0,99,0,0,0,0,0,0]},
     "food":{"add":[[7,2,10.0]],"del":[[10,11]]},
     "artifacts":{"add":[],"del":[]},
     "air":{"add":[[10,10,0.375],[10,11,1.0]],"del":[]},
     "food_total":4936.0,"n_agents":10,"n_infected":0,"n_sick":0,"n_bedridden":0}

``agents`` is written in full every step. ``food``, ``artifacts`` and ``air``
(the particle load of each cell with infectious air, empty without airborne
spread) are changes against the previous line, with a full ``key`` line at the
first step and every ``KEYFRAME_INTERVAL`` steps, so a reader seeks to any
step by replaying a bounded number of lines. Positions are ``[row, col]``. An
infinite energy is written as ``null``.

A fresh run truncates the file. A resumed run appends to it and starts with a
``key`` line; if the file is missing, it starts with the header.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

SCHEMA_VERSION = 7

#: Full world snapshot every this many steps.
KEYFRAME_INTERVAL = 50

#: Order of the per-being value arrays, mirrored in the ``meta`` header.
AGENT_FIELDS = [
    "row",
    "col",
    "energy",
    "time",
    "n_inv",
    "n_viral",
    "n_sick",
    "n_ppe",
    "n_recovered",
    "n_bedridden",
]


class WorldStateLogger:
    """Writes one JSON line per step to ``world_state.jsonl``."""

    def __init__(
        self,
        filepath: Path | str,
        grid_size: int,
        max_food_value: float,
        append: bool = False,
    ):
        self.save_path = Path(filepath)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            not append
            or not self.save_path.exists()
            or self.save_path.stat().st_size == 0
        )
        self.fp = open(self.save_path, "a" if append else "w", buffering=1)

        self._prev_food: Dict[Tuple[int, int], float] = {}
        self._prev_artifacts: Set[Tuple[int, int, str, str]] = set()
        self._prev_air: Dict[Tuple[int, int], float] = {}
        self._need_keyframe = True

        if header:
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
        artifacts: Iterable[Tuple[int, int, str, str]],
        food_total: float,
        n_infected: int,
        n_sick: int = 0,
        n_bedridden: int = 0,
        air: Iterable[list] = (),
    ):
        """Record the world at step ``t``.

        ``agents`` maps a being's tag to a list ordered as :data:`AGENT_FIELDS`.
        ``artifacts`` yields ``(row, col, name, kind)`` for every artifact on
        the map. ``air`` yields ``(row, col, load)`` for every cell with
        infectious air.
        """
        artifacts = set(artifacts)
        air = {(x, y): v for x, y, v in air}
        keyframe = self._need_keyframe or t % KEYFRAME_INTERVAL == 0

        if keyframe:
            food_part = {"set": [[x, y, v] for (x, y), v in food.items()]}
            art_part = {"set": [list(a) for a in sorted(artifacts)]}
            air_part = {"set": [[x, y, v] for (x, y), v in sorted(air.items())]}
        else:
            food_part = self._changes(self._prev_food, food)
            art_part = {
                "add": [list(a) for a in sorted(artifacts - self._prev_artifacts)],
                "del": [list(a) for a in sorted(self._prev_artifacts - artifacts)],
            }
            air_part = self._changes(self._prev_air, air)

        self._write(
            {
                "kind": "key" if keyframe else "delta",
                "t": t,
                "agents": agents,
                "food": food_part,
                "artifacts": art_part,
                "air": air_part,
                "food_total": food_total,
                "n_agents": len(agents),
                "n_infected": n_infected,
                "n_sick": n_sick,
                "n_bedridden": n_bedridden,
            }
        )

        self._prev_food = dict(food)
        self._prev_artifacts = artifacts
        self._prev_air = air
        self._need_keyframe = False

    @staticmethod
    def _changes(
        previous: Dict[Tuple[int, int], float], current: Dict[Tuple[int, int], float]
    ) -> dict:
        """Cell values that changed or appeared, and cells that went away."""
        return {
            "add": [
                [x, y, v] for (x, y), v in current.items() if previous.get((x, y)) != v
            ],
            "del": [[x, y] for (x, y) in previous if (x, y) not in current],
        }

    def close(self):
        if not self.fp.closed:
            self.fp.close()

    def _write(self, entry: dict):
        try:
            self.fp.write(json.dumps(entry, default=_jsonable) + "\n")
        except Exception as e:
            print(f"Failed logging world state at t={entry.get('t')}: {e}")


def _jsonable(obj):
    """Fallback for numpy scalars, which appear in energy and food values."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
