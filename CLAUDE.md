# TerraLingua — working notes

Multi-agent LLM simulation: "beings" on a toroidal grid explore, eat food, talk,
write text artifacts, and (optionally) infect each other. `main.py` →
`core/experiment/runner.py` drives it; `viz/` is a dashboard for watching runs.

Read the README first for install and CLI. This file is the stuff that will
otherwise cost you an hour.

## Running things

- Use `.venv/bin/python`. `pyproject.toml` pins `==3.12.*` (pygame has no 3.14
  wheels); the README's "Python 3.13+" is stale.
- `main.py` imports the whole runner chain at module level, so a missing optional
  dependency fails at `--help`, not at use.
- Tests are plain scripts at the repo root, run from the root, no pytest:
  `python test_world_state_log.py`, `test_viral_artifacts.py`,
  `test_init_artifacts.py`. They print `PASS:` lines and assert. Match that style
  for new ones.
- Lint: `.venv/bin/ruff check <paths>`. `core/` already fails `E501` in many
  places (122 at the time of writing) — check only what you touched, and don't
  treat the rest as regressions.
- **Don't spend API credits to test simulation changes.** `OpenGridWorld` takes a
  plain dict of actions, so you can drive a full run with scripted moves and no
  LLM calls — see `build_run()` in `test_world_state_log.py`.

## Where the data lives

Everything lands in `logs/<exp_name>/`. The split that matters:

**Written as the run goes** (safe to tail; this is all a live view can use)
`world_state.jsonl`, `open_gridworld.log`, `agent_logs/<tag>.jsonl`,
`agent_logs/token_counts.jsonl`, `agent_logs/<tag>_genome.json`, `frames/*.png`,
`params.json`.

**Only written when the run ends** (`env.close()`, the logger's own `close()`, and
ffmpeg afterwards) — absent from any run in progress or crashed
`messages.json`, `artifacts.json`, `food_counts.json`, `agent_names.json`,
`agent_events.json`, `agent_trajectories.pkl`, `env_state.pkl`, `video.mp4`.

Traps:

- **`timestamp` has three spellings.** Int in `open_gridworld.log`, **string** in
  `agent_logs/<tag>.jsonl`, and `timestep` (int) in `token_counts.jsonl`.
  `messages.json` keys are strings. Coerce on ingest.
- **`input_prompt` is ~12 KB per agent per tick** and fully redundant. Drop it
  when reading agent logs or you will hold hundreds of MB for nothing.
- **`AGENT_ADDED.position` is not where the agent starts.** It is the pre-reset
  random placement. `ENV_RESET.agent_poses` is the truth for t=0.
- **`agent_trajectories` is deduplicated** — a position is appended only when the
  agent actually moves, so index ≠ timestep. Never zip it against ticks.
- **All loggers append.** Re-running an `exp_name` concatenates into the existing
  files and rewinds the timestamps. `world_state.jsonl` is the exception: it
  truncates on a fresh start and appends only on `--resume`.
- Observation cells are relative-offset string keys `"(dr, dc)"` whose values mix
  stringified food floats, agent names, and `A(text): Name` artifact tokens.
- `env_state.pkl` / `checkpoint_latest.pkl` are pickles: only the final state, and
  loading one executes arbitrary code. `viz/` deliberately reads neither.

## Coordinates

- Positions are **`(row, col)`**. `MOVE_DICT` in `core/environment/env.py` has
  `up = (-1, 0)`, so component 0 is the row. The renderer matches
  (`Rect(y*w, x*h)`).
- **The prompt uses a different frame.** `LLMAgent._format_list` emits
  `(ry, -rx)`, so coordinates an agent talks about in its messages do not match
  the coordinates in the logs. Do not compare them directly.
- The grid is a torus (`env.wrap_xy`). When drawing paths, split segments that
  wrap or you get a line across the whole map.

## Known sharp edges (all live, all pre-existing)

- `runner.py` hardcodes the first two spawns at `(10, 10)` and `(12, 10)`, so
  `--grid_size <= 12` starts an agent off-grid.
- `_respawn_if_needed` builds `LLMAgent(...)` without `genome=` or
  `exogenous_motivation=`, so agents respawned to satisfy `--min_agents` silently
  get a random genome and `"base"` motivation regardless of your flags.
- **Agent placement is not reproducible from the seed** — the same seed gives
  different starting positions across processes. Don't write tests that assume a
  fixed layout; assert on properties instead.
- `_get_nearby_agents` and `_get_avail_actions` index raw `x+dx, y+dy` without
  wrapping, so `give` / `take` are unavailable across the torus seam even when the
  agents can see each other.
- `init_agent_energy < 0` means *infinite* energy: `agent_energy` becomes
  `np.inf`, which is not valid JSON. Guard before serializing.
- **"Infected" is derived state, not a flag** — it means "has a `ViralArtifact` in
  `agent_inventories[tag]`", read via `_count_viral`. It now gates three things:
  `move` (still offered, but restricted to `stay`), `take` (withdrawn), and the
  *implicit* eat in `step`. So `move` is no longer unconditionally available, and
  ending a turn on a food cell no longer implies eating it. Don't add a parallel
  `infected` flag; it will drift.
- `env.reset(agent_tag)` resets **one agent**. The whole-world reset is
  `env.restart_env(seed=..., **options)`.
- `env.add_agent` requires `agent_name` as well as `agent_tag`.
- `AgentLogger` does *not* add the `agent_logs/` subdirectory — `LLMAgent` appends
  it before constructing one. Pass `<exp_dir>/agent_logs` if you build one
  yourself.

## The dashboard (`viz/`)

`python -m viz` serves `./logs` on :8000. Reads JSON/JSONL only, never a pickle,
so it works mid-run and on untrusted downloads.

- `viz/reader.py` tails each file from a stored byte offset and skips partial
  final lines; it resets if a file shrinks (a re-run under the same name).
- `viz/backfill.py` reconstructs `world_state.jsonl` for runs predating it, by
  Viterbi-decoding "did this move land or was it blocked" against what every being
  observed. Exact on well-observed runs; inherently ambiguous for a being nobody
  can see. Output is tagged `provenance: reconstructed` and the UI says so.
- The world log runs **one step ahead of the agent decisions** — the frame at
  `last_step` has positions but no actions. Use `last_decision_step` for anything
  that pairs the map with what agents said or thought.

**Adding a field to `world_state.jsonl`:** bump `SCHEMA_VERSION` and extend
`AGENT_FIELDS` in `core/environment/world_logger.py`, emit it in
`OpenGridWorld._log_world_state`, and update `viz/backfill.py` so old runs still
produce the same shape.

## Charts

Chart work in `viz/static/` follows the `dataviz` skill: one y-axis per chart,
categorical hues assigned in fixed order and never cycled, a single-hue ramp for
magnitude. Green is reserved for the food ramp and kept out of the categorical
slots so a being never reads as the ground it stands on.

**Beings are not colored by identity.** Every being is `--s1`; `agentColor()` takes
no argument. Hue on the map is spent on state, not on who: infection is the dot's
**fill** (`--status-critical`), and selection is a neutral `--select` **bracket on
the cell box** (`drawSelection`), never a ring on the dot. Don't reintroduce a
per-being palette — reserved red sits below the legibility floor against both the
orange and the magenta slot (ΔE 6.8 and 9.0, floor 15), so a healthy being would
read as a sick one. Identity is carried by the brackets, the trail, the tooltip and
the Beings list. Selection and infection must also never share a geometry: they
were concentric rings a pixel apart, and at mid cell sizes the selection ring
painted straight over the infection ring.
