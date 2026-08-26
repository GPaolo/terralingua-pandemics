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
- Agent placement is reproducible only through `env.restart_env(seed=...)` (or by
  setting `env.rng` first) — it seeds the generator the placement helpers draw
  from. Placement used to hit the global `np.random`, which made layouts differ
  across processes and `test_world_state_log` flaky.
- `give` / `take` are **contact**: offered and executed only against a being on
  an adjacent cell (toroidal, so the seam works), and every transfer with a
  symptomatic party rolls an exposure at `viral_infection_probability x
  viral_contact_multiplier` (`_touch_exposure`, both directions, PPE applies).
  `give_artifact` still uses the old FOV gate, whose raw `x+dx, y+dy` loop
  (`_get_nearby_agents`, and the FOV scan in `_get_avail_actions`) does not
  wrap — so handing an *artifact* over still fails across the torus seam.
- `init_agent_energy < 0` means *infinite* energy: `agent_energy` becomes
  `np.inf`, which is not valid JSON. Guard before serializing.
- **"Infected" is derived state, not a flag** — it means "has a `ViralArtifact` in
  `agent_inventories[tag]`", read via `_count_viral`. Don't add a parallel
  `infected` flag; it will drift.
- **Recovery is permanent immunity.** An infection cleared alive (natural
  expiry or a health center) bumps `agent_recoveries[tag]`, and `infect_agent`
  refuses any agent with a recovery on record — spread, outbreak, touch and
  burial risk all pass through it. `agent_known_recoveries` is the subset the
  host lived through *symptomatic*: only those earn the persistent
  `RECOVERED_AGENT_NOTICE` ("you are immune") — announcing a recovery the host
  never felt would leak the silent infection. Death is a per-symptomatic-step
  roll whose hazard ramps 0 → `viral_death_probability` across the infectious
  window (`_death_hazard`), scaled by the best `hazard_multiplier` of any
  health center in reach (supportive care, min not product — the PPE rule);
  an incubating carrier never rolls. The ebola scenario sets the center's
  `heal_probability` to 0: care halves the hazard, recovery is surviving the
  12 days out. Every `AGENT_DIED` event carries `infected` (and
  `agent_events.json` `died_infected`) because a bedridden host that starves
  is a disease death filed under "hunger".
- **Infected ≠ sick ≠ bedridden.** A `ViralArtifact` carries an `incubation`
  countdown, a `symptomatic` property and a `days_symptomatic` counter. The
  illness has three states: incubating (silent, harmless), **dry** (the first
  `viral_mobile_days` symptomatic steps: feverish notice, full action set,
  transmits at `viral_mobile_infectiousness` x the rate, pays the drain) and
  **wet** (`_count_bedridden`): `move` restricted to `stay`, `take` and `bury`
  withdrawn, the *implicit* eat in `step` suppressed, fully infectious.
  `_count_viral` counts everything, `_count_sick` both symptomatic phases
  (the health notice, the drain, n_sick in the world log), `_count_bedridden`
  gates what the being can *do*. `viral_mobile_days = 0` is the old behaviour.
  The dry-day clock advances **after** the spread pass (`symptomatic_ticked`
  in `step`) so an infection transmits at the phase its host acted in;
  remains never take the dry discount (`_hosted_infectiousness` is only
  applied when the source has a host).
- **Incubation must stay silent, and that takes five edits, not one.** Nothing
  filters `infos` on the way to the prompt — `runner.py` pops `available_actions`
  and `LLMAgent._make_prompt` renders every remaining key verbatim. A carrier is
  hidden by: the `symptomatic` guard on the three `infos["Infection"]` messages
  (`_spread_viral_artifacts`, `_seed_viral_outbreak`, `_touch_exposure`), and
  `_is_hidden_infection` filtering both the inventory passive-effect loop and
  `_build_obs`'s `inventory_list`. The immune notice keys on
  `agent_known_recoveries` for the same reason. Add a new channel that names an
  artifact and you reopen the leak. Other beings can't tell either —
  observations render `artifacts_map` only, never someone else's inventory.
- `viral_lifespan` is the **symptomatic** period, not the total. `remaining_time`
  does not tick while `incubation > 0`, so latency sits in front of the infectious
  window rather than eating it. A corpse matures instantly (`_kill` zeroes
  `incubation`) so `viral_dropped_lifespan` keeps its plain meaning.
- With `--funeral_announcements`, a death that drops remains is announced
  **once** (`infos["Deaths"]`) to beings within `funeral_announcement_radius`
  (-1: everyone; further beings only learn by word of mouth), with walkable
  directions from `_compass_offset` ("3 cells up, 2 cells right" — MOVE_DICT
  words, shortest way around the torus) and **never the cause**. While
  `funeral_mourning_days` lasts (`buriable_after` on the artifact; 0 =
  buriable at once and the text says "may be buried", no "0 days") the
  `bury` action **stays offered but is refused with the days left** ("they
  may be buried in 2 days") — deliberate: attempting it is how a being
  learns how long to wait. The step the mourning ends a reminder goes out
  to earshot (`_pending_burial_reminders`) — skipped if the remains are
  already gone.
  **The burial is the funeral**: every other being within 1 cell of the grave
  rolls an exposure at `funeral_attendance_multiplier` (digger:
  `burial_infection_multiplier`; elsewhere: nothing), and the BURIAL event
  logs the `attendees`. Queued in `_kill` (`_pending_funerals`), flushed in
  `step`.
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
- **Any panel expands to full screen on a header double-click** (`setZoom` /
  `hookZoom`, Esc or the `#zoom-close` ✕ to leave). The panel is positioned out
  of the grid where it stands and never reparented, so every `draw*` keeps
  writing into the same nodes. `drawCharts` is the exception — it rebuilds its
  cards from scratch — so the expanded panel is tracked by **element id** in
  `state.zoom` and re-applied at the end of that redraw. Give any new chart card
  a stable `id` or it will drop out of full screen on the next step.

**Adding a field to `world_state.jsonl`:** bump `SCHEMA_VERSION` and extend
`AGENT_FIELDS` in `core/environment/world_logger.py`, emit it in
`OpenGridWorld._log_world_state`, and update `viz/backfill.py` so old runs still
produce the same shape. `AGENT_FIELDS` is **append-only** — the client reads the
array positionally (`healthOf`, the inspector destructure, the tooltip) and
ignores the `agent_fields` header it is sent. Guard new indices with `?? 0` so a
schema-1 file still renders. Schema 2 added `n_sick` at index 6 alongside the
per-step `n_sick` scalar; `n_viral - n_sick` is the silent-carrier count.
Schema 3 added `n_ppe` at index 7 (PPE artifacts carried; protection is the
min over the inventory, never a product — `_infection_protection`).
Schema 4 added `n_recovered` at index 8 (infections cleared alive, by cure or
natural expiry — deaths don't count). Schema 5 widened map artifact entries to
`[x, y, display_name, kind]` (kind: text/ppe/health_center/remains; ground
viral artifacts present as `remains_of_<host>` while their internal name — the
one transmission chains key on — never changes). "Recovered" on screen means
`n_recovered > 0 and n_viral == 0`, worn as a `--recovered` ring on the dot
and a dashed chart line, never a fill: no fill hue passes the legibility
floor against both the PPE blue and the base grey.

## Charts

Chart work in `viz/static/` follows the `dataviz` skill: one y-axis per chart,
categorical hues assigned in fixed order and never cycled, a single-hue ramp for
magnitude. Green is reserved for the food ramp and kept out of the categorical
slots so a being never reads as the ground it stands on.

**Beings are not colored by identity.** The base being is neutral `--being` grey;
`agentColor(a)` returns `--s1` blue only for a PPE carrier (state, still not
identity; validated grey↔blue ΔE 17.0 normal, worst CVD pair 9.1). Hue on the map
is spent on state, not on who: health is the dot's **fill** (and outranks the PPE
blue — a sick carrier reads sick), and selection is a neutral `--select` **bracket
on the cell box** (`drawSelection`), never a ring on the dot. Don't reintroduce a
per-being palette — reserved red sits below the legibility floor against both the
orange and the magenta slot (ΔE 6.8 and 9.0, floor 15), so a healthy being would
read as a sick one. Identity is carried by the brackets, the trail, the tooltip
and the Beings list. PPE is glyphed `⛨` (`PPE_GLYPH`) wherever health is glyphed,
so hue is never the only channel. Persona **roles** are the marker's *shape*
(`roleShape`: ▲ ■ ⬟ ⬢ assigned to roles alphabetically, never cycled — a fifth
role falls back to the circle; the square is axis-aligned so it never reads as
the artifact diamond). Fill, ring and glyphs behave identically on every shape. Selection and infection must also never share a geometry: they were
concentric rings a pixel apart, and at mid cell sizes the selection ring painted
straight over the infection ring.

**Health moves along the reserved status ramp, never into a categorical slot**:
`--status-warning` amber while incubating, `--status-critical` red once sick, via
`healthOf()` / `healthColor()` — one place, used by the map, the Beings pills, the
inspector and the tooltip. Amber sits ~48 ΔE from the red where the `--s2` orange
sits at 6.8, which is why "orange for incubating" is the wrong call however
natural it sounds. Both phases are always paired with a glyph (`⧖` / `☣`,
`HEALTH_GLYPH`) so hue is never the only channel — the tooltip is plain text and
has nothing else. On the population chart the two are plotted **disjoint**
(incubating = `n_infected - n_sick`), so they read as parts of one population
rather than one line drawn over another.
