# Ebola anthropologist

Post-hoc analysis for runs of `../run_viral_experiment.sh`. Everything reads
only the files a run writes as it goes (`params.json`, `world_state.jsonl`,
`open_gridworld.log`, `agent_logs/`), so it all works on a run still in
progress and never loads a pickle.

(The older `analysis_scripts/` AI Anthropologist pipeline was built for a
different set of experiments; this folder is self-contained and
epidemic-specific.)

## `dashboard.py` — the main interface

```
python scenarios/ebola_simulation/anthropologist/dashboard.py            # serves logs/
python scenarios/ebola_simulation/anthropologist/dashboard.py logs/<exp> # opens one run first
```

Serves http://127.0.0.1:8010: every run under the logs root is in the header
selector; the report's metric tiles and plots for the active run on the left
(with a "Regenerate report" button), the anthropologist chat on the right —
each run keeps its own chat session. Chat needs `ANTHROPIC_API_KEY` or a
`.env` (`--model` defaults to `claude-opus-5`, with server-side fallback to
`claude-opus-4-8` on refusals); the metrics/plots panes work without
credentials.

**Compare runs…** (also `#compare` in the URL) selects several runs and either
**averages them as seeds** of one configuration (mean line, min–max band) or
overlays them **side by side** (≤ 6 runs, one categorical color each), with a
summary table (attack rate, R0, peak, deaths, PPE protection per run). Same
thing from the CLI: `python …/compare.py logs/run_a logs/run_b --mode
average|sidebyside`; outputs land in `<logs root>/_comparisons/`.

The model navigates the logs with Claude Code-style read-only tools
(`list_files`, `read_file`, `grep_files` — host-side, no code execution, no
approval needed; they auto-strip the ~12 KB `input_prompt` fields from
`.jsonl` lines) and keeps persistent field notes per run (`write_note` →
`epidemic_analysis/notes.md`, reloaded into its context every session). Only
`run_python` — real computation — appears in the transcript and **waits for
your Approve/Deny click** before running (toggle "auto-run code" to skip the
click on trusted runs). Plots it makes show up inline and land in
`logs/<exp_name>/epidemic_analysis/chat/`. The stable system prompt + tool
prefix is prompt-cached, so multi-turn sessions stay cheap.

Also: a **Stop** button interrupts a running turn at the next tool boundary
(pending approvals are denied, an in-flight sandbox call is killed); typing
**@** in the input autocompletes run files (the model reads them with
read_file); and the conversation **resumes across restarts** — transcript and
history persist to `epidemic_analysis/chat_session.json` after every turn
("Reset chat" starts clean; sandbox variables are not restored on resume).

## How safe is the code execution?

Model-written code goes through `sandbox.py`, in layers:

1. **AST screen** — imports outside an allowlist (numpy, pandas, matplotlib,
   epidemic_utils, stdlib data modules), dunder attribute access, and
   `eval`/`exec`/`getattr`-style names are rejected before anything runs.
   Nothing network-capable is importable.
2. **Separate worker process** — per-call 30s timeout, memory cap where the
   OS honours it, crash isolation; killed workers restart cleanly.
3. **Filesystem confinement** — reads only inside the run, the repo and the
   interpreter; writes only under `epidemic_analysis/chat/` and tempdir;
   deletion disabled outright.
4. **Approve-before-run** — the default in both UIs, and the actual security
   boundary: Python cannot be fully sandboxed in-language, and the logs the
   model reads contain other LLMs' text (a prompt-injection vector). Layers
   1–3 stop accidents and casual injection; the approval click is the
   guarantee. Leave auto-run off for runs you don't trust.

`tests/test_anthropologist_sandbox.py` exercises all of this without LLM calls.

## `report.py` — metrics and plots, no LLM

```
python scenarios/ebola_simulation/anthropologist/report.py logs/<exp_name>
```

Prints a summary and writes to `logs/<exp_name>/epidemic_analysis/`:

- `metrics.json` — attack rate, peak, R0 per generation (same estimator as
  `analysis_scripts/compute_r0.py`), serial interval, incubation, mortality,
  PPE coverage/transfers/efficiency.
- `timeseries.csv` — per-day susceptible / incubating / sick / dead / PPE
  carriers / new + cumulative infections.
- `epidemic_curves.png`, `infections.png`, `transmission_tree.png`,
  `secondary_cases.png`, `ppe.png`.

PPE efficiency: the env does not log failed transmissions, so
`epidemic_utils.exposure_records` reconstructs every susceptible-next-to-
infectious contact from world-state positions and splits the realized
per-contact transmission rate by PPE carriage. Treat it as an estimate.

## `chat.py` — terminal fallback

Same anthropologist in the terminal, y/N approval per snippet
(`--auto-run` to skip):

```
python scenarios/ebola_simulation/anthropologist/chat.py logs/<exp_name>
```

## Layout

- `epidemic_utils.py` — loaders/derivations; `compute_all(run_dir)` returns
  `(metrics, series, infections, exposures)`.
- `agent.py` — system prompt (log-format traps included), the five tools,
  turn loop shared by dashboard and chat.
- `compare.py` — cross-run averaging and side-by-side comparison plots.
- `filetools.py` — the read-only navigation tools (scope confinement,
  input_prompt stripping, truncation).
- `sandbox.py` — the guarded executor described above.
- `static/` — the dashboard UI (tokens mirror `viz/static/style.css`).

Tests, from the repo root, no LLM calls:

```
python tests/test_epidemic_analysis.py
python tests/test_anthropologist_sandbox.py
```
