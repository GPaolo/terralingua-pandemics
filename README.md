# pandemics

Contact-sickness scenarios for [TerraLingua](https://github.com/cognizant-ai-lab/terralingua), a multi-agent simulation in which LLM-powered beings live on a shared grid. This package adds a sickness that spreads by contact, with incubation, protective equipment, a health center, remains and burials. It ships two tools that read a run folder: a run viewer and an epidemic anthropologist that writes a report and answers questions about a run.

Most beings are told nothing about the sickness. A few personas, such as the health workers, know about it and may tell the others. The rest learn from their own symptoms, from funerals, and from each other.

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/GPaolo/terralingua-pandemics.git
cd terralingua-pandemics
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs terralingua from its main branch, plus this package
cp .env.example .env        # put your model key there
```

## Run a scenario

Each disease is one preset at the top of the repository. Run from this folder:

```bash
terralingua ebola                              # the Ebola preset, 150 beings, 200 days
terralingua ebola --max_ts 20 --init_agents 12 # any setting can be overridden
terralingua --list                             # every preset found under this folder, plus the two built-in ones
```

A run writes `logs/<exp_name>/` under the working directory. Set `TL_LOGS_DIR` in the shell to write and read runs somewhere else. `--resume` restarts a run from its latest checkpoint.

### From the launcher

The [TerraLingua launcher](https://github.com/GPaolo/terralingua_launcher) is a web page that configures and starts runs. Install it in the same environment and point it at this folder:

```bash
pip install git+https://github.com/GPaolo/terralingua_launcher.git
terralingua-launcher --workdir .
```

Pick the `ebola` preset, change settings in the form, and launch. The page also starts the viewer and the anthropologist below with its "Open viewer" and "Open anthropologist" buttons.

## The scenario

`run.scenario: pandemics` selects the package. `run.scenario_options` in the preset holds every sickness parameter. The code is in `pandemics/`:

- `epidemic.py`: the `Epidemic` mechanic and its `EpidemicOptions`. Infection state lives in the mechanic and is saved with the world checkpoint.
- `artifacts.py`: `ppe` (protective equipment), `health_center`, and `remains`. Beings cannot create them; the scenario seeds them.
- `personas.json`: the personas of the Ebola setting, named by `agent.personas_path` in the preset and handed out by TerraLingua. Each entry has a `persona` text, an optional `name`, a `count`, and a `role`, which the scenario reads from the entry. The first beings created get them in file order. Every other being gets a human first name and no persona. Beings with the role named by `ppe_role` start with protective equipment. Some personas know about the sickness.
- `instructions.md`: empty on purpose.
- `state_log.py`: writes the per-step world state file the two tools read.

The package also declares which options apply only when another option turns a rule on, for example the burial multipliers without burials. A run that sets such an option gets a warning at start, and `python -m terralingua.config evaluate --preset ebola` lists every option with its state.

### How the sickness works

1. At `outbreak_step`, `init_infected` beings catch it. The infection incubates for a random number of steps between `incubation_min` and `incubation_max`. An incubating being passes nothing on and is told nothing.
2. Then the being is feverish for `mobile_days` steps. It can still move and act. It passes the sickness on at `feverish_multiplier` times the base chance. From the first sick step it loses `energy_multiplier` minus one extra energy per step, on top of the normal drain. Over the sick period a share `case_fatality` of the sick die. The daily chance is derived from it and rises with the number of sick days, so deaths come late in the sickness.
3. After `mobile_days` steps it is bedridden. It cannot move or take energy, and food under it stays untouched. After `infection_duration` sick steps it recovers and is immune.
4. Transmission: each step, every being within `infection_radius` of a sick host or of unburied remains catches it with chance `infection_probability`, times the host factor (`feverish_multiplier` for a feverish host, 1 for a bedridden host or for remains), times the being's protective equipment factor. Giving energy, taking energy, and handing over an artifact reach only a being on an adjacent cell. Each is a contact: with a sick host it is an extra exposure at `contact_multiplier` times the base chance, times the host factor.
5. A health center heals each infected being within its radius, incubating or sick, with its `heal_probability` per step, and multiplies the daily death chance of the sick ones by its `hazard_multiplier`.
6. A being that dies sick leaves remains at the end of the following step. Beings within `funeral_announcement_radius` hear of the death and where the remains lie. A being next to remains may bury them. The burier takes one exposure at `burial_infection_multiplier` times the base chance, and every other being on a cell next to the remains takes one at `burial_bystander_multiplier`, whether or not it meant to attend. Both come on top of the exposure the remains give each step. Remains spread for `remains_lifespan` minus one steps, then vanish.

The health center and the protective equipment are placed at the end of step 0.

### What the scenario writes

Next to the files every TerraLingua run writes, the mechanic adds:

- `world_state.jsonl`: one line per step with each being's position, energy, remaining time, inventory size and infection status, the food cells and the artifacts on the map. The format is documented in `pandemics/state_log.py`. Grid worlds only. The grid size is written once, so keep `dynamic_grid_scaling: false`, as the preset does.
- World log events: `VIRAL_INFECTION` (with `infection_id`, `source_kind`, `source_infection_id`, `incubation`), `VIRAL_EXPOSURE` (one line per chance to catch the sickness, with `probability`, `protection`, `infected`), `VIRAL_HEALED`, `BURIAL` (with `remains`, `attendees`, `infected`), and `IDENTITY` (a being's name, role and persona). Deaths from the sickness are `AGENT_DIED` events with `reason: sickness`.

## Watch a run

The viewer is a local web page for following a run while it is going and for scrubbing through it afterwards. It is a separate process from the run:

```bash
python -m pandemics.viewer                      # serves logs/ on http://127.0.0.1:8000
python -m pandemics.viewer --logs /data/runs --port 9999
```

It shows the world map with food, artifacts, beings and their infection status, a being inspector with energy, time left, inventory and genome, the action and the private memory behind it, the chat feed, the artifacts, and charts of population, sickness, artifacts and model spend. The transmission chain and an R0 estimate appear for runs with infections. The viewer reads no pickle, so a run copied from elsewhere can be opened without executing its contents.

## Analyze a run

The anthropologist reads only the files a run writes as it goes, so it works on a run still in progress.

```bash
python -m pandemics.anthropologist              # serves logs/ on http://127.0.0.1:8010
python -m pandemics.anthropologist logs/<exp>   # opens one run first
```

The page lists every run under the logs root that has a state file. For the active run it shows the report's metric tiles and plots, with a button to regenerate them, and a chat with the anthropologist. The chat needs `ANTHROPIC_API_KEY` in the environment or in `.env`; the metrics and plots work without it. **Compare runs** averages several runs as seeds of one configuration, or lays them side by side, with a summary table of attack rate, R0, peak, deaths and realized protection.

The same report and comparison exist as commands:

```bash
python -m pandemics.anthropologist.report logs/<exp>              # metrics.json, timeseries.csv and plots in logs/<exp>/epidemic_analysis/
python -m pandemics.anthropologist.compare logs/run_a logs/run_b --mode average
python -m pandemics.anthropologist.chat logs/<exp>                # the chat in the terminal
```

### How the chat works

The model navigates the run folder with read-only tools (`list_files`, `read_file`, `grep_files`) and keeps field notes per run in `epidemic_analysis/notes.md`, reloaded into its context every session. Every tool call appears in the transcript. Only `run_python`, real computation, waits for your approval before it runs; a switch turns on automatic approval for trusted runs. Plots it makes show up inline and land in `logs/<exp>/epidemic_analysis/chat/`. The web chat resumes across restarts.

Model-written code runs in a guarded worker process: imports outside a short allowlist (numpy, pandas, matplotlib, the metrics module and the standard library's data modules), dunder attribute access and `eval`-style names are rejected before anything runs; each call has a 30 second limit and a memory cap; file reads are confined to the run folder, this package, the interpreter's files, system fonts and the temp folder; writes to the chat plots folder, the temp folder and matplotlib's own cache. Python cannot be fully sandboxed in-language, so the approval step is the real security boundary. Keep it on for runs you do not trust.

## Calibrate the transmission probability

`python -m pandemics.calibrate_r0` runs scripted epidemics without a model on a small world with the preset's sickness settings, and prints the realized R0 for each candidate `infection_probability`, pooled over seeds. Beings drift toward food, give energy to bedridden neighbours and sometimes bury remains, with no protective equipment and no health center. Options: `--preset`, `--probs`, `--seeds`, `--steps`, `--index` (index cases per run), `--agents`, `--grid`, `--food`, `--spawn`, and `--keep DIR` to keep the runs.

## Add a disease

Copy `ebola.preset.yaml` to `<name>.preset.yaml`, set `name:` and `exp_name:` to the new name, change the `scenario_options` and the world settings, and run `terralingua <name>`. A new personas file is named by `agent.personas_path` in the preset, relative to the preset's folder. The package is general: every parameter of the sickness is a preset value.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The tests drive the world with scripted actions. No test calls a model.

## License

Apache 2.0, see `LICENSE`.
