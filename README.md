# TerraLingua

**Paper:** [Link](https://www.researchgate.net/publication/402263491_TerraLingua_Emergence_and_Analysis_of_Open-endedness_in_LLM_Ecologies) - [ArXiv](https://arxiv.org/abs/2603.16910)

**Dataset:** https://huggingface.co/datasets/GPaolo/TerraLingua

**Dataset dashboard:** https://aianthropology.decisionai.ml/

![TerraLingua agents](assets/environment.gif)

A multi-agent simulation framework for studying emergent behavior, artifact creation, and cultural evolution.

LLM-powered agents (Claude or other models) interact in a shared 2D grid environment — foraging for resources, creating text artifacts, reproducing, and communicating — enabling research into how language-using agents develop social structure and culture over time.

After each experiment, the **AI Anthropologist** — itself an LLM agent — analyzes the simulation logs to annotate agent behaviors, infer group dynamics, classify artifacts, and trace cultural lineages, providing a qualitative and quantitative account of what emerged.

An overview of the TerraLingua system and of the AI-Anthropologist is shown in the figure below.

![TerraLingua and the AI Anthropologist](assets/whole.png)


## Installation

Requires **Python 3.13+**.

**Using venv:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Using conda:**

```bash
conda create -n terralinguia python=3.13
conda activate terralinguia
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API key(s):

```bash
cp .env.example .env
```

## Running Experiments

Run directly with `main.py` using CLI flags:

```bash
python main.py --exp_name my_experiment --init_agents 10 --max_ts 200 --model claude-haiku-4-5
```

Or use `run_experiment.sh`, a fully annotated template with all available options documented:

```bash
bash run_experiment.sh
```

Logs are written to `logs/<exp_name>/`.

## Watching an experiment

`viz/` is a local web dashboard for following a run as it happens and scrubbing
back through it afterwards. Start it in a second terminal — it is deliberately a
separate process from the experiment:

```bash
python -m viz                     # serves ./logs on http://127.0.0.1:8000
python -m viz --logs /data/logs --port 9999
```

It shows the world map (food, artifacts, beings, movement trails, the selected
being's vision), a being inspector with energy, age, inventory and OCEAN-5
genome, the action and the private `internal_memory` behind it, the chat feed,
the artifacts, and charts of food, population, artifact creation and token spend.
Viral runs additionally get infection status on the map — amber ⧖ for a being
still incubating, red ☣ once it is sick — and R₀ per generation.

Runs are picked up automatically: a run is **live** until `open_gridworld.log`
records `END_RUN`, and the view follows new steps as they are written.

Keyboard: `space` play/pause, `←`/`→` step (hold shift for 10), `home`/`end`,
`esc` closes a dialog.

### Where the data comes from

Per-step world state is written to `logs/<exp_name>/world_state.jsonl` — agent
positions, energy, age, infection, plus food and artifact deltas with a keyframe
every 50 steps. Disable with `--no-log_world_state`; it costs roughly 20 MB for a
3000-step, 100-agent run.

Runs recorded before that file existed are reconstructed on first open (see
`viz/backfill.py`), and the UI marks them **reconstructed**: positions are
inferred from what the beings observed, and the food map shows only cells
somebody has visited. The dashboard reads no `.pkl` file, so a run downloaded
from elsewhere can be opened without executing its contents.

### Seeding artifacts

The environment can seed text artifacts by itself, at the start of a run or at a chosen timestep, via `--init_artifacts path/to/file.json`. The file is a JSON list of entries (see [`init_artifacts_example.json`](init_artifacts_example.json)):

```json
{
    "name": "welcome_stone",
    "payload": "Welcome to this world.",
    "pose": [10, 10],
    "lifespan": -1,
    "step": 0
}
```

| Field | Required | Default | Meaning |
|---|---|---|---|
| `name` | yes | — | Artifact name (uniquified with `_1` suffixes on collision) |
| `payload` | no | `""` | Text content |
| `pose` | no | random free cell | `[x, y]` map cell where the artifact appears |
| `agent` | no | — | Agent tag or name whose inventory receives the artifact (mutually exclusive with `pose`) |
| `lifespan` | no | `-1` | Steps before the artifact expires (`-1` = never) |
| `step` | no | `0` | Timestep at which the artifact is seeded (`0` = before the first observation) |

Seeded artifacts are ordinary text artifacts (agents can read, pick up, modify, destroy them), are logged as `ARTIFACT_ADDED` events with creator `environment`, and pending seeds survive checkpoint resumes without duplication.

### Viral artifacts

A virus-like artifact type for epidemic experiments: it has no content, cannot be created or acted on by agents, spreads probabilistically by contact — by default to agents standing in one of the 8 directly adjacent cells, diagonals and grid wraparound included — multiplies its host's energy consumption, and drops on the map for a limited time when its host dies. Enable it by setting `--viral_init_infected` > 0; widen the transmission range with `--viral_infection_radius` if you want an airborne virus instead. See `run_viral_experiment.sh` for an annotated example and the `viral_*` flags in `python main.py --help` for all knobs (outbreak step, incubation range, infection radius/probability, infection and corpse lifespans, energy multiplier). Every transmission is logged as a `VIRAL_INFECTION` event, from which the empirical R0 can be computed with `analysis_scripts/compute_r0.py`.

One timestep is one day, and an infection runs in two phases.

**Incubating.** For `--viral_incubation_min`..`--viral_incubation_max` days (2–21 by default, drawn per infection) the host is a **silent carrier**. It moves, eats and takes energy exactly as a healthy being does, it infects nobody, it pays no extra energy — and it is told nothing at all. The virus is filtered out of its own inventory listing, so nothing in its prompt reveals that it is carrying anything. Nor can anyone else tell: inventories are not observable, so a carrier looks identical to a healthy being right up until it collapses. Set both bounds to `0` for the old behaviour, where infection and illness are simultaneous.

**Sick.** Once the incubation runs out the being is told so in plain language every step. A sick being cannot move, cannot take energy from others, and loses its appetite — it will not eat the food it is standing on, and that food is left untouched for others (and for itself, once it recovers). It can still broadcast messages and still *receive* energy, so its energy strictly declines until either the infection expires or another being walks over and gives it some: asking for help is the only way out. Only now does it transmit, and only now does `--viral_energy_multiplier` apply.

`--viral_lifespan` measures the **symptomatic** period alone: the latency sits in front of it rather than eating into it, so every host stays infectious for the same number of days whether it incubated for 2 or for 21. Because hosts stop moving and stop foraging once ill, they become stationary sources and realised R0 falls below the free-mixing estimate annotated in `run_viral_experiment.sh` — measure it rather than assuming it. Budget enough timesteps for the latency: each generation is pushed a mean ~11 days later, so short runs will report far more censored infections and may show no outbreak at all.

### Reproducing paper experiments

The `paper_experiment_scripts/` folder contains the exact scripts used to run each experiment from the paper. All scripts must be run from the project root:

```bash
bash paper_experiment_scripts/run_core.sh
```

## Supported Agent Models

Pass any of the following keys via `--model`:

| Key | Provider | Notes |
|---|---|---|
| `claude-haiku-4-5` | Anthropic | Fast, cost-effective |
| `claude-sonnet-4-6` | Anthropic | Default |
| `o4-mini` | OpenAI | |
| `o3-mini` | OpenAI | |
| `gpt-5.1` | OpenAI | |
| `gpt-5-mini` | OpenAI | |
| `QWEN2.5` | Local (vLLM) | Qwen2.5-32B-Instruct |
| `QWEN3` | Local (vLLM) | Qwen3-32B |
| `DeepSeek-R1-32` | Local (vLLM) | DeepSeek-R1-Distill-Qwen-32B |
| `DeepSeek-R1-70` | Local (vLLM) | DeepSeek-R1-Distill-Llama-70B |

### Local models (vLLM)

Local models require a running [vLLM](https://github.com/vllm-project/vllm) server. Start one (or more) on any of the default ports (`9000–9003`, `9010–9012`):

```bash
vllm serve Qwen/Qwen3-32B --port 9000
```

Then pass the ports via `--ports` (defaults to `9000 9001 9002 9003 9010 9011 9012`):

```bash
python main.py --model QWEN3 --ports 9000 9001
```

TerraLingua will auto-discover which ports are hosting the requested model and load-balance across them.

## Data Analysis

Analysis is performed by the **AI Anthropologist**, a post-hoc LLM-based framework that annotates agent behaviors, infers group dynamics, classifies artifacts, and traces cultural lineages. See [`analysis_scripts/AI_ANTHROPOLOGIST.md`](analysis_scripts/AI_ANTHROPOLOGIST.md) for a detailed description of the pipeline.

Scripts follow a numbered order and must be run from the **project root** (they import from `core` and `analysis_scripts` as packages):

| Script | Description |
|---|---|
| `001_llm_agent_analyser.py` | Annotate agent logs with LLM-generated behavior labels |
| `002_make_graph.py` | Build interaction graphs and compute network metrics |
| `003_llm_group_analyser.py` | Group-level behavioral analysis |
| `004_artifact_analysis.py` | Compute artifact complexity metrics |
| `005_artifact_classification.py` | Classify artifacts into behavioral categories |
| `006_artifact_philogeny.py` | Analyze artifact genealogy and conceptual ancestry |

```bash
python analysis_scripts/001_llm_agent_analyser.py
```

## Data Visualization

Notebooks in `analysis_scripts/notebooks/` mirror the analysis pipeline:

| Notebook | Description |
|---|---|
| `n000_general_stats.ipynb` | Overall experiment statistics |
| `n001_llm_agent_analyser.ipynb` | Per-agent behavior visualization |
| `n002_graph_analysis.ipynb` | Interaction network plots |
| `n003_llm_group_analysis.ipynb` | Group dynamics |
| `n004_artifact_analysis.ipynb` | Artifact complexity over time |
| `n005_artifact_categories.ipynb` | Classification results |
| `n006_artifact_phylogeny.ipynb` | Artifact lineage trees |
| `n007_interactive_phylogeny.ipynb` | Interactive phylogeny explorer |

```bash
jupyter notebook analysis_scripts/notebooks/
```

## Citation

If you use TerraLingua in your research, please cite:

```bibtex
@techreport{paolo26terralingua,
title = "TerraLingua: Emergence and Analysis of Open-Endedness in LLM Ecologies",
author = "Giuseppe Paolo and Jamieson Warner and Hormoz Shahrzad and Babak Hodjat and Risto Miikkulainen and Elliot Meyerson",
year = 2026,
month = jan,
institution = "Cognizant AI Lab",
url = "https://arxiv.org/abs/2603.16910",
doi = "10.48550/arXiv.2603.16910",
number = "2026-01",
}
```
