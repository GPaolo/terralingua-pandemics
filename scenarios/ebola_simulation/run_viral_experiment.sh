#!/bin/bash

# Runs from anywhere: main.py lives at the repo root, the scenario files here.
SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCENARIO_DIR/../.."

python main.py \
    \
    `# Experiment` \
    --exp_name              "ebola_50_agents_30" \
    --exp_description       "Ebola-calibrated contact virus with incubation" \
    --max_ts                300  \
    --max_parallel_workers 20 \
    --no-save_video \
    --verbose               1 `# terminal chatter: 0 warnings only, 1 key events, 2 per-step debug` \
    \
    `# Agent LLM` \
    --model                 "gpt-5.6-luna" \
    \
    `# Agents` \
    --agents_name_prefix    "being" `# name prefix for agents, e.g. being_0, being_1, etc.` \
    --exogenous_motivation  "none" `# motivation mechanism for agents, e.g. "base", "creative", "survival", "none".` \
    --genome                "ocean_5" `# genome configuration for agents, e.g. "ocean_5", "no_traits", "sentence_directed".` \
    --max_history           1 `# number of past timesteps to include in agent observations` \
    --internal_memory_size  150 `# size of the internal memory for agents` \
    --max_message_size      50 `# max tokens per message; longer ones are cut off and the agent is told (-1: unlimited)` \
    --max_text_artifact_size 200 `# max tokens a text artifact can hold` \
    --use_internal_memory `# flag to enable internal memory for agents` \
    --use_inventory `# flag to enable inventory for agents` \
    --no-use_colors `# flag to disable color usage for agents (agents can set a color for themselves that other agents can see)` \
    --no-hereditary_persona \
    --personas       "$SCENARIO_DIR/personas_ebola.json" `# personas for the initial agents, assigned in file order` \
    --init_artifacts "$SCENARIO_DIR/init_artifacts_ebola.json" `# environment-seeded artifacts: 20 PPE in Miriam's (health worker) inventory` \
    \
    `# Environment` \
    --grid_size             30 `# size of the grid environment (grid_size x grid_size)` \
    --init_agents           50 `# initial number of agents in the environment` \
    --init_human_agents     0 `# initial number of human agents in the environment` \
    --min_agents            25 `# minimum number of agents in the environment` \
    --init_agent_energy     100 `# initial energy for each agent` \
    --init_food             500 `# initial amount of food in the environment` \
    --food_zones            3 `# number of food zones in the environment (areas where food can spawn more frequently)` \
    --food_spawn_rate       10 `# mean food cells spawned per step (Poisson); occupied draws fail silently` \
    --food_mechanism `# flag to enable the food mechanism` \
    --agent_lifespan        150 `# lifespan of agents in the environment` \
    --vision_radius         6 `# vision radius of agents` \
    --dead_agent_food       "none" \
    --artifact_creation \
    --artifact_creation_cost 0 \
    --no-inert_artifacts  \
    --no-reproduction_allowed \
    --reproduction_cost     50 \
    \
    `# Viral artifacts — a contact-transmitted virus with R0 ~= 2.5, from a single index case` \
    `# One timestep is one day. An infection has three phases: it incubates silently for` \
    `# viral_incubation_min..max days (the host moves, eats and infects nobody, and is told` \
    `# nothing), turns symptomatic-but-ambulatory for viral_mobile_days ("dry": feverish,` \
    `# draining energy, mildly infectious, still moving — this is how it travels), then` \
    `# goes "wet" for the rest of viral_lifespan — frozen, no appetite, fully infectious.` \
    `# Latency does NOT eat the infectious window.` \
    `#` \
    `# R0 ~= probability x avg_agents_within_radius x infectious_steps, where infectious_steps` \
    `# is viral_lifespan. At radius 1 the neighbourhood is the 8 adjacent cells. Ebola's` \
    `# R0 is 1.5-2.5; after changing any viral knob, measure the realized R0 with:` \
    `#   python analysis_scripts/compute_r0.py logs/<exp_name>` \
    `# then rescale: new_probability = probability x (2.5 / measured_R0).` \
    `# Full epidemic report (plots, PPE efficiency) and interactive analyst:` \
    `#   python scenarios/ebola_simulation/anthropologist/report.py logs/<exp_name>` \
    `#   python scenarios/ebola_simulation/anthropologist/dashboard.py logs/<exp_name>` \
    `# With one index case an outbreak fizzles out by chance roughly 1 run in 3 — expected,` \
    `# not a bug. Widen the grid or raise the probability if you need every run to take off.` \
    `#` \
    `# BUDGET FOR THE LATENCY: nothing at all transmits for the first few days, and each` \
    `# generation is pushed a mean ~11 days later. At --max_ts 25 a worst-case 21-day` \
    `# incubation means the index case never even falls ill, so you see no outbreak at all.` \
    `# Use several hundred steps for a real epidemic, and expect compute_r0.py to report` \
    `# many more censored (still-active) infections than before at the same horizon.` \
    --viral_init_infected   1 `# index cases at the outbreak (0 disables viral artifacts entirely)` \
    --viral_outbreak_step   0 `# timestep at which the outbreak infects the index cases` \
    --viral_incubation_min  2 `# min days between catching the virus and falling ill (silent, not infectious)` \
    --viral_incubation_max  21 `# max days between catching the virus and falling ill. Set both to 0 for no latency` \
    --viral_lifespan        12 `# infectious period: days an infection stays symptomatic (-1: permanent until the host dies). Ebola: onset -> death 6-16d, onset -> recovery ~2 weeks` \
    --viral_mobile_days     4 `# the first symptomatic days are "dry" (fever, weakness): the host still moves, eats and acts — this is when the disease travels` \
    --viral_mobile_infectiousness 0.3 `# transmission multiplier during those dry days; "wet" symptoms afterwards are bedridden and fully infectious` \
    --viral_dropped_lifespan 10 `# steps a viral artifact dropped at its host's death survives on the map, still spreading (-1: forever)` \
    --viral_infection_radius 1 `# max distance in cells at which a viral artifact can spread (1 = contact: the 8 adjacent cells)` \
    --viral_infection_probability 0.15 `# per-step probability of catching it from a symptomatic being on an adjacent cell. Calibrate this to hit the target R0` \
    --viral_contact_multiplier 4 `# touching (give/take energy) scales that exposure: 0.15 x 4 = 0.6 per contact — caregiving contact was the main Ebola transmission route` \
    --viral_energy_multiplier 6 `# energy drain multiplier once symptomatic (K): 6/day for 12 days with no eating. Deliberately lethal on its own, so that gifted energy (a contact risk) genuinely buys survival — supportive care roughly halved the real CFR` \
    --viral_death_probability 0.11 `# hazard ramps 0 -> this over the 12 sick days; 1-prod(1-0.11*t/12) ~= 50% CFR from the roll alone, ~70% combined with the energy drain above (untreated Zaire ebolavirus)` \
    --ppe_protection        0.1 `# multiplier on the infection probability of a being carrying PPE (0: immune, 1: no protection)` \
    --burials `# beings next to remains (a dropped viral artifact) can bury them, removing the artifact from the ground` \
    --burial_infection_multiplier 2.5 `# burying scales the burier's infection probability by this factor for that exposure (PPE still applies). Viral load peaks at death: corpses were the most infectious contacts of the 2014-16 outbreak`
