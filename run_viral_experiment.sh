#!/bin/bash

python main.py \
    \
    `# Experiment` \
    --exp_name              "oliv_7" \
    --exp_description       "Test experiment contact virus incubation" \
    --max_ts                25  \
    --max_parallel_workers 20 \
    --no-save_video \
    \
    `# Agent LLM` \
    --model                 "claude-haiku-4-5" \
    \
    `# Agents` \
    --agents_name_prefix    "being" `# name prefix for agents, e.g. being_0, being_1, etc.` \
    --exogenous_motivation  "base" `# motivation mechanism for agents, e.g. "base", "creative", "survival", "none".` \
    --genome                "ocean_5" `# genome configuration for agents, e.g. "ocean_5", "no_traits", "sentence_directed".` \
    --max_history           1 `# number of past timesteps to include in agent observations` \
    --internal_memory_size  150 `# size of the internal memory for agents` \
    --use_internal_memory `# flag to enable internal memory for agents` \
    --use_inventory `# flag to enable inventory for agents` \
    --no-use_colors `# flag to disable color usage for agents (agents can set a color for themselves that other agents can see)` \
    \
    `# Environment` \
    --grid_size             25 `# size of the grid environment (grid_size x grid_size)` \
    --init_agents           15 `# initial number of agents in the environment` \
    --init_human_agents     0 `# initial number of human agents in the environment` \
    --min_agents            15 `# minimum number of agents in the environment` \
    --init_agent_energy     100 `# initial energy for each agent` \
    --init_food             500 `# initial amount of food in the environment` \
    --food_zones            2 `# number of food zones in the environment (areas where food can spawn more frequently)` \
    --food_mechanism `# flag to enable the food mechanism` \
    --agent_lifespan        100 `# lifespan of agents in the environment` \
    --vision_radius         6 `# vision radius of agents` \
    --dead_agent_food       "none" `# food type from dead agents ("single": dead agent leaves all its energy as food in its cell, "none": dead agents do not leave food, "area": a 3x3 area around the dead agent position is filled with food)` \
    --artifact_creation \
    --artifact_creation_cost 0 \
    --no-inert_artifacts  \
    --no-reproduction_allowed \
    --reproduction_cost     50 \
    \
    `# Viral artifacts — a contact-transmitted virus with R0 ~= 2.5, from a single index case` \
    `# One timestep is one day. An infection has two phases: it incubates silently for` \
    `# viral_incubation_min..max days (the host moves, eats and infects nobody, and is told` \
    `# nothing), then turns symptomatic for viral_lifespan days — frozen, no appetite,` \
    `# infectious, draining energy fast. Latency does NOT eat the infectious window.` \
    `#` \
    `# R0 ~= probability x avg_agents_within_radius x infectious_steps, where infectious_steps` \
    `# is viral_lifespan. At radius 1 the neighbourhood is the 8 adjacent cells, so on a` \
    `# 25x25 grid with 15 beings that is 0.5 x 0.19 x 30 ~= 2.9. Measure the realized R0 with:` \
    `#   python analysis_scripts/compute_r0.py logs/<exp_name>` \
    `# then rescale: new_probability = probability x (2.5 / measured_R0).` \
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
    --viral_lifespan        30 `# infectious period: days an infection stays symptomatic (-1: permanent until the host dies)` \
    --viral_dropped_lifespan 10 `# steps a viral artifact dropped at its host's death survives on the map, still spreading (-1: forever)` \
    --viral_infection_radius 1 `# max distance in cells at which a viral artifact can spread (1 = contact: the 8 adjacent cells)` \
    --viral_infection_probability 0.5 `# per-step, per-neighbor transmission probability. Calibrate this to hit the target R0` \
    --viral_energy_multiplier 5.0 `# energy drain multiplier once symptomatic (K). Incubating hosts pay the normal 1/step`
