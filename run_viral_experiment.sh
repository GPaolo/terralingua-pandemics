#!/bin/bash

python main.py \
    \
    `# Experiment` \
    --exp_name              "viral_experiment_name" \
    --exp_description       "Experiment description" \
    --max_ts                300 `# outbreak at 20 + several 30-step infection generations` \
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
    --grid_size             100 `# size of the grid environment (grid_size x grid_size)` \
    --init_agents           100 `# initial number of agents in the environment` \
    --init_human_agents     0 `# initial number of human agents in the environment` \
    --min_agents            80 `# minimum number of agents in the environment` \
    --init_agent_energy     100 `# initial energy for each agent` \
    --init_food             500 `# initial amount of food in the environment` \
    --food_zones            4 `# number of food zones in the environment (areas where food can spawn more frequently)` \
    --food_mechanism `# flag to enable the food mechanism` \
    --agent_lifespan        100 `# lifespan of agents in the environment` \
    --vision_radius         6 `# vision radius of agents` \
    --dead_agent_food       "none" `# food type from dead agents ("single": dead agent leaves all its energy as food in its cell, "none": dead agents do not leave food, "area": a 3x3 area around the dead agent position is filled with food)` \
    --artifact_creation `# flag to enable artifact creation` \
    --artifact_creation_cost 0 `# cost of creating artifacts` \
    --no-inert_artifacts `# flag to disable inert artifacts. To have inert artifacts use the inert_artifacts flag instead of this one.` \
    --no-reproduction_allowed `# flag to disallow agents to reproduce. To enable reproduction, use the reproduction_allowed flag instead.` \
    --reproduction_cost     50 `# energy cost for agents to reproduce` \
    \
    `# Viral artifacts — starting point for a virus with R0 ~= 2.5` \
    `# R0 ~= probability x avg_agents_within_radius x infectious_steps. Measure the` \
    `# realized R0 with: python analysis_scripts/compute_r0.py logs/<exp_name>` \
    `# then rescale: new_probability = probability x (2.5 / measured_R0)` \
    --viral_init_infected   3 `# index cases at the outbreak (0 disables viral artifacts entirely)` \
    --viral_outbreak_step   1 `# timestep at which the outbreak infects the index cases` \
    --viral_lifespan        30 `# infectious period: steps an infection lasts in an agent's inventory (-1: permanent until the host dies)` \
    --viral_dropped_lifespan 10 `# steps a viral artifact dropped at its host's death survives on the map, still spreading (-1: forever)` \
    --viral_infection_radius 2 `# max distance in cells at which a viral artifact can spread to other agents` \
    --viral_infection_probability 0.1 `# per-step, per-neighbor transmission probability. Calibrate this to hit the target R0` \
    --viral_energy_multiplier 2.0 `# energy consumption multiplier per hosted viral artifact (K). Keep mild so hosts survive the infectious period`
