"""Calibrate ``infection_probability`` against a realized R0.

Scripted epidemics without a model, on a small world with the sickness
settings of a preset. Beings drift toward food, so they cluster as in real
runs; they give energy to bedridden neighbours (a caregiving contact) and
sometimes bury remains, so proximity, touch and burial exposures all happen.
No protective equipment and no health center: this measures the raw
transmission. Each candidate probability runs on several seeds. The realized
R0 is the mean number of secondary infections of the completed infections in
generations 0 and 1, pooled over the seeds.

Run it from the folder that holds the preset:

    python -m pandemics.calibrate_r0
    python -m pandemics.calibrate_r0 --probs 0.3 0.5 --seeds 6 --steps 300
"""

import argparse
import logging
import random
import shutil
import tempfile
from pathlib import Path

from terralingua.config.compose import compose
from terralingua.environment.grid_env import OpenGridWorld

from pandemics.anthropologist import epidemic_utils as eu
from pandemics.epidemic import Epidemic, EpidemicOptions

DIRECTIONS = ["up", "down", "left", "right"]
STEP_OF = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
STAY = {"action": "move", "params": {"direction": "stay"}}


def signed_delta(a, b, g):
    """Shortest step from a to b along one axis of a grid that wraps around, signed."""
    return (b - a + g // 2) % g - g // 2


def build_run(options: EpidemicOptions, world: dict, seed: int, out: Path, steps: int, agents: int):
    """One scripted epidemic, written to ``out`` like a real run."""
    env = OpenGridWorld(log_path=out, headless=True, **world)
    mechanic = Epidemic(options)
    env.attach(mechanic)
    rng = random.Random(seed)
    g = env.grid_size
    cells = rng.sample([(r, c) for r in range(g) for c in range(g)], agents)
    poses = {f"b{i}": cell for i, cell in enumerate(cells)}
    for tag, cell in poses.items():
        env.add_agent(tag, tag, "text", position=cell)
    env.restart_env(seed=seed, agent_poses=poses)

    for _ in range(steps):
        if not env.agent_registry:
            break
        actions = {}
        for tag in sorted(env.agent_registry):
            if mechanic.health(tag) == "bedridden":
                actions[tag] = STAY
                continue
            pos = env.agent_pos[tag]
            neighbors = env.agents_within(tag, 1)
            stricken = [t for t in neighbors if mechanic.health(t) == "bedridden"]
            remains = mechanic.remains_in_reach(env, tag)
            if stricken and env.agent_energy[tag] > 40 and rng.random() < 0.25:
                target = env.agent_names[rng.choice(stricken)]
                actions[tag] = {"action": "give", "params": {"target": target, "amount": 5}}
            elif remains and rng.random() < 0.10:
                actions[tag] = {"action": "bury", "params": {"name": remains[0]}}
            elif neighbors and rng.random() < 0.02:
                target = env.agent_names[rng.choice(neighbors)]
                actions[tag] = {"action": "give", "params": {"target": target, "amount": 1}}
            else:
                direction = rng.choice(DIRECTIONS)
                if env.food and rng.random() < 0.7:
                    target = min(env.food, key=lambda c: env.distance(pos, c))
                    dr = signed_delta(pos[0], target[0], g)
                    dc = signed_delta(pos[1], target[1], g)
                    toward = [
                        d for d, (r, c) in STEP_OF.items()
                        if (r and r * dr > 0) or (c and c * dc > 0)
                    ]
                    if toward:
                        direction = rng.choice(toward)
                actions[tag] = {"action": "move", "params": {"direction": direction}}
        env.step(actions)
    env.logger.fp.flush()
    mechanic.world_log.close()


def measure(run_dir: Path) -> dict:
    """The R0 table of one run, from its event log."""
    return eu.r0_table(eu.infection_records(eu.load_events(run_dir)))


def calibrate(probs, seeds, steps, index, agents, options: EpidemicOptions, world: dict, root: Path, log=print):
    """[(probability, pooled R0 or None, early cases, completed infections), ...]."""
    results = []
    for prob in probs:
        cases = infections = 0
        weighted = 0.0
        for seed in range(seeds):
            run = root / f"p{prob:g}_s{seed}"
            run.mkdir(parents=True, exist_ok=True)
            candidate = options.model_copy(update={"infection_probability": prob, "init_infected": index})
            build_run(candidate, world, seed, run, steps, agents)
            table = measure(run)
            early = [row for row in table["per_generation"] if row["generation"] in (0, 1)]
            cases += sum(row["cases"] for row in early)
            weighted += sum(row["cases"] * row["mean_secondary"] for row in early)
            infections += table["completed"]
            log(f"  p={prob:g} seed={seed}: {table['total_infections']} infections, "
                f"{table['completed']} completed, {table['censored']} active")
        r0 = weighted / cases if cases else None
        results.append((prob, r0, cases, infections))
        log(f"p={prob:g}: pooled R0 (generations 0-1) = "
            f"{'none' if r0 is None else f'{r0:.2f}'} "
            f"({cases} early cases, {infections} completed infections)\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="ebola", help="preset whose sickness settings are used")
    parser.add_argument("--probs", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--index", type=int, default=5,
                        help="index cases per run; more gives a larger generation-0 sample while the world is still susceptible")
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--grid", type=int, default=30)
    parser.add_argument("--food", type=int, default=500, help="initial food cells")
    parser.add_argument("--spawn", type=int, default=10, help="food cells added per step")
    parser.add_argument("--keep", type=Path, default=None, help="keep the runs under this folder")
    args = parser.parse_args()

    # The world's food warnings say nothing about transmission.
    logging.getLogger("terralingua.environment").setLevel(logging.ERROR)
    cfg = compose(args.preset)
    options = EpidemicOptions(**{
        **cfg.run.scenario_options,
        "ppe_per_worker": 0,
        "health_centers_path": None,
    })
    world = dict(
        grid_size=args.grid,
        vision_radius=cfg.env.vision_radius,
        init_agent_energy=cfg.env.init_agent_energy,
        lifespan=cfg.env.agent_lifespan,
        init_food=args.food,
        food_zones=cfg.env.food_zones,
        food_spawn_rate=args.spawn,
        food_mechanism=True,
        drop_food_on_death=cfg.env.drop_food_on_death,
        use_inventory=True,
        use_colors=False,
        reproduction_cost=-1,
        artifact_creation_cost=0,
    )
    root = args.keep or Path(tempfile.mkdtemp(prefix="r0_calib_"))
    results = calibrate(args.probs, args.seeds, args.steps, args.index, args.agents, options, world, root)
    print(f"{'probability':>12} {'R0 (gens 0-1)':>14} {'early cases':>12}")
    for prob, r0, cases, _ in results:
        print(f"{prob:>12g} {('none' if r0 is None else f'{r0:.2f}'):>14} {cases:>12}")
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
