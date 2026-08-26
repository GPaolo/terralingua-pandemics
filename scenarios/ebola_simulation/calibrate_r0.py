"""Calibrate viral_infection_probability against a target realized R0.

Scripted no-LLM epidemics on the ebola scenario's world (30x30, 50 beings,
one index case, knobs mirrored from run_viral_experiment.sh), each measured
by analysis_scripts/compute_r0.py and pooled across seeds per candidate
probability. Beings drift toward food (so they cluster like real runs),
give energy to adjacent bedridden beings (caregiving touch) and sometimes
bury remains, so proximity, touch and burial exposures all realize.
No PPE: this measures the raw transmission economy.

Usage:
    python scenarios/ebola_simulation/calibrate_r0.py
    python scenarios/ebola_simulation/calibrate_r0.py --probs 0.15 0.25 --seeds 6
"""

import argparse
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.environment.env import OpenGridWorld  # noqa: E402

DIRECTIONS = ["up", "down", "left", "right"]
STEP_OF = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# Scenario knobs from run_viral_experiment.sh (except the probability under test).
ENV = dict(
    grid_size=30, vision_radius=6, init_agent_energy=100, lifespan=150,
    init_food=500, food_zones=3, food_spawn_rate=10, dead_agent_food="none",
    use_inventory=True, verbose=0,
    viral_init_infected=1, viral_outbreak_step=0,
    viral_incubation_min=2, viral_incubation_max=21,
    viral_lifespan=12, viral_mobile_days=4, viral_mobile_infectiousness=0.3,
    viral_dropped_lifespan=10, viral_infection_radius=1,
    viral_contact_multiplier=1.8, viral_energy_multiplier=6,
    viral_death_probability=0.11,
    burials=True, burial_infection_multiplier=2.5, funeral_announcements=True,
    funeral_announcement_radius=10, funeral_attendance_multiplier=1.5,
)
N_AGENTS = 50


def signed_delta(a, b, g):
    """Shortest toroidal step from a to b along one axis, signed."""
    return (b - a + g // 2) % g - g // 2


def build_run(prob, seed, out, steps, index_cases):
    env = OpenGridWorld(log_path=out, viral_infection_probability=prob,
                        **{**ENV, "viral_init_infected": index_cases})
    for i in range(N_AGENTS):
        env.add_agent(agent_tag=f"b{i}", agent_name=f"b{i}", agent_type="text")
    env.restart_env(seed=seed)
    rng = random.Random(seed)
    g = env.grid_size

    for _ in range(steps):
        actions = {}
        for tag in list(env.agent_registry):
            if env._count_bedridden(tag):
                actions[tag] = {"action": "move", "message": "",
                                "params": {"direction": "stay"}}
                continue
            pos = env.agent_pos[tag]
            neighbors = [
                t for t in env.agent_registry
                if t != tag and env._toroidal_distance(pos, env.agent_pos[t]) <= 1
            ]
            stricken = [t for t in neighbors if env._count_bedridden(t)]
            remains = env._ground_viral_nearby(tag)
            if stricken and env.agent_energy[tag] > 40 and rng.random() < 0.25:
                actions[tag] = {"action": "give", "message": "", "params":
                                {"target": env.agent_names[rng.choice(stricken)],
                                 "amount": 5}}
            elif remains and rng.random() < 0.10:
                actions[tag] = {"action": "bury", "message": "",
                                "params": {"name": next(iter(remains))}}
            elif neighbors and rng.random() < 0.02:
                actions[tag] = {"action": "give", "message": "", "params":
                                {"target": env.agent_names[rng.choice(neighbors)],
                                 "amount": 1}}
            else:
                direction = rng.choice(DIRECTIONS)
                if env.food and rng.random() < 0.7:
                    target = min(env.food, key=lambda c: env._toroidal_distance(pos, c))
                    dr = signed_delta(pos[0], target[0], g)
                    dc = signed_delta(pos[1], target[1], g)
                    options = ([d for d, (r, c) in STEP_OF.items()
                                if (r and r * dr > 0) or (c and c * dc > 0)])
                    if options:
                        direction = rng.choice(options)
                actions[tag] = {"action": "move", "message": "",
                                "params": {"direction": direction}}
        env.step(actions)
    env.close()


ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+\d+\s*$")


def measure(run_dir):
    """Run compute_r0.py on the run; return (per-gen {gen: (cases, mean)}, raw)."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "analysis_scripts/compute_r0.py"), str(run_dir)],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout
    gens = {int(m[1]): (int(m[2]), float(m[3]))
            for line in out.splitlines() if (m := ROW.match(line))}
    return gens, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs", type=float, nargs="+",
                        default=[0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--index", type=int, default=5,
                        help="index cases per run; more tightens the gen-0 "
                             "sample while the world is still ~fully susceptible")
    parser.add_argument("--keep", type=Path, default=None,
                        help="keep the runs under this directory")
    args = parser.parse_args()

    root = args.keep or Path(tempfile.mkdtemp(prefix="r0_calib_"))
    results = []
    for prob in args.probs:
        cases = infections = 0
        weighted = 0.0
        for seed in range(args.seeds):
            run = root / f"p{prob:g}_s{seed}"
            run.mkdir(parents=True, exist_ok=True)
            build_run(prob, seed, run, args.steps, args.index)
            gens, raw = measure(run)
            for gen in (0, 1):
                n, mean = gens.get(gen, (0, 0.0))
                cases += n
                weighted += n * mean
            infections += sum(n for n, _ in gens.values())
            print(f"  p={prob:g} seed={seed}: "
                  + (raw.splitlines()[0] if raw.strip() else "no outbreak"))
        r0 = weighted / cases if cases else None
        results.append((prob, r0, cases, infections))
        print(f"p={prob:g}: pooled R0 (gens 0-1) = "
              f"{'—' if r0 is None else f'{r0:.2f}'} "
              f"({cases} early cases, {infections} completed infections)\n")

    print(f"{'probability':>12} {'R0 (gens 0-1)':>14} {'early cases':>12}")
    for prob, r0, cases, _ in results:
        print(f"{prob:>12g} {('—' if r0 is None else f'{r0:.2f}'):>14} {cases:>12}")
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
