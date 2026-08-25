"""Computes the empirical R0 of a viral artifact run from the environment log.

Every infection is logged as a VIRAL_INFECTION event whose `artifact` is the
new copy and whose `source_artifact` is the artifact that transmitted it
(null for environment-seeded index cases). Each artifact therefore is one
host-infection episode, and its number of secondary infections is the number
of events naming it as source_artifact.

R0 is estimated as the mean number of secondary infections per *completed*
infection (artifact already removed by the end of the log), reported per
generation: generation 0 are the seeded index cases, generation N+1 are
infections caused by generation N. The early generations (0 and 1), when the
population is still mostly susceptible, are the best R0 estimate; later
generations trend toward the effective reproduction number as susceptible
agents run out.

Usage:
    python analysis_scripts/compute_r0.py logs/<exp_name>
    python analysis_scripts/compute_r0.py logs/<exp_name>/open_gridworld.log
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_events(log_path: Path):
    infections = []
    removed = set()
    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") == "VIRAL_INFECTION":
                infections.append(entry)
            elif entry.get("event") == "ARTIFACT_REMOVED":
                artifact = entry.get("artifact", {})
                if artifact.get("art_type") == "viral":
                    removed.add(artifact.get("name"))
    return infections, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log", type=Path, help="Path to open_gridworld.log or the experiment log dir"
    )
    args = parser.parse_args()

    log_path = args.log
    if log_path.is_dir():
        log_path = log_path / "open_gridworld.log"

    infections, removed = load_events(log_path)
    if not infections:
        print(f"No VIRAL_INFECTION events found in {log_path}")
        return

    secondary = defaultdict(int)  # artifact name -> secondary infections caused
    source_of = {}  # artifact name -> source artifact name (None for seeds)
    infected_at = {}  # artifact name -> timestep of infection
    for event in infections:
        name = event["artifact"]["name"]
        source = event.get("source_artifact")
        source_of[name] = source
        infected_at[name] = event.get("timestamp")
        if source is not None:
            secondary[source] += 1

    def generation(name):
        gen = 0
        while source_of.get(name) is not None:
            name = source_of[name]
            gen += 1
        return gen

    by_generation = defaultdict(list)
    censored = 0
    for name in source_of:
        if name in removed:
            by_generation[generation(name)].append(secondary[name])
        else:
            censored += 1  # still infectious at the end of the log

    print(f"Infections: {len(source_of)} total, {censored} still active (excluded)")
    print(f"{'gen':>4} {'cases':>6} {'mean secondary (R)':>20} {'max':>5}")
    overall = []
    for gen in sorted(by_generation):
        counts = by_generation[gen]
        overall.extend(counts)
        mean = sum(counts) / len(counts)
        print(f"{gen:>4} {len(counts):>6} {mean:>20.2f} {max(counts):>5}")
    if overall:
        print(f"\nOverall mean R (completed infections): {sum(overall) / len(overall):.2f}")
    early = by_generation.get(0, []) + by_generation.get(1, [])
    if early:
        print(f"Empirical R0 (generations 0-1):        {sum(early) / len(early):.2f}")


if __name__ == "__main__":
    main()
