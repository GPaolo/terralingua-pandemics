"""Sanity checks for the ebola anthropologist's loaders, metrics and report
(scenarios/ebola_simulation/anthropologist/).

Run from the repo root with: python tests/test_epidemic_analysis.py

No LLM calls: a scripted epidemic drives the real environment, then the
analysis is checked against the logs it wrote.
"""

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from core.environment.env import OpenGridWorld

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scenarios/ebola_simulation/anthropologist"))
import epidemic_utils as eu  # noqa: E402
import report  # noqa: E402

DIRECTIONS = ["up", "down", "left", "right", "stay"]


def build_epidemic_run(out: Path, steps: int = 60, grid: int = 9, n_agents: int = 8,
                       seed: int = 3):
    """A contact epidemic with PPE, driven by scripted random walks."""
    env = OpenGridWorld(
        grid_size=grid, vision_radius=3, init_agent_energy=400, lifespan=500,
        init_food=5, food_zones=None, dead_agent_food="none",
        use_inventory=True, log_path=out, verbose=0,
        viral_init_infected=1, viral_outbreak_step=0,
        viral_incubation_min=0, viral_incubation_max=2,
        viral_lifespan=25, viral_dropped_lifespan=8,
        viral_infection_radius=1, viral_infection_probability=0.7,
        viral_energy_multiplier=3.0, viral_death_probability=0.1,
        ppe_protection=0.1,
        init_artifacts=[
            {"name": "ppe_01", "type": "ppe", "agent": "b1"},
            {"name": "ppe_02", "type": "ppe", "agent": "b2"},
        ],
    )
    tags = [f"b{i}" for i in range(n_agents)]
    for tag in tags:
        env.add_agent(agent_tag=tag, agent_name=tag, agent_type="text")

    json.dump(
        {
            "agent": {"model": "scripted"},
            "env": {"grid_size": grid, "viral_infection_radius": 1,
                    "viral_infection_probability": 0.7, "ppe_protection": 0.1,
                    "viral_lifespan": 25, "viral_incubation_min": 0,
                    "viral_incubation_max": 2},
            "run": {"exp_name": out.name, "max_ts": steps},
        },
        open(out / "params.json", "w"),
    )

    # Cluster everyone so the outbreak reliably takes off.
    poses = {tag: (2 + i // 3, 2 + i % 3) for i, tag in enumerate(tags)}
    env.restart_env(agent_poses=poses)
    env.rng = np.random.default_rng(seed)
    rng = random.Random(seed)

    for _ in range(steps):
        actions = {}
        for tag in list(env.agent_registry):
            direction = "stay" if env._count_sick(tag) else rng.choice(DIRECTIONS)
            actions[tag] = {"action": "move", "message": "",
                            "params": {"direction": direction}}
        env.step(actions)
    env.close()


def main():
    tmp = Path(tempfile.mkdtemp())
    run = tmp / "epidemic_test"
    try:
        steps = 60
        build_epidemic_run(run, steps=steps)

        meta, frames = eu.load_frames(run)
        assert meta["kind"] == "meta"
        assert len(frames) == steps + 1, \
            f"expected {steps + 1} frames, got {len(frames)}"
        for fr in frames:
            for tag, a in fr["agents"].items():
                assert a["n_sick"] <= a["n_viral"], (fr["t"], tag)
        print("PASS: world frames parse, one per step, n_sick <= n_viral throughout")

        events = eu.load_events(run)
        infections = eu.infection_records(events)
        seeds = [r for r in infections if r["source_artifact"] is None]
        assert len(seeds) == 1, f"expected 1 index case, got {len(seeds)}"
        assert seeds[0]["generation"] == 0
        assert len(infections) >= 3, (
            f"outbreak fizzled ({len(infections)} infections) — pick another seed"
        )
        by_name = {r["artifact"]: r for r in infections}
        for r in infections:
            if r["source_artifact"] is not None:
                assert r["source_artifact"] in by_name, r["artifact"]
                assert r["generation"] >= 1
        assert sum(r["secondary"] for r in infections) == len(infections) - len(seeds)
        print(f"PASS: transmission tree closed over {len(infections)} infections")

        for r in infections:
            assert r["outcome"] in ("died", "recovered", "active"), r
            if r["outcome"] == "recovered":
                assert r["removed_at"] is not None, r
            elif r["outcome"] == "active":
                assert r["removed_at"] is None, r
        outcomes = {r["outcome"] for r in infections}
        assert {"died", "recovered"} <= outcomes, (
            f"need both fates for the tree ({outcomes}) — reseed"
        )
        print("PASS: episode outcomes split died / recovered / active")

        deaths = eu.death_records(events)
        series = eu.status_series(frames, infections, deaths)
        assert series[-1]["cum_infections"] == len(infections)
        assert series[-1]["cum_deaths"] == len(deaths)
        assert series[-1]["cum_deaths_virus"] == sum(
            1 for d in deaths if d["reason"] == "sickness"
        )
        for s, fr in zip(series, frames):
            assert s["alive"] == len(fr["agents"])
            assert (s["susceptible"] + s["incubating"] + s["sick"]
                    + s["recovered"] == s["alive"])
            assert s["cum_deaths_virus"] + s["cum_deaths_other"] == s["cum_deaths"]
        assert series[-1]["cum_deaths_virus"] > 0, \
            "no virus deaths — raise viral_death_probability or reseed"
        assert any(s["recovered"] > 0 for s in series), \
            "nobody recovered — lower viral_death_probability or reseed"
        print("PASS: status series is a disjoint partition, deaths split by "
              "cause, recoveries counted")

        exposures = eu.exposure_records(frames, infections, grid_size=9, radius=1)
        hits = {(e["t"], e["tag"]) for e in exposures
                if e["infected"] and e["contacts"] > 0}
        for r in infections:
            if r["source_artifact"] is not None:
                assert (r["t"], r["host_tag"]) in hits, (
                    f"infection of {r['host_tag']} at t={r['t']} "
                    "has no reconstructed exposure"
                )
        print(f"PASS: every transmitted infection matches a reconstructed exposure "
              f"({len(exposures)} exposure-steps)")

        assert series[1]["ppe_carriers"] == 2, "seeded PPE should show at frame 1"
        eff = eu.ppe_efficiency(exposures, 0.1)
        assert eff["without_ppe"]["contacts"] > 0
        print(f"PASS: PPE seeded and exposure split computed "
              f"(with: {eff['with_ppe']['contacts']}, "
              f"without: {eff['without_ppe']['contacts']} contacts)")

        r0 = eu.r0_table(infections)
        assert r0["total_infections"] == len(infections)
        assert r0["completed"] + r0["censored"] == len(infections)
        intervals = eu.serial_intervals(infections)
        assert all(i >= 0 for i in intervals)
        print(f"PASS: R0 table consistent (overall mean R {r0['overall_mean_r']}, "
              f"{r0['censored']} censored)")

        out = tmp / "analysis"
        metrics = report.generate(run, out)
        json.dumps(metrics)
        expected = ["metrics.json", "timeseries.csv", "epidemic_curves.png",
                    "infections.png", "transmission_tree.png",
                    "secondary_cases.png", "ppe.png"]
        for name in expected:
            path = out / name
            assert path.exists() and path.stat().st_size > 0, name
        assert sum(1 for _ in open(out / "timeseries.csv")) == len(series) + 1
        print("PASS: report generates all plots, metrics.json and timeseries.csv")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
