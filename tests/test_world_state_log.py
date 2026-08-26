"""Sanity checks for the per-step world log and the legacy-run reconstruction.

Run from the repo root with: python test_world_state_log.py

No LLM calls: the environment is driven with scripted actions, so this is safe
to run in CI.
"""

import json
import random
import shutil
import tempfile
from pathlib import Path

from core.agents.agent_logger import AgentLogger
from core.environment.env import OpenGridWorld
from core.environment.world_logger import KEYFRAME_INTERVAL
from viz.backfill import backfill
from viz.reader import RunReader

DIRECTIONS = ["up", "down", "left", "right", "stay"]


def build_run(out: Path, steps: int, grid: int = 14, n_agents: int = 5, seed: int = 0):
    """Drive a real environment with scripted actions, writing the real logs."""
    env = OpenGridWorld(
        grid_size=grid, vision_radius=4, init_food=60, log_path=out,
        food_zones=None, verbose=0, lifespan=500, init_agent_energy=100,
    )
    tags = [f"being{i}" for i in range(n_agents)]
    for tag in tags:
        env.add_agent(tag, agent_name=tag, agent_type="text")

    json.dump(
        {
            "agent": {"model": "scripted"},
            "env": {"grid_size": grid, "vision_radius": 4, "agent_lifespan": 500,
                    "init_agent_energy": 100},
            "run": {"exp_name": out.name, "max_ts": steps},
        },
        open(out / "params.json", "w"),
    )

    loggers = {t: AgentLogger(out / "agent_logs", t) for t in tags}
    rng = random.Random(seed)
    obs, infos = env.restart_env(seed=seed)
    truth = {0: dict(env.food)}

    for t in range(steps):
        actions = {}
        for tag in list(env.agent_registry):
            action = {"action": "move", "message": f"{tag} at {t}",
                      "params": {"direction": rng.choice(DIRECTIONS)}}
            actions[tag] = action
            loggers[tag].log(
                agent_name=tag, agent_tag=tag, time=str(t), action=action,
                observation=obs.get(tag, {}), internal_memory=f"tick {t}",
                available_actions=infos.get(tag, {}).get("available_actions", {}),
                input_prompt="(scripted)",
            )
        obs, _, _, _, infos = env.step(actions)
        truth[env.step_count] = dict(env.food)

    positions = {t: list(p) for t, p in env.agent_trajectories.items()}
    env.close()
    return truth, positions


def replay(path: Path):
    """Apply the deltas the same way a consumer would."""
    rows = [json.loads(line) for line in open(path)]
    meta, steps = rows[0], rows[1:]
    food, out = {}, {}
    for r in steps:
        if r["kind"] == "key":
            food = {(x, y): v for x, y, v in r["food"]["set"]}
        else:
            for x, y, v in r["food"]["add"]:
                food[(x, y)] = v
            for x, y in r["food"]["del"]:
                food.pop((x, y), None)
        out[r["t"]] = dict(food)
    return meta, steps, out


def test_deltas_replay_exactly():
    tmp = Path(tempfile.mkdtemp())
    try:
        steps = KEYFRAME_INTERVAL * 2 + 5   # force several keyframes
        truth, _ = build_run(tmp / "run", steps)
        meta, rows, replayed = replay(tmp / "run" / "world_state.jsonl")

        assert meta["kind"] == "meta", "first line must be the schema header"
        assert meta["provenance"] == "recorded"
        assert len(rows) == steps + 1, f"expected {steps + 1} frames, got {len(rows)}"
        print("PASS: one frame per step plus the initial state")

        keys = [r["t"] for r in rows if r["kind"] == "key"]
        assert keys == list(range(0, steps + 1, KEYFRAME_INTERVAL)), keys
        print(f"PASS: keyframes at every {KEYFRAME_INTERVAL} steps ({keys})")

        for t, expected in truth.items():
            got = {(int(a), int(b)): float(v) for (a, b), v in replayed[t].items()}
            want = {(int(a), int(b)): float(v) for (a, b), v in expected.items()}
            assert got == want, f"food mismatch replaying step {t}"
        print(
            "PASS: replaying deltas reproduces the food map at all "
            f"{steps + 1} steps"
        )

        totals = json.load(open(tmp / "run" / "food_counts.json"))
        logged = [r["food_total"] for r in rows]
        assert all(abs(a - b) < 1e-6 for a, b in zip(totals, logged)), "totals diverge"
        print("PASS: food_total agrees with food_counts.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


#: Thresholds, not equality. A being nobody can see leaves the same trace whether
#: it moved and came back or was blocked and moved once, so perfect recovery is
#: not always available -- see the module docstring in viz/backfill.py. Measured
#: over random scripted runs: ~99% of sightings and ~90% of paths. Agent
#: placement is not reproducible from the seed, so these must hold on any run.
MIN_SIGHTING_AGREEMENT = 0.95
MIN_EXACT_PATHS = 0.75


def test_backfill_recovers_positions():
    """A legacy run has no world log; reconstruct it and check it is right."""
    tmp = Path(tempfile.mkdtemp())
    try:
        run = tmp / "legacy"
        _, positions = build_run(run, 30, n_agents=6, seed=11)

        # Pretend the run predates the world log.
        (run / "world_state.jsonl").unlink()
        result = backfill(run)
        assert result is not None, "backfill should have produced a file"
        ok, total = result["sighting_agreement"]
        assert total > 0, "no sighting constraints -- test is not exercising anything"
        ratio = ok / total
        assert ratio >= MIN_SIGHTING_AGREEMENT, f"sightings disagree: {ok}/{total}"
        print(f"PASS: reconstruction agrees with sightings ({ok}/{total}, {ratio:.1%})")

        meta, rows, _ = replay(run / "world_state.jsonl")
        assert meta["provenance"] == "reconstructed"
        assert meta["food_source"] == "observed"
        print("PASS: reconstruction is labelled as such for the UI")

        # Collapsing repeats should reproduce the environment's own trajectories.
        recon = {}
        for r in rows:
            for tag, v in r["agents"].items():
                recon.setdefault(tag, []).append((v[0], v[1]))
        exact = 0
        for tag, want in positions.items():
            seen = []
            for p in recon[tag]:
                if not seen or seen[-1] != p:
                    seen.append(p)
            exact += seen == [tuple(map(int, w)) for w in want]
        share = exact / len(positions)
        assert share >= MIN_EXACT_PATHS, (
            f"only {exact}/{len(positions)} paths recovered exactly"
        )
        print(
            f"PASS: paths match the environment for {exact}/{len(positions)} agents "
            f"({share:.0%})"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reader_survives_a_rerun():
    """Re-running an exp_name must not splice two runs together."""
    tmp = Path(tempfile.mkdtemp())
    try:
        run = tmp / "rerun"
        build_run(run, 20)
        reader = RunReader(run)
        reader.refresh()
        assert reader.last_step == 20, reader.last_step

        shutil.rmtree(run)
        build_run(run, 6)
        reader.refresh()
        assert reader.last_step == 6, (
            f"stale state leaked in: last_step={reader.last_step}"
        )
        assert reader.meta()["last_decision_step"] <= 6
        print("PASS: reader resets when a run is rewritten under the same name")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_deltas_replay_exactly()
    test_backfill_recovers_positions()
    test_reader_survives_a_rerun()
    print("\nAll world-state log checks passed ✅")
