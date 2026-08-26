"""Food seeding and respawn behave as designed around the food-zone density.

Semantics under test:
- init_food is a hard count: zones fill first, surplus intentionally spills
  onto low-probability tail cells (init_food is NOT the carrying capacity);
- respawn draws i.i.d. from the fixed density and fails on occupied cells,
  so it never drifts out of the zones as they fill (the old renormalize-
  over-free-cells approach did, and scanned every cell per spawn).

Run from the repo root: python test_food_spawning.py
No LLM calls, no API credits.
"""

import shutil
import tempfile

import numpy as np

from core.environment.env import OpenGridWorld

GRID = 25
CENTERS = [(6, 6), (18, 18)]


def make_env(tmp, seed=0, **kwargs):
    defaults = dict(
        grid_size=GRID,
        food_zones=CENTERS,
        log_world_state=False,
        verbose=0,
        log_path=tmp,
    )
    defaults.update(kwargs)
    env = OpenGridWorld(**defaults)
    env.rng = np.random.default_rng(seed)
    return env


def dist_to_nearest_center(pos):
    best = np.inf
    for c in CENTERS:
        dx = abs(pos[0] - c[0])
        dy = abs(pos[1] - c[1])
        dx = min(dx, GRID - dx)
        dy = min(dy, GRID - dy)
        best = min(best, (dx * dx + dy * dy) ** 0.5)
    return best


tmp = tempfile.mkdtemp(prefix="tl_food_test_")
try:
    # --- init food follows the zones ---------------------------------------
    env = make_env(tmp, init_food=100)
    env._seed_initial_food()
    dists = np.array([dist_to_nearest_center(p) for p in env.food])
    assert len(env.food) == 100, f"expected 100 tiles, got {len(env.food)}"
    near = (dists <= 6).mean()
    assert near >= 0.9, f"only {near:.0%} of init food within r<=6 of a zone"
    print(f"PASS: init food concentrated in zones ({near:.0%} within r<=6)")

    # --- over-asking keeps the exact count: zones fill first, surplus spills
    # onto low-probability tail cells --------------------------------------
    env = make_env(tmp, init_food=500)
    env._seed_initial_food()
    assert len(env.food) == 500, f"expected 500 tiles, got {len(env.food)}"
    core_cells = [
        (i, j)
        for i in range(GRID)
        for j in range(GRID)
        if dist_to_nearest_center((i, j)) <= 4
    ]
    core_filled = sum(p in env.food for p in core_cells) / len(core_cells)
    spilled = sum(dist_to_nearest_center(p) > 8 for p in env.food)
    assert core_filled >= 0.95, f"zone cores only {core_filled:.0%} filled"
    assert spilled > 0, "surplus food never spilled into the tail"
    print(
        f"PASS: over-asked init placed all 500 tiles — zone cores "
        f"{core_filled:.0%} filled, {spilled} tiles spilled into the tail"
    )

    # --- uniform density keeps the exact count ------------------------------
    env = make_env(tmp, init_food=500, food_zones=None)
    env._seed_initial_food()
    assert len(env.food) == 500, f"uniform init placed {len(env.food)}/500"
    assert all(v == env._max_food_value for v in env.food.values())
    print("PASS: uniform init still places the exact requested count")

    # --- respawn does not drift out of the zones as they fill ---------------
    env = make_env(tmp, init_food=100)
    env._seed_initial_food()
    spawn_dists = []
    before = set(env.food)
    for _ in range(600):
        env._decay_and_respawn_food()
        now = set(env.food)
        spawn_dists.extend(dist_to_nearest_center(p) for p in now - before)
        before = now
    early = np.mean(spawn_dists[: len(spawn_dists) // 4])
    late = np.mean(spawn_dists[-len(spawn_dists) // 4 :])
    assert late <= early + 1.0, (
        f"spawn distance drifted out of the zones: early {early:.2f} -> late {late:.2f}"
    )
    assert len(env.food) < GRID * GRID * 0.7, (
        f"food saturated the grid: {len(env.food)}/{GRID * GRID}"
    )
    print(
        f"PASS: respawn stays on the zones (mean dist early {early:.2f}, "
        f"late {late:.2f}) and food self-limits at {len(env.food)} tiles"
    )

    # --- respawn never lands on an occupied cell -----------------------------
    env = make_env(tmp, init_food=100)
    env._seed_initial_food()
    occupied_before = set(env.food)
    for _ in range(200):
        env._respawn_food_one()
    assert all(v == env._max_food_value for v in env.food.values())
    assert occupied_before <= set(env.food), "respawn overwrote existing food"
    print("PASS: respawn never overwrites an occupied cell")

    # --- food recovers after total extinction --------------------------------
    env = make_env(tmp, init_food=100, food_spawn_rate=3)
    env._seed_initial_food()
    env.food.clear()
    for _ in range(50):
        env._decay_and_respawn_food()
    assert len(env.food) > 0, "food never respawned after hitting zero"
    print("PASS: food respawns after the map goes bare")

    # --- sampling survives a checkpoint round-trip ---------------------------
    env = make_env(tmp, init_food=100)
    env._seed_initial_food()
    env2 = make_env(tmp, init_food=100, seed=1)
    env2.set_state_ckpt(env.get_state_ckpt())
    env2.rng = np.random.default_rng(1)
    p = env2._sample_density_cell()
    assert dist_to_nearest_center(p) <= 8 or env2.food_distribution is not None
    for _ in range(50):
        env2._respawn_food_one()
    print("PASS: density sampling works after checkpoint restore")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("All food spawning tests passed.")
