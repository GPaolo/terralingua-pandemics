"""The calibration runs scripted epidemics and pools R0 over seeds."""

from pandemics.calibrate_r0 import calibrate
from pandemics.epidemic import EpidemicOptions
from tests.test_epidemic import FAST

WORLD = dict(
    grid_size=8, vision_radius=2, init_agent_energy=100, lifespan=500, init_food=10,
    food_spawn_rate=1, food_mechanism=True, drop_food_on_death=False, use_inventory=True,
    use_colors=False, reproduction_cost=-1, artifact_creation_cost=0,
)


def test_calibration_pools_r0_over_seeds(tmp_path):
    options = EpidemicOptions(**{**FAST, "lifespan": 3, "infection_probability": 0.5})
    lines = []
    results = calibrate([1.0], seeds=2, steps=20, index=1, agents=8, options=options, world=WORLD, root=tmp_path, log=lines.append)
    (prob, r0, cases, infections), = results
    assert prob == 1.0
    assert cases >= 2  # one index case per seed, completed after three sick days
    assert infections >= cases
    assert r0 is None or r0 >= 0
    assert len(lines) == 3
    assert (tmp_path / "p1_s0" / "world_state.jsonl").exists()
    assert (tmp_path / "p1_s1" / "open_gridworld.log").exists()
