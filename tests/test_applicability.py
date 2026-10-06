"""The engine knows which sickness options are inert under the current switches."""

import pytest
from terralingua.config.compose import compose, compose_input
from terralingua.config.inspection import inspect_config


def report(overrides, preset="ebola"):
    requested = compose_input(preset, overrides)
    return inspect_config(compose(preset, overrides), requested)


@pytest.mark.parametrize("preset", ["ebola", "covid"])
def test_the_presets_set_no_inert_option(preset):
    diagnostics = report({}, preset)["diagnostics"]
    assert all(d["code"] != "inactive_setting" for d in diagnostics)
    assert all(d["code"] != "scenario_validation_deferred" for d in diagnostics)


def test_the_air_options_are_inert_without_airborne_spread():
    result = report({"scenario_options": {"airborne": False}}, "covid")
    assert result["fields"]["run.scenario_options.airborne_decay"] == {"active": False, "reason": "Requires airborne spread."}
    # Only the inert values that differ from their defaults warn: the preset's decay is the default.
    warnings = sorted(d["field"] for d in result["diagnostics"] if d["code"] == "inactive_setting")
    assert warnings == [f"run.scenario_options.airborne_{name}" for name in ("multiplier", "presymptomatic_days", "radius")]


def test_an_inert_option_set_by_a_run_warns():
    result = report({"scenario_options": {"burials": False, "burial_infection_multiplier": 2.5}})
    fields = result["fields"]
    assert fields["run.scenario_options.burial_infection_multiplier"] == {"active": False, "reason": "Requires burials."}
    assert fields["run.scenario_options.funeral_mourning_days"]["active"] is True  # announcements stay on
    warnings = [d["field"] for d in result["diagnostics"] if d["code"] == "inactive_setting"]
    assert warnings == ["run.scenario_options.burial_infection_multiplier"]
