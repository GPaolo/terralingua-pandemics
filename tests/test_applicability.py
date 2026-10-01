"""The engine knows which sickness options are inert under the current switches."""

from terralingua.config.compose import compose, compose_input
from terralingua.config.inspection import inspect_config


def report(overrides):
    requested = compose_input("ebola", overrides)
    return inspect_config(compose("ebola", overrides), requested)


def test_the_preset_sets_no_inert_option():
    diagnostics = report({})["diagnostics"]
    assert all(d["code"] != "inactive_setting" for d in diagnostics)
    assert all(d["code"] != "scenario_validation_deferred" for d in diagnostics)


def test_an_inert_option_set_by_a_run_warns():
    result = report({"scenario_options": {"burials": False, "burial_infection_multiplier": 2.5}})
    fields = result["fields"]
    assert fields["run.scenario_options.burial_infection_multiplier"] == {"active": False, "reason": "Requires burials."}
    assert fields["run.scenario_options.funeral_mourning_days"]["active"] is True  # announcements stay on
    assert fields["run.scenario_options.health_center.radius"]["active"] is True
    warnings = [d["field"] for d in result["diagnostics"] if d["code"] == "inactive_setting"]
    assert warnings == ["run.scenario_options.burial_infection_multiplier"]


def test_options_without_a_health_center_are_inert():
    result = report({"scenario_options": {"health_center": None}})
    assert result["fields"]["run.scenario_options.health_center.radius"]["active"] is False
    assert result["inactive_values"]["run.scenario_options.health_center.radius"] is None
