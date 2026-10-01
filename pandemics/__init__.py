"""The Ebola scenario. Select it with run.scenario: pandemics."""

from terralingua.config.dependencies import either, when

from pandemics import artifacts  # noqa: F401  registers the artifact types
from pandemics.epidemic import Epidemic, EpidemicOptions

Options = EpidemicOptions

BURIALS = when("burials", const=True)
ANNOUNCEMENTS = when("funeral_announcements", const=True)
EQUIPMENT = when("ppe_per_worker", minimum=1)
CENTER = when("health_center", type="object")

# Which options apply only when another option turns a rule on. The engine
# warns at start when a run sets an inert option, and lists them in
# `python -m terralingua.config evaluate --preset <name>`.
APPLICABILITY = {
    "outbreak_step": (when("init_infected", minimum=1), "Requires a being infected at the outbreak."),
    "mobile_infectiousness": (when("mobile_days", minimum=1), "Requires feverish days before the host is bedridden."),
    "burial_infection_multiplier": (BURIALS, "Requires burials."),
    "funeral_attendance_multiplier": (BURIALS, "Requires burials."),
    "funeral_mourning_days": (either(BURIALS, ANNOUNCEMENTS), "Requires burials or funeral announcements."),
    "funeral_announcement_radius": (ANNOUNCEMENTS, "Requires funeral announcements."),
    "ppe_protection": (EQUIPMENT, "Requires protective equipment for a role."),
    "ppe_role": (EQUIPMENT, "Requires protective equipment for a role."),
    "health_center.radius": (CENTER, "Requires a health center."),
    "health_center.heal_probability": (CENTER, "Requires a health center."),
    "health_center.hazard_multiplier": (CENTER, "Requires a health center."),
}


def build(options: EpidemicOptions):
    return [Epidemic(options)]
