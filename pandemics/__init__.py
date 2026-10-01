"""The Ebola scenario. Select it with run.scenario: pandemics."""

from pandemics import artifacts  # noqa: F401  registers the artifact types
from pandemics.epidemic import Epidemic, EpidemicOptions

Options = EpidemicOptions


def build(options: EpidemicOptions):
    return [Epidemic(options)]
