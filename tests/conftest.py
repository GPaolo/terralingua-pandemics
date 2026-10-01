"""Every test runs from the repository root.

The preset is found under the working directory, the package is imported from
it, and the logs folder lives under it.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
