import pytest

from pm.config import ARTIFACTS


@pytest.fixture(scope="session", autouse=True)
def _need_artifacts():
    if not (ARTIFACTS / "meta.json").exists():
        pytest.skip("Run `python -m pm.run_all` first to create artifacts.")
