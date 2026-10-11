"""No test may reach W&B by accident.

`wandb_enabled()` calls `wandb.setup()`, which creates W&B's process-wide singleton and
freezes its settings. A test that ran it before `WANDB_MODE` was set made every later
`wandb.init` try to log in (CI has no key): the run-order dependence behind the failed
CI on PR #53. `tests/conftest.py` disables W&B for every test and tears the singleton
down afterwards.

The two tests below are an ORDERED pair (pytest runs a file's tests in definition
order): the first deliberately freezes the singleton in the "online" state, the second
checks the next test starts with a clean, disabled one. Removing the teardown in the
conftest fixture makes the second test fail.
"""

from __future__ import annotations

import os

from src.training.inference_results import wandb_enabled


def test_every_test_starts_with_wandb_disabled():
    assert os.environ.get("WANDB_MODE") == "disabled"
    assert wandb_enabled() is False


def test_a_test_that_freezes_the_wandb_singleton_online_(monkeypatch):
    """Pollute: with no WANDB_MODE, wandb_enabled() creates the singleton as 'online'."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    wandb_enabled()


def test_the_next_test_gets_a_fresh_disabled_singleton():
    import wandb

    assert os.environ.get("WANDB_MODE") == "disabled"
    assert str(wandb.setup().settings.mode).lower() == "disabled"
