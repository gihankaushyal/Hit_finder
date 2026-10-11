"""No test may reach W&B by accident.

`wandb_enabled()` calls `wandb.setup()`, which creates W&B's process-wide singleton and
freezes its settings. A test that ran it before `WANDB_MODE` was set made every later
`wandb.init` try to log in (CI has no key): the run-order dependence behind the failed
CI on PR #53. `tests/conftest.py` therefore disables W&B for every test.
"""

from __future__ import annotations

import os

from src.training.inference_results import wandb_enabled


def test_every_test_starts_with_wandb_disabled():
    assert os.environ.get("WANDB_MODE") == "disabled"


def test_wandb_enabled_is_false_unless_a_test_opts_in():
    assert wandb_enabled() is False
