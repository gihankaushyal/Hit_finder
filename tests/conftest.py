"""Shared fixtures for the test suite.

Jungfrau 4M assembly needs a full (2164, 2068) canvas for the real
PADAssembler; most unit tests use small synthetic frames for speed, so
assemble_only is monkeypatched to fall back to the old _to_2d() passthrough
for undersized Jungfrau frames only. Real-shaped frames still go through the
real assembler.

Two variants exist because assemble_only is imported differently by its two
callers:
- src.data.dataset imports assemble_only into its own module namespace at
  import time, so the patch must target that already-bound reference.
- src.evaluation.benchmark.run_patch_agg imports assemble_only inside the
  function body on every call, so the patch must target the source module
  (src.preprocessing.pipeline) directly.
"""

from __future__ import annotations

import pytest


def _fake_assemble_only(frame, pads, detector_desc, assembler=None):
    from src.preprocessing.pipeline import _to_2d
    from src.preprocessing.pipeline import assemble_only as _real_assemble_only

    if detector_desc == "Jungfrau 4M" and frame.shape != (2164, 2068):
        return _to_2d(frame)
    return _real_assemble_only(frame, pads, detector_desc, assembler=assembler)


@pytest.fixture(autouse=True)
def wandb_disabled(monkeypatch: pytest.MonkeyPatch):
    """Keep every test away from W&B, whatever order the tests run in.

    `wandb.setup()` (reached through `wandb_enabled()`) creates a process-wide singleton
    and freezes its settings, so a test that ran before `WANDB_MODE` was set made every
    later `wandb.init` try to log in; CI has no API key and failed. Setting the mode for
    every test, and dropping the singleton afterwards, removes the order dependence.
    A test that needs a different mode sets or deletes `WANDB_MODE` itself.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
    yield
    try:
        import wandb

        wandb.teardown()
    except (
        Exception
    ):  # best effort: a test that never touched W&B has nothing to tear down
        pass


@pytest.fixture(autouse=True)
def fake_jungfrau_assembly_via_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch assemble_only as bound into src.data.dataset's namespace.

    autouse so every test file gets this for free — no file can forget the
    opt-in (one did: tests/test_train_ssl.py, 2026-10). Safe to apply
    globally: it only intercepts undersized synthetic Jungfrau frames and
    delegates to the real assembler for real (2164, 2068) frames, and no
    test exercises undersized Jungfrau frames through src.data.dataset on
    purpose.
    """
    import src.data.dataset as dataset_module

    monkeypatch.setattr(dataset_module, "assemble_only", _fake_assemble_only)


@pytest.fixture
def fake_jungfrau_assembly_via_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch assemble_only at its source module (src.preprocessing.pipeline)."""
    monkeypatch.setattr("src.preprocessing.pipeline.assemble_only", _fake_assemble_only)
