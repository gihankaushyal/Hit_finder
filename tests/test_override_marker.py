"""Durable record that --override-training was already applied by this launch."""

from __future__ import annotations

import json

import pytest

from src.training import override_marker as om

RUN = "resnet18-asymmetric-v2-fold1-seed42"


def test_no_job_id_means_no_marker_and_never_applied(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert om.current_job_id() is None
    assert om.write_override_marker(tmp_path, RUN) is False
    assert not (tmp_path / om.OVERRIDE_MARKER_FILE).exists()
    assert om.override_already_applied(tmp_path) is False


def test_empty_job_id_counts_as_unset(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "")
    assert om.current_job_id() is None


def test_write_then_same_job_is_already_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert om.write_override_marker(tmp_path, RUN) is True
    data = json.loads((tmp_path / om.OVERRIDE_MARKER_FILE).read_text())
    assert data["job_id"] == "123" and data["run_name"] == RUN
    assert data["applied_at"].endswith("+00:00")
    assert om.override_already_applied(tmp_path) is True


def test_other_job_is_not_already_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    om.write_override_marker(tmp_path, RUN)
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    assert om.override_already_applied(tmp_path) is False


def test_corrupt_marker_is_not_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    (tmp_path / om.OVERRIDE_MARKER_FILE).write_text("{not json")
    assert om.override_already_applied(tmp_path) is False


def test_write_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    om.write_override_marker(tmp_path, RUN)
    before = (tmp_path / om.OVERRIDE_MARKER_FILE).read_text()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(om.os, "replace", boom)
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    with pytest.raises(OSError):
        om.write_override_marker(tmp_path, RUN)
    assert (tmp_path / om.OVERRIDE_MARKER_FILE).read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == [om.OVERRIDE_MARKER_FILE]
