"""W&B run identity across override attempts: id file, rotation, tagging."""

from __future__ import annotations

import sys
import types

import pytest

from src.training import wandb_identity as wi

RUN = "resnet18-asymmetric-v2-fold1-seed42"


class TestResolveAndNext:
    def test_no_file_means_run_name(self, tmp_path):
        assert wi.resolve_wandb_id(tmp_path, RUN) == RUN

    def test_empty_file_means_run_name(self, tmp_path):
        (tmp_path / wi.WANDB_ID_FILE).write_text("\n")
        assert wi.resolve_wandb_id(tmp_path, RUN) == RUN

    def test_reads_file(self, tmp_path):
        (tmp_path / wi.WANDB_ID_FILE).write_text(f"{RUN}-o2\n")
        assert wi.resolve_wandb_id(tmp_path, RUN) == f"{RUN}-o2"

    def test_next_from_plain_name(self):
        assert wi.next_wandb_id(RUN, RUN) == f"{RUN}-o2"

    def test_next_increments(self):
        assert wi.next_wandb_id(RUN, f"{RUN}-o2") == f"{RUN}-o3"
        assert wi.next_wandb_id(RUN, f"{RUN}-o9") == f"{RUN}-o10"

    def test_next_ignores_unrelated_suffix(self):
        assert wi.next_wandb_id(RUN, "other-o4") == f"{RUN}-o2"


class TestRotate:
    def test_rotate_writes_file_and_returns_pair(self, tmp_path):
        old, new = wi.rotate_wandb_id(tmp_path, RUN)
        assert (old, new) == (RUN, f"{RUN}-o2")
        assert wi.resolve_wandb_id(tmp_path, RUN) == f"{RUN}-o2"
        old, new = wi.rotate_wandb_id(tmp_path, RUN)
        assert (old, new) == (f"{RUN}-o2", f"{RUN}-o3")


class FakeRun:
    def __init__(self, tags):
        self.tags = tags
        self.updated = 0

    def update(self):
        self.updated += 1


class FakeApi:
    runs: dict = {}
    paths: list = []

    def __init__(self, **kwargs):
        pass

    def run(self, path):
        FakeApi.paths.append(path)
        if path not in FakeApi.runs:
            raise RuntimeError("not found")
        return FakeApi.runs[path]


@pytest.fixture
def fake_wandb(monkeypatch):
    FakeApi.runs, FakeApi.paths = {}, []
    mod = types.SimpleNamespace(Api=FakeApi)
    monkeypatch.setitem(sys.modules, "wandb", mod)
    monkeypatch.setattr(wi, "wandb_enabled", lambda: True)
    return mod


class TestTagOverridden:
    def test_adds_tag_once(self, fake_wandb):
        FakeApi.runs["proj/run1"] = FakeRun(["a"])
        assert wi.tag_overridden("proj", None, "run1") is True
        assert FakeApi.runs["proj/run1"].tags == ["a", wi.OVERRIDDEN_TAG]
        assert wi.tag_overridden("proj", None, "run1") is True
        assert FakeApi.runs["proj/run1"].updated == 1  # already tagged: no 2nd write

    def test_entity_in_path(self, fake_wandb):
        FakeApi.runs["me/proj/run1"] = FakeRun([])
        assert wi.tag_overridden("proj", "me", "run1") is True

    def test_failure_warns_only(self, fake_wandb, capsys):
        assert wi.tag_overridden("proj", None, "missing") is False
        assert "could not tag" in capsys.readouterr().out

    def test_disabled_skips_api(self, fake_wandb, monkeypatch):
        monkeypatch.setattr(wi, "wandb_enabled", lambda: False)
        assert wi.tag_overridden("proj", None, "run1") is False
        assert FakeApi.paths == []

    def test_no_project_skips_api(self, fake_wandb):
        assert wi.tag_overridden(None, None, "run1") is False
        assert FakeApi.paths == []


class TestOverrideHook:
    def test_rotates_then_tags(self, tmp_path, fake_wandb):
        FakeApi.runs[f"proj/{RUN}"] = FakeRun([])
        hook = wi.OverrideHook("proj", None)
        hook(RUN, tmp_path)
        assert hook.rotations == [(RUN, RUN, f"{RUN}-o2")]
        assert wi.resolve_wandb_id(tmp_path, RUN) == f"{RUN}-o2"
        assert FakeApi.runs[f"proj/{RUN}"].tags == [wi.OVERRIDDEN_TAG]

    def test_tag_failure_still_rotates(self, tmp_path, fake_wandb):
        hook = wi.OverrideHook("proj", None)
        hook(RUN, tmp_path)  # run missing in the fake API: tagging fails quietly
        assert wi.resolve_wandb_id(tmp_path, RUN) == f"{RUN}-o2"

    def test_from_cfg(self):
        hook = wi.override_hook_from_cfg({"wandb": {"project": "p", "entity": "e"}})
        assert (hook.project, hook.entity) == ("p", "e")
        assert wi.override_hook_from_cfg({}).project is None
