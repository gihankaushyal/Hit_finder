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


class TestRotateIsAtomic:
    def test_failed_replace_keeps_old_file(self, tmp_path, monkeypatch):
        (tmp_path / wi.WANDB_ID_FILE).write_text(f"{RUN}-o2\n")

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(wi.os, "replace", boom)
        with pytest.raises(OSError):
            wi.rotate_wandb_id(tmp_path, RUN)
        assert wi.resolve_wandb_id(tmp_path, RUN) == f"{RUN}-o2"
        assert [p.name for p in tmp_path.iterdir()] == [wi.WANDB_ID_FILE]


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

    def test_rollback_restores_missing_file_and_untags(self, tmp_path, fake_wandb):
        FakeApi.runs[f"proj/{RUN}"] = FakeRun([])
        hook = wi.OverrideHook("proj", None)
        rollback = hook(RUN, tmp_path)
        assert FakeApi.runs[f"proj/{RUN}"].tags == [wi.OVERRIDDEN_TAG]
        rollback()
        assert not (tmp_path / wi.WANDB_ID_FILE).exists()
        assert FakeApi.runs[f"proj/{RUN}"].tags == []

    def test_rollback_restores_previous_content(self, tmp_path, fake_wandb):
        (tmp_path / wi.WANDB_ID_FILE).write_text(f"{RUN}-o2\n")
        FakeApi.runs[f"proj/{RUN}-o2"] = FakeRun(["x"])
        rollback = wi.OverrideHook("proj", None)(RUN, tmp_path)
        assert wi.resolve_wandb_id(tmp_path, RUN) == f"{RUN}-o3"
        rollback()
        assert wi.resolve_wandb_id(tmp_path, RUN) == f"{RUN}-o2"
        assert FakeApi.runs[f"proj/{RUN}-o2"].tags == ["x"]

    def test_from_cfg(self):
        hook = wi.override_hook_from_cfg({"wandb": {"project": "p", "entity": "e"}})
        assert (hook.project, hook.entity) == ("p", "e")
        assert wi.override_hook_from_cfg({}).project is None


class _R:
    def __init__(self, name, id, tags=()):
        self.name, self.id, self.tags = name, id, list(tags)


class TestSelectCurrentRuns:
    def test_drops_tagged(self):
        runs = [_R("a", "a", [wi.OVERRIDDEN_TAG]), _R("b", "b")]
        assert [r.id for r in wi.select_current_runs(runs)] == ["b"]

    def test_highest_attempt_wins_without_any_tag(self):
        runs = [_R("a", "a"), _R("a", "a-o3"), _R("a", "a-o2")]
        assert [r.id for r in wi.select_current_runs(runs)] == ["a-o3"]

    def test_unrelated_names_untouched(self):
        runs = [_R("a", "a"), _R("b", "b"), _R("c", "c-o2")]
        assert {r.id for r in wi.select_current_runs(runs)} == {"a", "b", "c-o2"}

    def test_unparseable_id_counts_as_one(self):
        runs = [_R("a", "weird"), _R("a", "a-o2")]
        assert [r.id for r in wi.select_current_runs(runs)] == ["a-o2"]


def test_plot_script_uses_shared_selection():
    from pathlib import Path

    text = (
        Path(__file__).resolve().parent.parent / "scripts" / "plot_hit_frac.py"
    ).read_text()
    assert "from src.training.wandb_identity import" in text
    assert "select_current_runs" in text
    assert 'OVERRIDDEN_TAG = "' not in text


def test_wandb_identity_imports_no_heavy_deps():
    import subprocess

    code = (
        "import sys; import src.training.wandb_identity; "
        "bad=[m for m in ('torch','wandb','numpy','h5py') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


class TestEnsureFreshWandbId:
    def test_no_existing_run_keeps_id(self, tmp_path, fake_wandb):
        assert wi.ensure_fresh_wandb_id(tmp_path, RUN, "proj", None) == RUN
        assert not (tmp_path / wi.WANDB_ID_FILE).exists()

    def test_skips_existing_chain_and_writes_file(self, tmp_path, fake_wandb):
        FakeApi.runs[f"proj/{RUN}"] = FakeRun([])
        FakeApi.runs[f"proj/{RUN}-o2"] = FakeRun([])
        got = wi.ensure_fresh_wandb_id(tmp_path, RUN, "proj", None)
        assert got == f"{RUN}-o3"
        assert wi.resolve_wandb_id(tmp_path, RUN) == f"{RUN}-o3"
        assert FakeApi.runs[f"proj/{RUN}"].tags == [wi.OVERRIDDEN_TAG]
        assert FakeApi.runs[f"proj/{RUN}-o2"].tags == [wi.OVERRIDDEN_TAG]

    def test_unknown_error_keeps_id_and_warns(
        self, tmp_path, fake_wandb, capsys, monkeypatch
    ):
        def boom(self, path):
            raise ConnectionError("network down")

        monkeypatch.setattr(FakeApi, "run", boom)
        got = wi.ensure_fresh_wandb_id(tmp_path, RUN, "proj", None)
        assert got == RUN
        assert "[wandb]" in capsys.readouterr().out

    def test_disabled_makes_no_api_call(self, tmp_path, fake_wandb, monkeypatch):
        monkeypatch.setattr(wi, "wandb_enabled", lambda: False)
        assert wi.ensure_fresh_wandb_id(tmp_path, RUN, "proj", None) == RUN
        assert FakeApi.paths == []

    def test_no_project_makes_no_api_call(self, tmp_path, fake_wandb):
        assert wi.ensure_fresh_wandb_id(tmp_path, RUN, None, None) == RUN
        assert FakeApi.paths == []


class TestWandbIdForTraining:
    def test_resume_keeps_current_id_without_api(self, tmp_path, fake_wandb):
        FakeApi.runs[f"proj/{RUN}"] = FakeRun([])
        got = wi.wandb_id_for_training(tmp_path, RUN, "proj", None, resuming=True)
        assert got == RUN and FakeApi.paths == []

    def test_fresh_start_rotates_past_existing_run(self, tmp_path, fake_wandb):
        FakeApi.runs[f"proj/{RUN}"] = FakeRun([])
        got = wi.wandb_id_for_training(tmp_path, RUN, "proj", None, resuming=False)
        assert got == f"{RUN}-o2"


class TestMissingRunMessage:
    """wandb 0.27.0 words a missing run as 'Could not find run ...' (seen live)."""

    def test_real_wandb_wording_counts_as_missing(self, tmp_path, monkeypatch):
        class RealWordingApi:
            def __init__(self, **kwargs):
                pass

            def run(self, path):
                raise ValueError(f"Could not find run <Run {path} (None)>")

        monkeypatch.setitem(
            sys.modules, "wandb", types.SimpleNamespace(Api=RealWordingApi)
        )
        monkeypatch.setattr(wi, "wandb_enabled", lambda: True)
        assert wi.ensure_fresh_wandb_id(tmp_path, RUN, "proj", None) == RUN
        assert not (tmp_path / wi.WANDB_ID_FILE).exists()


def test_real_wording_prints_no_warning(tmp_path, monkeypatch, capsys):
    class RealWordingApi:
        def __init__(self, **kwargs):
            pass

        def run(self, path):
            raise ValueError("Could not find run <Run x (None)>")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=RealWordingApi))
    monkeypatch.setattr(wi, "wandb_enabled", lambda: True)
    wi.ensure_fresh_wandb_id(tmp_path, RUN, "proj", None)
    assert "could not check" not in capsys.readouterr().out
