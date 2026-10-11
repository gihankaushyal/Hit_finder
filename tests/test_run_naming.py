"""Unit tests for the shared run-naming standard and checkpoint collision gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.run_naming import (
    ASYMMETRIC_CONVENTION,
    ASYMMETRIC_EXAMPLE,
    ASYMMETRIC_PREFIX_RE,
    SSL_FINETUNE_CONVENTION,
    SSL_FINETUNE_EXAMPLE,
    SSL_FINETUNE_PREFIX_RE,
    SSL_PRETRAIN_CONVENTION,
    SSL_PRETRAIN_EXAMPLE,
    SSL_PRETRAIN_PREFIX_RE,
    check_checkpoint_collisions,
    effective_resume,
    expand_finetune_prefix,
    fold_run_name,
    validate_run_name_prefix,
)

PATTERNS = {
    "asymmetric": (ASYMMETRIC_PREFIX_RE, ASYMMETRIC_CONVENTION, ASYMMETRIC_EXAMPLE),
    "pretrain": (SSL_PRETRAIN_PREFIX_RE, SSL_PRETRAIN_CONVENTION, SSL_PRETRAIN_EXAMPLE),
    "finetune": (SSL_FINETUNE_PREFIX_RE, SSL_FINETUNE_CONVENTION, SSL_FINETUNE_EXAMPLE),
}


class TestValidateRunNamePrefix:
    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_accepts_own_example(self, kind):
        pattern, convention, example = PATTERNS[kind]
        validate_run_name_prefix(example, pattern, convention, example)

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_rejects_other_entry_points_examples(self, kind):
        pattern, convention, example = PATTERNS[kind]
        for other in sorted(set(PATTERNS) - {kind}):
            with pytest.raises(SystemExit):
                validate_run_name_prefix(
                    PATTERNS[other][2], pattern, convention, example
                )

    @pytest.mark.parametrize(
        "bad",
        [
            "mae-vits16",  # no version
            "mae-vits16-fold1-seed42-v2",  # version in suffix position
            "MAE-vits16-v2",  # uppercase
            "mae-vits16-v",  # version number missing
            "mae-vits16-v2-",  # trailing dash
            "",
        ],
    )
    def test_pretrain_rejects_malformed(self, bad):
        with pytest.raises(SystemExit):
            validate_run_name_prefix(
                bad,
                SSL_PRETRAIN_PREFIX_RE,
                SSL_PRETRAIN_CONVENTION,
                SSL_PRETRAIN_EXAMPLE,
            )

    @pytest.mark.parametrize(
        "bad",
        [
            "vits16-mae",  # no version
            "vits16-mae-finetune-v2",  # mode is inserted by the code, not typed
            "vits16-mae-finetune",  # the pre-standard hardcoded prefix
            "Vits16-mae-v2",  # uppercase
        ],
    )
    def test_finetune_rejects_malformed(self, bad):
        with pytest.raises(SystemExit):
            validate_run_name_prefix(
                bad,
                SSL_FINETUNE_PREFIX_RE,
                SSL_FINETUNE_CONVENTION,
                SSL_FINETUNE_EXAMPLE,
            )

    @pytest.mark.parametrize(
        "bad",
        [
            "resnet18-asymmetric",  # no version
            "resnet18-asymmetric-fold1-seed42-v2",  # version in suffix position
            "ResNet18-asymmetric-v2",  # uppercase inside the backbone
            "resnet18-asymmetric-v",  # bare -v
            "resnet18-asymmetric-v2-fold1",  # trailing junk
            "res-net18-asymmetric-v2",  # dash in backbone
        ],
    )
    def test_asymmetric_rejects_malformed(self, bad):
        with pytest.raises(SystemExit):
            validate_run_name_prefix(
                bad,
                ASYMMETRIC_PREFIX_RE,
                ASYMMETRIC_CONVENTION,
                ASYMMETRIC_EXAMPLE,
            )

    @pytest.mark.parametrize(
        "bad",
        [
            "vits16-mae-fold1-seed42-v2",  # version in suffix position
            "vits16-mae-v",  # bare -v
            "vits16-mae-v2-fold1-seed42",  # trailing junk
            "vit-small-mae-v2",  # dash in backbone
        ],
    )
    def test_finetune_rejects_more_malformed(self, bad):
        with pytest.raises(SystemExit):
            validate_run_name_prefix(
                bad,
                SSL_FINETUNE_PREFIX_RE,
                SSL_FINETUNE_CONVENTION,
                SSL_FINETUNE_EXAMPLE,
            )

    def test_pretrain_rejects_uppercase_in_backbone(self):
        with pytest.raises(SystemExit):
            validate_run_name_prefix(
                "mae-ViTS16-v2",
                SSL_PRETRAIN_PREFIX_RE,
                SSL_PRETRAIN_CONVENTION,
                SSL_PRETRAIN_EXAMPLE,
            )

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_rejects_trailing_newline(self, kind):
        pattern, convention, example = PATTERNS[kind]
        with pytest.raises(SystemExit):
            validate_run_name_prefix(example + "\n", pattern, convention, example)

    @pytest.mark.parametrize("kind", sorted(PATTERNS))
    def test_rejects_non_ascii_digit(self, kind):
        pattern, convention, example = PATTERNS[kind]
        with pytest.raises(SystemExit):
            validate_run_name_prefix(example[:-1] + "٢", pattern, convention, example)

    def test_expand_rejects_trailing_newline_and_unicode_digit(self):
        for bad in ("vits16-mae-v2\n", "vits16-mae-v٢"):
            with pytest.raises(ValueError):
                expand_finetune_prefix(bad, linear_probe=False)

    def test_error_message_names_convention_and_example(self):
        with pytest.raises(SystemExit) as exc:
            validate_run_name_prefix(
                "nope",
                SSL_PRETRAIN_PREFIX_RE,
                SSL_PRETRAIN_CONVENTION,
                SSL_PRETRAIN_EXAMPLE,
            )
        message = str(exc.value)
        assert "'nope'" in message
        assert SSL_PRETRAIN_CONVENTION in message
        assert f"--run-name-prefix {SSL_PRETRAIN_EXAMPLE}" in message


class TestRunNameConstruction:
    def test_fold_run_name(self):
        assert fold_run_name("mae-vits16-v2", 3, 42) == "mae-vits16-v2-fold3-seed42"

    def test_fold_run_name_with_suffix(self):
        assert (
            fold_run_name("resnet18-asymmetric-v2", 1, 7, "-x")
            == "resnet18-asymmetric-v2-fold1-seed7-x"
        )

    def test_expand_finetune_prefix(self):
        assert (
            expand_finetune_prefix("vits16-mae-v2", linear_probe=False)
            == "vits16-mae-finetune-v2"
        )

    def test_expand_probe_prefix(self):
        assert (
            expand_finetune_prefix("vits16-mae-v12", linear_probe=True)
            == "vits16-mae-probe-v12"
        )

    def test_expand_rejects_invalid_prefix(self):
        with pytest.raises(ValueError):
            expand_finetune_prefix("mae-vits16-v2", linear_probe=False)


def _make_run_dir(root: Path, run_name: str, files: list[str]) -> Path:
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    for name in files:
        (run_dir / name).write_bytes(b"x")
    return run_dir


class TestCheckCheckpointCollisions:
    def test_no_checkpoint_passes(self, tmp_path):
        check_checkpoint_collisions(
            {1: "run-fold1-seed42"}, "last.pt", False, False, checkpoint_root=tmp_path
        )

    def test_existing_checkpoint_without_flag_exits(self, tmp_path):
        _make_run_dir(tmp_path, "run-fold1-seed42", ["last.pt"])
        with pytest.raises(SystemExit) as exc:
            check_checkpoint_collisions(
                {1: "run-fold1-seed42"},
                "last.pt",
                False,
                False,
                checkpoint_root=tmp_path,
            )
        message = str(exc.value)
        assert "fold 1" in message
        assert "--resume-training" in message
        assert "--override-training" in message

    def test_resume_keeps_everything(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-fold1-seed42", ["last.pt", "epoch20.pt"])
        check_checkpoint_collisions(
            {1: "run-fold1-seed42"},
            "last.pt",
            True,
            False,
            extra_delete=("epoch*.pt",),
            checkpoint_root=tmp_path,
        )
        assert (run_dir / "last.pt").exists()
        assert (run_dir / "epoch20.pt").exists()

    def test_override_deletes_checkpoint_and_extra_globs(self, tmp_path):
        run_dir = _make_run_dir(
            tmp_path,
            "run-fold1-seed42",
            ["last.pt", "epoch20.pt", "epoch40.pt", "keep.txt"],
        )
        check_checkpoint_collisions(
            {1: "run-fold1-seed42"},
            "last.pt",
            False,
            True,
            extra_delete=("epoch*.pt",),
            checkpoint_root=tmp_path,
        )
        assert not (run_dir / "last.pt").exists()
        assert not list(run_dir.glob("epoch*.pt"))
        assert (run_dir / "keep.txt").exists()

    def test_collision_in_later_fold_is_found(self, tmp_path):
        _make_run_dir(tmp_path, "run-fold1-seed42", ["README"])
        _make_run_dir(tmp_path, "run-fold2-seed42", ["last.pt"])
        with pytest.raises(SystemExit) as exc:
            check_checkpoint_collisions(
                {1: "run-fold1-seed42", 2: "run-fold2-seed42"},
                "last.pt",
                False,
                False,
                checkpoint_root=tmp_path,
            )
        assert "fold 2" in str(exc.value)

    def test_exit_message_includes_checkpoint_path(self, tmp_path):
        _make_run_dir(tmp_path, "run-fold1-seed42", ["last.pt"])
        with pytest.raises(SystemExit) as exc:
            check_checkpoint_collisions(
                {1: "run-fold1-seed42"},
                "last.pt",
                False,
                False,
                checkpoint_root=tmp_path,
            )
        assert str(tmp_path / "run-fold1-seed42" / "last.pt") in str(exc.value)

    def test_both_flags_raise_and_delete_nothing(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-fold1-seed42", ["last.pt", "epoch20.pt"])
        with pytest.raises(ValueError):
            check_checkpoint_collisions(
                {1: "run-fold1-seed42"},
                "last.pt",
                True,
                True,
                extra_delete=("epoch*.pt",),
                checkpoint_root=tmp_path,
            )
        assert (run_dir / "last.pt").exists()
        assert (run_dir / "epoch20.pt").exists()

    def test_override_skips_directories_matching_extra_glob(self, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, "run-fold1-seed42", ["last.pt", "epoch20.pt", "epoch40.pt"]
        )
        backup = run_dir / "epoch_backup.pt"
        backup.mkdir()
        (backup / "inner.txt").write_bytes(b"x")
        check_checkpoint_collisions(
            {1: "run-fold1-seed42"},
            "last.pt",
            False,
            True,
            extra_delete=("epoch*.pt",),
            checkpoint_root=tmp_path,
        )
        assert not (run_dir / "last.pt").exists()
        assert not (run_dir / "epoch20.pt").exists()
        assert not (run_dir / "epoch40.pt").exists()
        assert backup.is_dir()
        assert (backup / "inner.txt").exists()

    def test_override_deletes_extras_before_checkpoint(self, tmp_path, monkeypatch):
        run_dir = _make_run_dir(tmp_path, "run-fold1-seed42", ["last.pt", "epoch20.pt"])
        real_unlink = Path.unlink

        def failing_unlink(self, *args, **kwargs):
            if self.name == "epoch20.pt":
                raise OSError("boom")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", failing_unlink)
        with pytest.raises(OSError):
            check_checkpoint_collisions(
                {1: "run-fold1-seed42"},
                "last.pt",
                False,
                True,
                extra_delete=("epoch*.pt",),
                checkpoint_root=tmp_path,
            )
        # the primary checkpoint must survive so a rerun still sees the collision
        assert (run_dir / "last.pt").exists()

    def test_override_without_primary_checkpoint_deletes_nothing(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-fold1-seed42", ["epoch20.pt"])
        check_checkpoint_collisions(
            {1: "run-fold1-seed42"},
            "last.pt",
            False,
            True,
            extra_delete=("epoch*.pt",),
            checkpoint_root=tmp_path,
        )
        assert (run_dir / "epoch20.pt").exists()

    def test_only_colliding_fold_is_touched(self, tmp_path):
        keep = _make_run_dir(
            tmp_path, "other-fold2-seed42", ["best.pt", "results.json"]
        )
        _make_run_dir(tmp_path, "run-fold1-seed42", ["best.pt", "results.json"])
        check_checkpoint_collisions(
            {1: "run-fold1-seed42"},
            "best.pt",
            False,
            True,
            extra_delete=("results.json",),
            checkpoint_root=tmp_path,
        )
        assert (keep / "best.pt").exists()
        assert (keep / "results.json").exists()
        assert not (tmp_path / "run-fold1-seed42" / "best.pt").exists()
        assert not (tmp_path / "run-fold1-seed42" / "results.json").exists()


class TestTrack1Wrapper:
    """train_asymmetric keeps its private wrapper; it must delegate to the shared gate."""

    def test_override_removes_best_and_results(self, tmp_path, monkeypatch):
        import src.training.train_asymmetric as ta

        assert ta.check_checkpoint_collisions is check_checkpoint_collisions
        monkeypatch.chdir(tmp_path)
        run_dir = _make_run_dir(
            tmp_path / "checkpoints",
            "resnet18-asymmetric-v2-fold1-seed42",
            ["best.pt", "results.json"],
        )
        ta._check_checkpoint_collisions(
            [1], {"seed": 42}, "resnet18-asymmetric-v2", False, True
        )
        assert not (run_dir / "best.pt").exists()
        assert not (run_dir / "results.json").exists()

    def test_existing_checkpoint_without_flag_exits(self, tmp_path, monkeypatch):
        import src.training.train_asymmetric as ta

        monkeypatch.chdir(tmp_path)
        _make_run_dir(
            tmp_path / "checkpoints",
            "resnet18-asymmetric-v2-fold1-seed42-x",
            ["best.pt"],
        )
        with pytest.raises(SystemExit):
            ta._check_checkpoint_collisions(
                [1],
                {"seed": 42, "wandb": {"run_suffix": "-x"}},
                "resnet18-asymmetric-v2",
                False,
                False,
            )


class TestDryRun:
    def test_dry_run_override_deletes_nothing(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-fold1-seed42", ["last.pt", "epoch20.pt"])
        check_checkpoint_collisions(
            {1: "run-fold1-seed42"},
            "last.pt",
            False,
            True,
            extra_delete=("epoch*.pt",),
            checkpoint_root=tmp_path,
            dry_run=True,
        )
        assert (run_dir / "last.pt").exists()
        assert (run_dir / "epoch20.pt").exists()

    def test_dry_run_still_exits_on_unresolved_collision(self, tmp_path):
        _make_run_dir(tmp_path, "run-fold1-seed42", ["last.pt"])
        with pytest.raises(SystemExit):
            check_checkpoint_collisions(
                {1: "run-fold1-seed42"},
                "last.pt",
                False,
                False,
                checkpoint_root=tmp_path,
                dry_run=True,
            )


class TestInferenceOnlyGate:
    def test_existing_checkpoint_passes_and_nothing_is_deleted(self, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, "run-fold1-seed42", ["best.pt", "results.json"]
        )
        check_checkpoint_collisions(
            {1: "run-fold1-seed42"},
            "best.pt",
            False,
            False,
            extra_delete=("results.json",),
            checkpoint_root=tmp_path,
            inference_only=True,
        )
        assert (run_dir / "best.pt").exists()
        assert (run_dir / "results.json").exists()

    def test_missing_checkpoint_exits_naming_the_path(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            check_checkpoint_collisions(
                {2: "run-fold2-seed42"},
                "best.pt",
                False,
                False,
                checkpoint_root=tmp_path,
                inference_only=True,
            )
        message = str(exc.value)
        assert "fold 2" in message
        assert str(tmp_path / "run-fold2-seed42" / "best.pt") in message
        assert "--inference-only" in message

    def test_missing_checkpoint_in_a_later_fold_is_found(self, tmp_path):
        _make_run_dir(tmp_path, "run-fold1-seed42", ["best.pt"])
        with pytest.raises(SystemExit):
            check_checkpoint_collisions(
                {1: "run-fold1-seed42", 2: "run-fold2-seed42"},
                "best.pt",
                False,
                False,
                checkpoint_root=tmp_path,
                inference_only=True,
            )

    def test_dry_run_still_exits_when_the_checkpoint_is_missing(self, tmp_path):
        with pytest.raises(SystemExit):
            check_checkpoint_collisions(
                {1: "run-fold1-seed42"},
                "best.pt",
                False,
                False,
                checkpoint_root=tmp_path,
                dry_run=True,
                inference_only=True,
            )

    @pytest.mark.parametrize(
        "resume,override", [(True, False), (False, True), (True, True)]
    )
    def test_combined_with_another_flag_raises_and_deletes_nothing(
        self, tmp_path, resume, override
    ):
        run_dir = _make_run_dir(tmp_path, "run-fold1-seed42", ["best.pt"])
        with pytest.raises(ValueError, match="mutually exclusive"):
            check_checkpoint_collisions(
                {1: "run-fold1-seed42"},
                "best.pt",
                resume,
                override,
                checkpoint_root=tmp_path,
                inference_only=True,
            )
        assert (run_dir / "best.pt").exists()

    def test_collision_message_offers_inference_only(self, tmp_path):
        _make_run_dir(tmp_path, "run-fold1-seed42", ["best.pt"])
        with pytest.raises(SystemExit) as exc:
            check_checkpoint_collisions(
                {1: "run-fold1-seed42"},
                "best.pt",
                False,
                False,
                checkpoint_root=tmp_path,
            )
        assert "--inference-only" in str(exc.value)


class TestOnOverride:
    def test_on_override_called_before_deletion(self, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, "run-v2-fold1-seed42", ["best.pt", "results.json"]
        )
        seen = []

        def hook(name, directory):
            seen.append((name, directory, (directory / "best.pt").exists()))

        check_checkpoint_collisions(
            {1: "run-v2-fold1-seed42"},
            "best.pt",
            False,
            True,
            extra_delete=("results.json",),
            checkpoint_root=tmp_path,
            on_override=hook,
        )
        assert seen == [("run-v2-fold1-seed42", run_dir, True)]
        assert not (run_dir / "best.pt").exists()

    @pytest.mark.parametrize("mode", ["resume", "inference", "dry"])
    def test_on_override_not_called_otherwise(self, tmp_path, mode):
        _make_run_dir(tmp_path, "run-v2-fold1-seed42", ["best.pt"])
        calls = []
        check_checkpoint_collisions(
            {1: "run-v2-fold1-seed42"},
            "best.pt",
            resume_training=mode == "resume",
            override_training=mode == "dry",
            checkpoint_root=tmp_path,
            dry_run=mode == "dry",
            inference_only=mode == "inference",
            on_override=lambda *a: calls.append(a),
        )
        assert calls == []

    def test_on_override_not_called_without_checkpoint(self, tmp_path):
        calls = []
        check_checkpoint_collisions(
            {1: "run-v2-fold1-seed42"},
            "best.pt",
            False,
            True,
            checkpoint_root=tmp_path,
            on_override=lambda *a: calls.append(a),
        )
        assert calls == []


class TestOverrideRollback:
    def test_failed_delete_calls_rollback_and_keeps_checkpoint(
        self, tmp_path, monkeypatch
    ):
        run_dir = _make_run_dir(tmp_path, "run-fold1-seed42", ["last.pt", "epoch20.pt"])
        rolled = []
        real_unlink = Path.unlink

        def flaky_unlink(self, *a, **k):
            if self.name == "epoch20.pt":
                raise OSError("read-only")
            return real_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", flaky_unlink)
        with pytest.raises(OSError):
            check_checkpoint_collisions(
                {1: "run-fold1-seed42"},
                "last.pt",
                False,
                True,
                extra_delete=("epoch*.pt",),
                checkpoint_root=tmp_path,
                on_override=lambda n, d: (lambda: rolled.append(n)),
            )
        assert rolled == ["run-fold1-seed42"]
        assert (run_dir / "last.pt").exists()

    def test_success_does_not_roll_back(self, tmp_path):
        _make_run_dir(tmp_path, "run-fold1-seed42", ["last.pt"])
        rolled = []
        check_checkpoint_collisions(
            {1: "run-fold1-seed42"},
            "last.pt",
            False,
            True,
            checkpoint_root=tmp_path,
            on_override=lambda n, d: (lambda: rolled.append(n)),
        )
        assert rolled == []

    def test_hook_returning_none_is_fine(self, tmp_path):
        _make_run_dir(tmp_path, "run-fold1-seed42", ["last.pt"])
        check_checkpoint_collisions(
            {1: "run-fold1-seed42"},
            "last.pt",
            False,
            True,
            checkpoint_root=tmp_path,
            on_override=lambda n, d: None,
        )


class TestIdempotentOverride:
    NAME = "run-v2-fold1-seed42"

    def _gate(self, tmp_path, calls=None, **kw):
        return check_checkpoint_collisions(
            {1: self.NAME},
            "best.pt",
            False,
            True,
            extra_delete=("results.json",),
            checkpoint_root=tmp_path,
            on_override=(lambda n, d: calls.append(n)) if calls is not None else None,
            **kw,
        )

    def test_first_override_writes_marker_and_reports_nothing_continued(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        run_dir = _make_run_dir(tmp_path, self.NAME, ["best.pt", "results.json"])
        assert self._gate(tmp_path) == set()
        assert not (run_dir / "best.pt").exists()
        marker = json.loads((run_dir / "override_applied.json").read_text())
        assert marker["job_id"] == "123"

    def test_same_job_again_keeps_the_new_checkpoint(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        run_dir = _make_run_dir(tmp_path, self.NAME, ["best.pt"])
        self._gate(tmp_path)  # first start: overrides
        (run_dir / "best.pt").write_bytes(b"new")  # the new attempt's checkpoint
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")  # the requeue
        calls = []
        assert self._gate(tmp_path, calls) == {self.NAME}  # requeue: same job id
        assert (run_dir / "best.pt").read_bytes() == b"new"
        assert calls == []  # no second id rotation
        assert "already applied by job 123" in capsys.readouterr().out

    def test_other_job_overrides_again_and_updates_the_marker(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        run_dir = _make_run_dir(tmp_path, self.NAME, ["best.pt"])
        self._gate(tmp_path)
        (run_dir / "best.pt").write_bytes(b"new")
        monkeypatch.setenv("SLURM_JOB_ID", "456")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        assert self._gate(tmp_path) == set()
        assert not (run_dir / "best.pt").exists()
        marker = json.loads((run_dir / "override_applied.json").read_text())
        assert marker["job_id"] == "456"

    def test_without_a_job_id_it_always_overrides_and_writes_no_marker(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        run_dir = _make_run_dir(tmp_path, self.NAME, ["best.pt"])
        assert self._gate(tmp_path) == set()
        assert not (run_dir / "best.pt").exists()
        assert not (run_dir / "override_applied.json").exists()

    def test_dry_run_writes_no_marker_and_reports_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        run_dir = _make_run_dir(tmp_path, self.NAME, ["best.pt"])
        assert self._gate(tmp_path, dry_run=True) == set()
        assert (run_dir / "best.pt").exists()
        assert not (run_dir / "override_applied.json").exists()

    def test_failed_delete_writes_no_marker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        run_dir = _make_run_dir(tmp_path, self.NAME, ["best.pt", "results.json"])
        real_unlink = Path.unlink

        def failing_unlink(self, *args, **kwargs):
            if self.name == "results.json":
                raise OSError("boom")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", failing_unlink)
        with pytest.raises(OSError):
            self._gate(tmp_path)
        assert (run_dir / "best.pt").exists()
        assert not (run_dir / "override_applied.json").exists()

    def test_a_marker_survives_a_later_override(self, tmp_path, monkeypatch):
        """The marker is not in extra_delete: it is overwritten, never removed first."""
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        run_dir = _make_run_dir(tmp_path, self.NAME, ["best.pt"])
        self._gate(tmp_path)
        assert (run_dir / "override_applied.json").exists()

    def test_second_launch_in_one_allocation_overrides_again(
        self, tmp_path, monkeypatch
    ):
        """Same job id, no requeue (salloc / srun --pty): a deliberate relaunch."""
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        run_dir = _make_run_dir(tmp_path, self.NAME, ["best.pt"])
        self._gate(tmp_path)
        (run_dir / "best.pt").write_bytes(b"new")
        calls = []
        assert self._gate(tmp_path, calls) == set()
        assert not (run_dir / "best.pt").exists()
        assert calls == [self.NAME]

    def test_requeue_before_the_new_checkpoint_exists_is_continued(
        self, tmp_path, monkeypatch, capsys
    ):
        """Preempted after the delete, before the first save: no checkpoint to find."""
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        run_dir = _make_run_dir(tmp_path, self.NAME, ["best.pt"])
        self._gate(tmp_path)  # deletes best.pt, writes the marker
        assert not (run_dir / "best.pt").exists()
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        calls = []
        assert self._gate(tmp_path, calls) == {self.NAME}
        assert calls == []  # no second id rotation
        assert "already applied by job 123" in capsys.readouterr().out

    def test_marker_alone_does_not_make_a_dry_run_continue(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        _make_run_dir(tmp_path, self.NAME, ["best.pt"])
        self._gate(tmp_path)
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        assert self._gate(tmp_path, dry_run=True) == set()

    def test_marker_write_failure_warns_and_does_not_fail_the_launch(
        self, tmp_path, monkeypatch, capsys
    ):
        """The delete already happened: failing now protects nothing."""
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        run_dir = _make_run_dir(tmp_path, self.NAME, ["best.pt"])

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("src.training.run_naming.write_override_marker", boom)
        assert self._gate(tmp_path) == set()
        assert not (run_dir / "best.pt").exists()
        err = capsys.readouterr().err
        assert "could not write" in err and "disk full" in err

    def test_no_checkpoint_after_a_crash_between_delete_and_marker_is_a_noop(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        _make_run_dir(tmp_path, self.NAME, [])  # checkpoint already deleted, no marker
        assert self._gate(tmp_path) == set()

    def test_resume_and_inference_return_an_empty_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        _make_run_dir(tmp_path, self.NAME, ["best.pt"])
        assert (
            check_checkpoint_collisions(
                {1: self.NAME}, "best.pt", True, False, checkpoint_root=tmp_path
            )
            == set()
        )
        assert (
            check_checkpoint_collisions(
                {1: self.NAME},
                "best.pt",
                False,
                False,
                checkpoint_root=tmp_path,
                inference_only=True,
            )
            == set()
        )


def test_effective_resume():
    assert effective_resume(True, "a", set()) is True
    assert effective_resume(False, "a", {"a"}) is True
    assert effective_resume(False, "a", {"b"}) is False
    assert effective_resume(False, "a", set()) is False
