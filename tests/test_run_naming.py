"""Unit tests for the shared run-naming standard and checkpoint collision gate."""

from __future__ import annotations

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
