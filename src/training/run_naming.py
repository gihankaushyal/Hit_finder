"""Run-naming standard and checkpoint collision gate shared by training entry points.

Every training entry point takes a required ``--run-name-prefix`` ending in
``-v<N>``, where <N> is the pipeline generation that produced the run. The
prefix becomes both the wandb run id/name and the checkpoint directory
(``checkpoints/<prefix>-fold{N}-seed{S}/``), so two pipeline generations can
never collide under the same name.

    Track 1 (train_asymmetric):     <backbone>-asymmetric-v<N>   resnet18-asymmetric-v2
    Track 2 pretrain:               mae-<backbone>-v<N>          mae-vits16-v2
    Track 2 fine-tune / probe:      <backbone>-mae-v<N>          vits16-mae-v2

For fine-tune / probe the mode is inserted before the version, giving
``vits16-mae-finetune-v2`` or ``vits16-mae-probe-v2``.

If a checkpoint already exists under the resolved run name, the entry point
exits unless ``--resume-training`` (continue), ``--override-training``
(discard and restart) or ``--inference-only`` (evaluate it without training;
the checkpoint must exist) was passed. The three are mutually exclusive.
"""

from __future__ import annotations

import re
from pathlib import Path

CHECKPOINT_ROOT = Path("checkpoints")

ASYMMETRIC_PREFIX_RE = re.compile(r"^[a-z0-9]+-asymmetric-v[0-9]+$")
ASYMMETRIC_CONVENTION = "<backbone>-asymmetric-v<N>"
ASYMMETRIC_EXAMPLE = "resnet18-asymmetric-v2"

SSL_PRETRAIN_PREFIX_RE = re.compile(r"^mae-[a-z0-9]+-v[0-9]+$")
SSL_PRETRAIN_CONVENTION = "mae-<backbone>-v<N>"
SSL_PRETRAIN_EXAMPLE = "mae-vits16-v2"

SSL_FINETUNE_PREFIX_RE = re.compile(
    r"^(?P<backbone>[a-z0-9]+)-mae-v(?P<version>[0-9]+)$"
)
SSL_FINETUNE_CONVENTION = "<backbone>-mae-v<N>"
SSL_FINETUNE_EXAMPLE = "vits16-mae-v2"

SSL_MODE_FINETUNE = "finetune"
SSL_MODE_PROBE = "probe"


def validate_run_name_prefix(
    prefix: str, pattern: re.Pattern[str], convention: str, example: str
) -> None:
    """Exit with the naming convention if `prefix` does not match `pattern`."""
    if not pattern.fullmatch(prefix):
        raise SystemExit(
            f"Invalid --run-name-prefix: {prefix!r}\n\n"
            "Run names must follow the convention:\n\n"
            f"    {convention}\n\n"
            "<N> identifies the pipeline generation that produced this run, incremented\n"
            "whenever the preprocessing/pipeline changes in a way that invalidates a direct\n"
            "numeric comparison with the previous generation (e.g. v1 = pre-frame-cache,\n"
            "v2 = frame-cache-backed).\n\n"
            f"Example: --run-name-prefix {example}"
        )


def fold_run_name(prefix: str, fold_id: int, seed: int, run_suffix: str = "") -> str:
    """Run name for one fold: wandb run id/name and checkpoint directory name."""
    return f"{prefix}-fold{fold_id}-seed{seed}{run_suffix}"


def expand_finetune_prefix(prefix: str, linear_probe: bool) -> str:
    """Insert the mode before the version: vits16-mae-v2 -> vits16-mae-finetune-v2."""
    m = SSL_FINETUNE_PREFIX_RE.fullmatch(prefix)
    if m is None:
        raise ValueError(
            f"{prefix!r} does not match the fine-tune convention "
            f"{SSL_FINETUNE_CONVENTION}"
        )
    mode = SSL_MODE_PROBE if linear_probe else SSL_MODE_FINETUNE
    return f"{m['backbone']}-mae-{mode}-v{m['version']}"


def check_checkpoint_collisions(
    run_names: dict[int, str],
    checkpoint_name: str,
    resume_training: bool,
    override_training: bool,
    extra_delete: tuple[str, ...] = (),
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
    dry_run: bool = False,
    inference_only: bool = False,
) -> None:
    """Gate on checkpoints that already exist under the resolved run names.

    `run_names` maps fold id -> run name. For each run whose
    `<checkpoint_root>/<run_name>/<checkpoint_name>` exists: with
    `override_training` the checkpoint and every match of the `extra_delete`
    glob patterns in that run directory are deleted; with `resume_training` or
    `inference_only` nothing is touched; with none of the three the process
    exits. Extras are only removed when the primary checkpoint exists.

    `inference_only` inverts the missing case: every run must already have the
    checkpoint (there is nothing to evaluate otherwise), so a missing one exits.

    Passing more than one of `resume_training`, `override_training` and
    `inference_only` is an error (ValueError). With `dry_run` nothing is ever
    deleted: an unresolved collision (or a missing inference checkpoint) still
    exits, so callers can fail early and defer the destructive part until just
    before training starts.
    """
    if sum((resume_training, override_training, inference_only)) > 1:
        raise ValueError(
            "resume_training, override_training and inference_only are mutually exclusive"
        )
    for fold_id, run_name in run_names.items():
        run_dir = Path(checkpoint_root) / run_name
        ckpt_path = run_dir / checkpoint_name
        if not ckpt_path.exists():
            if inference_only:
                raise SystemExit(
                    f"--inference-only needs an existing checkpoint, none found for "
                    f"fold {fold_id}: {ckpt_path}\n\n"
                    "Train this fold first, or pass --resume-training / "
                    "--override-training instead."
                )
            continue
        if override_training:
            if dry_run:
                continue
            # Extras first, primary checkpoint last: if an extra fails to delete,
            # the checkpoint survives so a rerun still sees the collision.
            for pattern in extra_delete:
                for stale in run_dir.glob(pattern):
                    if stale.is_file() or stale.is_symlink():
                        stale.unlink()
            ckpt_path.unlink()
        elif not resume_training and not inference_only:
            raise SystemExit(
                f"Checkpoint already exists for fold {fold_id}: {ckpt_path}\n\n"
                "Re-run with --resume-training to continue training from it, "
                f"--override-training to discard it and start fold {fold_id} from "
                "scratch, or --inference-only to evaluate it without training."
            )
