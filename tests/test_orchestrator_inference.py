"""Orchestrator behaviour for the `inference` answer, run against a stub sbatch.

Every script runs in a temp directory with a fake `sbatch` first on PATH that only
logs its arguments; no test here can reach the real scheduler.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

STUB_SBATCH = """#!/bin/bash
n=$(cat "$STUB_DIR/counter" 2>/dev/null || echo 100); n=$((n+1)); echo "$n" > "$STUB_DIR/counter"
echo "$n: $*" >> "$STUB_DIR/sbatch.log"
echo "$n"
"""

FINETUNE_ALL = "submit_ssl_finetune_all.sh"
TRACK1_ALL = "submit_asymmetric_lodo_all.sh"
PRETRAIN_PREFIX = "mae-vits16-v2"
FINETUNE_RUN = "vits16-mae-finetune-v2-fold1-seed42"
PROBE_RUN = "vits16-mae-probe-v2-fold1-seed42"
TRACK1_RUN = "resnet18-asymmetric-v2-fold1-seed42"


@pytest.fixture
def sandbox(tmp_path):
    work = tmp_path / "repo"
    (work / "scripts").mkdir(parents=True)
    for sub in ("ssl", "supervised"):
        (work / "configs" / sub).mkdir(parents=True)
        for cfg in (REPO / "configs" / sub).glob("*.yaml"):
            shutil.copy(cfg, work / "configs" / sub / cfg.name)
    shutil.copy(REPO / "configs" / "base.yaml", work / "configs" / "base.yaml")
    for script in (FINETUNE_ALL, TRACK1_ALL):
        shutil.copy(SCRIPTS / script, work / "scripts" / script)
    stub_dir, bin_dir = tmp_path / "stub", tmp_path / "bin"
    stub_dir.mkdir()
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(STUB_SBATCH)
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IEXEC)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_DIR": str(stub_dir),
    }
    for var in ("RESUME_FLAG", "SLURM_RESTART_COUNT"):
        env.pop(var, None)
    # A broken fixture must fail loudly instead of reaching the real scheduler.
    resolved = subprocess.run(
        ["bash", "-c", "command -v sbatch"], env=env, capture_output=True, text=True
    ).stdout.strip()
    assert resolved == str(sbatch)
    return work, env, stub_dir


def _run(work, env, script, *args, stdin=""):
    return subprocess.run(
        ["bash", f"scripts/{script}", *args],
        cwd=work,
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _log(stub_dir) -> list[str]:
    path = stub_dir / "sbatch.log"
    return path.read_text().splitlines() if path.exists() else []


def _best_pt(work: Path, run_name: str) -> None:
    run_dir = work / "checkpoints" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"x")


def _pretrain_ckpt(work: Path, fold: int = 1) -> None:
    run_dir = work / "checkpoints" / f"{PRETRAIN_PREFIX}-fold{fold}-seed42"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "last.pt").write_bytes(b"x")


FT_ARGS = [
    "--run-name-prefix",
    "vits16-mae-v2",
    "--pretrain-run-prefix",
    PRETRAIN_PREFIX,
    "--folds",
    "1",
]


class TestFinetuneOrchestrator:
    def test_inference_answer_needs_no_pretrain_checkpoint(self, sandbox):
        work, env, stub = sandbox
        _best_pt(work, FINETUNE_RUN)
        proc = _run(
            work, env, FINETUNE_ALL, *FT_ARGS, "--no-probe", stdin="inference\n"
        )
        assert proc.returncode == 0, proc.stderr
        jobs = [line for line in _log(stub) if "submit_ssl_finetune.sh" in line]
        assert len(jobs) == 1
        assert "RESUME_FLAG=--inference-only" in jobs[0]

    def test_a_fold_that_will_train_still_needs_its_pretrain_checkpoint(self, sandbox):
        work, env, stub = sandbox
        proc = _run(work, env, FINETUNE_ALL, *FT_ARGS, "--no-probe")
        assert proc.returncode == 1
        assert "pretrain checkpoint not found" in proc.stderr
        assert _log(stub) == []

    def test_mixed_answers_one_trains_one_infers(self, sandbox):
        work, env, stub = sandbox
        _best_pt(work, FINETUNE_RUN)  # fine-tune finished -> answer inference
        _pretrain_ckpt(work)  # probe has no best.pt -> trains, needs this
        proc = _run(work, env, FINETUNE_ALL, *FT_ARGS, stdin="inference\n")
        assert proc.returncode == 0, proc.stderr
        flags = {
            ("probe" if "--linear-probe" in line else "finetune"): line
            for line in _log(stub)
            if "submit_ssl_finetune.sh" in line
        }
        assert "RESUME_FLAG=--inference-only" in flags["finetune"]
        assert "RESUME_FLAG=--resume-training" in flags["probe"]

    def test_all_inference_answers_skip_the_pretrain_requirement(self, sandbox):
        work, env, stub = sandbox
        _best_pt(work, FINETUNE_RUN)
        _best_pt(work, PROBE_RUN)
        proc = _run(work, env, FINETUNE_ALL, *FT_ARGS, stdin="inference\ninference\n")
        assert proc.returncode == 0, proc.stderr
        jobs = [line for line in _log(stub) if "submit_ssl_finetune.sh" in line]
        assert len(jobs) == 2
        assert all("RESUME_FLAG=--inference-only" in j for j in jobs)

    def test_prompt_names_the_three_answers_and_rejects_others(self, sandbox):
        work, env, stub = sandbox
        _best_pt(work, FINETUNE_RUN)
        proc = _run(
            work,
            env,
            FINETUNE_ALL,
            *FT_ARGS,
            "--no-probe",
            stdin="bogus\ninference\n",
        )
        assert proc.returncode == 0, proc.stderr
        assert "'inference'" in proc.stdout
        assert "Please type exactly" in proc.stdout

    def test_a_fold_that_trains_anyway_is_checked_before_any_prompt(self, sandbox):
        """Fine-tune finished (would prompt), probe never ran (trains) and the pretrain
        checkpoint is missing: report that first, not after the user answered."""
        work, env, stub = sandbox
        _best_pt(work, FINETUNE_RUN)
        proc = _run(work, env, FINETUNE_ALL, *FT_ARGS, stdin="")
        assert proc.returncode == 1
        assert "pretrain checkpoint not found" in proc.stderr
        assert "No answer read" not in proc.stderr
        assert "Type 'resume'" not in proc.stdout
        assert _log(stub) == []

    def test_eof_at_the_prompt_submits_nothing(self, sandbox):
        work, env, stub = sandbox
        _best_pt(work, FINETUNE_RUN)
        proc = _run(work, env, FINETUNE_ALL, *FT_ARGS, "--no-probe", stdin="")
        assert proc.returncode == 1
        assert _log(stub) == []


class TestTrack1Orchestrator:
    ARGS = ["--run-name-prefix", "resnet18-asymmetric-v2", "--folds", "1"]

    def test_inference_answer_reaches_the_fold_job(self, sandbox):
        work, env, stub = sandbox
        _best_pt(work, TRACK1_RUN)
        proc = _run(work, env, TRACK1_ALL, *self.ARGS, stdin="inference\n")
        assert proc.returncode == 0, proc.stderr
        jobs = [line for line in _log(stub) if "submit_asymmetric_lodo_fold.sh" in line]
        assert len(jobs) == 1
        assert "RESUME_FLAG=--inference-only" in jobs[0]

    def test_resume_and_override_answers_still_work(self, sandbox):
        work, env, stub = sandbox
        _best_pt(work, TRACK1_RUN)
        for answer, flag in (
            ("resume", "--resume-training"),
            ("override", "--override-training"),
        ):
            (stub / "sbatch.log").unlink(missing_ok=True)
            proc = _run(work, env, TRACK1_ALL, *self.ARGS, stdin=f"{answer}\n")
            assert proc.returncode == 0, proc.stderr
            jobs = [
                line for line in _log(stub) if "submit_asymmetric_lodo_fold.sh" in line
            ]
            assert f"RESUME_FLAG={flag}" in jobs[0]


class TestJobScripts:
    def test_finetune_job_skips_the_pretrain_requirement_for_inference(self):
        text = (SCRIPTS / "submit_ssl_finetune.sh").read_text()
        assert '[[ "${RESUME_FLAG}" == "--inference-only" ]]' in text
        assert "PRETRAIN_ARGS=()" in text
        assert '"${PRETRAIN_ARGS[@]}"' in text
        # the unconditional --pretrain-checkpoint argument is gone
        assert '--pretrain-checkpoint "${PRETRAIN_CKPT}" \\' not in text

    @pytest.mark.parametrize(
        "script", ["submit_ssl_finetune.sh", "submit_asymmetric_lodo_fold.sh"]
    )
    def test_job_scripts_document_the_third_value(self, script):
        assert "--inference-only" in (SCRIPTS / script).read_text()

    def test_finetune_orchestrator_runs_its_preflight_after_the_prompts(self):
        text = (SCRIPTS / FINETUNE_ALL).read_text()
        assert text.index("PROBE_FLAGS[${fold}]=") < text.rindex("MISSING=0")
