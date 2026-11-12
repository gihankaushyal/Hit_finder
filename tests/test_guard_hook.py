"""Tests for .claude/hooks/guard_destructive.py (the PreToolUse Bash guard)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / ".claude" / "hooks" / "guard_destructive.py"
ROOT = "/repo"

_spec = importlib.util.spec_from_file_location("guard_destructive", HOOK)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


class StubCtx:
    """Stands in for GitContext so decide() runs without a git repository."""

    def __init__(self, main=True, branch="phase-05-demo", report="ready", existing=()):
        self.main, self.branch, self.report, self.existing = (
            main,
            branch,
            report,
            existing,
        )

    def is_main_checkout(self, cwd):
        return self.main

    def checkout_roots(self, cwd):
        return [ROOT]

    def current_branch(self, cwd):
        return self.branch

    def path_exists(self, cwd, name):
        return name in self.existing

    def eval_report_status(self, cwd):
        return self.report


def action(command, **ctx):
    return guard.decide(command, ROOT, StubCtx(**ctx))[0]


@pytest.mark.parametrize(
    "command",
    [
        "git push --force origin main",
        "git push -f origin phase-05",
        "git push origin +phase-05",
        "git push --force-with-lease origin HEAD:refs/heads/main",
        "git push --force --all",
        "git add -f PLANNING.md",
        "git add --force docs/superpowers/specs/x.md",
        "git add -f phase-05-kanban.md MEMORY.md",
        "git add -f .",
        "git add .secrets/wandb.env",
        "cat .secrets/wandb.env",
        "source .secrets/wandb.env && grep KEY .secrets/wandb.env",
        "export K=$(cat .secrets/wandb.env)",
        "bash -c 'head -1 .secrets/wandb.env'",
    ],
)
def test_denied(command):
    assert action(command) == guard.DENY


def test_force_push_without_refspec_uses_current_branch():
    assert action("git push --force", branch="phase-05") == guard.DENY
    assert action("git push --force", branch="phase-05-demo") == guard.ALLOW


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf checkpoints/resnet18-asymmetric-v2-fold1-seed42",
        "rm -r logs",
        "rm -fr ./data/processed",
        "rm -rf /data/bioxfel/user/gihan/Hit_finder_cache/eiger",
        "rm -rf /tmp/sfx_frame_cache",
        "rm -rf *",
        "rm -rf ..",
        'rm -rf "$RUN_DIR/checkpoints"',
        "cd checkpoints && rm -rf old-run",
        "ls checkpoints | xargs rm -rf",
        "git reset --hard HEAD~1",
        "git clean -fd",
        "git checkout main",
        "git switch -c phase-05-new",
        "git checkout -b phase-05-new",
        "scancel 779452 779453",
        "scancel -u gketawal",
        "scancel 779452_[1-4]",
        "squeue -h -o %i | xargs scancel",
        "touch .secrets/new.env",
    ],
)
def test_asks(command):
    assert action(command) == guard.ASK


@pytest.mark.parametrize(
    "command",
    [
        "git push origin phase-05-demo",
        "git push --force-with-lease origin phase-05-demo",
        "git push origin phase-05",
        "git add src/evaluation/metrics.py",
        "git add -f tests/fixtures/sample.bin",
        "source .secrets/wandb.env && python -m src.training.train_asymmetric --config c.yaml",
        "ls -l .secrets/",
        "stat -c %a .secrets/wandb.env",
        "git check-ignore -v .secrets/wandb.env",
        "ln -s /repo/.secrets .secrets",
        "rm -rf /tmp/scratch/run1",
        "rm -rf dashboard/node_modules",
        "rm checkpoints/run/tmp.pt",
        "rm -rf tests/data",
        "git clean -nfd",
        "git reset --soft HEAD~1",
        "git checkout -- src/models/ssl.py",
        "git checkout HEAD src/models/ssl.py",
        "git status && git log --oneline -3",
        "scancel 779452",
        "pytest tests/ -v 2>&1 | tail -5",
    ],
)
def test_allowed(command):
    assert action(command) == guard.ALLOW


def test_checkout_of_existing_file_is_a_restore():
    assert action("git checkout README.md", existing=("README.md",)) == guard.ALLOW


def test_branch_switch_is_allowed_in_a_worktree():
    assert action("git checkout phase-05", main=False) == guard.ALLOW
    assert action("cd /repo && git switch phase-05", main=False) == guard.ALLOW


@pytest.mark.parametrize("status", ["missing", "not-ready", "ready-after-fixes"])
def test_pr_create_asks_without_ready_report(status):
    assert action("gh pr create --base phase-05 --title t", report=status) == guard.ASK


def test_pr_create_allowed_with_ready_report():
    assert (
        action("gh pr create --base phase-05 --title t", report="ready") == guard.ALLOW
    )


def _write_report(directory, name, branch, commit, verdict):
    text = "---\nworkflow: evaluate-code\nbranch: %s\ncommit: %s\nverdict: %s\n---\n# body\n"
    path = directory / name
    path.write_text(text % (branch, commit, verdict))
    return path


def test_report_status_matches_branch_and_commit(tmp_path):
    sha = "0123456789abcdef0123456789abcdef01234567"
    assert guard.report_status(str(tmp_path), "b", sha) == "missing"
    _write_report(tmp_path, "2026-11-11-evaluate-code-a.md", "other", sha, "ready")
    _write_report(tmp_path, "2026-11-11-evaluate-code-b.md", "b", "fffffff", "ready")
    _write_report(tmp_path, "2026-11-11-debug-c.md", "b", sha, "ready")
    assert guard.report_status(str(tmp_path), "b", sha) == "missing"
    _write_report(tmp_path, "2026-11-11-evaluate-code-d.md", "b", sha[:10], "ready")
    assert guard.report_status(str(tmp_path), "b", sha) == "ready"


def test_report_status_takes_the_newest_report(tmp_path):
    sha = "0123456789abcdef0123456789abcdef01234567"
    old = _write_report(tmp_path, "2026-11-10-evaluate-code-a.md", "b", sha, "ready")
    new = _write_report(
        tmp_path, "2026-11-11-evaluate-code-a.md", "b", sha, "not-ready"
    )
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    assert guard.report_status(str(tmp_path), "b", sha) == "not-ready"


def test_cache_dir_names_match_base_config():
    cache = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())["cache"]
    expected = {Path(cache["root"]).name, Path(cache["nvme_root"]).name}
    assert set(guard.CACHE_DIR_NAMES) == expected


def _run_hook(stdin_text):
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def test_cli_is_silent_for_harmless_commands(tmp_path):
    payload = {"tool_input": {"command": "ls -la"}, "cwd": str(tmp_path)}
    assert _run_hook(json.dumps(payload)) is None


def test_cli_emits_the_decision_json(tmp_path):
    payload = {
        "tool_input": {"command": "cat .secrets/wandb.env"},
        "cwd": str(tmp_path),
    }
    out = _run_hook(json.dumps(payload))["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert out["permissionDecision"] == "deny"
    assert out["permissionDecisionReason"].startswith("guard: ")


def test_cli_asks_when_the_hook_itself_fails():
    out = _run_hook("this is not json")["hookSpecificOutput"]
    assert out["permissionDecision"] == "ask"
    assert "guard hook error" in out["permissionDecisionReason"]
