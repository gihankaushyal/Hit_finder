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

    """None for `main` or `branch` stands for a git call that failed."""

    def __init__(
        self,
        main=True,
        branch="phase-05-demo",
        report="ready",
        existing=(),
        roots=(ROOT,),
    ):
        self.main, self.branch, self.report, self.existing, self.roots = (
            main,
            branch,
            report,
            existing,
            roots,
        )

    def is_main_checkout(self, cwd):
        return self.main

    def checkout_roots(self, cwd):
        return list(self.roots)

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
        "grep -r . .secrets/",
        "grep -r . .secrets",
        "tar -cf - .secrets | tar -xO",
        "cat .secret?/wandb.env",
        "cat .s*/wandb.env",
        "cat /repo/.secrets//wandb.env",
        "cd .secrets && cat wandb.env",
        "cd /repo/.secrets; head -1 *",
        "source .secrets/wandb.env && printenv WANDB_API_KEY",
        "source .secrets/wandb.env && echo $WANDB_API_KEY",
        'source .secrets/wandb.env && printf "%s" "${WANDB_API_KEY}"',
        "source .secrets/wandb.env && env",
        "source .secrets/wandb.env && printenv",
        "source .secrets/wandb.env && export -p",
        "source .secrets/wandb.env && set",
        "source .secrets/wandb.env && declare -p",
        "set -x; source .secrets/wandb.env",
        "set -o xtrace && . .secrets/wandb.env",
        "echo $WANDB_API_KEY",
        "printenv WANDB_API_KEY",
        # quoting that only the shell removes
        'cat ."secrets"/wandb.env',
        "cat .\\secrets/wandb.env",
        "cat '.'secrets/wandb.env",
        "printenv WANDB_API''_KEY",
        'echo $"WANDB_API_KEY"',
        # cd options, pushd
        "cd -P .secrets && cat wandb.env",
        "cd -- .secrets; cat wandb.env",
        "pushd .secrets && cat wandb.env",
        # subshells and groups
        "source .secrets/wandb.env; (env)",
        "source .secrets/wandb.env && ( set )",
        "(source .secrets/wandb.env; env)",
        "{ source .secrets/wandb.env; env; }",
        # other dumps and traces
        "source .secrets/wandb.env && set -v",
        "set -o verbose; source .secrets/wandb.env",
        "set -euxo pipefail; source .secrets/wandb.env",
        "source .secrets/wandb.env && declare -x",
        "source .secrets/wandb.env && declare -px",
        "source .secrets/wandb.env && typeset -x",
        "source .secrets/wandb.env && export",
        "source .secrets/wandb.env && env -u PATH",
        "source .secrets/wandb.env && env FOO=1",
        "declare -p WANDB_API_KEY",
        "bash -xc 'source .secrets/wandb.env'",
        "bash -x -c 'source .secrets/wandb.env'",
        "bash -o xtrace -c '. .secrets/wandb.env'",
        "sh -ec 'cat .secrets/wandb.env'",
        # other ways in
        "builtin echo $WANDB_API_KEY",
        "eval 'echo $WANDB_API_KEY'",
        'eval "cat .secrets/wandb.env"',
        "cat /proc/self/environ",
        "strings /proc/1234/environ",
        "cat .{s,}ecrets/wandb.env",
        "cat .secret{s,}/wandb.env",
        "head {.secrets,x}/wandb.env",
        "grep -r -e x .secrets/",
        "git grep -n KEY -- .secrets/wandb.env",
        # a quoted ; or | is not a command separator
        "cat 'x| ls ' .secrets/wandb.env",
        "cat 'x; source ' .secrets/wandb.env",
        'git push -f origin "main" -o "x;y"',
        # wrappers and git global options
        "timeout 30 git push --force origin main",
        "env git push -f origin main",
        "env -u X git push -f origin main",
        "git --git-dir .git push -f origin main",
        # a key written into the command
        "WANDB_API_KEY=abc123 python -m src.training.train_asymmetric",
        "export WANDB_API_KEY=abc123",
        "sbatch --export=ALL,WANDB_API_KEY=abc123 job.sh",
        # git add -f of local-only files
        "git add -f docs/*",
        "git add -f session-tail.md",
        "git add -f docs/progress_notes.md",
        "git add -f $(git ls-files -o --exclude-standard)",
        # find -exec bodies are checked
        "find . -maxdepth 0 -exec cat .secrets/wandb.env \\;",
        # shell keywords, long shell options, sudo/command options
        "if git push -f origin main; then :; fi",
        "while git push -f origin main; do :; done",
        "until git push -f origin main; do :; done",
        "bash --norc -c 'git push -f origin main'",
        "bash -c -- 'git push -f origin main'",
        "bash --rcfile x -c 'git push -f origin main'",
        "sudo -u x git push -f origin main",
        "command -p git push -f origin main",
        "git push -f \\\norigin main",
        # pathspec magic in git add -f
        "git add -f :/docs",
        "git add -f ':(top)PLANNING.md'",
        "git add -f ':(glob)docs/**'",
        # deleting a protected branch on the remote
        "git push --delete origin main",
        "git push -d origin phase-05",
        "git push origin :phase-05",
        "git push origin +:refs/heads/main",
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
        "cd .secrets && touch new.env",
        'cat <<< "$WANDB_API_KEY"',
        """awk 'BEGIN{print ENVIRON["WANDB_API_KEY"]}'""",
        """python -c "import os; print(os.environ['WANDB_API_KEY'])\"""",
        'source .secrets/wandb.env && python -c "import os; print(os.environ)"',
        "d=.secrets; cat $d/other.env",
        "F=.s*/wandb.env python x.py",
        'git commit -m "$WANDB_API_KEY"',
        "tr a a < .secrets/wandb.env",
        'rm -rf "checkpoints" "a;b"',
        "nice rm -rf checkpoints",
        "nice -n 10 rm -rf checkpoints",
        'echo "unterminated',
        "source .secrets/wandb.env; v=$WANDB_API_KEY; echo $v",
        "k=WANDB_API_KEY; echo ${!k}",
        "find . -name wandb.env -exec cat {} \\;",
        "find . -name wandb.env | xargs cat",
        "find .secrets -type f -exec cat {} \\;",
        "find . -maxdepth 0 -exec rm -rf checkpoints \\;",
        "if rm -rf checkpoints; then :; fi",
        "sudo -u x rm -rf checkpoints",
        "command -p rm -rf checkpoints",
        "rm -rf \\\ncheckpoints",
        # discarding uncommitted work in many files at once
        "git checkout -- .",
        "git checkout .",
        "git checkout HEAD -- src/",
        "git restore .",
        "git restore --worktree --staged .",
        "git restore -s HEAD~1 '*.py'",
        "git stash clear",
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
        "ls .secrets",
        "ls -la .secrets/ && stat .secrets",
        "mkdir -p .secrets && chmod 700 .secrets",
        "cd .secrets && ls -l",
        "cat *",
        "cat docs/secrets.md",
        "source .secrets/wandb.env && wandb login --relogin",
        "set -euo pipefail; source .secrets/wandb.env && python -m src.training.lodo",
        "echo done",
        "env | grep CUDA",
        # search patterns and message text are not file names
        'grep -rn ".secrets" scripts/',
        "grep -n '^\\.secrets' .gitignore",
        "grep -rn -A 2 WANDB_API_KEY scripts/",
        "rg -e .secrets -e WANDB_API_KEY docs/",
        "git grep -n WANDB_API_KEY",
        'git commit -m "guard: block reads of .secrets/ and WANDB_API_KEY"',
        "git commit -am 'deny Read(**/.secrets)'",
        "git log --grep .secrets --oneline",
        'echo "WANDB_API_KEY not set"',
        'test -n "$WANDB_API_KEY"',
        '[ -z "$WANDB_API_KEY" ] && echo missing',
        "unset WANDB_API_KEY",
        "cd -P /tmp && ls",
        "(cd /tmp && ls)",
        "source .secrets/wandb.env && export WANDB_MODE=offline",
        "source .secrets/wandb.env && env CUDA_VISIBLE_DEVICES=0 python x.py",
        "source .secrets/wandb.env && bash -c 'python x.py'",
        "source .secrets/wandb.env && set +x",
        "echo ${HOME}",
        'echo "a; b | c && d"',
        'git commit -m "fix: don\'t split on a quoted ; or |"',
        "time -p python x.py",
        "timeout 600 python -m pytest tests/ -q",
        "nice -n 10 python x.py",
        "env -u DISPLAY python x.py",
        "git --git-dir .git status",
        "find . -name '*.pyc' -exec rm {} +",
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
        "if [ -f x ]; then echo y; fi",
        "while read -r f; do echo $f; done < list.txt",
        "sudo -u x ls",
        "command -v git",
        "bash --norc -c 'echo hi'",
        "echo a \\\n  b",
        "git push --delete origin phase-05-demo",
        "git push origin :phase-05-demo",
        "git restore src/models/ssl.py",
        "git restore --staged .",
        "git stash list",
    ],
)
def test_allowed(command):
    assert action(command) == guard.ALLOW


def test_at_sign_is_the_current_branch():
    assert action("git push -f origin @", branch="phase-05") == guard.DENY
    assert action("git push -f origin @", branch="phase-05-demo") == guard.ALLOW
    assert action("git push origin :@", branch="phase-05") == guard.DENY


def test_line_continuation_does_not_hide_the_pr_gate():
    assert action("gh \\\npr create --base phase-05", report="missing") == guard.ASK


def test_failed_git_calls_ask():
    assert action("git checkout phase-04", main=None) == guard.ASK
    assert action("git push --force", branch=None) == guard.ASK
    assert action("git push -f origin HEAD", branch=None) == guard.ASK


def test_recursive_rm_asks_without_roots():
    assert action("rm -rf build", roots=()) == guard.ASK


def test_bash_trace_of_a_script_that_sources_secrets(tmp_path):
    job = tmp_path / "job.sh"
    job.write_text("source .secrets/wandb.env\npython x.py\n")
    plain = tmp_path / "plain.sh"
    plain.write_text("python x.py\n")
    ctx = StubCtx()
    assert guard.decide("bash -x job.sh", str(tmp_path), ctx)[0] == guard.DENY
    assert guard.decide("sh -v %s" % job, ROOT, ctx)[0] == guard.DENY
    assert guard.decide("bash -x plain.sh", str(tmp_path), ctx)[0] == guard.ALLOW
    assert guard.decide("bash job.sh", str(tmp_path), ctx)[0] == guard.ALLOW


def test_checkout_roots_include_the_hook_checkout():
    roots = [os.path.realpath(r) for r in guard.GitContext().checkout_roots("/")]
    assert os.path.realpath(str(REPO)) in roots


def test_git_calls_stop_after_the_time_budget():
    ctx = guard.GitContext()
    ctx.start -= guard.GIT_BUDGET_S + 1
    with pytest.raises(guard.GuardTimeout):
        ctx.current_branch(str(REPO))


def test_checkout_of_existing_file_is_a_restore():
    assert action("git checkout README.md", existing=("README.md",)) == guard.ALLOW


def test_branch_switch_is_allowed_in_a_worktree():
    assert action("git checkout phase-05", main=False) == guard.ALLOW
    assert action("cd /repo && git switch phase-05", main=False) == guard.ALLOW


@pytest.mark.parametrize("status", ["missing", "not-ready", "ready-after-fixes"])
def test_pr_create_asks_without_ready_report(status):
    assert action("gh pr create --base phase-05 --title t", report=status) == guard.ASK


@pytest.mark.parametrize(
    "command",
    ["gh pr new --fill", "gh -R o/r pr create --fill", "gh --repo o/r pr new"],
)
def test_pr_create_aliases_and_repo_flag_are_gated(command):
    assert action(command, report="missing") == guard.ASK


def test_pr_create_allowed_with_ready_report():
    assert (
        action("gh pr create --base phase-05 --title t", report="ready") == guard.ALLOW
    )


def test_pr_text_may_name_secrets():
    command = (
        'gh pr create --base phase-05 --title "guard: .secrets reads" '
        '--body "blocks WANDB_API_KEY prints"'
    )
    assert action(command, report="ready") == guard.ALLOW


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


def test_report_status_breaks_mtime_ties_by_name(tmp_path):
    sha = "0123456789abcdef0123456789abcdef01234567"
    old = _write_report(tmp_path, "2026-11-11-evaluate-code-a.md", "b", sha, "ready")
    new = _write_report(
        tmp_path, "2026-11-12-evaluate-code-a.md", "b", sha, "not-ready"
    )
    os.utime(old, (1_000, 1_000))
    os.utime(new, (1_000, 1_000))
    assert guard.report_status(str(tmp_path), "b", sha) == "not-ready"


def test_report_status_reads_utf8_under_an_ascii_locale(tmp_path):
    sha = "0123456789abcdef0123456789abcdef01234567"
    path = _write_report(tmp_path, "2026-11-11-evaluate-code-a.md", "b", sha, "ready")
    path.write_text(path.read_text() + "| 🔴 | finding |\n" * 50, encoding="utf-8")
    code = (
        "import importlib.util, sys\n"
        "s = importlib.util.spec_from_file_location('g', sys.argv[1])\n"
        "g = importlib.util.module_from_spec(s); s.loader.exec_module(g)\n"
        "print(g.report_status(sys.argv[2], 'b', sys.argv[3]))\n"
    )
    # Disable 3.7+ locale coercion and UTF-8 mode, as under Python 3.6 with LANG=C.
    env = dict(
        os.environ, LC_ALL="C", LANG="C", PYTHONCOERCECLOCALE="0", PYTHONUTF8="0"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(HOOK), str(tmp_path), sha],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.stdout.strip() == "ready", proc.stderr


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


@pytest.mark.parametrize(
    "command",
    [
        "cat .secret?/wandb.env",
        "echo $WANDB_API_KEY",
        "head -1 .s*/w*",
        'cat ."secrets"/wandb.env',
        "printenv WANDB_API''_KEY",
    ],
)
def test_cli_checks_every_spelling(tmp_path, command):
    payload = {"tool_input": {"command": command}, "cwd": str(tmp_path)}
    out = _run_hook(json.dumps(payload))["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"


def test_cli_checks_commands_run_inside_secrets(tmp_path):
    payload = {
        "tool_input": {"command": "cat wandb.env"},
        "cwd": str(tmp_path / ".secrets"),
    }
    out = _run_hook(json.dumps(payload))["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"


def test_cli_asks_when_the_hook_itself_fails():
    out = _run_hook("this is not json")["hookSpecificOutput"]
    assert out["permissionDecision"] == "ask"
    assert "guard hook error" in out["permissionDecisionReason"]
