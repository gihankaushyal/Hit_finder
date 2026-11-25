"""backup-docs.sh copies the local-only docs into a git repo outside the checkout."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "backup-docs.sh"


def run(main: Path, home: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(
        ["bash", str(SCRIPT), str(main)], env=env, capture_output=True, text=True
    )


def backup_dir(main: Path, home: Path) -> Path:
    project_id = "".join(c if c.isalnum() else "-" for c in str(main))
    return home / ".claude" / "projects" / project_id / "doc-backup"


def git(dest: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(dest), *args], capture_output=True, text=True, check=True
    )
    return out.stdout


@pytest.fixture
def checkout(tmp_path):
    main = tmp_path / "main"
    (main / "docs" / "superpowers" / "specs").mkdir(parents=True)
    (main / "reports" / "workflows").mkdir(parents=True)
    (main / "PLANNING.md").write_text("plan v1\n")
    (main / "MEMORY.md").write_text("memory\n")
    (main / "phase-05-kanban.md").write_text("- [ ] item\n")
    (main / "docs" / "progress_notes.md").write_text("notes\n")
    (main / "docs" / "superpowers" / "specs" / "a.md").write_text("spec\n")
    (main / "reports" / "workflows" / "r.md").write_text("report\n")
    (main / "session-tail.md").write_text("regenerated every turn\n")
    (main / ".session-next").write_text("next step\n")
    return main


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    h.mkdir()
    return h


def test_first_run_copies_the_listed_files_and_commits(checkout, home):
    result = run(checkout, home)
    assert result.returncode == 0, result.stderr
    dest = backup_dir(checkout, home)
    assert (dest / "PLANNING.md").read_text() == "plan v1\n"
    assert (dest / "MEMORY.md").exists()
    assert (dest / "phase-05-kanban.md").exists()
    assert (dest / "docs" / "progress_notes.md").exists()
    assert (dest / "docs" / "superpowers" / "specs" / "a.md").exists()
    assert (dest / "reports" / "workflows" / "r.md").exists()
    assert len(git(dest, "log", "--format=%s").splitlines()) == 1


def test_session_state_files_are_not_backed_up(checkout, home):
    run(checkout, home)
    dest = backup_dir(checkout, home)
    assert not (dest / "session-tail.md").exists()
    assert not (dest / ".session-next").exists()


def test_second_run_without_changes_makes_no_commit(checkout, home):
    run(checkout, home)
    run(checkout, home)
    assert len(git(backup_dir(checkout, home), "log", "--format=%s").splitlines()) == 1


def test_a_changed_file_makes_a_commit_and_history_keeps_the_old_version(
    checkout, home
):
    run(checkout, home)
    (checkout / "PLANNING.md").write_text("plan v2\n")
    run(checkout, home)
    dest = backup_dir(checkout, home)
    assert len(git(dest, "log", "--format=%s").splitlines()) == 2
    assert (dest / "PLANNING.md").read_text() == "plan v2\n"
    assert git(dest, "show", "HEAD~1:PLANNING.md") == "plan v1\n"


@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")
def test_a_deleted_spec_leaves_the_tree_but_stays_in_history(checkout, home):
    run(checkout, home)
    (checkout / "docs" / "superpowers" / "specs" / "a.md").unlink()
    run(checkout, home)
    dest = backup_dir(checkout, home)
    assert not (dest / "docs" / "superpowers" / "specs" / "a.md").exists()
    assert git(dest, "show", "HEAD~1:docs/superpowers/specs/a.md") == "spec\n"


def test_a_missing_file_is_skipped_not_an_error(checkout, home):
    (checkout / "MEMORY.md").unlink()
    result = run(checkout, home)
    assert result.returncode == 0, result.stderr
    assert not (backup_dir(checkout, home) / "MEMORY.md").exists()


def test_an_unwritable_destination_never_fails_the_caller(checkout, tmp_path):
    bad_home = tmp_path / "home-is-a-file"
    bad_home.write_text("not a directory")
    result = run(checkout, bad_home)
    assert result.returncode == 0
    assert "backup-docs" in result.stderr


def test_a_missing_checkout_never_fails_the_caller(tmp_path, home):
    result = run(tmp_path / "does-not-exist", home)
    assert result.returncode == 0
    assert "backup-docs" in result.stderr


def test_the_commit_hook_calls_the_backup_and_cannot_be_blocked_by_it():
    hook = (SCRIPT.parent / "on-git-commit.sh").read_text()
    calls = [line for line in hook.splitlines() if "backup-docs.sh" in line]
    assert len(calls) == 1
    assert calls[0].startswith("timeout 30 ")
    assert "2>>" in calls[0] and "doc-backup.log" in calls[0]
    assert calls[0].rstrip().endswith("|| true")


def marker(main: Path, home: Path) -> Path:
    return backup_dir(main, home).parent / "doc-backup.FAILED"


def test_a_good_run_leaves_no_failure_marker(checkout, home):
    marker(checkout, home).parent.mkdir(parents=True)
    marker(checkout, home).write_text("left over from an earlier failure\n")
    run(checkout, home)
    assert not marker(checkout, home).exists()
    run(checkout, home)  # no changes: still a good run
    assert not marker(checkout, home).exists()


def test_a_stale_index_lock_is_reported_and_marked(checkout, home):
    run(checkout, home)
    (backup_dir(checkout, home) / ".git" / "index.lock").write_text("")
    (checkout / "PLANNING.md").write_text("plan v2\n")
    result = run(checkout, home)
    assert result.returncode == 0
    assert "git add failed" in result.stderr
    assert "index.lock" in marker(checkout, home).read_text()


def test_a_broken_backup_repo_is_reported_and_marked(checkout, home):
    run(checkout, home)
    (backup_dir(checkout, home) / ".git" / "HEAD").write_text("garbage\n")
    result = run(checkout, home)
    assert result.returncode == 0
    assert "not a git repository" in result.stderr
    assert "commit failed" not in result.stderr
    assert "not a git repository" in marker(checkout, home).read_text()


def test_a_killed_run_leaves_the_marker(checkout, home, tmp_path):
    slow = tmp_path / "slowbin"
    slow.mkdir()
    (slow / "rsync").write_text("#!/bin/sh\nsleep 5\n")
    (slow / "rsync").chmod(0o755)
    env = {**os.environ, "HOME": str(home), "PATH": f"{slow}:{os.environ['PATH']}"}
    result = subprocess.run(
        ["timeout", "1", "bash", str(SCRIPT), str(checkout)],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 124
    assert "not finished" in marker(checkout, home).read_text()


def test_a_symlinked_top_level_file_is_copied_as_a_link(checkout, home, tmp_path):
    secret = tmp_path / "secret.env"
    secret.write_text("TOKEN=do-not-copy\n")
    (checkout / "PLANNING.md").unlink()
    (checkout / "PLANNING.md").symlink_to(secret)
    run(checkout, home)
    copied = backup_dir(checkout, home) / "PLANNING.md"
    assert copied.is_symlink()
    assert "do-not-copy" not in git(
        backup_dir(checkout, home), "show", "HEAD:PLANNING.md"
    )


TAIL_SCRIPT = SCRIPT.parent / "write-session-tail.sh"


def write_tail(repo: Path, home: Path) -> str:
    hooks = repo / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    shutil.copy(TAIL_SCRIPT, hooks / TAIL_SCRIPT.name)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    env = {**os.environ, "HOME": str(home)}
    subprocess.run(
        ["bash", str(hooks / TAIL_SCRIPT.name)],
        env=env,
        input="",
        check=True,
        text=True,
    )
    return (repo / "session-tail.md").read_text()


def test_the_session_tail_shows_a_failed_backup(tmp_path, home):
    repo = tmp_path / "repo"
    marker(repo, home).parent.mkdir(parents=True)
    marker(repo, home).write_text("2026-11-25T07:00Z git add failed in x\n")
    tail = write_tail(repo, home)
    assert "backup: FAILED 2026-11-25T07:00Z git add failed in x" in tail


def test_the_session_tail_has_no_backup_line_when_all_is_well(tmp_path, home):
    tail = write_tail(tmp_path / "repo", home)
    assert "backup:" not in tail
