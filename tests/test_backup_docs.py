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
    assert calls[0].rstrip().endswith("|| true")
