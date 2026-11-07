"""Read-only helpers of the progress-report skill's digest script."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".claude/skills/progress-report/scripts/collect_since.py"
)
spec = importlib.util.spec_from_file_location("collect_since", SCRIPT)
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

NOTES = """# Notes

## 22. Old (2026-10-10)
text with no later dates

## 23. Mid (2026-11-01 … 2026-11-03)
a result, merged 2026-11-03

## 24. New (2026-11-04 … 2026-11-06)
more
"""


def test_split_sections_and_last_date():
    secs = cs.split_sections(NOTES)
    assert [h.split(".")[0] for h, _ in secs] == ["22", "23", "24"]
    assert [cs.section_last_date(b) for _, b in secs] == [
        date(2026, 10, 10),
        date(2026, 11, 3),
        date(2026, 11, 6),
    ]


def test_section_without_a_heading_date_has_none():
    assert cs.section_last_date("## 1. Undated\nnothing 2026-11-05 here") is None


def test_latest_report_is_the_newest_dated_folder(tmp_path):
    assert cs.latest_report(tmp_path) is None
    for name in ("2026-10-01", "2026-11-06", "figures", "2026-9-9"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "state.md").write_text("x")
    folder, when = cs.latest_report(tmp_path)
    assert folder.name == "2026-11-06" and when == date(2026, 11, 6)


def test_body_dates_do_not_make_an_old_section_recent():
    block = "## 12. Old result (2026-09-01)\nrerun planned 2026-11-20\n"
    assert cs.section_last_date(block) == date(2026, 9, 1)


def test_older_or_undated_sections_with_period_dates_are_listed(
    tmp_path, monkeypatch, capsys
):
    notes = tmp_path / "notes.md"
    notes.write_text(
        "## 1. Undated\nseeds rerun 2026-11-05\n## 2. Old (2026-09-01)\nx\n"
    )
    monkeypatch.setattr(cs, "NOTES_PATH", notes)
    cs.print_notes(date(2026, 11, 1), full=False)
    out = capsys.readouterr().out
    assert "1. Undated" in out and "undated heading" in out
    assert "2. Old" not in out


def test_hash_lines_inside_code_fences_are_not_headings():
    text = "## 1. A (2026-11-01)\n```\n## submit\n```\nmore\n## 2. B (2026-11-02)\n"
    assert [h for h, _ in cs.split_sections(text)] == [
        "1. A (2026-11-01)",
        "2. B (2026-11-02)",
    ]


def test_latest_report_ignores_unfinished_folder_and_prefers_same_day_suffix(
    tmp_path, capsys
):
    for name, done in [
        ("2026-10-05", True),
        ("2026-11-06", False),
        ("2026-10-05-2", True),
    ]:
        (tmp_path / name).mkdir()
        if done:
            (tmp_path / name / "state.md").write_text("x")
    folder, when = cs.latest_report(tmp_path)
    assert folder.name == "2026-10-05-2" and when == date(2026, 10, 5)
    assert "unfinished run" in capsys.readouterr().out


def test_failed_command_is_an_error_not_an_empty_result():
    out, err = cs._run(["git", "no-such-subcommand-xyz"])
    assert out == "" and err and "failed" in err


def test_latest_report_skips_impossible_dates(tmp_path, capsys):
    (tmp_path / "2026-11-31").mkdir()
    (tmp_path / "2026-10-05").mkdir()
    (tmp_path / "2026-10-05" / "state.md").write_text("x")
    folder, when = cs.latest_report(tmp_path)
    assert (folder.name, when) == ("2026-10-05", date(2026, 10, 5))
    assert "not a real calendar date" in capsys.readouterr().out


def test_heading_takes_only_the_leading_date_or_range():
    f = cs.section_last_date
    assert f("## 12. Old (2026-09-01, rerun planned 2026-11-20)") == date(2026, 9, 1)
    assert f("## 23. X (2026-11-01 … 2026-11-03)") == date(2026, 11, 3)


def test_fence_must_close_with_the_same_character_and_length():
    text = "## 1. A (2026-11-01)\n````\n```\n## fake\n```\n````\n## 2. B (2026-11-02)\n"
    assert [h for h, _ in cs.split_sections(text)] == [
        "1. A (2026-11-01)",
        "2. B (2026-11-02)",
    ]


def test_planning_status_regex_ignores_prose():
    assert cs.PLANNING_STATUS_RE.search("| **5** | SSL | **CURRENT** |")
    assert not cs.PLANNING_STATUS_RE.search("the current threshold; currently idle")


def test_newest_kanban_is_numeric(tmp_path, monkeypatch):
    for n in ("phase-9-kanban.md", "phase-10-kanban.md", "notes.md"):
        (tmp_path / n).write_text("x")
    monkeypatch.chdir(tmp_path)
    assert cs._newest_kanban().name == "phase-10-kanban.md"


def test_failed_gh_prints_error_not_none(monkeypatch, capsys):
    monkeypatch.setattr(cs, "_run", lambda cmd: ("", "gh failed (exit 4): auth"))
    cs.print_prs(date(2026, 11, 1))
    out = capsys.readouterr().out
    assert "ERROR" in out and "none" not in out


def test_pr_limit_warning(monkeypatch, capsys):
    lines = "\n".join(f"#{i} t" for i in range(cs.GH_PR_LIMIT))
    monkeypatch.setattr(cs, "_run", lambda cmd: (lines, None))
    cs.print_prs(date(2026, 11, 1))
    assert "older merged PRs are missing" in capsys.readouterr().out


def test_anchor_goes_to_the_main_checkout_from_a_worktree(tmp_path, monkeypatch):
    import subprocess

    main = tmp_path / "main"
    main.mkdir()
    git = ["git", "-C", str(main)]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(
        git
        + [
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "i",
        ],
        check=True,
    )
    wt = tmp_path / "wt"
    subprocess.run(git + ["worktree", "add", "-q", str(wt), "-b", "b"], check=True)
    monkeypatch.chdir(wt)
    cs.anchor_to_main_checkout()
    assert Path.cwd().resolve() == main.resolve()
