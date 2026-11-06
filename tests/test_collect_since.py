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


def test_section_without_a_date_is_ignored():
    assert cs.section_last_date("## 1. Undated\nnothing here") is None


def test_latest_report_is_the_newest_dated_folder(tmp_path):
    assert cs.latest_report(tmp_path) is None
    for name in ("2026-10-01", "2026-11-06", "figures", "2026-9-9"):
        (tmp_path / name).mkdir()
    folder, when = cs.latest_report(tmp_path)
    assert folder.name == "2026-11-06" and when == date(2026, 11, 6)


def test_body_dates_do_not_make_an_old_section_recent():
    block = "## 12. Old result (2026-09-01)\nrerun planned 2026-11-20\n"
    assert cs.section_last_date(block) == date(2026, 9, 1)


def test_latest_report_skips_impossible_dates(tmp_path, capsys):
    (tmp_path / "2026-11-31").mkdir()
    (tmp_path / "2026-10-05").mkdir()
    folder, when = cs.latest_report(tmp_path)
    assert (folder.name, when) == ("2026-10-05", date(2026, 10, 5))
    assert "not a real calendar date" in capsys.readouterr().out
