#!/usr/bin/env python3
"""Read-only digest of what happened since the last progress report.

Two modes (run from the repository root):

    collect_since.py --reports-dir reports
        Print the latest report folder, its date (the period start for the next report)
        and its state.md.

    collect_since.py --since 2026-11-01 [--full]
        Print progress-note sections dated on/after --since, merged PRs, commit subjects,
        and the kanban / planning files. --full prints the whole text of each section;
        without it only headings and sizes are shown.

Nothing is written. Missing optional inputs (gh, the local notes, the kanban) are reported
and skipped, never fatal.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date
from pathlib import Path

NOTES_PATH = Path("docs/progress_notes.md")
KANBAN_PATH = Path("phase-05-kanban.md")
PLANNING_PATH = Path("PLANNING.md")
REPORT_DIR_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
SECTION_HEADING_RE = re.compile(r"^## ", re.MULTILINE)
PLANNING_HEAD_LINES = 60
GH_PR_LIMIT = 100
COMMIT_LIMIT = 200
COMMAND_TIMEOUT_S = 120
KANBAN_MARKER_RE = re.compile(r"\s*<!--\s*gh:#\d+\s*-->")
PLANNING_STATUS_RE = re.compile(r"CURRENT|IN PROGRESS", re.IGNORECASE)


def _run(cmd: list[str]) -> str | None:
    """Stdout of a command, or None if it is missing or fails."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def latest_report(reports_dir: Path) -> tuple[Path, date] | None:
    """Newest dated report folder, or None if there is none yet."""
    found = []
    for d in reports_dir.glob("*"):
        if not (d.is_dir() and REPORT_DIR_RE.fullmatch(d.name)):
            continue
        try:
            found.append((date.fromisoformat(d.name), d))
        except ValueError:
            print(f"WARNING: ignoring {d}: not a real calendar date")
    if not found:
        return None
    when, folder = max(found)
    return folder, when


def split_sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) for each '## ' section of a markdown file."""
    starts = [m.start() for m in SECTION_HEADING_RE.finditer(text)]
    sections = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[s:end]
        heading, _, _ = block.partition("\n")
        sections.append((heading.lstrip("# ").strip(), block))
    return sections


def section_last_date(block: str) -> date | None:
    """Latest ISO date in a section's heading line; the body only if the heading has none.

    A body routinely mentions later dates ("rerun planned 2026-11-20", "superseded by the
    2026-11-04 run"), which would make an old section look recent.
    """
    heading = block.partition("\n")[0]
    found = ISO_DATE_RE.findall(heading) or ISO_DATE_RE.findall(block)
    valid = []
    for d in found:
        try:
            valid.append(date.fromisoformat(d))
        except ValueError:
            continue
    return max(valid) if valid else None


def print_reports_state(reports_dir: Path) -> None:
    if not reports_dir.is_dir():
        print(
            f"[first run] {reports_dir}/ does not exist: run the first-run interview."
        )
        return
    if not (reports_dir / "INSTRUCTIONS.md").exists():
        print(
            f"[first run] {reports_dir}/INSTRUCTIONS.md missing: run the first-run interview."
        )
    latest = latest_report(reports_dir)
    if latest is None:
        print(
            "No previous report folder: ask the user how far back report #1 should reach."
        )
        return
    folder, when = latest
    print(f"Latest report: {folder}  (period for the next report starts {when})")
    state = folder / "state.md"
    if state.exists():
        print(f"\n--- {state} ---\n{state.read_text()}")
    else:
        print(f"WARNING: {state} is missing: the next report cannot continue from it.")


def print_notes(since: date, full: bool) -> None:
    print(f"\n=== Progress-note sections dated {since} or later ===")
    print(
        f"(boundary day {since} is included on purpose so nothing is missed; anything "
        "dated that day may already be in the previous report's facts.md: dedupe against it)"
    )
    if not NOTES_PATH.exists():
        print(f"{NOTES_PATH} not found (local, untracked): skipped.")
        return
    shown = 0
    for heading, block in split_sections(NOTES_PATH.read_text()):
        last = section_last_date(block)
        if last is None or last < since:
            continue
        shown += 1
        print(f"\n## {heading}   (latest date {last}, {len(block)} chars)")
        if full:
            print(block)
    if not shown:
        print("none")


def print_prs(since: date) -> None:
    print(f"\n=== Pull requests merged since {since} (all base branches) ===")
    out = _run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "merged",
            "--limit",
            str(GH_PR_LIMIT),
            "--search",
            f"merged:>={since}",
            "--json",
            "number,title,mergedAt,baseRefName",
            "--template",
            "{{range .}}#{{.number}} {{.mergedAt}} [{{.baseRefName}}] {{.title}}\n{{end}}",
        ]
    )
    print(out.strip() if out and out.strip() else "none (or gh unavailable)")
    if out and len(out.strip().splitlines()) >= GH_PR_LIMIT:
        print(f"WARNING: hit the --limit {GH_PR_LIMIT}; older merged PRs are missing.")


def print_commits(since: date) -> None:
    print(f"\n=== Commit subjects since {since} (current branch) ===")
    out = _run(
        [
            "git",
            "log",
            f"--since={since}",
            f"--max-count={COMMIT_LIMIT}",
            "--pretty=%h %ad %s",
            "--date=short",
        ]
    )
    print(out.strip() if out and out.strip() else "none")
    if out and len(out.strip().splitlines()) >= COMMIT_LIMIT:
        print(f"WARNING: hit --max-count={COMMIT_LIMIT}; older commits are missing.")


def print_plan_state() -> None:
    print("\n=== Kanban and planning (local, untracked) ===")
    if KANBAN_PATH.exists():
        print(f"\n--- {KANBAN_PATH} ---")
        for line in KANBAN_PATH.read_text().splitlines():
            print(KANBAN_MARKER_RE.sub("", line))
    else:
        print(f"{KANBAN_PATH}: not found, skipped")
    if PLANNING_PATH.exists():
        lines = PLANNING_PATH.read_text().splitlines()
        print(f"\n--- {PLANNING_PATH} (first {PLANNING_HEAD_LINES} lines) ---")
        print("\n".join(lines[:PLANNING_HEAD_LINES]))
        later = [
            f"{n}: {line}"
            for n, line in enumerate(
                lines[PLANNING_HEAD_LINES:], PLANNING_HEAD_LINES + 1
            )
            if PLANNING_STATUS_RE.search(line)
        ]
        if later:
            print("\n--- later lines marking current / in-progress work ---")
            print("\n".join(later))
    else:
        print(f"{PLANNING_PATH}: not found, skipped")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--reports-dir", type=Path, default=None, help="print the latest report state"
    )
    p.add_argument(
        "--since", type=date.fromisoformat, default=None, help="YYYY-MM-DD period start"
    )
    p.add_argument(
        "--full", action="store_true", help="print full text of matching note sections"
    )
    args = p.parse_args()
    if args.reports_dir is None and args.since is None:
        p.error("give --reports-dir and/or --since")
    if args.reports_dir is not None:
        print_reports_state(args.reports_dir)
    if args.since is not None:
        print_notes(args.since, args.full)
        print_prs(args.since)
        print_commits(args.since)
        print_plan_state()


if __name__ == "__main__":
    main()
