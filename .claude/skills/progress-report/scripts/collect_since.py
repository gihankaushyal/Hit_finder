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
import os
import re
import subprocess
from datetime import date
from pathlib import Path

NOTES_PATH = Path("docs/progress_notes.md")
KANBAN_GLOB = "phase-*-kanban.md"
PLANNING_PATH = Path("PLANNING.md")
REPORT_DIR_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})(?:-(\d+))?"
)  # -N: extra report on one day
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_RANGE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})(?:\s*(?:…|\.\.\.|–|—|-|to)\s*(\d{4}-\d{2}-\d{2}))?"
)
ENCODING = "utf-8"
PLANNING_HEAD_LINES = 60
GH_PR_LIMIT = 100
COMMIT_LIMIT = 200
COMMAND_TIMEOUT_S = 120
KANBAN_MARKER_RE = re.compile(r"\s*<!--\s*gh:#\d+\s*-->")
PLANNING_STATUS_RE = re.compile(r"\b(CURRENT|IN PROGRESS)\b")
KANBAN_PHASE_RE = re.compile(r"phase-(\d+)-kanban\.md")
SECTION_NUMBER_RE = re.compile(r"^(\d+)\.")  # "25. Title (dates)"


def _run(cmd: list[str]) -> tuple[str, str | None]:
    """(stdout, error). error is None on success; a failed command is never read as 'no results'."""
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding=ENCODING,
            errors="replace",
            timeout=COMMAND_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", f"{cmd[0]} could not run ({exc.__class__.__name__})"
    if out.returncode != 0:
        detail = out.stderr.strip().splitlines()[-1:] or ["no message"]
        return "", f"{cmd[0]} failed (exit {out.returncode}): {detail[0]}"
    return out.stdout, None


def latest_report(reports_dir: Path) -> tuple[Path, date] | None:
    """Newest complete report folder (one that has a state.md), or None.

    Folders are YYYY-MM-DD, or YYYY-MM-DD-N for a further report on the same day. A folder
    without state.md is an unfinished run: it is reported and never taken as the baseline.
    """
    found = []
    for d in reports_dir.glob("*"):
        m = REPORT_DIR_RE.fullmatch(d.name)
        if not (d.is_dir() and m):
            continue
        try:
            when = date.fromisoformat(m.group(1))
        except ValueError:
            print(f"WARNING: ignoring {d}: not a real calendar date")
            continue
        if not (d / "state.md").exists():
            print(
                f"WARNING: {d} has no state.md (unfinished run): not used as baseline. "
                "Ask the user whether to resume it (reuse its facts.md) or delete it."
            )
            continue
        found.append(((when, int(m.group(2) or 0)), d))
    if not found:
        return None
    (when, _), folder = max(found)
    return folder, when


def split_sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) for each '## ' section; '## ' lines inside code fences are not headings.

    A fence closes only on the same character, at least as long as the one that opened it.
    """
    sections: list[tuple[str, list[str]]] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
        if line.startswith("## ") and fence is None:
            sections.append((line.lstrip("# ").strip(), []))
        if sections:
            sections[-1][1].append(line)
    return [(heading, "".join(lines)) for heading, lines in sections]


def _valid_dates(text: str) -> list[date]:
    out = []
    for d in ISO_DATE_RE.findall(text):
        try:
            out.append(date.fromisoformat(d))
        except ValueError:
            continue
    return out


def latest_section_number(headings: list[str]) -> int | None:
    """Highest "N." leading number among the section headings, or None if none is numbered."""
    numbers = [int(m.group(1)) for h in headings if (m := SECTION_NUMBER_RE.match(h))]
    return max(numbers) if numbers else None


def section_last_date(block: str) -> date | None:
    """Latest date of the heading's leading date or date range, or None if it has none.

    Only the heading is trusted: a body mentions later dates for other reasons ("rerun planned
    2026-11-20", "superseded by the 2026-11-04 run"). Sections whose heading has no date, or
    an older one, are listed separately by print_notes with their body dates for a human check.
    """
    m = HEADING_RANGE_RE.search(block.partition("\n")[0])
    if not m:
        return None
    dates = _valid_dates(" ".join(g for g in m.groups() if g))
    return max(dates) if dates else None


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
        print(f"\n--- {state} ---\n{state.read_text(encoding=ENCODING)}")
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
    to_check = []
    sections = split_sections(NOTES_PATH.read_text(encoding=ENCODING))
    latest = latest_section_number([h for h, _ in sections])
    for heading, block in sections:
        last = section_last_date(block)
        if last is not None and last >= since:
            shown += 1
            print(f"\n## {heading}   (heading date {last}, {len(block)} chars)")
            if full:
                print(block)
            continue
        later = sorted({d for d in _valid_dates(block) if d >= since})
        if later:
            to_check.append((heading, last, later))
    if not shown:
        print("none")
    if latest is not None:
        print(
            f"\nLatest numbered section: {latest}. The report's own section (SKILL.md step 6) is "
            f"number {latest + 1}."
        )
    if to_check:
        print(
            "\n--- Older or undated headings whose body mentions dates in the period: "
            "read these yourself (a section may have been appended to, or only mention a "
            "planned or superseded date) ---"
        )
        for heading, last, later in to_check:
            tag = f"heading date {last}" if last else "undated heading"
            print(
                f"## {heading}   ({tag}; body dates: {', '.join(map(str, later[-3:]))})"
            )


def print_prs(since: date) -> None:
    print(f"\n=== Pull requests merged since {since} (all base branches) ===")
    out, err = _run(
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
    if err:
        print(f"ERROR: PR list unavailable ({err}); do NOT read this as 'no PRs'.")
    else:
        print(out.strip() or "none")
    if len(out.strip().splitlines()) >= GH_PR_LIMIT:
        print(f"WARNING: hit the --limit {GH_PR_LIMIT}; older merged PRs are missing.")


def print_commits(since: date) -> None:
    print(f"\n=== Commit subjects since {since} (current branch) ===")
    out, err = _run(
        [
            "git",
            "log",
            f"--since={since}",
            f"--max-count={COMMIT_LIMIT}",
            "--pretty=%h %ad %s",
            "--date=short",
        ]
    )
    if err:
        print(f"ERROR: git log failed ({err}); do NOT read this as 'no commits'.")
    else:
        print(out.strip() or "none")
    if len(out.strip().splitlines()) >= COMMIT_LIMIT:
        print(f"WARNING: hit --max-count={COMMIT_LIMIT}; older commits are missing.")


def _newest_kanban() -> Path | None:
    """Kanban file of the highest phase number (numeric, so phase-10 beats phase-9)."""
    files = [
        (int(m.group(1)), f)
        for f in Path(".").glob(KANBAN_GLOB)
        if (m := KANBAN_PHASE_RE.fullmatch(f.name))
    ]
    return max(files)[1] if files else None


def print_plan_state() -> None:
    print("\n=== Kanban and planning (local, untracked) ===")
    kanban = _newest_kanban()
    if kanban:
        print(f"\n--- {kanban} ---")
        for line in kanban.read_text(encoding=ENCODING).splitlines():
            print(KANBAN_MARKER_RE.sub("", line))
    else:
        print(f"{KANBAN_GLOB}: none found, skipped")
    if PLANNING_PATH.exists():
        lines = PLANNING_PATH.read_text(encoding=ENCODING).splitlines()
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


def anchor_to_main_checkout() -> None:
    """chdir to the main checkout: the notes, kanban, planning and reports/ are gitignored, so
    they exist only there, and `git log` should describe the main branch, not a worktree.
    """
    out, err = _run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"])
    if err:
        raise SystemExit(f"not inside the Hit_finder repository ({err})")
    os.chdir(Path(out.strip()).parent)


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
    anchor_to_main_checkout()
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
