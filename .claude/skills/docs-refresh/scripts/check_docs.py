#!/usr/bin/env python3
"""Check tracked docs for links to ignored files, broken links/paths, stale phrases.

Run from anywhere inside the repo: python .claude/skills/docs-refresh/scripts/check_docs.py
Exit code 1 if any problem is found.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

STALE_PHRASES = [
    "Track 2 (planned)",
    "not slow",
    "ViT-Base",
    "run_lodo",
    "2167",
    "2160",
    "preprocess_assembled",
    "32x32",
]
# Phrases that are legitimate where the docs explain the removal or history.
STALE_ALLOW_FILES = {"docs/codebase/index.md"}

LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
PATH_RE = re.compile(
    r"`((?:src|scripts|tests|configs|docs|dashboard|\.github)/[\w./\-]+)`"
)


def run(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout


def ignored_paths(root: Path, rels: list[str]) -> set[str]:
    """The paths among `rels` that git ignores, from a single `git check-ignore --stdin` call."""
    if not rels:
        return set()
    out = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"],
        cwd=root,
        input="\0".join(rels) + "\0",
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return {p for p in out.split("\0") if p}


def main() -> int:
    root = Path(run("git", "rev-parse", "--show-toplevel").strip())
    docs = [
        "README.md",
        "docs/detailed_notes.md",
        "docs/architecture.md",
        "docs/data_spec.md",
        "docs/eval_protocol.md",
        *sorted(
            str(p.relative_to(root)) for p in (root / "docs/codebase").glob("*.md")
        ),
    ]
    problems: list[str] = []
    existing_links: list[tuple[str, str, str]] = []  # (where, target, repo-relative path)
    warnings: list[str] = (
        []
    )  # advisory: phrases are fine when the doc explains the history

    for rel in docs:
        path = root / rel
        if not path.exists():
            problems.append(f"{rel}: doc missing")
            continue
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            for target in LINK_RE.findall(line):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                file_part = target.split("#", 1)[0]
                # normpath, not resolve(): a link through a symlinked folder such as data/ stays inside the repo
                resolved = Path(os.path.normpath(path.parent / file_part))
                try:
                    repo_rel = str(resolved.relative_to(root))
                except ValueError:
                    problems.append(f"{rel}:{lineno}: link leaves repo: {target}")
                    continue
                if not resolved.exists():
                    problems.append(f"{rel}:{lineno}: broken link: {target}")
                else:
                    existing_links.append((f"{rel}:{lineno}", target, repo_rel))
            for repo_path in PATH_RE.findall(line):
                if any(ch in repo_path for ch in "*<>"):
                    continue
                candidate = root / repo_path.rstrip("/.,")
                if "/my_" in repo_path or candidate.with_suffix(".py").exists():
                    continue  # placeholder example or a module named without its extension
                if not candidate.exists():
                    problems.append(f"{rel}:{lineno}: path not found: {repo_path}")
            if rel not in STALE_ALLOW_FILES:
                for phrase in STALE_PHRASES:
                    if phrase in line:
                        warnings.append(
                            f"{rel}:{lineno}: stale phrase '{phrase}' (check context)"
                        )

    # one git call for every link, not one per link
    ignored = ignored_paths(root, [repo_rel for _, _, repo_rel in existing_links])
    for where, target, repo_rel in existing_links:
        if repo_rel in ignored:
            problems.append(f"{where}: links to gitignored file: {target}")

    for w in warnings:
        print("WARN", w)
    for p in problems:
        print("FAIL", p)
    print(f"{len(problems)} problem(s), {len(warnings)} warning(s) in {len(docs)} docs")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
