#!/usr/bin/env python3
"""Check tracked docs for links to ignored files, broken links/paths, stale phrases.

Run from anywhere inside the repo: python .claude/skills/docs-refresh/scripts/check_docs.py
Exit code 1 if any problem is found.
"""

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
]
# Phrases that are legitimate where the docs explain the removal or history.
STALE_ALLOW_FILES = {"docs/codebase/index.md"}

LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
PATH_RE = re.compile(
    r"`((?:src|scripts|tests|configs|docs|dashboard|\.github)/[\w./\-]+)`"
)


def run(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout


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
                resolved = (path.parent / file_part).resolve()
                try:
                    repo_rel = str(resolved.relative_to(root))
                except ValueError:
                    problems.append(f"{rel}:{lineno}: link leaves repo: {target}")
                    continue
                if not resolved.exists():
                    problems.append(f"{rel}:{lineno}: broken link: {target}")
                elif (
                    subprocess.run(
                        ["git", "check-ignore", "-q", repo_rel], cwd=root
                    ).returncode
                    == 0
                ):
                    problems.append(
                        f"{rel}:{lineno}: links to gitignored file: {target}"
                    )
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

    for w in warnings:
        print("WARN", w)
    for p in problems:
        print("FAIL", p)
    print(f"{len(problems)} problem(s), {len(warnings)} warning(s) in {len(docs)} docs")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
