"""The project hooks in .claude/settings.json: the graphify hint stays quiet."""

import json
import subprocess
from pathlib import Path

import pytest

SETTINGS = Path(__file__).resolve().parents[1] / ".claude" / "settings.json"


def _pre_tool_use_hooks():
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    for group in settings["hooks"]["PreToolUse"]:
        for hook in group["hooks"]:
            yield group.get("matcher"), hook.get("command", "")


def test_settings_file_is_valid_json_with_hooks():
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    assert "PreToolUse" in settings["hooks"]


def test_no_hook_injects_mandatory_text():
    # The old hints said "MANDATORY" and fired on every search and every read.
    assert [m for m, cmd in _pre_tool_use_hooks() if "MANDATORY" in cmd] == []


def test_no_graphify_hook_on_read_or_glob():
    assert [m for m, cmd in _pre_tool_use_hooks() if m == "Read|Glob"] == []
    assert [
        m for m, cmd in _pre_tool_use_hooks() if "graphify" in cmd and m != "Bash"
    ] == []


def test_one_graphify_hint_on_bash_is_kept():
    hints = [
        cmd for m, cmd in _pre_tool_use_hooks() if m == "Bash" and "graphify" in cmd
    ]
    assert len(hints) == 1
    assert "graphify query" in hints[0]
    assert "graphify-out/graph.json" in hints[0]


def _bash_hint_command():
    (cmd,) = [
        cmd for m, cmd in _pre_tool_use_hooks() if m == "Bash" and "graphify" in cmd
    ]
    return cmd


def _run_hint(command: str, cwd: Path, stdin: str | None = None):
    return subprocess.run(
        ["bash", "-c", _bash_hint_command()],
        input=(
            stdin
            if stdin is not None
            else json.dumps({"tool_input": {"command": command}})
        ),
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def _hint_fires(command: str, cwd: Path) -> bool:
    """Run the hook; it must always exit 0, stay quiet on stderr and print valid JSON."""
    result = _run_hint(command, cwd)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    if result.stdout.strip() == "":
        return False
    hint = json.loads(result.stdout)["hookSpecificOutput"]
    assert hint["hookEventName"] == "PreToolUse"
    assert "graphify query" in hint["additionalContext"]
    return True


@pytest.fixture
def with_graph(tmp_path):
    (tmp_path / "graphify-out").mkdir()
    (tmp_path / "graphify-out" / "graph.json").write_text("{}")
    return tmp_path


@pytest.mark.parametrize(
    "command",
    [
        "grep -rn foo src",
        "git grep foo",
        "ls src | grep foo",
        "cd src && find . -name '*.py'",
        "rg foo",
        "cat a.txt|rg foo",
        "fd name",
        "ag pattern src",
        "ack pattern",
        "egrep -r foo .",
        "echo start; grep foo f",
        "(grep x f)",
        "echo x | grep",
        "ls | rg",
        # the search tool in command position behind a wrapper, a path or a substitution
        "echo `grep x y`",
        'echo "$(find . -name a)"',
        "/usr/bin/grep foo x",
        "\\grep foo x",
        "FOO=1 rg x",
        "sudo find / -name a",
        "time grep x f",
        "xargs grep x",
        "ls\ngrep foo f",
    ],
)
def test_hint_fires_on_a_search_command(command, with_graph):
    assert _hint_fires(command, with_graph)


@pytest.mark.parametrize(
    "command",
    [
        "git tag foo",
        "git tag -l",
        "wget http://example.com/x",
        "cat flag.txt",
        "ls -la",
        "pgrep python",
        "cat /tmp/pdf-file",
        "echo agenda for monday",
        "pytest tests/ -q",
        "git log --oneline -3",
        # the tool's name as a plain word: an argument, a message, a file name
        "man find",
        "git checkout fd",
        "pip install ack",
        "git log --oneline -- ag",
        "sbatch --job-name find x.sh",
        "ls | wc -l # grep later",
        'git commit -m "fix the find step"',
        "echo find me",
        "git log --grep=x",
        "ls fd",
    ],
)
def test_hint_stays_silent_on_other_commands(command, with_graph):
    assert not _hint_fires(command, with_graph)


def test_hint_stays_silent_without_a_graph(tmp_path):
    assert not _hint_fires("grep -rn foo src", tmp_path)


@pytest.mark.parametrize("stdin", ["", "not json", "{}", '{"tool_input": {}}', "[]"])
def test_hint_survives_odd_hook_input(stdin, with_graph):
    result = _run_hint("", with_graph, stdin=stdin)
    assert result.returncode == 0
    assert result.stdout.strip() == ""
