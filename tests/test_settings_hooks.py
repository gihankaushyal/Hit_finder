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


def _hint_fires(command: str, cwd: Path) -> bool:
    out = subprocess.run(
        ["bash", "-c", _bash_hint_command()],
        input=json.dumps({"tool_input": {"command": command}}),
        capture_output=True,
        text=True,
        cwd=cwd,
    ).stdout
    return out.strip() != ""


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
    ],
)
def test_hint_stays_silent_on_other_commands(command, with_graph):
    assert not _hint_fires(command, with_graph)


def test_hint_stays_silent_without_a_graph(tmp_path):
    assert not _hint_fires("grep -rn foo src", tmp_path)
