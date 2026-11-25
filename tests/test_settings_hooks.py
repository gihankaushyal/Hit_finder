"""The project hooks in .claude/settings.json: the graphify hint stays quiet."""

import json
from pathlib import Path

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
