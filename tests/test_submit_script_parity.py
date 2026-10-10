"""The SSL submit scripts re-implement two Python rules in bash (run-name prefix
patterns and the top-level seed lookup) so a typo fails at submit time, before
anything is queued. These tests keep the bash copies from drifting away from
`src/training/run_naming.py` and `load_config()`.
"""

import re
import subprocess
from pathlib import Path

import pytest

from src.training import run_naming
from src.utils.config import load_config

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

PRETRAIN_CONFIG = "configs/ssl/mae_pretrain.yaml"
FINETUNE_CONFIG = "configs/ssl/mae_finetune.yaml"

# `[[ ! "${VAR}" =~ <regex> ]]` for a run-name prefix variable.
PREFIX_CHECK = re.compile(r'"\$\{(?P<var>\w*PREFIX)\}" =~ (?P<regex>\S+) \]\]')

# script -> {shell variable: Python pattern it must equal}
EXPECTED_PREFIX_CHECKS = {
    "submit_ssl_pretrain_all.sh": {
        "RUN_NAME_PREFIX": run_naming.SSL_PRETRAIN_PREFIX_RE,
    },
    "submit_ssl_finetune_all.sh": {
        "RUN_NAME_PREFIX": run_naming.SSL_FINETUNE_PREFIX_RE,
        "PRETRAIN_RUN_PREFIX": run_naming.SSL_PRETRAIN_PREFIX_RE,
    },
}


def _bare(pattern: re.Pattern[str]) -> str:
    """Python pattern text without named groups, as the shell copy has none."""
    return re.sub(r"\(\?P<\w+>", "(", pattern.pattern)


@pytest.mark.parametrize("script", sorted(EXPECTED_PREFIX_CHECKS))
def test_bash_prefix_regex_matches_run_naming(script):
    text = (SCRIPTS / script).read_text()
    found = {m["var"]: m["regex"] for m in PREFIX_CHECK.finditer(text)}
    expected = EXPECTED_PREFIX_CHECKS[script]
    assert set(found) == set(expected), f"{script}: prefix checks {set(found)}"
    for var, pattern in expected.items():
        # The shell copy writes the groups out as plain text.
        shell = found[var]
        python = _bare(pattern)
        assert re.sub(r"[()]", "", shell) == re.sub(
            r"[()]", "", python
        ), f"{script}: {var} regex {shell!r} differs from run_naming {python!r}"


def _shell_seed(config: str) -> str:
    """The seed exactly as the submit scripts' `read_seed` grep finds it."""
    out = subprocess.run(
        [
            "bash",
            "-c",
            f"{{ grep -hE '^seed:' {config} configs/base.yaml || true; }}"
            " | head -1 | awk '{print $2}'",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.mark.parametrize("config", [PRETRAIN_CONFIG, FINETUNE_CONFIG])
def test_shell_seed_equals_load_config_seed(config):
    assert _shell_seed(config) == str(load_config(REPO / config)["seed"])


def test_pretrain_and_finetune_configs_share_a_seed():
    """Fine-tune finds its pretrain checkpoint by seed; a mismatch is a config bug."""
    assert (
        load_config(REPO / PRETRAIN_CONFIG)["seed"]
        == load_config(REPO / FINETUNE_CONFIG)["seed"]
    )


@pytest.mark.parametrize(
    "script", ["submit_ssl_finetune.sh", "submit_ssl_finetune_all.sh"]
)
def test_finetune_scripts_take_pretrain_seed_from_pretrain_config(script):
    text = (SCRIPTS / script).read_text()
    assert f'PRETRAIN_CONFIG="{PRETRAIN_CONFIG}"' in text
    assert 'PRETRAIN_SEED="$(read_seed "${PRETRAIN_CONFIG}")"' in text
    # The pretrain checkpoint path must use that seed, not the fine-tune one.
    for line in text.splitlines():
        if (
            "/last.pt" in line
            and "checkpoints/" in line
            and "PRETRAIN_RUN_PREFIX" in line
        ):
            assert "seed${PRETRAIN_SEED}" in line, line
