"""The legacy v1 submit scripts refuse to run and point at their replacements.

Each runs in a temp directory with stub `sbatch` and `python` first on PATH; the gate
sits before any scheduler or training call, so both logs must stay empty.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

# script -> replacement script it must name
DEPRECATED = {
    "submit_all_lodo_folds.sh": "submit_asymmetric_lodo_all.sh",
    "submit_asymmetric_lodo.sh": "submit_asymmetric_lodo_all.sh",
    "submit_lodo_parallel.sh": "submit_asymmetric_lodo_all.sh",
    "submit_agipd_lodo.sh": "submit_asymmetric_lodo_all.sh",
    "submit_epix_smoketest.sh": "submit_epix_cache_smoketest.sh",
    "submit_agipd_smoketest.sh": "submit_asymmetric_lodo_all.sh",
    "submit_resonet_smoketest.sh": "submit_asymmetric_lodo_all.sh",
}

STUB = '#!/bin/bash\necho "$0 $*" >> "$STUB_DIR/calls.log"\n'


@pytest.mark.parametrize("script", sorted(DEPRECATED))
def test_script_refuses_to_run(script, tmp_path):
    bin_dir, stub_dir = tmp_path / "bin", tmp_path / "stub"
    bin_dir.mkdir()
    stub_dir.mkdir()
    for name in ("sbatch", "python", "module"):
        path = bin_dir / name
        path.write_text(STUB)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_DIR": str(stub_dir),
    }
    proc = subprocess.run(
        ["bash", str(SCRIPTS / script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1
    assert "DEPRECATED" in proc.stderr
    assert DEPRECATED[script] in proc.stderr
    assert not (stub_dir / "calls.log").exists()  # nothing was submitted or trained


@pytest.mark.parametrize("script", sorted(DEPRECATED))
def test_replacement_script_exists(script):
    assert (SCRIPTS / DEPRECATED[script]).is_file()
    assert (SCRIPTS / script).is_file()


def test_no_current_script_is_deprecated():
    """The gate only goes on the legacy scripts, never on the v2 entry points."""
    for name in (
        "submit_asymmetric_lodo_all.sh",
        "submit_asymmetric_lodo_fold.sh",
        "submit_epix_cache_smoketest.sh",
    ):
        assert "DEPRECATED" not in (SCRIPTS / name).read_text(), name
