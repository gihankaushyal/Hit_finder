"""scripts/recall_fpr_stability.py: bootstrap spread of recall at fixed FPR and the rule."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "recall_fpr_stability",
    Path(__file__).resolve().parents[1] / "scripts" / "recall_fpr_stability.py",
)
stab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stab)


def _scores(n_pos=500, n_neg=10_000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(np.int8)
    s = np.r_[rng.normal(2.0, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)]
    return y, s


def test_the_spread_shrinks_with_more_negatives():
    y, s = _scores()
    rows = stab.stability_rows(
        y, s, targets={"1e-2": 1e-2}, sizes=[500, 10_000], n_boot=200, seed=1
    )
    width = {r["n_neg"]: r["q95"] - r["q05"] for r in rows}
    assert width[500] > width[10_000]


def test_sizes_at_or_above_the_negative_count_collapse_to_all():
    y, s = _scores(n_neg=1_000)
    rows = stab.stability_rows(
        y, s, targets={"1e-2": 1e-2}, sizes=[500, 1_000, 5_000], n_boot=20, seed=1
    )
    assert [r["n_neg"] for r in rows] == [500, 1_000]


def test_the_same_seed_gives_the_same_rows():
    y, s = _scores(n_neg=1_000)
    kw = dict(targets={"1e-2": 1e-2}, sizes=[500], n_boot=20, seed=3)
    assert stab.stability_rows(y, s, **kw) == stab.stability_rows(y, s, **kw)


def test_the_rule_picks_the_strictest_target_that_is_narrow_enough():
    assert stab.decide({"1e-3": 0.08, "1e-2": 0.03}, max_width=0.05) == "1e-2"
    assert stab.decide({"1e-3": 0.04, "1e-2": 0.03}, max_width=0.05) == "1e-3"
    assert stab.decide({"1e-3": 0.2, "1e-2": 0.1}, max_width=0.05) == "interval"


def test_the_rule_fails_a_target_no_run_measured():
    nan = float("nan")
    assert stab.decide({"1e-3": nan, "1e-2": nan}, max_width=0.05) == "interval"
    assert stab.decide({"1e-3": nan, "1e-2": 0.01}, max_width=0.05) == "1e-2"


def test_the_rule_fails_a_target_with_an_underpowered_run():
    widths = {"1e-3": 0.0, "1e-2": 0.03}
    blocked = {"1e-3": ["r (400 negatives)"], "1e-2": []}
    assert stab.decide(widths, max_width=0.05, underpowered=blocked) == "1e-2"


def test_too_few_negatives_for_a_target_fails_it_in_main(tmp_path):
    run = tmp_path / "ckpt" / "m-fold1-seed42"
    run.mkdir(parents=True)
    y, s = _scores(n_neg=2_000)  # 2 expected false positives at 1e-3, 20 at 1e-2
    np.savez(run / "scores.npz", cross_y_true=y, cross_vote=s)
    data = stab.main(
        ["--checkpoints-dir", str(tmp_path / "ckpt"), "--n-boot", "20",
         "--sizes", "500", "--max-width", "1.0", "--out", str(tmp_path / "o.json")]
    )  # fmt: skip
    assert data["underpowered"] == {
        "1e-3": ["m-fold1-seed42 (2000 negatives)"],
        "1e-2": [],
    }
    assert data["decision"] == "1e-2"  # 1e-3 would pass on width alone


@pytest.mark.parametrize("flag", ["--sizes", "--n-boot"])
def test_sizes_and_n_boot_must_be_positive(tmp_path, flag):
    with pytest.raises(SystemExit):
        stab.main(["--checkpoints-dir", str(tmp_path), flag, "0"])


def test_main_reads_score_files_and_writes_json(tmp_path):
    run = tmp_path / "ckpt" / "m-fold1-seed42"
    run.mkdir(parents=True)
    y, s = _scores(n_neg=2_000)
    np.savez(
        run / "scores.npz",
        cross_y_true=y,
        cross_vote=s,
        cross_max=s,
        cross_session=np.array(["a"] * len(y)),
    )
    (tmp_path / "ckpt" / "no-scores-fold1-seed42").mkdir()  # listed, not an error
    out = tmp_path / "stab.json"
    stab.main(
        [
            "--checkpoints-dir", str(tmp_path / "ckpt"),
            "--n-boot", "50",
            "--sizes", "500",
            "--out", str(out),
        ]
    )  # fmt: skip
    data = json.loads(out.read_text())
    assert data["decision"] in {"1e-3", "1e-2", "interval"}
    assert {r["run"] for r in data["rows"]} == {"m-fold1-seed42"}
    assert {r["n_neg"] for r in data["rows"]} == {500, 2_000}  # "all" is always added
    assert {r["file"] for r in data["rows"]} == {"scores.npz"}
    assert data["missing_scores"] == ["no-scores-fold1-seed42"]


def test_main_without_score_files_exits(tmp_path):
    (tmp_path / "ckpt").mkdir()
    with pytest.raises(SystemExit):
        stab.main(["--checkpoints-dir", str(tmp_path / "ckpt")])


def test_a_run_without_positives_is_reported_not_counted(tmp_path):
    run = tmp_path / "ckpt" / "m-fold1-seed42"
    run.mkdir(parents=True)
    y = np.zeros(1_000, dtype=np.int8)
    np.savez(run / "scores.npz", cross_y_true=y, cross_vote=np.zeros(1_000))
    out = tmp_path / "stab.json"
    data = stab.main(
        [
            "--checkpoints-dir",
            str(tmp_path / "ckpt"),
            "--n-boot",
            "5",
            "--out",
            str(out),
        ]
    )
    assert data["undefined"] == ["m-fold1-seed42 (1e-3)", "m-fold1-seed42 (1e-2)"]
    assert data["widest_at_full"] == {"1e-3": None, "1e-2": None}
    assert data["decision"] == "interval"  # nothing measured: no target is approved
