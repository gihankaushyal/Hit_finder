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


def _write_run(run: Path, y, s, sessions=None, name="scores.npz", results=None):
    """A run dir as the evaluation leaves it: the score file, then its results file."""
    run.mkdir(parents=True, exist_ok=True)
    if sessions is None:
        sessions = np.array([f"s{i % 4}" for i in range(len(y))])
    np.savez(
        run / name, cross_y_true=y, cross_vote=s, cross_max=s, cross_session=sessions
    )
    results = results or (
        "results.json" if name == "scores.npz" else "results.inference.json"
    )
    (run / results).write_text("{}")


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
    assert stab.decide(widths, max_width=0.05, blocked=blocked) == "1e-2"


def test_too_few_negatives_for_a_target_fails_it_in_main(tmp_path):
    y, s = _scores(n_neg=2_000)  # 2 expected false positives at 1e-3, 20 at 1e-2
    _write_run(tmp_path / "ckpt" / "m-fold1-seed42", y, s)
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
    y, s = _scores(n_neg=2_000)
    _write_run(tmp_path / "ckpt" / "m-fold1-seed42", y, s)
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
    assert data["unit"] == "both"
    assert data["runs"]["m-fold1-seed42"]["n_sessions"] == 4
    assert {r["unit"] for r in data["rows"]} == {"frame", "session"}
    width = data["runs"]["m-fold1-seed42"]["width"]["1e-2"]
    assert width["used"] == max(("frame", "session"), key=lambda u: width[u])


def test_main_without_score_files_exits(tmp_path):
    (tmp_path / "ckpt").mkdir()
    with pytest.raises(SystemExit):
        stab.main(["--checkpoints-dir", str(tmp_path / "ckpt")])


def test_a_run_without_positives_is_reported_not_counted(tmp_path):
    y = np.zeros(1_000, dtype=np.int8)
    _write_run(tmp_path / "ckpt" / "m-fold1-seed42", y, np.zeros(1_000))
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
    assert data["undefined"] == ["m-fold1-seed42"]
    assert data["widest_at_full"] == {"1e-3": None, "1e-2": None}
    assert data["decision"] == "interval"  # nothing measured: no target is approved


def test_a_run_without_negatives_is_undefined_not_underpowered(tmp_path):
    y, s = _scores(n_neg=3_000)
    _write_run(tmp_path / "ckpt" / "a-fold1-seed42", y, s)
    _write_run(tmp_path / "ckpt" / "b-fold1-seed42", np.ones(50, np.int8), np.ones(50))
    data = stab.main(
        ["--checkpoints-dir", str(tmp_path / "ckpt"), "--n-boot", "20",
         "--sizes", "500", "--max-width", "1.0", "--out", str(tmp_path / "o.json")]
    )  # fmt: skip
    assert data["undefined"] == ["b-fold1-seed42"]
    assert data["underpowered"]["1e-2"] == []  # only a's 3000 negatives count
    assert data["decision"] == "1e-2"


def test_a_score_file_newer_than_its_results_is_incomplete(tmp_path):
    import os

    run = tmp_path / "ckpt" / "m-fold1-seed42"
    y, s = _scores(n_neg=2_000)
    _write_run(run, y, s)
    _write_run(run, y, s, name="scores.inference.npz")
    os.utime(run / "results.json", (1, 1))  # results older: the second write never ran
    data = stab.main(
        ["--checkpoints-dir", str(tmp_path / "ckpt"), "--n-boot", "5",
         "--sizes", "500", "--out", str(tmp_path / "o.json")]
    )  # fmt: skip
    assert data["incomplete"] == ["m-fold1-seed42/scores.npz"]
    assert {r["file"] for r in data["rows"]} == {"scores.inference.npz"}
    assert data["missing_scores"] == []


def test_session_resampling_is_wider_when_sessions_differ():
    # Two sessions of negatives that score differently: a frame bootstrap treats the
    # 4000 negatives as independent, a session bootstrap sees two units.
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(400), np.zeros(4_000)].astype(np.int8)
    s = np.r_[rng.normal(2.0, 1.0, 400), rng.normal(0.0, 1.0, 2_000),
              rng.normal(1.0, 1.0, 2_000)]  # fmt: skip
    sessions = np.r_[np.array(["p0", "p1"] * 200), ["n0"] * 2_000, ["n1"] * 2_000]
    kw = dict(targets={"1e-2": 1e-2}, sizes=[], n_boot=200, seed=1)
    frame = stab.stability_rows(y, s, **kw)[0]
    session = stab.stability_rows(y, s, sessions=sessions, **kw)[0]
    assert session["q95"] - session["q05"] > 2 * (frame["q95"] - frame["q05"])
    assert session["n_undefined"] > 0  # some resamples draw no negative session


def test_both_targets_come_from_the_same_resamples():
    y, s = _scores(n_neg=2_000)
    rows = stab.stability_rows(
        y, s, targets={"1e-3": 1e-3, "1e-2": 1e-2}, sizes=[], n_boot=50, seed=2
    )
    by_t = {r["target"]: r for r in rows}
    assert by_t["1e-3"]["median"] <= by_t["1e-2"]["median"]
    single = stab.stability_rows(
        y, s, targets={"1e-2": 1e-2}, sizes=[], n_boot=50, seed=2
    )
    assert single[0] == by_t["1e-2"]  # adding a target does not change the draws


def test_session_scale_widens_few_sessions():
    assert stab.session_scale(5) == pytest.approx((5 / 4) ** 0.5)
    assert stab.session_scale(100) == pytest.approx(1.0, abs=0.01)
    assert np.isnan(stab.session_scale(1))


def test_the_session_width_is_scaled_and_the_wider_unit_decides(tmp_path, monkeypatch):
    y, s = _scores(n_neg=3_000)
    _write_run(tmp_path / "ckpt" / "m-fold1-seed42", y, s)  # 4 sessions
    real = stab.stability_rows

    def fake(y, s, targets, sizes, n_boot, seed, sessions=None):
        rows = real(y, s, targets, sizes, n_boot, seed, sessions)
        half = 0.01 if sessions is None else 0.02  # session raw width 0.04
        for r in rows:
            r["q05"], r["q95"] = r["median"] - half, r["median"] + half
        return rows

    monkeypatch.setattr(stab, "stability_rows", fake)
    data = stab.main(
        ["--checkpoints-dir", str(tmp_path / "ckpt"), "--n-boot", "5",
         "--sizes", "500", "--out", str(tmp_path / "o.json")]
    )  # fmt: skip
    w = data["runs"]["m-fold1-seed42"]["width"]["1e-2"]
    assert w["frame"] == pytest.approx(0.02)
    assert w["session"] == pytest.approx(0.04 * (4 / 3) ** 0.5)  # 0.0462
    assert w["used"] == "session"
    assert data["widest_at_full"]["1e-2"] == pytest.approx(0.04 * (4 / 3) ** 0.5)


def test_too_many_one_class_resamples_fail_the_target(tmp_path):
    # Hits only in session h, misses only in sessions n0..n3: a session resample
    # often draws no hit session, which the frame interval would hide.
    y = np.r_[np.ones(300), np.zeros(4_000)].astype(np.int8)
    rng = np.random.default_rng(0)
    s = np.r_[rng.normal(2, 1, 300), rng.normal(0, 1, 4_000)]
    sessions = np.r_[["h"] * 300, [f"n{i % 4}" for i in range(4_000)]]
    _write_run(tmp_path / "ckpt" / "m-fold1-seed42", y, s, sessions=sessions)
    data = stab.main(
        ["--checkpoints-dir", str(tmp_path / "ckpt"), "--n-boot", "50",
         "--sizes", "500", "--max-width", "1.0", "--out", str(tmp_path / "o.json")]
    )  # fmt: skip
    assert data["undefined_draws"]["1e-2"]
    assert data["undefined_draws"]["1e-2"][0].startswith("m-fold1-seed42 (session: ")
    assert data["decision"] == "interval"


def test_smaller_sizes_draw_from_the_original_negatives(monkeypatch):
    # The pool for a size below the full count is the original frames, not the
    # resample (which would double the variance from resampling twice).
    y, s = _scores(n_pos=50, n_neg=1_000)
    seen = []
    real = stab._resample

    def spy(rng, y_, groups):
        idx, pool = real(rng, y_, groups)
        seen.append(pool)
        return idx, pool

    monkeypatch.setattr(stab, "_resample", spy)
    stab.stability_rows(y, s, {"1e-2": 1e-2}, sizes=[500], n_boot=3, seed=0)
    assert all(np.array_equal(pool, np.arange(len(y))) for pool in seen)


def test_session_unit_needs_session_ids(tmp_path):
    run = tmp_path / "ckpt" / "m-fold1-seed42"
    run.mkdir(parents=True)
    y, s = _scores(n_neg=1_000)
    np.savez(run / "scores.npz", cross_y_true=y, cross_vote=s)
    (run / "results.json").write_text("{}")
    with pytest.raises(SystemExit, match="--unit frame"):
        stab.main(["--checkpoints-dir", str(tmp_path / "ckpt"), "--n-boot", "5"])
    data = stab.main(
        ["--checkpoints-dir", str(tmp_path / "ckpt"), "--n-boot", "5", "--unit",
         "frame", "--out", str(tmp_path / "o.json")]
    )  # fmt: skip
    assert data["runs"]["m-fold1-seed42"]["n_sessions"] is None
