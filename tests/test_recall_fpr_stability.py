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
        sessions = np.array([f"s{i % 5}" for i in range(len(y))])  # MIN_SESSIONS
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
    assert data["runs"]["m-fold1-seed42"]["n_sessions"] == 5
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
    # sqrt(S/(S-1)) * t(S-1, 0.95) / z(0.95)
    assert stab.session_scale(5) == pytest.approx(1.449, abs=1e-3)
    assert stab.session_scale(10) == pytest.approx(1.175, abs=1e-3)
    assert stab.session_scale(1000) == pytest.approx(1.0, abs=0.01)
    assert np.isnan(stab.session_scale(1))


def test_the_session_width_is_scaled_and_the_wider_unit_decides(tmp_path, monkeypatch):
    y, s = _scores(n_neg=3_000)
    _write_run(tmp_path / "ckpt" / "m-fold1-seed42", y, s)  # 5 sessions
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
    assert w["session"] == pytest.approx(0.04 * stab.session_scale(5))  # 0.058
    assert w["used"] == "session"
    assert data["widest_at_full"]["1e-2"] == pytest.approx(0.04 * stab.session_scale(5))


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
    # hits in 1 of 5 sessions, misses in 4: P = (4/5)^5 + (1/5)^5 = 0.3280
    assert data["undefined_draws"]["1e-2"] == ["m-fold1-seed42 (P = 0.3280)"]
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


def _main(tmp_path, *extra):
    return stab.main(
        ["--checkpoints-dir", str(tmp_path / "ckpt"), "--n-boot", "20",
         "--sizes", "500", "--out", str(tmp_path / "o.json"), *extra]
    )  # fmt: skip


@pytest.mark.parametrize("n_sessions", [1, 2])
def test_fewer_than_min_sessions_fails_every_target(tmp_path, n_sessions):
    y, s = _scores(n_neg=3_000)
    sessions = np.array([f"s{i % n_sessions}" for i in range(len(y))])
    _write_run(tmp_path / "ckpt" / "m-fold2-seed42", y, s, sessions=sessions)
    data = _main(tmp_path, "--max-width", "1.0")
    expected = [f"m-fold2-seed42 ({n_sessions} sessions, {n_sessions} with positives)"]
    assert data["few_sessions"] == {"1e-3": expected, "1e-2": expected}
    assert data["decision"] == "interval"  # the frame width alone does not decide


def test_one_class_probability_with_pure_sessions():
    # hits only in h, misses only in n0..n3: P = (4/5)^5 + (1/5)^5
    y = np.r_[np.ones(30), np.zeros(400)]
    sessions = np.r_[["h"] * 30, [f"n{i % 4}" for i in range(400)]]
    assert stab.one_class_probability(y, sessions) == pytest.approx(0.8**5 + 0.2**5)


def test_one_class_probability_counts_mixed_sessions_that_draw_no_positive():
    # 4 sessions of 500 frames with 1 hit each, 1 session of misses only. A drawn mixed
    # session misses its hit with chance (499/500)^500 ~ e^-1, so P(no positive) is
    # ((1 + 4 (499/500)^500) / 5)^5 ~ 0.029, not (1/5)^5.
    y = np.zeros(2_500)
    y[[0, 500, 1_000, 1_500]] = 1
    sessions = np.repeat([f"s{i}" for i in range(5)], 500)
    expected = ((1 + 4 * (499 / 500) ** 500) / 5) ** 5
    assert stab.one_class_probability(y, sessions) == pytest.approx(expected)
    assert stab.one_class_probability(y, sessions) > stab.MAX_UNDEFINED_SHARE


def test_one_class_probability_matches_the_resamples():
    y = np.zeros(400)
    y[[0, 1, 100, 200]] = 1
    sessions = np.repeat([f"s{i}" for i in range(5)], 80)
    rng = np.random.default_rng(0)
    groups = [np.nonzero(sessions == g)[0] for g in np.unique(sessions)]
    n = 20_000
    one_class = 0
    for _ in range(n):
        idx, _pool = stab._resample(rng, y, groups)
        one_class += len(np.unique(y[idx])) < 2
    p = stab.one_class_probability(y, sessions)
    assert one_class / n == pytest.approx(p, abs=4 * np.sqrt(p * (1 - p) / n))


def test_the_resample_count_backs_up_the_exact_probability(tmp_path, monkeypatch):
    # Hits only in session h: with the exact probability forced to 0, the full-size
    # resamples without a hit still fail the target.
    y = np.r_[np.ones(300), np.zeros(4_000)].astype(np.int8)
    rng = np.random.default_rng(0)
    s = np.r_[rng.normal(2, 1, 300), rng.normal(0, 1, 4_000)]
    sessions = np.r_[["h"] * 300, [f"n{i % 4}" for i in range(4_000)]]
    _write_run(tmp_path / "ckpt" / "m-fold1-seed42", y, s, sessions=sessions)
    monkeypatch.setattr(stab, "one_class_probability", lambda y_, s_: 0.0)
    monkeypatch.setattr(stab, "MIN_SESSIONS", 1)
    data = _main(tmp_path, "--max-width", "1.0", "--n-boot", "50")
    entry = data["undefined_draws"]["1e-2"]
    assert len(entry) == 1 and entry[0].startswith("m-fold1-seed42 (")
    assert entry[0].endswith("of 50 resamples)")
    assert data["decision"] == "interval"


def test_sessions_without_positives_do_not_count(tmp_path):
    # 10 sessions, hits in 4: the session minimum fails although S = 10.
    y, s = _scores(n_pos=400, n_neg=3_000)
    sessions = np.r_[
        [f"s{i % 4}" for i in range(400)], [f"s{i % 10}" for i in range(3_000)]
    ]
    _write_run(tmp_path / "ckpt" / "m-fold2-seed42", y, s, sessions=sessions)
    data = _main(tmp_path, "--max-width", "1.0")
    expected = ["m-fold2-seed42 (10 sessions, 4 with positives)"]
    assert data["few_sessions"] == {"1e-3": expected, "1e-2": expected}
    assert data["runs"]["m-fold2-seed42"]["n_pos_sessions"] == 4


def test_the_session_scale_uses_the_sessions_with_positives(tmp_path, monkeypatch):
    # 10 sessions, hits in 6: the width is scaled by session_scale(6), not (10).
    y, s = _scores(n_pos=600, n_neg=3_000)
    sessions = np.r_[
        [f"s{i % 6}" for i in range(600)], [f"s{i % 10}" for i in range(3_000)]
    ]
    _write_run(tmp_path / "ckpt" / "m-fold2-seed42", y, s, sessions=sessions)
    real = stab.stability_rows

    def fake(y_, s_, targets, sizes, n_boot, seed, sessions=None):
        rows = real(y_, s_, targets, sizes, n_boot, seed, sessions)
        for r in rows:
            r["q05"], r["q95"] = r["median"] - 0.02, r["median"] + 0.02
        return rows

    monkeypatch.setattr(stab, "stability_rows", fake)
    data = _main(tmp_path, "--max-width", "1.0")
    w = data["runs"]["m-fold2-seed42"]["width"]["1e-2"]
    assert w["session"] == pytest.approx(0.04 * stab.session_scale(6))
    assert data["few_sessions"]["1e-2"] == []


@pytest.mark.parametrize("wide_first", [True, False])
def test_the_widest_run_decides(tmp_path, monkeypatch, wide_first):
    y, s = _scores(n_neg=3_000)
    names = ["a-fold1-seed42", "b-fold1-seed42"]
    halves = {names[0]: 0.04, names[1]: 0.005}
    if not wide_first:
        halves = {names[0]: 0.005, names[1]: 0.04}
    for name in names:
        _write_run(tmp_path / "ckpt" / name, y, s + halves[name])  # tag the run
    real = stab.stability_rows

    def fake(y_, s_, targets, sizes, n_boot, seed, sessions=None):
        half = 0.04 if s_[0] - s[0] > 0.02 else 0.005
        rows = real(y_, s_, targets, sizes, n_boot, seed, sessions)
        for r in rows:
            r["q05"], r["q95"] = r["median"] - half, r["median"] + half
        return rows

    monkeypatch.setattr(stab, "stability_rows", fake)
    data = _main(tmp_path, "--max-width", "0.05")
    assert data["widest_at_full"]["1e-2"] == pytest.approx(0.08 * stab.session_scale(5))
    assert data["decision"] == "interval"


def test_split_and_score_pick_the_matching_arrays(tmp_path):
    y = np.r_[np.ones(200), np.zeros(2_000)].astype(np.int8)
    separating = np.r_[np.ones(200), np.zeros(2_000)]  # recall 1 at FPR 0
    tied = np.zeros(len(y))  # all tied: recall 0
    sessions = np.array([f"s{i % 5}" for i in range(len(y))])
    run = tmp_path / "ckpt" / "m-fold1-seed42"
    run.mkdir(parents=True)
    np.savez(
        run / "scores.npz",
        cross_y_true=y, cross_vote=separating, cross_max=tied, cross_session=sessions,
        in_domain_y_true=y, in_domain_vote=tied, in_domain_max=separating,
        in_domain_session=sessions, aggregation=np.array("vote"),
    )  # fmt: skip
    (run / "results.json").write_text("{}")

    def full_median(data):
        return {r["median"] for r in data["rows"] if r["n_neg"] == 2_000}

    assert full_median(_main(tmp_path)) == {1.0}  # stored aggregation: vote
    assert full_median(_main(tmp_path, "--score", "max")) == {0.0}
    assert full_median(_main(tmp_path, "--split", "in_domain")) == {0.0}
    data = _main(tmp_path, "--split", "in_domain", "--score", "max")
    assert full_median(data) == {1.0}
    assert data["score_mismatch"] == ["m-fold1-seed42 (evaluated with vote)"]


def test_the_stored_aggregation_is_the_default_score(tmp_path):
    y, s = _scores(n_neg=2_000)
    run = tmp_path / "ckpt" / "m-fold1-seed42"
    _write_run(run, y, s)
    with np.load(run / "scores.npz") as z:
        arrays = dict(z)
    np.savez(run / "scores.npz", **arrays, aggregation=np.array("max"))
    (run / "results.json").write_text("{}")  # not older than the score file
    data = _main(tmp_path)
    assert data["runs"]["m-fold1-seed42"]["score"] == "max"
    assert data["score_mismatch"] == []


def test_equal_mtimes_count_as_complete(tmp_path):
    import os

    run = tmp_path / "ckpt" / "m-fold1-seed42"
    y, s = _scores(n_neg=1_000)
    _write_run(run, y, s)
    for name in ("scores.npz", "results.json"):
        os.utime(run / name, (100, 100))
    assert stab._score_file(run) == (run / "scores.npz", [])


def test_a_run_with_no_usable_pair_is_incomplete_not_missing(tmp_path):
    import os

    run = tmp_path / "ckpt" / "m-fold1-seed42"
    y, s = _scores(n_neg=1_000)
    _write_run(run, y, s)
    _write_run(run, y, s, name="scores.inference.npz")
    for name in ("results.json", "results.inference.json"):
        os.utime(run / name, (1, 1))
    _write_run(tmp_path / "ckpt" / "ok-fold1-seed42", y, s)
    data = _main(tmp_path)
    assert data["incomplete"] == [
        "m-fold1-seed42/scores.npz",
        "m-fold1-seed42/scores.inference.npz",
    ]
    assert data["missing_scores"] == []


def test_the_json_is_strict_and_its_folder_is_created(tmp_path):
    _write_run(
        tmp_path / "ckpt" / "m-fold1-seed42", np.zeros(500, np.int8), np.zeros(500)
    )
    out = tmp_path / "new" / "dir" / "o.json"
    stab.main(["--checkpoints-dir", str(tmp_path / "ckpt"), "--n-boot", "5",
               "--out", str(out)])  # fmt: skip
    data = json.loads(out.read_text(), parse_constant=lambda c: pytest.fail(c))
    assert data["rows"][0]["median"] is None


def test_decision_covers_only_runs_with_a_measured_width(tmp_path, monkeypatch):
    y, s = _scores(n_neg=3_000)
    _write_run(tmp_path / "ckpt" / "a-fold1-seed42", y, s)
    _write_run(tmp_path / "ckpt" / "b-fold1-seed42", y, s + 1.0)  # tag the run
    real = stab.stability_rows

    def fake(y_, s_, targets, sizes, n_boot, seed, sessions=None):
        rows = real(y_, s_, targets, sizes, n_boot, seed, sessions)
        if s_[0] - s[0] > 0.5:  # run b: no width measured
            for r in rows:
                r["q05"] = r["q95"] = float("nan")
        return rows

    monkeypatch.setattr(stab, "stability_rows", fake)
    data = _main(tmp_path)
    assert data["decision_covers"] == {"runs": 1, "of": 2}


def test_an_old_score_file_without_aggregation_defaults_to_vote(tmp_path):
    y = np.r_[np.ones(200), np.zeros(2_000)].astype(np.int8)
    run = tmp_path / "ckpt" / "m-fold1-seed42"
    run.mkdir(parents=True)
    np.savez(
        run / "scores.npz",
        cross_y_true=y, cross_vote=y.astype(float), cross_max=np.zeros(len(y)),
        cross_session=np.array([f"s{i % 5}" for i in range(len(y))]),
    )  # fmt: skip
    (run / "results.json").write_text("{}")
    data = _main(tmp_path)
    info = data["runs"]["m-fold1-seed42"]
    assert info["score"] == "vote" and info["aggregation"] is None
    assert {r["median"] for r in data["rows"] if r["n_neg"] == 2_000} == {1.0}
    assert data["score_mismatch"] == []


def test_the_session_minimum_does_not_apply_to_frame_resampling(tmp_path):
    y, s = _scores(n_neg=3_000)
    sessions = np.array([f"s{i % 2}" for i in range(len(y))])
    _write_run(tmp_path / "ckpt" / "m-fold1-seed42", y, s, sessions=sessions)
    data = _main(tmp_path, "--unit", "frame", "--max-width", "1.0")
    assert data["few_sessions"] == {"1e-3": [], "1e-2": []}
    assert (
        data["decision"] == "1e-2"
    )  # 1e-3 fails only as underpowered (3,000 negatives)
