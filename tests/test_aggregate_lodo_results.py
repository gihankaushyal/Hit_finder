import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_lodo_results",
    Path(__file__).resolve().parents[1] / "scripts" / "aggregate_lodo_results.py",
)
agg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(agg)


def _run(root: Path, name: str, fold: int, ap: float) -> None:
    d = root / name
    d.mkdir()
    metrics = {"ap": ap, "auc_roc": ap, "f1": ap}
    (d / "results.json").write_text(
        json.dumps({"fold_id": fold, "test_detector": "x", "cross": metrics,
                    "in_domain": metrics})
    )  # fmt: skip


def test_suffixed_runs_do_not_mix_into_the_baseline(tmp_path):
    _run(tmp_path, "vits16-mae-finetune-v2-fold4-seed42", 4, 0.89)
    _run(tmp_path, "vits16-mae-finetune-v2-fold4-seed42-careful", 4, 0.99)
    _run(tmp_path, "vits16-mae-finetune-fold4-seed42-v2", 4, 0.50)

    groups = agg.discover_prefixes(tmp_path)
    assert "vits16-mae-finetune-v2" in groups
    assert len(groups) == 3

    base = agg.load_results(tmp_path, "vits16-mae-finetune-v2")
    assert [r["cross"]["ap"] for r in base] == [0.89]
    careful_key = next(g for g in groups if g.endswith("-careful"))
    assert [r["cross"]["ap"] for r in agg.load_results(tmp_path, careful_key)] == [0.99]


def test_missing_checkpoints_dir_gives_no_groups(tmp_path):
    assert agg.discover_prefixes(tmp_path / "does-not-exist") == []


def test_other_seeds_are_separate_groups_not_pooled(tmp_path):
    _run(tmp_path, "resnet18-asymmetric-v2-fold1-seed42", 1, 0.80)
    _run(tmp_path, "resnet18-asymmetric-v2-fold1-seed43", 1, 0.90)
    _run(tmp_path, "resnet18-asymmetric-v2-fold1-seed43-jitter", 1, 0.95)

    groups = agg.discover_prefixes(tmp_path)
    assert len(groups) == 3
    assert [
        r["cross"]["ap"] for r in agg.load_results(tmp_path, "resnet18-asymmetric-v2")
    ] == [0.80]
    s43 = "resnet18-asymmetric-v2-fold*-seed43"
    assert [r["cross"]["ap"] for r in agg.load_results(tmp_path, s43)] == [0.90]
    assert [r["cross"]["ap"] for r in agg.load_results(tmp_path, s43 + "-jitter")] == [
        0.95
    ]


def test_legacy_seed_star_key_still_means_seed_42(tmp_path):
    _run(tmp_path, "m-fold1-seed42-careful", 1, 0.80)
    _run(tmp_path, "m-fold1-seed43-careful", 1, 0.90)
    legacy = agg.load_results(tmp_path, "m-fold*-seed*-careful")
    assert [r["cross"]["ap"] for r in legacy] == [0.80]


def test_pool_seeds_pools_every_seed_of_one_suffix(tmp_path):
    _run(tmp_path, "m-fold1-seed42", 1, 0.80)
    _run(tmp_path, "m-fold1-seed43", 1, 0.90)
    _run(tmp_path, "m-fold1-seed43-jitter", 1, 0.95)
    pooled = agg.load_results(tmp_path, "m-fold*-seed42", pool_seeds=True)
    assert [(r["n_seeds"], round(r["cross"]["ap"], 6)) for r in pooled] == [(2, 0.85)]


def test_pool_seeds_gives_one_averaged_row_per_fold(tmp_path):
    for fold in (1, 2):
        _run(tmp_path, f"m-fold{fold}-seed42", fold, 0.80)
        _run(tmp_path, f"m-fold{fold}-seed43", fold, 0.90)
        _run(tmp_path, f"m-fold{fold}-seed44", fold, 1.00)
    rows = agg.load_results(tmp_path, "m-fold*-seed*", pool_seeds=True)
    assert [r["fold_id"] for r in rows] == [1, 2]
    assert all(r["n_seeds"] == 3 for r in rows)
    assert all(abs(r["cross"]["ap"] - 0.90) < 1e-9 for r in rows)


def test_pool_seeds_menu_lists_each_group_once(tmp_path):
    _run(tmp_path, "m-fold1-seed42", 1, 0.80)
    _run(tmp_path, "m-fold1-seed43", 1, 0.90)
    _run(tmp_path, "m-fold1-seed43-jitter", 1, 0.95)
    assert agg.discover_prefixes(tmp_path, pool_seeds=True) == [
        "m-fold*-seed*",
        "m-fold*-seed*-jitter",
    ]


def test_metric_falls_back_to_the_inference_results(tmp_path, capsys):
    _run(tmp_path, "m-fold1-seed42", 1, 0.9)
    (tmp_path / "m-fold1-seed42" / "results.inference.json").write_text(json.dumps(
        {"cross": {"recall_at_fpr_1e-2": 0.61, "ap": 0.1},
         "in_domain": {"recall_at_fpr_1e-2": 0.7}}
    ))  # fmt: skip
    results = agg.load_results(tmp_path, "m", metric="recall_at_fpr_1e-2")
    assert results[0]["cross"]["recall_at_fpr_1e-2"] == 0.61
    assert results[0]["cross"]["ap"] == 0.9  # results.json wins for everything else
    agg.print_metric_table("m", results, "recall_at_fpr_1e-2")
    out = capsys.readouterr().out
    assert "0.6100*" in out and "0.7000*" in out  # marked as from the side file
    assert "* from results.inference.json" in out


def test_results_json_wins_when_it_has_the_metric(tmp_path):
    _run(tmp_path, "m-fold1-seed42", 1, 0.9)
    (tmp_path / "m-fold1-seed42" / "results.inference.json").write_text(
        json.dumps({"cross": {"ap": 0.1}, "in_domain": {"ap": 0.1}})
    )
    results = agg.load_results(tmp_path, "m", metric="ap")
    assert results[0]["cross"]["ap"] == 0.9
    assert agg.FROM_INFERENCE not in results[0]  # nothing filled, nothing marked


def test_a_fold_without_the_metric_shows_na(tmp_path, capsys):
    _run(tmp_path, "m-fold1-seed42", 1, 0.9)
    results = agg.load_results(tmp_path, "m", metric="recall_at_fpr_1e-2")
    mu, _, n = agg.print_metric_table("m", results, "recall_at_fpr_1e-2")
    assert "n/a" in capsys.readouterr().out and mu != mu and n == 0  # NaN mean


def test_the_metric_flag_replaces_the_default_table(tmp_path, capsys):
    _run(tmp_path, "m-fold1-seed42", 1, 0.9)
    agg.main(["--checkpoints-dir", str(tmp_path), "--run-prefix", "m",
              "--metric", "auc_roc"])  # fmt: skip
    out = capsys.readouterr().out
    assert "(auc_roc)" in out and "Cross AP" not in out


def test_an_unknown_metric_name_is_rejected(tmp_path):
    _run(tmp_path, "m-fold1-seed42", 1, 0.9)
    with pytest.raises(SystemExit):
        agg.main(["--checkpoints-dir", str(tmp_path), "--run-prefix", "m",
                  "--metric", "recall_at_fpr_1e3"])  # fmt: skip


def test_metric_pools_seeds_across_results_and_inference_files(tmp_path, capsys):
    _run(tmp_path, "m-fold1-seed42", 1, 0.9)
    _run(tmp_path, "m-fold1-seed43", 1, 0.9)
    key = "recall_at_fpr_1e-2"
    r42 = tmp_path / "m-fold1-seed42" / "results.json"
    data = json.loads(r42.read_text())
    data["cross"][key] = data["in_domain"][key] = 0.4
    r42.write_text(json.dumps(data))
    (tmp_path / "m-fold1-seed43" / "results.inference.json").write_text(
        json.dumps({"cross": {key: 0.6}, "in_domain": {key: 0.8}})
    )
    rows = agg.load_results(tmp_path, "m-fold*-seed*", pool_seeds=True, metric=key)
    assert rows[0]["n_seeds"] == 2
    assert rows[0]["cross"][key] == pytest.approx(0.5)
    assert rows[0]["in_domain"][key] == pytest.approx(0.6)
    agg.print_metric_table("m", rows, key)
    assert "0.5000*" in capsys.readouterr().out


def test_a_pooled_fold_keeps_the_metric_from_the_seeds_that_have_it(tmp_path, capsys):
    key = "recall_at_fpr_1e-3"
    for seed, value in ((42, None), (43, 0.4), (44, 0.6)):
        _run(tmp_path, f"m-fold1-seed{seed}", 1, 0.9)
        if value is not None:
            path = tmp_path / f"m-fold1-seed{seed}" / "results.json"
            data = json.loads(path.read_text())
            data["cross"][key] = data["in_domain"][key] = value
            path.write_text(json.dumps(data))
    rows = agg.load_results(tmp_path, "m-fold*-seed*", pool_seeds=True, metric=key)
    assert rows[0]["cross"][key] == pytest.approx(0.5)
    assert rows[0][agg.PARTIAL_SEEDS] == {"cross": {key: 2}, "in_domain": {key: 2}}
    assert rows[0]["cross"]["ap"] == pytest.approx(0.9)  # every seed: not marked
    agg.print_metric_table("m", rows, key)
    out = capsys.readouterr().out
    assert "0.5000+" in out and "fold 1 cross: 2/3 seeds" in out


def test_the_metric_summary_shows_how_many_folds_each_mean_covers(tmp_path, capsys):
    key = "recall_at_fpr_1e-2"
    for prefix, folds in (("a", (1, 2)), ("b", (1,))):
        for fold in folds:
            _run(tmp_path, f"{prefix}-fold{fold}-seed42", fold, 0.9)
            path = tmp_path / f"{prefix}-fold{fold}-seed42" / "results.json"
            data = json.loads(path.read_text())
            data["cross"][key] = data["in_domain"][key] = 0.5
            path.write_text(json.dumps(data))
    agg.main(["--checkpoints-dir", str(tmp_path), "--select", "1,2", "--metric", key])
    summary = capsys.readouterr().out.split("Cross-variant summary")[1]
    assert "Folds" in summary and "2/4" in summary and "1/4" in summary
