import importlib.util
import json
from pathlib import Path

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


def test_seed_star_pools_every_seed_of_one_suffix(tmp_path):
    _run(tmp_path, "m-fold1-seed42", 1, 0.80)
    _run(tmp_path, "m-fold1-seed43", 1, 0.90)
    _run(tmp_path, "m-fold1-seed43-jitter", 1, 0.95)
    pooled = agg.load_results(tmp_path, "m-fold*-seed*")
    assert sorted(r["cross"]["ap"] for r in pooled) == [0.80, 0.90]
