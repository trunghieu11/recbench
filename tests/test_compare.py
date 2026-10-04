"""The paired comparison (python -m recbench.compare), the error analysis and the lab scoreboard."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from recbench.compare import Ref, _positions, compare, load_side, paired_difference, promotion, render, segments, verdict


def test_a_real_gain_has_an_interval_above_zero():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 5000)
    gained = paired_difference(a, a + 0.01 + rng.normal(0, 0.05, 5000))
    assert gained["low"] > 0 and verdict(gained["low"], gained["high"]) == "better"
    noise = paired_difference(a, a + rng.normal(0, 0.05, 5000))
    assert noise["low"] < 0 < noise["high"] and verdict(noise["low"], noise["high"]) == "no clear difference"
    lost = paired_difference(a, a - 0.02)
    assert verdict(lost["low"], lost["high"]) == "worse" and lost["worse_users"] == 1.0


def test_users_without_a_value_are_left_out():
    result = paired_difference(np.array([0.1, np.nan, 0.3, 0.5]), np.array([0.2, 0.4, np.nan, 0.5]))
    assert result["users"] == 2 and result["diff"] == pytest.approx(0.05)


def test_users_are_paired_by_id_not_by_position():
    ids = np.array([7, 3, 9, 1])
    assert list(ids[_positions(ids, np.array([1, 3, 7, 9]))]) == [1, 3, 7, 9]


def test_the_promotion_rule():
    def rule(*verdicts):
        return promotion(pd.DataFrame({"verdict": list(verdicts)}))

    assert "new default" in rule(*["better"] * 5)
    assert "search-space option" in rule("better", "better", "no clear difference", "no clear difference", "no clear difference")
    assert "do not promote" in rule("better", "better", "better", "better", "worse")
    assert "no evidence" in rule(*["no clear difference"] * 5)
    assert "all 5 datasets" in rule("better", "better", "-", "-", "-")


def test_references_parse():
    assert Ref.parse("ease") == Ref("ease", "baseline") and str(Ref.parse("ease:edlae")) == "ease:edlae"


# ----- on real (toy) lab results ---------------------------------------------------------------------------------------

def _baselines(lab, methods=("most_popular", "itemknn", "bpr_mf")):
    from recbench.lab import runs
    from recbench.tuning.job import run_job

    _, resolved, spaces, settings = runs.lab_config()
    return {m: run_job("toy", m, resolved, settings, spaces[m], isolate=False) for m in methods}


def test_compare_pairs_the_same_test_users(lab):
    _baselines(lab, ("most_popular", "itemknn"))
    frame = compare("most_popular", "itemknn", ["toy"], tier="full")
    row = frame.iloc[0]
    assert row["users"] > 0 and row["verdict"] in ("better", "worse", "no clear difference")
    a, b = load_side(Ref("most_popular"), "toy", tier="full"), load_side(Ref("itemknn"), "toy", tier="full")
    common = np.intersect1d(a.users, b.users)
    diff = b.values["ndcg_at_10"][_positions(b.users, common)] - a.values["ndcg_at_10"][_positions(a.users, common)]
    assert row["diff"] == pytest.approx(np.nanmean(diff))
    assert a.top10 is not None and a.top10.shape == (len(a.users), 10)  # the lab saves each user's top-10 list
    assert "Promotion rule" in render(frame, Ref("most_popular"), Ref("itemknn"), "ndcg_at_10")


def test_random_methods_are_averaged_over_their_seeds(lab):
    summaries = _baselines(lab, ("bpr_mf",))
    side = load_side(Ref("bpr_mf"), "toy", tier="full")
    assert side.seeds == 2
    assert np.nanmean(side.values["ndcg_at_10"]) == pytest.approx(summaries["bpr_mf"]["test"]["ndcg_at_10"], rel=1e-6)


def test_segments_break_the_difference_down(lab):
    _baselines(lab, ("most_popular", "itemknn"))
    parts = segments("most_popular", "itemknn", "toy", tier="full")
    assert {"activity", "recency", "taste", "items", "gained", "lost"} <= set(parts)
    assert parts["activity"]["users"].sum() > 0
    assert set(parts["items"]["test items"]) == {"popular head (top 20% of items)", "long tail", "new (no history before the test)"}


def test_a_missing_side_says_how_to_produce_it(lab):
    frame = compare("itemknn", "itemknn:no-time-knobs", ["toy"], tier="full")
    assert frame.iloc[0]["verdict"] == "-" and "python -m recbench.lab baseline" in frame.iloc[0]["note"]


def test_the_scoreboard_chooses_experiments_by_validation_not_test(lab):
    from recbench.lab import runs, scoreboard
    from recbench.tuning.job import summary_path

    _baselines(lab, ("most_popular", "itemknn"))
    experiments = lab / "labs" / "01-itemknn" / "experiments.yaml"
    experiments.write_text(experiments.read_text() + "\n  few-neighbours:\n    fixed: {knn_neighbors: 2}\n")
    for label in ("no-time-knobs", "few-neighbours"):
        runs.run_experiment("itemknn", label, ["toy"], log=lambda _: None)
    # Make the experiment with the WORSE test score look better on validation: the scoreboard must pick it anyway.
    tests = {label: json.loads(summary_path("full", "toy", "itemknn", label=label).read_text())["test"]["ndcg_at_10"]
             for label in ("no-time-knobs", "few-neighbours")}
    worse = min(tests, key=tests.get)
    for label in tests:
        path = summary_path("full", "toy", "itemknn", label=label)
        found = json.loads(path.read_text())
        found["best_val"] = 0.99 if label == worse else 0.01
        path.write_text(json.dumps(found))
    fragments = scoreboard.build(docs_dir=lab / "docs")
    assert f"({worse})" in fragments["scoreboard"]
    assert "few-neighbours" in fragments["itemknn"] and "no-time-knobs" in fragments["itemknn"]
    assert (lab / "docs" / "generated" / "lab" / "scoreboard.md").exists() and (lab / "docs" / "generated" / "lab" / "itemknn.md").exists()
    assert (lab / "reports" / "lab" / "scoreboard.md").exists()
