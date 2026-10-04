"""Template: tests for a variant of a lab method. Copy it next to the other tests and fill in the four TODOs.

    cp labs/templates/test_variant_template.py tests/test_ease_variant.py
    pytest -q tests/test_ease_variant.py

A variant is a new setting of an existing method, for example `ease_variant: edlae`. Its default must keep today's
behaviour, so the baseline and every earlier result stay valid (tests/test_lab_defaults.py checks that as well).
The tests run on the small toy data of tests/conftest.py and take a few seconds.

SANSA only: on macOS, SANSA cannot run in a process that has loaded PyTorch, and pytest loads it for other tests. Run
SANSA's checks in a child process, as tests/test_new_methods.py::test_sansa_ranks_like_exact_ease does.
"""

from __future__ import annotations

import numpy as np
import pytest

from recbench.lab import checks

METHOD = "ease"  # TODO 1: the method you changed
VARIANT = {"ease_variant": "edlae", "ease_dropout": 0.5}  # TODO 2: the settings that switch your variant on
DEFAULT = {"ease_variant": "ease"}  # TODO 3: your new setting(s) at the default value ({} if there is no switch)


def test_the_variant_keeps_the_scores_contract(toy):
    view, _ = toy
    checks.check_scores(checks.fit(METHOD, view, **VARIANT), view)


def test_the_default_is_unchanged(toy):
    """Writing the default out must give exactly the model you get without it."""
    view, _ = toy
    checks.check_same_scores(METHOD, view, {}, DEFAULT)


def test_the_variant_changes_the_model(toy):
    """Identical scores usually mean your new setting is never read (a typo in cfg.get, or the wrong branch)."""
    view, _ = toy
    checks.check_differs(METHOD, view, {}, VARIANT)


def test_the_variant_is_deterministic(toy):
    """Delete this test for methods whose training is random (iALS, BPR-MF, the re-ranker)."""
    view, _ = toy
    checks.check_deterministic(METHOD, view, **VARIANT)


def test_the_variant_still_learns(toy):
    """On the toy data, a working variant ranks better than random and keeps most of the method's default quality; a
    broken change usually does not."""
    view, split = toy
    variant = checks.toy_ndcg(METHOD, view, split, **VARIANT)
    assert variant > checks.toy_ndcg("random", view, split) and variant > 0.8 * checks.toy_ndcg(METHOD, view, split)


def test_the_new_math_on_a_case_small_enough_to_check_by_hand():
    """TODO 4: check your change on a tiny example you can compute by hand or with a few lines of numpy.

    Example for EDLAE: build a 4 x 3 interaction matrix X, compute G = X.T @ X, the penalty
    lambda + p / (1 - p) * diag(G), and B from the closed form; compare with the weights your variant learns."""
    np.testing.assert_allclose(1.0, 1.0)  # replace this line with your check
    pytest.skip("write this check for your variant")
