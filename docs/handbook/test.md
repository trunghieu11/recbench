# Test

!!! abstract "In plain words"
    Tests are small programs that check the code still does what it should. You need three layers. **Your variant's
    tests** check the code you wrote. **The lab's checks** confirm that every method's default behaviour is unchanged.
    **The whole suite** guards everything else, including the docs. They run on tiny made-up data, so they take
    seconds to minutes and need no downloads.

| Layer | Command | When | Time |
|---|---|---|---|
| your variant's tests | `pytest -q tests/test_ease_variant.py` | while you write the code | seconds |
| the lab's checks | `./scripts/check.sh --quick` | before each commit | about two minutes |
| everything | `./scripts/check.sh` | before a pull request | several minutes |

## Your variant's tests: start from the template

```bash
cp labs/templates/test_variant_template.py tests/test_itemknn_variant.py
```

Open the copy and fill in the four `TODO`s: the method, the settings that switch your variant on, the new setting at
its default value, and one check of the new math on a tiny case. Then:

```bash
pytest -q tests/test_itemknn_variant.py
```

!!! success "You should see"
    ```text
    ......                                                                   [100%]
    ```
    One dot per passing test (an `s` for a skipped one: the template skips TODO 4 until you write it).

What each test catches:

| Test | Catches |
|---|---|
| `test_the_variant_keeps_the_scores_contract` | scores of the wrong shape or type, NaN (often `0/0` or `log(0)`), users with nothing to rank |
| `test_the_default_is_unchanged` | a default that silently changed the method, which would make every earlier result wrong |
| `test_the_variant_changes_the_model` | a setting that is never read (a typo in `cfg.get("...")`, or code in the wrong branch) |
| `test_the_variant_is_deterministic` | randomness that makes results irreproducible (delete it for iALS, BPR-MF and the re-ranker, which are random by design) |
| `test_the_variant_still_learns` | a change so broken that the method ranks no better than random |
| `test_the_new_math_on_a_case_small_enough_to_check_by_hand` | the formula itself, on an example you can verify by hand |

### Writing the hand-computed check (TODO 4)

The best checks use a fact you know from the math. For example, asymmetric cosine with α = 0.5 *is* ordinary cosine,
so the variant at α = 0.5 must give exactly the baseline's scores:

```python
def test_the_new_math_on_a_case_small_enough_to_check_by_hand(toy):
    view, _ = toy
    checks.check_same_scores("itemknn", view, {}, {"knn_similarity": "asymmetric", "knn_alpha": 0.5})
```

Other kinds of check: compute the expected value with a few lines of numpy on a 4 × 3 matrix and compare; or check
that the result moves in the expected direction (a larger penalty must give smaller weights). The helpers in
`src/recbench/lab/checks.py` (`fit`, `scores`, `check_same_scores`, `check_differs`, `toy_ndcg`) make each check a
few lines.

## The default-behaviour pins

`tests/test_lab_defaults.py` fits each deterministic lab method (ItemKNN, RP3beta, EASE, SLIM, SANSA, PureSVD, GF-CF,
V-SKNN) at its default settings on the toy data. Their 10 best scores and items for 30 users must equal those
stored in `tests/golden/lab_defaults.json`. If you change a method and its defaults move, the test fails:

```text
ease's default behaviour changed:
  user 3: top scores [0.3141, 0.2432, 0.2419]... were [0.3141, 0.2401, 0.2387]...
Make your new setting's default keep the old behaviour. If the change is meant (a promotion), bump the method's
impl_version and regenerate the golden file (see the top of this test).
```

Usually this means the default is not quite the old behaviour, for example the new code runs even when the setting is
off. Fix the code. Only when you [promote](promote.md) a winner to be the new default do you raise the method's
`impl_version` and regenerate the file:

```bash
RECBENCH_UPDATE_GOLDEN=1 pytest tests/test_lab_defaults.py
```

## The rest of the suite

`./scripts/check.sh` also runs every fast test of the project, including the docs checks: every command and flag
mentioned in the docs must exist, every code pointer must resolve, and the site must build without warnings.
[Testing](../codebase/testing.md) describes each test file.

!!! tip "SANSA is special on macOS"
    SANSA (SuiteSparse) crashes in a process that has loaded PyTorch, and pytest loads it for other tests. Run SANSA
    checks in a child process, as `tests/test_new_methods.py::test_sansa_ranks_like_exact_ease` does. Lab 5 shows
    how.
