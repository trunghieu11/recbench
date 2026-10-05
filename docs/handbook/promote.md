# Promote a winner

!!! abstract "In plain words"
    When an experiment wins, it should not stay in the lab. **Promoting** it makes the improvement part of the
    method: either as a new option the bake-off's tuner can choose, or as the method's new default. Then the method's
    bake-off job is run again, so the official comparison reflects it.

## Decide

Run the comparison on all five datasets:

```bash
python -m recbench.compare ease ease:edlae
```

Its last line applies the rule:

| Result | Promote as | What changes |
|---|---|---|
| better on all five | the **new default** | the code's default, the method's `impl_version`, the golden file, and the search space |
| better on some, worse on none | a **search-space option** | the search space only; the default stays |
| worse on any, or no clear difference | not promoted | nothing; record what you learned on the lab page |

## A search-space option

1. Add the new setting to the method's entry in `configs/tuning/quick.yaml`, keeping the old behaviour as one of the
   choices, so the tuner can still pick it:

    ```yaml
    ease:
      params:
        ease_lambda: {type: log, low: 1.0, high: 20000.0}
        ease_variant: {type: choice, values: [ease, edlae]}
        ease_dropout: {type: float, low: 0.05, high: 0.75}
        decay_half_life_days: {type: choice, values: [null, 30, 90, 365]}
    ```

2. Raise `version:` at the top of that file by one. It is part of every tuning study's name, so new jobs never mix
   with trials of the old search space.
3. Mark the experiment `promoted: true` in `labs/<nn>-<method>/experiments.yaml`.
4. Update the method page's **Hyperparameters** table.

## A new default

All of the above, plus:

1. Change the default in the code, for example `cfg.get("ease_variant", "edlae")`.
2. Raise the method's `impl_version` in its `MethodSpec` (for example `"3"` to `"4"`). Results from the old code are
   then marked as older on the docs pages, and nothing is mistaken for the new behaviour.
3. Regenerate the default-behaviour pins: `RECBENCH_UPDATE_GOLDEN=1 pytest tests/test_lab_defaults.py`.
4. Update the method page's **In recbench** section.

## Re-run its bake-off job

The bake-off's results come from a fixed search space and code. After a promotion, the method's job must run again,
on a rented box like the bake-off itself. The box has the bake-off's machine profile, which matters for EASE: its
item cap is 30,000 on the box and 20,000 on a laptop. It also keeps training off your laptop.

1. **Push** your merged `main`, so the box can clone it.
2. **Rent and set up** a box ([5b](../start/box-2-rent-and-set-up.md)).
3. **Copy the data and the job summaries** from the laptop. Without the summaries, the queue would tune every method
   again:

    ```bash
    ./scripts/upload_splits.sh vast-gpu && ./scripts/upload_results.sh vast-gpu
    ```

4. **Re-run the method**, on the box inside tmux:

    ```bash
    ./scripts/run_quick_box.sh --methods ease --rerun
    ```

5. **Fetch the results** on the laptop with `./scripts/fetch_results.sh vast-gpu`. It also rebuilds the
   bake-off's reports and docs pages.

- `--rerun` moves the method's finished summaries to `archive/` and starts fresh studies. It needs `--methods`, so
  a whole bake-off is never re-run by accident. If the command is interrupted, resume it *without* `--rerun`.
- It also archives the method's full-data confirmations. Where the re-run method is still among a dataset's top 3
  by validation, the queue confirms it again on the full splits.
- Promoting EASE or ItemKNN changes the LightGBM and DCN-V2 re-rankers too, since their candidates come from these
  two. Re-run them in the same command: `--methods ease,lgbm_rerank,dcnv2_rerank`. The queue runs the re-rankers
  after EASE, with its new settings.
- Several promotions can share one box session: `--methods itemknn,ease,lgbm_rerank,dcnv2_rerank`.

Commit everything in the lab's pull request (see the checklist in the PR template).
