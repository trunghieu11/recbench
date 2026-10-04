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

The bake-off's results come from a fixed search space and code. After a promotion, the method's job must run again:

```bash
python -m recbench.queue run --config configs/benchmarks/quick.yaml --methods ease --rerun --hardware configs/hardware/local-cpu.yaml --cpu-workers 2
```

- `--rerun` moves the method's finished summaries to `archive/` and starts fresh studies. It needs `--methods`, so
  a whole bake-off is never re-run by accident. If the command is interrupted, resume it *without* `--rerun`.
- `--hardware configs/hardware/local-cpu.yaml` runs it on this laptop. Fine for the CPU methods: their accuracy does
  not depend on the machine, only their training times do, and the report records the hardware. EASE is the
  exception: this laptop caps it at 20,000 items and the box at 30,000. Re-run EASE on the next rented box instead.
- If the bake-off has not run yet, skip this step: it will use the promoted code when it runs.

Then rebuild the bake-off's reports and docs:

```bash
python -m recbench.report.build --tier quick --tuning tuned --out reports/quick-tuned --docs
python -m recbench.dictionary.build
python -m recbench.report.overall --docs --out reports/overall
```

Commit everything in the lab's pull request (see the checklist in the PR template).
