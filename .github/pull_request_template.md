## What this changes

<!-- One or two sentences: the method, the idea, and why. Link the lab page, for example docs/labs/03-ease.md. -->

## Results

<!-- Paste the output of: python -m recbench.compare <method> <method>:<experiment> -->

```text

```

## Checklist

- [ ] The change is a setting whose default keeps today's behaviour, or `impl_version` is bumped and
      `tests/golden/lab_defaults.json` regenerated (`RECBENCH_UPDATE_GOLDEN=1 pytest tests/test_lab_defaults.py`).
- [ ] `tests/test_<method>_variant.py`, from `labs/templates/test_variant_template.py`, tests the change.
- [ ] The experiment is in `labs/<nn>-<method>/experiments.yaml` with a `note`, and ran on all five datasets.
- [ ] Settings were chosen on the validation fold; the test split was used only by the experiment's final runs.
- [ ] `python -m recbench.lab scoreboard --docs` was rerun; the lab page's "Record your results" table is filled in.
- [ ] Notebooks are saved without outputs.
- [ ] `./scripts/check.sh` passes.
- [ ] If promoted: `promoted: true` in the experiment, `configs/tuning/quick.yaml` updated (with `version` raised), and
      the bake-off job re-run (`python -m recbench.queue run --config configs/benchmarks/quick.yaml --methods <m> --rerun`).
