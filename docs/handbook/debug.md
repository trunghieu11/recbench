# Debug

!!! abstract "In plain words"
    Most problems announce themselves in a message. This page lists the usual ones in the lab, what they mean, and
    what to do. When there is no message, only a strange number, the second half of the page shows how to look
    inside: fit the model in a notebook, open the run in MLflow, read the job's summary.

## Messages and what they mean

| You see | It means | Do this |
|---|---|---|
| `already done (test NDCG@10 …); --rerun runs it again` | the experiment finished with the same definition and code | nothing, or `--rerun` |
| `(stored result of an identical run; --fresh computes it again)` | `lab once` found a stored run with exactly these settings and this code | nothing, or `--fresh` |
| `itemknn's code never reads knn_similarity` | no code of the method reads that setting: the variant is not written yet, or the name is misspelled | write the variant, or fix the spelling in `experiments.yaml` / `--set` |
| `itemknn has no baseline on movielens-25m yet` | `--best` needs the baseline's best setting | `python -m recbench.lab baseline --methods itemknn --datasets movielens-25m` |
| `cannot remove 'x': the baseline does not search it` | `remove:` names a setting that is not in the search space | check the method's entry in `configs/tuning/quick.yaml` |
| `OMP: Error #15` or a crashed kernel while fitting SANSA or LightGBM in a notebook | PyTorch's OpenMP runtime was loaded in the same process | restart the kernel and run `lab.setup()` first; never `import torch` in a lab notebook |
| a job `over_budget` | it did not finish within 3 hours | a result about cost; look at which settings were slow (`lab.trials(...)`) and narrow the range if you have a reason |
| a job `failed` | an error in a trial or the final run | read the reason in `lab status`; the full traceback is the run's `error.txt` in MLflow (below) |
| `MemoryError` or the Mac becomes very slow | two big jobs at once | run with `--workers 1` |
| `RuntimeWarning: overflow encountered in divide` | a number became too large for float32, often from dividing by a tiny weight | worth a look: [lab 2](../labs/02-rp3beta.md) investigates one |
| your numbers differ from a "You should see" box | deterministic methods should match exactly; random ones (iALS, BPR-MF, the re-ranker) vary a little by seed | if a deterministic method differs, check that the settings and the dataset are the same |

## Look inside

### Fit it in a notebook

```python
from recbench import lab
lab.setup()
data, split = lab.load("movielens-25m")
model = lab.fit("ease", data, ease_lambda=300.0)
model.weights.shape, model.fit_info        # the item x item weights, and what fit() recorded
```

Every lab method keeps its learned parts as attributes: `model.sim` (ItemKNN, RP3beta), `model.weights` (EASE,
SLIM), `model.v` (PureSVD), `model.user_factors` / `model.item_factors` (iALS, BPR-MF). You can print them, plot
them, or score one user by hand and compare with `model.score_users(...)`.

To stop inside the code and look around, put `breakpoint()` on a line of the method's `fit`, then call `lab.fit` in
the notebook. Python pauses there: type a variable's name to see it, `n` for the next line, `c` to continue. In
VS Code, the **Debug Cell** command does the same with buttons.

### Open the run in MLflow

```bash
mlflow ui --backend-store-uri "file://$PWD/runs/lab/mlflow" --port 5002
```

Open <http://127.0.0.1:5002> and find the run by its tags (`method`, `dataset`, `stage`: `search` for a tuning
trial, `final` for a test run, `once` for your own runs). **Parameters** shows exactly which settings it received.
That is the quickest way to prove a setting reached the method. For a failed run, **Artifacts → error.txt** holds
the full traceback.

### Read the job's summary

`runs/lab/tuning/quick/<dataset>/<method>[@experiment].json` holds every trial: settings, validation score, seconds,
status and the reason for a failure. In a notebook, `lab.trials("ease", "movielens-25m")` shows it as a table, best
first.

## "My change does nothing"

1. **Is the setting reaching the method?** In MLflow, the run's Parameters must list it. If not, check the spelling
   in `experiments.yaml`.
2. **Is it read?** Search the method's code for `cfg.get("your_setting"`. The lab refuses unread settings, but a
   setting can be read and then never used.
3. **Is the new branch taken?** Put `print("variant on")` in it, or `breakpoint()`, and call `lab.fit` with the
   setting.
4. **Is the difference just too small to see?** Run `lab sweep` on the new setting. If NDCG@10 moves less than the
   width of its interval, that is the answer.
