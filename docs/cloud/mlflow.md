# MLflow

**MLflow** is an open-source tool for tracking machine-learning experiments. recbench uses its **tracking**
component only: every (dataset, method) pair becomes one MLflow **run** with tags, parameters, metrics, and
artifacts. Reports, the dashboard, and the generated docs all read their numbers back from MLflow, so it is the
single source of truth for results.

## Vocabulary

| Term | Meaning | In recbench |
|---|---|---|
| tracking store | where runs are saved | the folder `runs/mlflow/` (a local file store) |
| experiment | a group of runs | one experiment, `recbench` |
| run | one execution | one (dataset, method, tier) fit and evaluation |
| tag | a text label | dataset, method, tier, status, `config_hash`, ... |
| param | a setting | `dim`, `lr`, `ease_lambda`, ... |
| metric | a number | `ndcg_at_10`, `train_seconds`, ... |
| artifact | a file attached to a run | `per_user_metrics.npz`, `explanations.json`, ... |

The full list of what is logged is in [training and evaluation](../codebase/training-and-evaluation.md#what-is-logged-per-run).

## Where the store is

`src/recbench/results.py::tracking_uri` decides:

1. `MLFLOW_TRACKING_URI` if set (the run scripts set it to `file://<repo>/runs/mlflow`);
2. otherwise `<RECBENCH_ROOT or current folder>/runs/mlflow` as an absolute `file://` URI.

An absolute URI matters: the runner starts a child process per method, and a relative path would resolve
differently if the working directory changed.

## Open the UI

```bash
mlflow ui --backend-store-uri "file://$PWD/runs/mlflow" --port 5001
# then open http://127.0.0.1:5001
```

Port 5001 avoids a clash with macOS AirPlay, which uses port 5000. Alternatively,
`docker compose -f deploy/compose.yaml up` starts an MLflow container on the same port (see [Docker](docker.md)).

Useful UI actions:

- **Search runs** with a filter such as `tags.dataset = "movielens-25m" and tags.status = "finished"`.
- **Compare**: select several runs and click *Compare* to see parameters and metrics side by side.
- **Artifacts**: open a run to download `per_user_metrics.npz` or read `explanations.json`.

## Query from Python

```python
import mlflow
from recbench.results import load_runs, leaderboard, tracking_uri

frame = load_runs("smoke")                       # latest finished protocol-v2 run per (dataset, method)
board = leaderboard(frame, "movielens-25m", "ndcg_at_10")
print(board[["rank", "tags.method", "metrics.ndcg_at_10", "tied_with_best"]])

mlflow.set_tracking_uri(tracking_uri())
run = mlflow.get_run(frame.iloc[0]["run_id"])   # everything about one run
```

## Status tags

Every run has `tags.status`:

| Status | Meaning |
|---|---|
| `running` | in progress, or the process died (the runner marks stale runs when it next sees them) |
| `finished` | metrics are valid; this run can appear on leaderboards |
| `unsupported` | the method does not apply (reason in `tags.reason`, e.g. no images) |
| `failed` | an error (traceback in the `error.txt` artifact) |
| `timeout` | the wall-clock limit was reached |

## Resume

Before fitting, the runner computes a `config_hash` from the protocol version, code version, method, split, and
every setting, and skips the pair if a finished run with that hash exists. Changing anything that affects results
gives a new hash, so the pair runs again. See [configuration](../codebase/configuration.md).

## Backups and sharing

The file store is a plain folder, so a backup is one command:

```bash
tar czf mlflow-$(date +%F).tgz runs/mlflow
```

**Combining two machines' results needs an import, not a copy.** Each store has its own `recbench`
experiment with a random id, and its runs record absolute artifact paths from the machine that wrote them. If
you copy the GPU machine's folder into `runs/mlflow/`, MLflow sees two experiments named `recbench` and reads
only one, so half the results silently disappear. Copy it elsewhere and import it:

```bash
rsync -a gpu-box:recommendation_benchmark/runs/mlflow/ runs/mlflow-gpu/
python -m recbench.import_runs runs/mlflow-gpu
```

`src/recbench/import_runs.py::import_runs` re-creates each run in your store with the same tags, parameters,
metrics, start and end times, and artifact files, and tags it `imported_from`. Running it again skips runs it
already imported.

!!! note "Why MLflow is pinned below 3"
    `pyproject.toml` pins `mlflow<3`. MLflow 3 changed defaults and APIs; pinning keeps the store format and the
    query code stable. Upgrading is a [roadmap](../results/roadmap.md) item.

## Remote tracking (optional, not set up)

For a team, MLflow can use a database plus a bucket for artifacts behind an MLflow server. That needs an
always-on server, which conflicts with the budget, so recbench keeps a local store and publishes results through
reports and the docs site instead.
