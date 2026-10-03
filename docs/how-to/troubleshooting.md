# Troubleshooting

## Installation and environment

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: recbench` | the virtual environment is not active | `source .venv/bin/activate` |
| `Unsupported: third_party/SELFRec is missing` | third-party code not fetched | `bash scripts/fetch_third_party.sh` |
| `fuxictr is not installed` / RecBole import errors | installed with the `bench` or `serve` extra only | `uv pip install -e ".[cpu]"` |
| many pandas FutureWarnings from RecBole | RecBole 1.2 uses older pandas APIs | harmless; the run scripts set `PYTHONWARNINGS=ignore` |
| `No module named 'numba'` (LightGCN, XSimGCL) or `'kmeans_pytorch'` (BERT4Rec, S3-Rec) | an older install without these extras | reinstall with `uv pip install -e ".[cpu]"` (or `[gpu]`); both are now in the extras |

## Data preparation

| Symptom | Cause | Fix |
|---|---|---|
| "Kaggle credentials missing" | no `~/.kaggle/kaggle.json` | see [install, step 5](../start/install.md#step-5-a-kaggle-token-for-hm-and-retailrocket) |
| Kaggle 403 for H&M | competition rules not accepted | accept them on the competition page |
| "Authentication required to call the Kaggle API" with a new-style key (`KGAT...`) | Kaggle CLI 2.x reads `~/.kaggle/access_token` | the run scripts copy the key there automatically (`ensure_kaggle_token` in `scripts/_common.sh`); or create the file yourself |
| Last.fm download hangs or fails | the original host does not answer from some cloud machines | nothing: the adapter falls back to the Zenodo mirror |
| `SplitError: only N warm eval users` | the split is too small to evaluate reliably | use a bigger tier, or lower `min_eval_users` via `tier_overrides` in the config (results get noisier) |
| "Need at least 30 GB free" | disk space | free space, or `export DATA_DIR=/Volumes/Big/recbench-data` |
| a dataset failed but others continued | by design | read the reason in `data/unavailable/<dataset>-<tier>.txt` |
| changes to an adapter have no effect | old clean files reused | bump `CLEAN_VERSION` in the adapter, or delete `data/clean/<dataset>/` |

## Running methods

| Status / symptom | Meaning | What to do |
|---|---|---|
| `unsupported: no item images on disk` | the multimodal tower needs real images | download H&M images (see the [H&M page](../dictionary/datasets/hm.md)) or ignore |
| `unsupported: S3-Rec needs item attributes` | the dataset has no categories (Last.fm) | expected |
| `unsupported: RecBole's sequence dataset would need ~X GB` | not enough RAM for RecBole at this size | lower `seq_len`, use SASRec, or a bigger machine |
| `timeout` | exceeded `timeout_minutes` (wall clock) | raise `timeout_minutes` in the config, or run the method on the GPU tier |
| iALS fails with exit code -11 (segfault) on a many-core machine | OpenBLAS was built for at most 128 threads; the machine has more | the runner caps `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` at 32 per child process; set them lower if needed |
| `pkill -f recbench.runner` kills your own SSH command | the pattern also matches the remote shell running it | use `pkill -f "[r]ecbench\.runner"` |
| `failed` | an exception | open the run in MLflow and read `error.txt` |
| `skipped_existing` | an identical run already finished | change a setting, bump the version in `src/recbench/__init__.py`, or delete the run in MLflow to force a re-run |
| results near Random for a new sequence model | reading padding instead of the newest item | use the last column of right-aligned histories (see [sequential models](../dictionary/concepts/sequential-and-session.md)) |
| very slow DIN or DCN-V2 on a laptop | full-catalog ranking scores every pair | use the GPU tier, or lower `eval.max_eval_users` |

## Apple Silicon GPU (MPS)

recbench uses the CPU on Macs by default (`device: cpu` in `configs/hardware/local-cpu.yaml`). You can try
`device: mps`, but some operations (sparse matrices in the graph models) are not supported on MPS, so expect
`unsupported` or failed runs. The CPU is reliable for the smoke tier.

## MLflow and results

| Symptom | Fix |
|---|---|
| the MLflow UI shows no runs | point it at the right folder: `mlflow ui --backend-store-uri file://$PWD/runs/mlflow` |
| old v0.1 runs appear | they live in `runs/mlflow_v1_archive`; reports and dashboards read only protocol v2 runs |
| the report misses a method | it did not finish; check the "Did not run" table |

## Docs

| Symptom | Fix |
|---|---|
| `mkdocs build --strict` fails on a snippet | run `python -m recbench.dictionary.build` and the report builder with `--docs` first, so `docs/generated/` exists |
| a broken link warning | links must be relative paths to pages listed in `mkdocs.yml` |
| `tests/test_docs.py` fails on a code pointer | a `path::Symbol` in the docs no longer matches the code; update the page |
