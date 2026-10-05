# Serve recommendations locally

The API serves precomputed **bundles** (top-100 lists per user) written by the runner. It needs no model code
and no PyTorch. See [serving and bundles](../codebase/serving-and-bundles.md) for the design.

## 1. Make sure bundles exist

```bash
ls data/bundles/*/smoke/          # one folder per method that finished
```

If empty, run the benchmark first (bundles are exported automatically when `export_bundles: true`, the default).

## 2. Start the API (from the virtual environment)

```bash
uvicorn recbench.serving.app:app --port 8080
```

Useful environment variables: `RECBENCH_TIER` (default `smoke`), `RECBENCH_BUNDLES` (default `data/bundles`), and
`RECBENCH_SERVE_METHODS` (an allow-list, for example `ease,sasrec`).

## 3. Call it

```bash
curl localhost:8080/health
# {"ok": true, "version": "0.3.0", "tier": "smoke", "bundles": 70}    (HTTP 503 and "ok": false when there are none)

curl localhost:8080/methods
# [{"dataset": "hm", "tier": "smoke", "method": "bpr_mf"}, ...]

curl -X POST localhost:8080/recommend -H 'content-type: application/json' \
     -d '{"dataset": "movielens-25m", "method": "ease", "user_id": "123", "k": 3}'
```

A response looks like:

```json
{
  "dataset": "movielens-25m",
  "method": "ease",
  "user_id": "123",
  "fallback": false,
  "recommendations": [
    {"item_id": "2571", "title": "Matrix, The (1999)", "score": 0.84},
    {"item_id": "296", "title": "Pulp Fiction (1994)", "score": 0.79},
    {"item_id": "318", "title": "Shawshank Redemption, The (1994)", "score": 0.75}
  ]
}
```

(Illustrative values.) Unknown users get the popularity list and `"fallback": true`. The terminal running uvicorn
prints one JSON line per call (dataset, method, status, latency, fallback); `RECBENCH_REQUEST_LOG=0` turns it off.

Check what is served and how fresh it is:

```bash
curl -s localhost:8080/stats | python -m json.tool | head -30
python -m recbench.serving.monitor --base-url http://127.0.0.1:8080
```

[Monitoring the API](../cloud/monitoring.md) explains both. Open <http://localhost:8080/docs> for the interactive
API documentation, and <http://localhost:8080/dashboard> for leaderboards (needs the `bench` extra, which provides
MLflow; `?tuning=defaults` or `?tuning=tuned` keeps one kind of run).

## 4. With Docker

```bash
docker build -t recbench .
docker run --rm -p 8080:8080 -v "$PWD/data/bundles:/bundles:ro" recbench
```

or the compose file, which also starts a read-only MLflow UI on port 5001:

```bash
docker compose -f deploy/compose.yaml up --build
```

The image installs only the `serve` extra. Details in [Docker](../cloud/docker.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `"bundles": 0` | wrong `RECBENCH_TIER` or `RECBENCH_BUNDLES`; check the folder layout `<dataset>/<tier>/<method>/manifest.json` |
| 404 "No bundle for ..." | that method did not finish on that dataset (see the report's "Did not run" table) |
| dashboard 404 | install the `bench` extra; the Docker image deliberately does not include MLflow (`GET /` lists what the service offers) |
