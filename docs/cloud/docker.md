# Docker

A **container image** packages an application with its operating-system libraries and Python packages, so it
runs the same on your laptop and on Cloud Run. **Docker** builds and runs images. recbench has one image: the
serving API.

## The image, line by line

```dockerfile
FROM python:3.12-slim
```

Start from a small official Debian image with Python 3.12, the same Python version as the development
environment.

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    RECBENCH_BUNDLES=/bundles \
    RECBENCH_TIER=smoke
```

Do not write `.pyc` files; print logs immediately (Cloud Run collects standard output); listen on port 8080 (Cloud
Run's convention, and it sets `PORT` itself); look for bundles in `/bundles`; serve the smoke tier unless told
otherwise.

```dockerfile
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]" && useradd --create-home app
USER app
```

Copy only what installation needs and install the package with the **serve** extra: NumPy, pandas, PyArrow,
FastAPI, Uvicorn, Jinja2. No PyTorch, no MLflow, no RecBole. Then create an unprivileged user and switch to it,
so a bug in the API cannot modify the image's system files.

```dockerfile
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn recbench.serving.app:app --host 0.0.0.0 --port ${PORT}"]
```

Start the API on all network interfaces. `exec` makes Uvicorn the main process, so it receives the stop signal
when the platform shuts the container down.

## What is not in the image

`.dockerignore` excludes `.venv`, `.git`, `data`, `runs`, `reports`, `site`, `docs`, `third_party`, `tests`,
caches, and `.env*` files. Two benefits: the build context is small (fast uploads to Cloud Build), and secrets in
`.env` files can never be copied into an image.

**Bundles are not in the image either.** They are mounted at run time. The same image can serve any tier or
dataset, and updating recommendations does not require a rebuild.

## Run it locally

```bash
docker build -t recbench-api .
docker run --rm -p 8080:8080 -v "$PWD/data/bundles:/bundles:ro" recbench-api
curl localhost:8080/health
```

`-v host:container:ro` mounts your bundles read-only. Set `-e RECBENCH_TIER=full` to serve full-tier bundles.

## Docker Compose: API plus MLflow

`deploy/compose.yaml` starts two containers:

| Service | URL | What it does |
|---|---|---|
| `api` | <http://localhost:8080> | the image above, with `data/bundles` mounted read-only |
| `mlflow` | <http://localhost:5001> | the official MLflow 2.22 image showing `runs/mlflow` read-only |

```bash
docker compose -f deploy/compose.yaml up --build
```

The `/dashboard` page does not work inside the `api` container, because the dashboard needs MLflow and the image
does not include it. Use the MLflow container, or run the API from the virtual environment.

## Why the image is small

| Choice | Effect |
|---|---|
| `python:3.12-slim` base | no compilers or documentation |
| `serve` extra only | no PyTorch (about 1 GB or more with CUDA) |
| precomputed bundles | no model code at serving time |
| `.dockerignore` | no data or virtual environment in the build context |

A smaller image is pulled faster, so a new Cloud Run instance starts faster after scaling from zero (a **cold
start**). See [serving and latency](../dictionary/concepts/serving-and-latency.md).

## Pitfalls

!!! warning "Apple Silicon builds"
    On an M-series Mac, `docker build` produces an `arm64` image by default. Cloud Run needs `linux/amd64`. The
    deploy script avoids this problem because Cloud Build builds the image on Google's machines. If you push an
    image you built yourself, use `docker build --platform linux/amd64`.

!!! tip "Nothing in the container changes"
    The API only reads files. Everything it writes (logs) goes to standard output, so the container can be
    stopped and replaced at any time.
