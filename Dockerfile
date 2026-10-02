# Serving image: FastAPI + precomputed recommendation bundles.
# It installs only the "serve" extra (numpy, pandas, pyarrow, fastapi): no torch, no MLflow,
# so the image is small and cold starts are fast. Models are never loaded here; bundles are.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    RECBENCH_BUNDLES=/bundles \
    RECBENCH_TIER=smoke

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]" && useradd --create-home app
USER app

EXPOSE 8080
# Bundles are mounted at run time, e.g. docker run -v "$PWD/data/bundles:/bundles:ro" ...
CMD ["sh", "-c", "exec uvicorn recbench.serving.app:app --host 0.0.0.0 --port ${PORT}"]
