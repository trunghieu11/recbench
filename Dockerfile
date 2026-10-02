FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY dictionary ./dictionary
RUN pip install --no-cache-dir -e ".[cpu]"

ENV CUDA_VISIBLE_DEVICES=""
ENV RECBENCH_SERVE_METHODS=xsimgcl,bert4rec,dcnv2,din
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn recbench.serving.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
