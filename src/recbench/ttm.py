"""Time to market, measured: from a raw interaction log to a healthy API returning recommendations.

    python -m recbench.ttm --method ease

Stages timed on a synthetic raw log (so it runs anywhere, no downloads):
  raw_to_split_s     clean + split the log
  fit_eval_s         fit the method and evaluate it
  export_s           write the serving bundle
  start_to_first_s   start the API until the first successful POST /recommend
This is the automated part of the "time to market" metric; the human part
(writing an adapter, tuning) is scored with the rubric in dictionary/catalog.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from recbench.data import TrainView
from recbench.evaluation import EvalSplit, Evaluator
from recbench.pipeline.materialize import materialize
from recbench.pipeline.toy import write_toy_clean
from recbench.registry import ensure_loaded
from recbench.serving.bundle import export_bundle


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(root: Path, method: str = "ease", steps: int = 50) -> dict[str, object]:
    root = Path(root).resolve()
    began = time.perf_counter()
    write_toy_clean(root / "clean" / "toy")
    split_dir = materialize(root / "clean" / "toy", root / "splits" / "toy" / "smoke", dataset="toy", tier="smoke",
                            tier_overrides={"target_events": None, "min_eval_users": 1})
    split_done = time.perf_counter()
    data = TrainView(split_dir)
    model = ensure_loaded().create_method(method)
    cfg = {"seed": 42, "dim": 16, "layers": 1, "heads": 2, "seq_len": 20, "batch_size": 64, "max_steps": steps, "device": "cpu"}
    model.fit(data, cfg)
    metrics = Evaluator(EvalSplit(split_dir), data, cfg).run(model).metrics
    fitted = time.perf_counter()
    export_bundle(model, data, root / "bundles" / "toy" / "smoke" / method, k=20)
    exported = time.perf_counter()
    port = _free_port()
    env = {**os.environ, "RECBENCH_BUNDLES": str(root / "bundles"), "RECBENCH_TIER": "smoke"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "recbench.serving.app:app", "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    first = None
    try:
        deadline = time.perf_counter() + 60
        while time.perf_counter() < deadline:
            try:
                response = httpx.post(f"http://127.0.0.1:{port}/recommend",
                                      json={"dataset": "toy", "method": method, "user_id": "u1", "k": 5}, timeout=2)
                if response.status_code == 200 and response.json()["recommendations"]:
                    first = time.perf_counter()
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    if first is None:
        raise RuntimeError("the API never returned recommendations")
    result = {
        "method": method,
        "raw_to_split_s": round(split_done - began, 3),
        "fit_eval_s": round(fitted - split_done, 3),
        "export_s": round(exported - fitted, 3),
        "start_to_first_s": round(first - exported, 3),
        "time_to_endpoint_s": round(first - began, 3),
        "toy_ndcg_at_10": round(float(metrics.get("ndcg_at_10", float("nan"))), 4),
    }
    (root / "ttm.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Time raw data to a working recommendation endpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("runs/ttm"))
    parser.add_argument("--method", default="ease")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.method, args.steps), indent=2))


if __name__ == "__main__":
    main()
