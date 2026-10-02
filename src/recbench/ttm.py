"""Time the path from a raw file to a healthy endpoint."""

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

from recbench.pipeline.materialize import materialize
from recbench.pipeline.toy import write_toy_clean
from recbench.registry import ensure_loaded
from recbench.runner import run_pair
from recbench.store import SplitStore


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(root: Path, steps: int = 20) -> dict[str, float]:
    raw_at = time.perf_counter()
    clean = root / "clean" / "toy"
    write_toy_clean(clean)
    out = root / "splits" / "toy" / "smoke"
    materialize(clean, out, dataset="toy", tier="smoke", split_mode="quantile")
    split_ready = time.perf_counter()
    os.environ["DATA_DIR"] = str(root)
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{root / 'mlflow'}")
    store = SplitStore(out, "toy", "smoke")
    ensure_loaded()
    resolved = {
        "tier": "smoke",
        "preset": "cpu",
        "max_steps": steps,
        "hardware_name": "local-cpu",
        "model_config": "smoke_cpu",
        "dim": 8,
        "layers": 1,
        "seq_len": 8,
        "batch_size": 32,
        "lr": 1e-3,
        "managed_services": False,
        "continue_on_error": False,
        "resume": False,
    }
    train_started = time.perf_counter()
    status = run_pair(store, "xsimgcl", resolved, resume=False)
    if status != "finished":
        status = run_pair(store, "bert4rec", resolved, resume=False)
    trained_at = time.perf_counter()
    port = _free_port()
    env = os.environ.copy()
    env["DATA_DIR"] = str(root)
    env["RECBENCH_DATASET"] = "toy"
    env["RECBENCH_TIER"] = "smoke"
    env["RECBENCH_PRESET"] = "cpu"
    env["RECBENCH_SERVE_METHODS"] = "xsimgcl,bert4rec"
    env["MLFLOW_TRACKING_URI"] = os.environ["MLFLOW_TRACKING_URI"]
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "recbench.serving.app:app", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.perf_counter() + 30
        while time.perf_counter() < deadline:
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            raise RuntimeError("endpoint did not become healthy")
        healthy_at = time.perf_counter()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    result = {
        "method": "xsimgcl" if status == "finished" else "bert4rec",
        "raw_to_split_s": split_ready - raw_at,
        "split_to_train_s": trained_at - train_started,
        "train_to_endpoint_s": healthy_at - trained_at,
        "time_to_endpoint_s": healthy_at - raw_at,
        "status": status,
    }
    dest = root / "ttm.json"
    dest.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Time raw data to a healthy endpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("runs/ttm"))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, steps=args.steps), indent=2))


if __name__ == "__main__":
    main()
