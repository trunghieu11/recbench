"""Load benchmark and hardware YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from recbench.protocol import PRESET_STEPS, SMOKE_STEPS


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return data


def resolve_run_config(
    benchmark: dict[str, Any],
    *,
    preset: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    hardware_path = benchmark.get("hardware", "configs/hardware/local-cpu.yaml")
    hardware = load_yaml(root / hardware_path)
    model_config = hardware.get("model_config", "smoke_cpu")
    if model_config == "smoke_cpu":
        chosen = "cpu"
        max_steps = int(hardware.get("max_steps", SMOKE_STEPS))
    else:
        chosen = preset or "24gb"
        presets = hardware.get("presets") or {}
        block = presets.get(chosen) or {}
        max_steps = int(block.get("max_steps", PRESET_STEPS.get(chosen, 10_000)))
    if benchmark.get("max_steps") is not None:
        max_steps = int(benchmark["max_steps"])
    dims = {
        "cpu": {"dim": 16, "layers": 1, "seq_len": 20, "batch_size": 64, "lr": 1e-3},
        "24gb": {"dim": 64, "layers": 2, "seq_len": 50, "batch_size": 256, "lr": 1e-3},
        "48gb": {"dim": 128, "layers": 3, "seq_len": 200, "batch_size": 512, "lr": 1e-3},
    }[chosen if chosen in ("cpu", "24gb", "48gb") else "24gb"]
    return {
        "tier": benchmark.get("tier", "smoke"),
        "hardware_name": hardware.get("name", "local-cpu"),
        "requires_gpu": bool(hardware.get("requires_gpu", False)),
        "model_config": model_config,
        "preset": chosen,
        "max_steps": max_steps,
        "resume": bool(benchmark.get("resume", True)),
        "continue_on_error": bool(benchmark.get("continue_on_error", True)),
        "managed_services": bool(benchmark.get("managed_services", False)),
        "datasets": list(benchmark.get("datasets") or []),
        "methods": list(benchmark.get("methods") or []),
        "metrics": benchmark.get("metrics", "all"),
        **dims,
    }
