"""Load benchmark + hardware YAML into one resolved run configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from recbench import __version__
from recbench.protocol import PROTOCOL_VERSION

# Model sizes per hardware preset. Every neural method reads these keys.
PRESETS: dict[str, dict[str, Any]] = {
    "cpu": {"dim": 32, "layers": 2, "heads": 2, "seq_len": 50, "batch_size": 128, "lr": 1e-3, "max_steps": 400},
    "24gb": {"dim": 64, "layers": 2, "heads": 2, "seq_len": 50, "batch_size": 256, "lr": 1e-3, "max_steps": 10_000},
    "48gb": {"dim": 128, "layers": 3, "heads": 4, "seq_len": 200, "batch_size": 512, "lr": 1e-3, "max_steps": 30_000},
}


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
    """Merge the benchmark file, its hardware profile, and the model-size preset."""
    root = repo_root or Path.cwd()
    hardware = load_yaml(root / benchmark.get("hardware", "configs/hardware/local-cpu.yaml"))
    chosen = preset or hardware.get("default_preset", "cpu")
    if chosen not in PRESETS:
        raise ValueError(f"Unknown preset {chosen}. Known: {sorted(PRESETS)}")
    resolved: dict[str, Any] = {
        "tier": benchmark.get("tier", "smoke"),
        "hardware_name": hardware.get("name", "local-cpu"),
        "device": hardware.get("device", "auto"),
        "preset": chosen,
        **PRESETS[chosen],
        **(hardware.get("overrides") or {}),
        "seed": int(benchmark.get("seed", 42)),
        "resume": bool(benchmark.get("resume", True)),
        "continue_on_error": bool(benchmark.get("continue_on_error", True)),
        "timeout_minutes": float(benchmark.get("timeout_minutes", 240)),
        "managed_services": bool(benchmark.get("managed_services", False)),
        "export_bundles": bool(benchmark.get("export_bundles", True)),
        "datasets": list(benchmark.get("datasets") or []),
        "methods": list(benchmark.get("methods") or []),
        "method_params": dict(benchmark.get("method_params") or {}),
    }
    if benchmark.get("max_steps") is not None:
        resolved["max_steps"] = int(benchmark["max_steps"])
    resolved.update(benchmark.get("eval") or {})
    return resolved


def method_config(resolved: dict[str, Any], method: str) -> dict[str, Any]:
    """The flat dict a method's fit() receives: shared settings plus that method's own parameters."""
    shared = {k: v for k, v in resolved.items() if k not in {"datasets", "methods", "method_params"}}
    return {**shared, **(resolved.get("method_params", {}).get(method) or {})}


def config_hash(cfg: dict[str, Any], method: str, split_hash: str) -> str:
    """Identity of one run. Changing the protocol, the split, the code version, or any setting changes it."""
    ignored = {"resume", "continue_on_error", "timeout_minutes", "export_bundles", "managed_services"}
    payload = {
        "protocol": PROTOCOL_VERSION,
        "code": __version__,
        "method": method,
        "split": split_hash,
        "cfg": {k: v for k, v in sorted(cfg.items()) if k not in ignored},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
