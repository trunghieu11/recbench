"""Download, clean, and split datasets.

Usage:
    python -m recbench.pipeline.prepare --config configs/benchmarks/smoke-cpu.yaml
    python -m recbench.pipeline.prepare --config ... --datasets hm --tier full

Cleaning is skipped when data/clean/<dataset> was produced by the same adapter
version (CLEAN_VERSION); splitting always runs, because it is cheap and its
settings may have changed.
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any

from recbench.config import load_yaml
from recbench.pipeline.materialize import materialize
from recbench.registry import ensure_loaded


def data_root() -> Path:
    return Path(os.environ.get("DATA_DIR", Path.cwd() / "data")).resolve()


def prepare_one(name: str, tier: str, root: Path | None = None, tier_overrides: dict[str, Any] | None = None) -> Path:
    reg = ensure_loaded()
    dataset = reg.create_dataset(name)
    spec = dataset.spec
    root = root or data_root()
    raw = root / "raw" / name
    clean = root / "clean" / name
    out = root / "splits" / name / tier
    marker = root / "unavailable" / f"{name}-{tier}.txt"
    raw.mkdir(parents=True, exist_ok=True)
    try:
        dataset.download(raw)
        version = str(getattr(dataset, "CLEAN_VERSION", "1"))
        stamp = clean / "_clean_version.txt"
        if not (clean / "interactions.parquet").exists() or not stamp.exists() or stamp.read_text().strip() != version:
            print(f"{name}: cleaning raw files (adapter version {version})")
            dataset.to_clean(raw, clean)
            stamp.write_text(version)
        materialize(
            clean,
            out,
            dataset=name,
            tier=tier,
            split_rule=spec.split_rule,
            test_days=spec.test_days,
            repeat_policies=spec.repeat_policies,
            tier_overrides=tier_overrides,
        )
    except Exception:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(traceback.format_exc())
        raise
    marker.unlink(missing_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, clean, and split datasets.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--tier", type=str, default="")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    names = [n for n in args.datasets.split(",") if n] or list(cfg.get("datasets") or [])
    tier = args.tier or cfg.get("tier", "smoke")
    failures = []
    for name in names:
        try:
            path = prepare_one(name, tier, tier_overrides=(cfg.get("tier_overrides") or {}).get(tier))
            print(f"prepared {name} -> {path}")
        except Exception as exc:  # noqa: BLE001 - recorded in data/unavailable/
            failures.append(name)
            print(f"prepare failed for {name}: {type(exc).__name__}: {exc}")
    if failures and len(failures) == len(names):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
