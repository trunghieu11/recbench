"""Land a dataset and materialize one tier. A failure is the caller's to skip."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from recbench.config import load_yaml
from recbench.registry import ensure_loaded


def data_root() -> Path:
    import os

    return Path(os.environ.get("DATA_DIR", Path.cwd() / "data"))


def prepare_one(name: str, tier: str, root: Path | None = None) -> Path:
    reg = ensure_loaded()
    dataset = reg.create_dataset(name)
    root = root or data_root()
    raw = root / "raw" / name
    clean = root / "clean" / name
    out = root / "splits" / name / tier
    raw.mkdir(parents=True, exist_ok=True)
    try:
        dataset.download(raw)
        dataset.to_clean(raw, clean)
        split_mode = dataset.spec.official_split
        from recbench.pipeline.materialize import materialize

        materialize(clean, out, dataset=name, tier=tier, split_mode=split_mode)
        if name == "hm":
            from recbench.datasets.hm import attach_slice_images

            attach_slice_images(out, raw, tier)
    except Exception:
        marker = root / "unavailable" / f"{name}.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(traceback.format_exc())
        raise
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and split one or more datasets.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", type=str, default="")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    names = [n for n in args.datasets.split(",") if n] or list(cfg.get("datasets") or [])
    tier = cfg.get("tier", "smoke")
    failures = []
    for name in names:
        try:
            path = prepare_one(name, tier)
            print(f"prepared {name} -> {path}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"prepare failed for {name}: {exc}")
    if failures and len(failures) == len(names):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
