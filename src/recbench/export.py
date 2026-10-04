"""Export a serving bundle for one method, with the settings the quick-tier bake-off chose for it.

    python -m recbench.export --dataset hm --method ease --from-confirm
    python -m recbench.export --dataset hm --method ease --tier full --config configs/benchmarks/quick.yaml

The confirmation step writes bundles itself, in its first final run. Use this command when that run had already
finished in an earlier session (finished runs are never repeated, and `export_bundles` is not part of a run's
identity), after a code change, or to serve a method that was not confirmed. It takes the settings from:

1. the method's confirmation summary (runs/tuning/<tier>/<dataset>/<method>.confirm.json) with --from-confirm, else
2. its quick-tier tuning summary (runs/tuning/<quick tier>/<dataset>/<method>.json), else
3. the method's defaults.

It fits the method on data/splits/<dataset>/<tier> exactly as a benchmark run does (same training window, same
epoch count) and writes data/bundles/<dataset>/<tier>/<method>/. Nothing is evaluated or logged to MLflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recbench.config import method_config
from recbench.data import TrainView
from recbench.registry import ensure_loaded
from recbench.runner import _seed_everything, data_root, run_hash_for
from recbench.tuning.job import _base_cfg, read_summary


def chosen_settings(dataset: str, method: str, *, tier: str, quick_tier: str, from_confirm: bool) -> tuple[dict[str, Any], int | None, str]:
    """(settings, epoch count or None, where they came from)."""
    if from_confirm:
        confirm = read_summary(tier, dataset, method, "confirm")
        if confirm and confirm.get("best_params"):
            return dict(confirm["best_params"]), confirm.get("best_epoch"), f"confirmation on {tier}"
    tuned = read_summary(quick_tier, dataset, method)
    if tuned and tuned.get("best_params"):
        return dict(tuned["best_params"]), tuned.get("best_epoch"), f"{quick_tier}-tier tuning"
    return {}, None, "defaults"


def export(dataset: str, method: str, *, config: Path, tier: str = "full", from_confirm: bool = True,
           hardware: str | None = None) -> Path:
    from recbench.serving.bundle import export_bundle
    from recbench.tuning.__main__ import load_benchmark

    _, resolved, _, settings = load_benchmark(config, hardware)
    params, epochs, source = chosen_settings(dataset, method, tier=tier, quick_tier=settings.tier, from_confirm=from_confirm)
    run_cfg = {**_base_cfg(resolved, method, set(params)), **params, "tuning": "tuned" if params else "defaults", "stage": "export"}
    if epochs:
        run_cfg["epochs"] = int(epochs)
    data = TrainView(data_root() / "splits" / dataset / tier)
    model = ensure_loaded().create_method(method)
    cfg = method_config(run_cfg, method)
    _seed_everything(int(cfg.get("seed", 42)))
    model.fit(data.restrict(cfg.get("train_window_days"), int(cfg.get("train_window_keep_last", 10))), cfg)
    out = data_root() / "bundles" / dataset / tier / method
    extra = {"config_hash": run_hash_for(run_cfg, method, data.split_hash), "stage": "export", "settings_from": source,
             "settings": {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool)) or v is None}}
    return export_bundle(model, data, out, k=int(cfg.get("bundle_k", 100)), max_users=int(cfg.get("bundle_users", 20_000)), extra=extra)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a serving bundle with the bake-off's chosen settings.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--tier", default="full", help="split to fit on and bundle tier to write (default: full)")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks/quick.yaml"))
    parser.add_argument("--hardware", default=None, help="override the config's hardware profile")
    parser.add_argument("--from-confirm", action="store_true", help="use the confirmation's settings when they exist")
    args = parser.parse_args()
    out = export(args.dataset, args.method, config=args.config, tier=args.tier, from_confirm=args.from_confirm, hardware=args.hardware)
    manifest = json.loads((out / "manifest.json").read_text())
    print(json.dumps({"bundle": str(out), **{k: manifest.get(k) for k in ("settings_from", "n_users", "k", "data_cutoff")}}, indent=2))


if __name__ == "__main__":
    main()
