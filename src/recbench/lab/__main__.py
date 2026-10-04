"""The improvement lab: compute baselines, try settings, run experiments, and keep score.

    python -m recbench.lab baseline                       # the 11 lab methods (+ MostPopular) on all five datasets
    python -m recbench.lab status
    python -m recbench.lab once --method itemknn --dataset movielens-25m --set knn_neighbors=200
    python -m recbench.lab sweep --method itemknn --dataset movielens-25m --param knn_neighbors --values 10,50,200,1000
    python -m recbench.lab run --method itemknn --experiment no-time-knobs
    python -m recbench.lab scoreboard --docs

Everything is written under runs/lab/ and reports/lab/, apart from the bake-off's results. `once` and `sweep` use the
validation fold only; `run` tunes on the validation fold and tests the chosen setting once, like the baseline.
Compare results with python -m recbench.compare. The guide: docs/handbook/index.md.
"""

from __future__ import annotations

import argparse
import sys

from recbench.lab import experiments as ex
from recbench.lab import runs


def _list(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def _datasets(text: str) -> list[str] | None:
    return None if text in ("", "all") else _list(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m recbench.lab", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    base = sub.add_parser("baseline", help="tune every lab method on every dataset (the numbers to beat)")
    base.add_argument("--datasets", default="", help="comma-separated, default all five")
    base.add_argument("--methods", default="", help="comma-separated, default all of configs/benchmarks/lab.yaml")
    base.add_argument("--cpu-workers", type=int, default=None, help="jobs at a time (default 2)")
    base.add_argument("--stop-after-dataset", action="store_true")
    base.add_argument("--retry-failed", action="store_true")

    sub.add_parser("status", help="the baseline's progress and every experiment's results")

    once = sub.add_parser("once", help="score one setting on the validation fold")
    once.add_argument("--method", required=True)
    once.add_argument("--dataset", required=True)
    once.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a setting (repeat for several)")
    once.add_argument("--best", action="store_true", help="start from the baseline's best setting on this dataset")
    once.add_argument("--users", type=int, default=None, help="validation users to score (default 3000, as in tuning)")
    once.add_argument("--fresh", action="store_true", help="compute again even if the same run is stored")

    sweep = sub.add_parser("sweep", help="vary one setting, everything else at the baseline's best (validation fold)")
    sweep.add_argument("--method", required=True)
    sweep.add_argument("--dataset", required=True)
    sweep.add_argument("--param", required=True)
    sweep.add_argument("--values", required=True, help="comma-separated, for example 10,50,200 or null,30,90")
    sweep.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    sweep.add_argument("--from-defaults", action="store_true", help="start from the method's defaults, not the baseline's best")
    sweep.add_argument("--users", type=int, default=None)

    run = sub.add_parser("run", help="run an experiment from labs/<nn>-<method>/experiments.yaml")
    run.add_argument("--method", required=True)
    run.add_argument("--experiment", required=True)
    run.add_argument("--datasets", default="", help="comma-separated, default all five")
    run.add_argument("--workers", type=int, default=2, help="datasets at a time (default 2)")
    run.add_argument("--rerun", action="store_true", help="run again even if done")

    board = sub.add_parser("scoreboard", help="rebuild the scoreboard (reports/lab/scoreboard.md)")
    board.add_argument("--docs", action="store_true", help="also write docs/generated/lab/*.md")

    args = parser.parse_args(argv)
    try:
        if args.command == "baseline":
            from recbench.queue import Queue, print_status
            from recbench.paths import repo_root

            config = repo_root() / ex.LAB_CONFIG
            Queue(config, datasets=_list(args.datasets) or None, methods=_list(args.methods) or None, cpu_workers=args.cpu_workers,
                  stop_after_dataset=args.stop_after_dataset, retry_failed=args.retry_failed).run()
            print_status(config)
        elif args.command == "status":
            runs.status()
        elif args.command == "once":
            result = runs.once(args.method, args.dataset, runs.parse_settings(args.set), best=args.best, users=args.users, fresh=args.fresh)
            _print_once(args.method, args.dataset, result, best=args.best)
        elif args.command == "sweep":
            values = [runs.parse_value(v) for v in _list(args.values)]
            print(f"{args.method} on {args.dataset} (validation fold): {args.param} = {values}")
            runs.sweep(args.method, args.dataset, args.param, values, settings=runs.parse_settings(args.set),
                       start="defaults" if args.from_defaults else "best", users=args.users)
        elif args.command == "run":
            runs.run_experiment(args.method, args.experiment, _datasets(args.datasets), workers=args.workers, rerun=args.rerun)
        elif args.command == "scoreboard":
            from recbench.lab.scoreboard import build
            from recbench.paths import repo_root

            fragments = build(docs_dir=repo_root() / "docs" if args.docs else None)
            print(fragments["scoreboard"])
    except (ex.ExperimentError, FileNotFoundError) as exc:
        sys.exit(f"error: {exc}")


def _print_once(method: str, dataset: str, result: dict, *, best: bool) -> None:
    m = result.get("metrics") or {}
    if result["status"] not in ("finished", "cached"):
        print(f"{method} on {dataset}: {result['status']} {result.get('reason', '')}")
        return
    settings = ", ".join(f"{k}={v}" for k, v in sorted(result["settings"].items())) or "defaults"
    print(f"{method} on {dataset}, validation fold ({int(m.get('n_eval_users', 0))} users), {settings}")
    print(f"  NDCG@10   {m.get('ndcg_at_10', float('nan')):.4f}  [{m.get('ndcg_at_10_ci_low', float('nan')):.4f}, "
          f"{m.get('ndcg_at_10_ci_high', float('nan')):.4f}]")
    print(f"  Recall@10 {m.get('recall_at_10', float('nan')):.4f}   coverage@10 {m.get('coverage_at_10', float('nan')):.3f}   "
          f"train {m.get('train_seconds', float('nan')):.1f} s   scoring {m.get('score_seconds_per_1k_users', float('nan')):.2f} s per 1,000 users")
    if best:
        found = runs.summary(method, dataset) or {}
        if found.get("best_val") is not None:
            print(f"  (the baseline's best trial scored {found['best_val']:.4f} on the same users)")
    if result["status"] == "cached":
        print("  (stored result of an identical run; --fresh computes it again)")


if __name__ == "__main__":
    main()
