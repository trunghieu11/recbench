# Archived run scripts

These scripts and `configs/benchmarks/archive/gpu-12h.yaml` produced the **untuned v0.2 full-tier results** on a
rented GPU box on 2026-10-03/04 (the numbers on `docs/results/leaderboards.md`). They are kept so those results can
be traced and repeated, not for new runs:

- `run_12h.sh`: a 12-hour breadth pass, cheapest methods first, every pair capped by `timeout_minutes`.
- `run_queue.sh`: an ordered job list with a wall-clock deadline (`runs/BREADTH_DEADLINE`).

They assume the box they ran on (for example its core count) and run every method once with default settings.
New runs use the quick-tier queue instead: `python -m recbench.queue` and `scripts/run_quick_box.sh`
(see `docs/start/quick-tier-box.md`).
