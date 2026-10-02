"""Build the static dictionary from YAML records and the live registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from recbench.protocol import PROTOCOL_NOTE
from recbench.registry import ensure_loaded

OMITTED = """
# Methods left off the runner

These stay out of the leaderboard. Popularity is the exception: its Recall@10 is logged on the same candidate file and tagged so it is not ranked.

- Item-kNN
- Matrix factorization
- Content-based filtering
- LightFM
- DeepFM
- LightGCN
- SASRec
- A generic two-tower without images
- A cross-encoder-only reranker

Adding one later is a class, a `@register_method` decorator, and a block in `dictionary/catalog.yaml`.
"""

HOWTO = """
# Add a metric, a method, or a dataset

## Metric

```python
from recbench.protocol import Metric, MetricSpec, Task
from recbench.registry import register_metric

@register_metric
class Example(Metric):
    spec = MetricSpec("example_at_10", {Task.topn}, description="What the number means.")

    def compute(self, scores, store, context):
        return 0.0
```

## Method

```python
from recbench.protocol import MethodSpec, Task
from recbench.registry import register_method
from recbench.methods._common import TorchMethod

@register_method
class Example(TorchMethod):
    spec = MethodSpec(name="example", tasks={Task.topn}, feedback={"implicit"})
```

Then add an `example:` block under `methods` in `dictionary/catalog.yaml` with the complexity rubric and a prose paragraph.

## Dataset

Subclass nothing. Implement `download(raw_dir)` and `to_clean(raw_dir, clean_dir)` and decorate the class with `@register_dataset`. The clean tables must match the feature contract: interactions, items, users. Add a `datasets` block in the catalog.
"""


def _load_catalog(root: Path) -> dict[str, Any]:
    path = root / "dictionary" / "catalog.yaml"
    with path.open() as handle:
        return yaml.safe_load(handle)


def _mean_complexity(block: dict[str, Any]) -> float | None:
    scores = block.get("complexity") or {}
    if not scores:
        return None
    values = [float(value) for value in scores.values()]
    return sum(values) / len(values)


def build(root: Path | None = None) -> Path:
    root = root or Path.cwd()
    catalog = _load_catalog(root)
    reg = ensure_loaded()
    docs = root / "site" / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "index.md").write_text(
        "# Recommendation dictionary\n\n"
        f"{PROTOCOL_NOTE}\n\n"
        "Method rank comes from the full GPU tier, one task at a time. "
        "The laptop smoke run is a wiring test.\n\n"
        "See [methods](methods.md), [datasets](datasets.md), [metrics](metrics.md), "
        "[decision](decision.md), and [how to extend](how-to-extend.md).\n"
    )
    (docs / "omitted.md").write_text(OMITTED)
    (docs / "how-to-extend.md").write_text(HOWTO)
    method_lines = ["# Methods\n", PROTOCOL_NOTE, ""]
    decision = ["# Decision page\n", "Filter by cost, images, and task. Smoke runs are not a ranking.\n"]
    for name, block in (catalog.get("methods") or {}).items():
        page = docs / "methods" / f"{name}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        mean = _mean_complexity(block)
        tasks = ", ".join(block.get("tasks") or [])
        body = [
            f"# {block.get('title', name)}",
            "",
            PROTOCOL_NOTE,
            "",
            block.get("summary", ""),
            "",
            f"- Tasks: {tasks}",
            f"- Cost band: {block.get('cost_band', '')}",
            f"- Images required: {block.get('requires_images', False)}",
            f"- Complexity mean: {mean:.2f}" if mean is not None else "- Complexity mean: n/a",
            f"- Explainability rubric: {block.get('explainability_rubric', '')}",
            f"- Upstream: {block.get('upstream', '')}",
            "",
            block.get("prose", ""),
            "",
        ]
        page.write_text("\n".join(body))
        method_lines.append(f"- [{block.get('title', name)}](methods/{name}.md) — {block.get('summary', '')}")
        decision.append(
            f"- **{name}** cost={block.get('cost_band')} images={block.get('requires_images')} tasks={tasks}"
        )
    (docs / "methods.md").write_text("\n".join(method_lines) + "\n")
    (docs / "decision.md").write_text("\n".join(decision) + "\n")
    dataset_lines = ["# Datasets\n", PROTOCOL_NOTE, ""]
    for name, block in (catalog.get("datasets") or {}).items():
        page = docs / "datasets" / f"{name}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(
            f"# {block.get('title', name)}\n\n{PROTOCOL_NOTE}\n\n"
            f"Domain: {block.get('domain', '')}\n\n{block.get('summary', '')}\n\n{block.get('prose', '')}\n"
        )
        dataset_lines.append(f"- [{block.get('title', name)}](datasets/{name}.md) — {block.get('summary', '')}")
    (docs / "datasets.md").write_text("\n".join(dataset_lines) + "\n")
    metric_lines = ["# Metrics\n", PROTOCOL_NOTE, ""]
    matrix = ["# Capability matrix\n", PROTOCOL_NOTE, ""]
    task_names = ["topn", "rating", "ctr", "sequential", "session", "similar_items"]
    for method_name, cls in sorted(reg.methods.items()):
        spec = cls.spec
        flags = ", ".join(task for task in task_names if any(t.value == task for t in spec.tasks))
        matrix.append(f"- {method_name}: {flags}; images={spec.requires_images}; managed={spec.managed}")
    (docs / "capability.md").write_text("\n".join(matrix) + "\n")
    for metric_cls in reg.metrics.values():
        spec = metric_cls.spec
        tasks = ", ".join(task.value for task in spec.tasks)
        leaderboard = "ranked" if spec.leaderboard else "not ranked"
        metric_lines.append(f"- **{spec.name}** ({tasks}, {leaderboard}): {spec.description}")
    (docs / "metrics.md").write_text("\n".join(metric_lines) + "\n")
    mkdocs = root / "site" / "mkdocs.yml"
    mkdocs.write_text(
        "site_name: Recommendation dictionary\n"
        "docs_dir: docs\n"
        "theme:\n"
        "  name: material\n"
        "nav:\n"
        "  - Home: index.md\n"
        "  - Methods: methods.md\n"
        "  - Datasets: datasets.md\n"
        "  - Metrics: metrics.md\n"
        "  - Capability: capability.md\n"
        "  - Decision: decision.md\n"
        "  - Omitted: omitted.md\n"
        "  - Extend: how-to-extend.md\n"
    )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the dictionary site.")
    parser.add_argument("--tracking-uri", default="")
    args = parser.parse_args()
    if args.tracking_uri:
        import os

        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    print(build(Path.cwd()))


if __name__ == "__main__":
    main()
