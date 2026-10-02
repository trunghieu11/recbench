# Registry and catalog

Two mechanisms make recbench extensible: a **registry** of classes (code) and a **catalog** of facts (YAML).

## The registry

`src/recbench/registry.py` holds three dictionaries (`methods`, `metrics`, `datasets`) filled by decorators:

```python
from recbench.registry import register_method

@register_method
class EASE(Recommender):
    spec = MethodSpec(name="ease", ...)
```

Decorators run when the module is **imported**. `src/recbench/registry.py::ensure_loaded` imports the three
packages, whose `__init__.py` files import every module:

| Package | Registers |
|---|---|
| `src/recbench/methods/__init__.py` | all methods |
| `src/recbench/metrics/__init__.py` | all metrics (through `catalog.py`) |
| `src/recbench/datasets/__init__.py` | all dataset adapters |

Everything else (runner, evaluator, reports, docs generator) iterates over the registry, so adding a class and
importing it is enough for it to appear everywhere.

```python
from recbench.registry import ensure_loaded
reg = ensure_loaded()
print(sorted(reg.methods))      # ['bert4rec', 'bpr_mf', 'dcnv2', ...]
method = reg.create_method("ease")
```

## The catalog

`dictionary/catalog.yaml` holds facts that are not code:

| Section | Per entry |
|---|---|
| `ladder` | rung number → name (0 Baselines ... 7 Managed services) |
| `methods` | title, family, rung, year, paper {title, venue, url}, code, fidelity, summary, rubric |
| `services` | documented-only managed services: title, status, why not benchmarked, url |
| `datasets` | title, domain, source, citation, licence, commercial use, feedback, side info, pitfalls |

A rubric entry is `dimension: [score, "reason"]` for the five dimensions; see the
[qualitative rubric](../dictionary/metrics/qualitative-rubric.md).

## From catalog and registry to docs

`python -m recbench.dictionary.build` (`src/recbench/dictionary/build.py::build`) writes `docs/generated/`:

| File | Built from |
|---|---|
| `methods/<name>.md` | catalog entry + the method's `MethodSpec` (facts and rubric) |
| `methods/<name>-results.md` | the latest MLflow runs of that method (smoke and full) |
| `capability.md` | all methods' specs |
| `metrics.md` | all registered metrics |
| `datasets/<name>.md` | catalog entry + `meta.json` of every split on this machine |
| `services.md` | the `services` section |

The report builder adds `leaderboards/<tier>/<dataset>.md` (with `--docs`). Hand-written pages embed these
fragments with `--8<-- "generated/methods/ease.md"`.

The build **fails** (`CatalogError`) if a registered method or dataset has no catalog entry, and
`tests/test_docs.py` fails if a method has no docs page or a docs page references code that does not exist. The
docs cannot silently fall out of date.
