"""The documentation stays in sync with the code: pages exist, code pointers and snippets resolve."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from recbench.registry import ensure_loaded

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
POINTER = re.compile(r"`(src/recbench/[\w/]+\.py)(?:::([\w.]+))?`")
SNIPPET = re.compile(r'^--8<-- "([^"]+)"\s*$', re.MULTILINE)


def _pages() -> list[Path]:
    return [p for p in DOCS.rglob("*.md") if "generated" not in p.parts]


def test_every_method_and_dataset_has_a_page():
    reg = ensure_loaded()
    missing = [f"algorithms/{n.replace('_', '-')}.md" for n in reg.methods
               if not (DOCS / "dictionary" / "algorithms" / f"{n.replace('_', '-')}.md").exists()]
    missing += [f"datasets/{n}.md" for n in reg.datasets if not (DOCS / "dictionary" / "datasets" / f"{n}.md").exists()]
    assert not missing, missing


def test_code_pointers_resolve():
    broken = []
    for page in _pages():
        for path, symbol in POINTER.findall(page.read_text()):
            source = ROOT / path
            if not source.exists():
                broken.append(f"{page.relative_to(DOCS)}: {path}")
                continue
            if symbol:
                text = source.read_text()
                for part in symbol.split("."):
                    if not re.search(rf"^\s*(class|def)\s+{re.escape(part)}\b|^{re.escape(part)}\s*=", text, re.MULTILINE):
                        broken.append(f"{page.relative_to(DOCS)}: {path}::{symbol}")
                        break
    assert not broken, "\n".join(broken)


def test_snippets_point_at_generated_fragments():
    reg = ensure_loaded()
    allowed = {"generated/capability.md", "generated/metrics.md", "generated/services.md"}
    allowed |= {f"generated/methods/{n}.md" for n in reg.methods} | {f"generated/methods/{n}-results.md" for n in reg.methods}
    allowed |= {f"generated/datasets/{n}.md" for n in reg.datasets}
    allowed |= {f"generated/leaderboards/{tier}/{n}.md" for tier in ("smoke", "full") for n in reg.datasets}
    bad = [f"{p.relative_to(DOCS)}: {s}" for p in _pages() for s in SNIPPET.findall(p.read_text()) if s not in allowed]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("name", sorted(ensure_loaded().methods))
def test_catalog_has_every_method(name):
    import yaml

    catalog = yaml.safe_load((ROOT / "dictionary" / "catalog.yaml").read_text())
    entry = catalog["methods"][name]
    assert set(entry["rubric"]) == {"implementation", "tuning", "data_hunger", "controllability", "explainability"}
    assert all(1 <= score <= 5 and reason for score, reason in entry["rubric"].values())
