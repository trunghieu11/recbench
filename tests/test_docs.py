"""The documentation stays in sync with the code: pages exist, code pointers and snippets resolve."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from recbench.registry import ensure_loaded

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
POINTER = re.compile(r"`(src/recbench/[\w/]+\.py)(?:::([\w.]+))?`")
SNIPPET = re.compile(r'^[ \t]*--8<-- "([^"]+)"\s*$', re.MULTILINE)  # also inside tabs (indented)


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
    allowed = {"generated/capability.md", "generated/metrics.md", "generated/services.md", "generated/ladder.md", "generated/glance.md"}
    allowed |= {f"generated/methods/{n}.md" for n in reg.methods} | {f"generated/methods/{n}-results.md" for n in reg.methods}
    allowed |= {f"generated/datasets/{n}.md" for n in reg.datasets}
    tiers = ("smoke", "full", "quick-tuned", "full-tuned")
    allowed |= {f"generated/leaderboards/{tier}/{n}.md" for tier in tiers for n in reg.datasets}
    from recbench.report.overall import SOURCES

    allowed |= {"generated/overall/headline.md", "generated/overall/scorecard.md"}
    allowed |= {f"generated/overall/{part}-{key}.md" for part in ("matrix", "cost", "beyond", "status") for key, *_ in SOURCES}
    bad = [f"{p.relative_to(DOCS)}: {s}" for p in _pages() for s in SNIPPET.findall(p.read_text()) if s not in allowed]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("name", sorted(ensure_loaded().methods))
def test_catalog_has_every_method(name):
    import yaml

    catalog = yaml.safe_load((ROOT / "dictionary" / "catalog.yaml").read_text())
    entry = catalog["methods"][name]
    assert set(entry["rubric"]) == {"implementation", "tuning", "data_hunger", "controllability", "explainability"}
    assert all(1 <= score <= 5 and reason for score, reason in entry["rubric"].values())


# ----- keeping the docs correct automatically -------------------------------------------------------------------------

COMMAND = re.compile(r"python -m (recbench(?:\.[a-z_]+)+)((?:[ \t]+(?:\\\n[ \t]*)?[^\s`|;&]+)*)")
SCRIPT = re.compile(r"\b((?:scripts|deploy)/[\w./-]+\.sh)\b")
INTERNAL_FLAGS = {("recbench.runner", "--single")}  # the runner's child-process entry, parsed before argparse


def _texts() -> list[tuple[str, str]]:
    """(name, text) of every hand-written page and the README."""
    found = [(str(p.relative_to(ROOT)), p.read_text()) for p in _pages()]
    return found + [("README.md", (ROOT / "README.md").read_text())]


def _help(module: str, sub: str | None) -> str:
    import subprocess
    import sys

    args = [sys.executable, "-m", module] + ([sub] if sub else []) + ["--help"]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert out.returncode == 0, f"{' '.join(args[2:])} --help failed: {out.stderr[-500:]}"
    return out.stdout


def test_documented_command_flags_exist():
    """Every `python -m recbench.<module> [run|status] --flag` in the docs is accepted by that module."""
    wanted: dict[tuple[str, str | None], set[str]] = {}
    where: dict[tuple[str, str | None, str], str] = {}
    for name, text in _texts():
        for match in COMMAND.finditer(text):
            module, rest = match.group(1), match.group(2).replace("\\\n", " ").split()
            sub = rest[0] if module == "recbench.queue" and rest and rest[0] in ("run", "status") else None
            for token in rest:
                if token.startswith("--"):
                    flag = token.split("=", 1)[0]
                    wanted.setdefault((module, sub), set()).add(flag)
                    where.setdefault((module, sub, flag), name)
    assert wanted, "no documented commands found"
    unknown = []
    for (module, sub), flags in sorted(wanted.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        text = _help(module, sub)
        unknown += [f"{where[(module, sub, f)]}: python -m {module} {sub or ''} {f}" for f in sorted(flags)
                    if f not in text and (module, f) not in INTERNAL_FLAGS]
    assert not unknown, "flags the code does not accept:\n" + "\n".join(unknown)


def test_documented_scripts_exist():
    missing = sorted({f"{name}: {path}" for name, text in _texts() for path in SCRIPT.findall(text) if not (ROOT / path).exists()})
    assert not missing, "\n".join(missing)


def _counts() -> dict[str, set[int]]:
    """Counts the docs may state, all derived from the code and configs."""
    import yaml

    reg = ensure_loaded()
    quick = yaml.safe_load((ROOT / "configs" / "benchmarks" / "quick.yaml").read_text())
    queued = quick["queue"]["methods"]
    smoke = yaml.safe_load((ROOT / "configs" / "benchmarks" / "smoke-cpu.yaml").read_text())
    full = yaml.safe_load((ROOT / "configs" / "benchmarks" / "gpu-full.yaml").read_text())
    methods = {
        len(reg.methods),                                                   # registered
        sum(not c.spec.managed for c in reg.methods.values()),              # local
        len(queued),                                                        # in the bake-off
        sum(e.get("resource", "cpu") == "cpu" for e in queued),             # bake-off on CPU
        sum(e.get("resource") == "gpu" for e in queued),                    # bake-off on GPU
        len(quick.get("held_back") or []),                                  # held back
        len(smoke["methods"]), len(full["methods"]),                        # the older configs
    }
    return {"methods": methods, "datasets": {len(reg.datasets)}}


def test_method_and_dataset_counts_match_the_code():
    """A stated number of methods (10 or more) or datasets must be one the code produces. Smaller numbers are
    usually examples ("the top 3 methods"); the review log is historical."""
    allowed = _counts()
    qualifiers = "low-budget|recommendation|public|registered|original|newer|new|local|ranked|tuned|held-back"
    claim = re.compile(rf"\b(\d+) (?:(?:{qualifiers}) )?(methods|datasets)\b")
    wrong = []
    for name, text in _texts():
        if name.startswith("docs/review/"):
            continue
        for number, kind in claim.findall(text):
            n = int(number)
            if (kind == "methods" and n >= 10 and n not in allowed["methods"]) or (kind == "datasets" and n not in allowed["datasets"]):
                wrong.append(f"{name}: {number} {kind} (the code has {sorted(allowed[kind])})")
    assert not wrong, "\n".join(wrong)


TEMPLATE = list(range(1, 13))


@pytest.mark.parametrize("name", sorted(ensure_loaded().methods))
def test_method_pages_follow_the_template(name):
    import yaml

    text = (DOCS / "dictionary" / "algorithms" / f"{name.replace('_', '-')}.md").read_text()
    assert [int(n) for n in re.findall(r"^## (\d+)\. ", text, re.M)] == TEMPLATE, "sections 1-12, in order"
    queued = {e["name"] for e in yaml.safe_load((ROOT / "configs" / "benchmarks" / "quick.yaml").read_text())["queue"]["methods"]}
    if name in queued:
        assert '!!! abstract "In plain words"' in text, "bake-off methods start with an 'In plain words' box"


def test_readme_method_table_is_current():
    from recbench.dictionary.build import README_END, README_START, load_catalog, quick_methods, readme_block

    text = (ROOT / "README.md").read_text()
    current = text[text.index(README_START): text.index(README_END) + len(README_END)]
    assert current == readme_block(load_catalog(ROOT), ensure_loaded(), quick_methods(ROOT)), \
        "run python -m recbench.dictionary.build"


def test_code_and_catalog_agree_on_fidelity():
    import yaml

    catalog = yaml.safe_load((ROOT / "dictionary" / "catalog.yaml").read_text())["methods"]
    wrong = [f"{n}: code says {c.spec.fidelity}, catalog says {catalog[n]['fidelity']}"
             for n, c in ensure_loaded().methods.items() if c.spec.fidelity != catalog[n]["fidelity"]]
    assert not wrong, wrong


def test_every_method_is_in_the_bake_off_or_held_back():
    """The gate: a local method is queued in quick.yaml with a search space, or listed under held_back."""
    import yaml

    from recbench.tuning.spaces import load_spaces

    quick = yaml.safe_load((ROOT / "configs" / "benchmarks" / "quick.yaml").read_text())
    queued = [e["name"] for e in quick["queue"]["methods"]]
    held = set(quick.get("held_back") or [])
    spaces, _ = load_spaces(ROOT / "configs" / "tuning" / "quick.yaml")
    reg = ensure_loaded()
    local = {n for n, c in reg.methods.items() if not c.spec.managed}
    assert not (set(queued) & held), "a method cannot be both queued and held back"
    assert sorted(local - set(queued) - held) == [], "add each new method to queue.methods or held_back in quick.yaml"
    assert sorted(set(queued) - set(spaces)) == [], "every queued method needs a search space in configs/tuning/quick.yaml"
    assert sorted((set(queued) | held) - set(reg.methods)) == [], "quick.yaml names a method that is not registered"


def test_docs_build_strictly():
    import shutil
    import subprocess
    import sys
    import tempfile

    if shutil.which("mkdocs") is None and not (Path(sys.executable).parent / "mkdocs").exists():
        pytest.skip("mkdocs is not installed (the docs extra)")
    mkdocs = str(Path(sys.executable).parent / "mkdocs") if (Path(sys.executable).parent / "mkdocs").exists() else "mkdocs"
    with tempfile.TemporaryDirectory() as site:
        out = subprocess.run([mkdocs, "build", "--strict", "--site-dir", site], capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
