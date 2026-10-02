# Qualitative rubric

Some important properties cannot be computed from a log file: how hard a method is to implement, tune,
control, or explain. recbench scores them on a **1–5 scale with written anchors**, so two people scoring the
same method get similar results. Each score lives in `dictionary/catalog.yaml` together with a one-line reason,
and the generated facts box on every algorithm page shows them.

!!! warning "Directions differ"
    For **implementation**, **tuning**, and **data hunger**, *higher means more work or more data needed*
    (worse). For **controllability** and **explainability**, *higher is better*.

## Implementation effort (higher = more work)

| Score | Anchor |
|---|---|
| 1 | a few lines with standard libraries (a count, a sort) |
| 2 | one library call or a short closed-form computation |
| 3 | a training loop, or an adapter around a library model |
| 4 | several components or stages (pre-training + fine-tuning, codebooks, custom attention) |
| 5 | custom kernels, distributed training, or a multi-team system |

Automated proxies: lines of code in the method file; number of dependencies; whether it needs a GPU.

## Tuning effort (higher = more work)

| Score | Anchor |
|---|---|
| 1 | nothing, or one robust knob |
| 2 | 2–3 knobs with good defaults |
| 3 | 4–6 knobs; defaults usually reasonable |
| 4 | many interacting knobs; sensitive to the training length |
| 5 | extensive search needed; results unstable without it |

Automated proxies: number of hyperparameters read by the method (`cfg.get(...)` calls); the spread of results
across settings.

## Data hunger (higher = needs more data)

| Score | Anchor |
|---|---|
| 1 | works with no or very little data |
| 2 | works on small, sparse data |
| 3 | needs several interactions per user and item |
| 4 | needs many ordered sequences or labelled examples |
| 5 | shines only at very large scale |

## Controllability (higher = easier to steer)

| Score | Anchor |
|---|---|
| 1 | black box; only post-filtering possible |
| 2 | latent factors; hard to steer beyond filtering |
| 3 | business features can be added as model inputs |
| 4 | inspectable structure (similarity lists, weights) or rule support |
| 5 | fully rule-based or directly editable |

## Explainability (higher = clearer reasons)

| Score | Anchor |
|---|---|
| 1 | no reason available |
| 2 | post-hoc reasons only (embedding similarity) |
| 3 | partial or coarse reasons (popularity, attention, shared codes) |
| 4 | mostly faithful reasons |
| 5 | exact: the reason *is* the computation |

Automated proxy: `personal_explanation_rate` (see [explainability](../concepts/explainability.md)).

## Changing a score

Edit the method's `rubric` block in `dictionary/catalog.yaml`. Keep the format `[score, "reason"]`, and give a
reason that cites the anchor. Then run `python -m recbench.dictionary.build` to refresh the docs.
`tests/test_docs.py` checks that every method has all five scores between 1 and 5, each with a reason.

## Check your understanding

??? question "Why is EASE's implementation score 2 rather than 1?"
    The computation is short, but it needs a dense item × item matrix, a matrix inverse, and a catalog cap to fit
    in memory, which is more than a count and a sort.

??? question "Which dimension would you weight highest for a regulated industry?"
    Explainability, and controllability next: you must be able to justify and steer recommendations.
