# Explainability

## Why it matters

"Because you watched *Up*" makes users trust recommendations, helps engineers debug them, and is sometimes
legally required. Some models can explain themselves exactly; others can only be explained after the fact;
managed services usually give no reason at all.

## Intuition: three levels

| Level | Meaning | Example |
|---|---|---|
| **Exact** | the explanation *is* the model's computation | ItemKNN: score = sum of similarities to your items, so the largest terms are the reason |
| **Post-hoc** | a plausible story computed next to the model | "this is close (in embedding space) to items you liked" for a matrix factorisation model |
| **None** | the system returns only a ranking | most managed services |

Exact explanations are faithful by construction. Post-hoc ones can be wrong: the model may have ranked the
item highly for a different reason than the one cited.

## A small example

User history: *Toy Story*, *Up*. Recommended: *Inside Out*.

- **ItemKNN (exact):** sim(Toy Story, Inside Out) = 0.6, sim(Up, Inside Out) = 0.3, so the score is 0.9, and the
  explanation "because you watched Toy Story (0.6) and Up (0.3)" is literally the score.
- **BPR-MF (post-hoc):** the score is a dot product of latent vectors. recbench reports the history items whose
  vectors are closest to *Inside Out* (cosine). This is informative, but not the model's actual computation.
- **MostPopular:** "popular now: 312 interactions in the last 28 days". Honest, but not personal.

## In recbench

- Every method can return `Explanation` objects (`src/recbench/protocol.py::Explanation`) with a `kind`, a
  `text`, and `evidence`. An explanation is **personal** when its evidence cites one of the user's own history items.
- Exact explanations: `src/recbench/methods/_explain.py::contribution_explanations` (ItemKNN, EASE, RP3beta, SLIM:
  every score is a sum over the user's history items, so each item's share is the reason).
- The re-rankers cite a co-visit with the user's last item when there is one; text kNN cites the history items with
  the most similar text.
- Post-hoc: `src/recbench/methods/_explain.py::embedding_explanations` (embedding models). TIGER-lite cites
  shared semantic-ID prefixes.
- Metric: `personal_explanation_rate`, the share of sampled recommendations (top 3 for 50 users) with a
  personal explanation. Example explanations are saved with every MLflow run (`explanations.json`).
- The qualitative rubric scores explainability from 1 to 5 per method, with a reason
  ([qualitative rubric](../metrics/qualitative-rubric.md)).

## Pitfalls

- **Equating "has an explanation" with "is explainable".** A post-hoc story can be convincing and wrong.
- **Fixed templates.** recbench v0.1 produced the same sentence for every recommendation, which made its
  explanation metric meaningless. v0.2 requires evidence from the user's own history.
- **Explaining with private data:** "because your partner bought X" can leak information.

## Check your understanding

??? question "Why is an EASE explanation exact but an iALS explanation post-hoc?"
    EASE's score is a sum of weights over history items, so citing the largest weights reproduces the score.
    iALS's score is a dot product of latent factors; citing similar history items is an interpretation added afterwards.

??? question "Why does MostPopular get a personal-explanation rate of 0?"
    Its reason cites popularity, not the user's history.

## Further reading

- Zhang & Chen (2020), [Explainable Recommendation: A Survey and New Perspectives](https://arxiv.org/abs/1804.11192).
- [ItemKNN](../algorithms/itemknn.md) and [EASE](../algorithms/ease.md).
