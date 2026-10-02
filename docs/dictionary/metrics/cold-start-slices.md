# Cold-start slices

Averages over all users hide where a method fails. recbench reports two *slices* focused on
[cold start](../concepts/cold-start.md).

## Cold-item recall

**Question:** when a user's relevant items include brand-new items (no interactions before the cutoff), how
many of those does the method find?

$$
\text{ColdItemRecall@10}(u) = \frac{|L_u^{(10)} \cap R_u^{\text{cold}}|}{\min(10, |R_u^{\text{cold}}|)}
$$

| Symbol | Meaning |
|---|---|
| $R_u^{\text{cold}}$ | user $u$'s relevant items that had no pre-test interactions |

It is averaged only over users who have at least one relevant cold item. Example: a user's relevant items
include 2 new items, and the top 10 contains 1 of them, so the score is 1/2 = 0.5.

What to expect: pure ID methods score **0**, because the evaluator removes cold items from their rankings (they
cannot score them). Content methods ([text hash tower](../algorithms/text-hash-tower.md),
[multimodal tower](../algorithms/multimodal-tower.md)) can score above 0. Code:
`src/recbench/metrics/catalog.py::_item_cold_recall`.

## Cold-user metrics

**Question:** how well does a method serve users who had **no** history before the cutoff?

Only methods that declare `handles_cold_users=True` (MostPopular, Random) are evaluated on these users. Every
other method would fall back to popularity anyway. They are reported as `cold_users/ndcg_at_10`,
`cold_users/recall_at_10`, and `cold_users/n_users`, with all of a cold user's test items counted as relevant.
Code: `src/recbench/evaluation.py::Evaluator._cold_users`.

## Why the primary population is warm users

Mixing warm and cold users in one average mostly measures the **share** of cold users in a dataset, not model
quality. RetailRocket, for example, has many one-visit users. Keeping them separate makes each number interpretable.

## Check your understanding

??? question "Why is EASE's cold-item recall always 0 in recbench?"
    EASE cannot score items without pre-test interactions, so the evaluator removes them from its rankings.

??? question "Why does recbench not report cold-user NDCG for SASRec?"
    SASRec needs a history; for a cold user it would return the popularity fallback, the same as MostPopular.

## Further reading

- [Cold start](../concepts/cold-start.md) and [evaluation protocols](../concepts/evaluation-protocols.md).
