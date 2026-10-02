# Cold start

## Why it matters

Every product has new users and new items every day. A recommender that only works for people and products
with long histories fails exactly where first impressions are made.

## Intuition

- **Cold user:** someone who just signed up. There is no history to personalise from.
- **Cold item:** a product added today. Nobody has interacted with it, so collaborative models know
  nothing about it.
- **Warm** is the opposite: enough history to work with.

## A small example

| Case | MostPopular | EASE / matrix factorisation | Text hash tower |
|---|---|---|---|
| Cold user, warm items | works (same list for everyone) | no history → no personal scores | needs at least one history item |
| Warm user, cold item | the item has 0 recent interactions → never recommended | no co-occurrence or trained embedding → cannot score it | scores it from its text |

## Strategies

- **Cold users:** popularity (globally, or by region or entry page), onboarding questions, session-based
  models after the first clicks, user attributes.
- **Cold items:** content-based features (text, images), exploration slots that show new items to some users,
  or hybrid models with an ID part that starts at zero (see the [multimodal tower](../algorithms/multimodal-tower.md)).

## In recbench

- **Users.** Evaluation targets *warm* users: users with pre-test history and at least one relevant test item.
  Users without pre-test history are a separate **cold-user slice**. Only methods that declare
  `handles_cold_users=True` (MostPopular, Random) are scored on it, in the `cold_users/*` metrics. For every
  other method, the cold-user answer would be a popularity fallback anyway.
- **Items.** Items with no pre-test interactions are **cold items**. For methods that cannot score them
  (`scores_cold_items=False`), the evaluator removes them from the ranking. Content models keep them. The
  metric `item_cold_recall_at_10` measures how many relevant cold items a method finds.
- **Serving.** Unknown users get the popularity list stored in each bundle (`popular.npy`), marked
  `"fallback": true` in the API response.
- Code: `src/recbench/evaluation.py::Evaluator` and `src/recbench/pipeline/materialize.py::materialize`
  (where `is_cold` flags are computed).

## Pitfalls

- **Scoring cold users with ID models.** A cold user's ID embedding is random, so such results are noise.
- **Reporting one number for everyone.** Mixing warm and cold users hides where a model fails; report them separately.
- **Cold items in sampled evaluation.** recbench draws sampled negatives only from warm items, so ID models
  are not unfairly helped.

## Check your understanding

??? question "Why is RetailRocket especially cold-start-heavy?"
    Most visitors appear in only one session. Many test-period users have no pre-test history at all, so
    they fall into the cold-user slice.

??? question "Why can't the evaluator simply let EASE score cold items?"
    EASE's weight matrix has no learned relation to them. Their scores would be zero or meaningless, and
    recbench caps EASE's catalog to items with pre-test history.

## Further reading

- [Cold-start slices](../metrics/cold-start-slices.md) and [collaborative vs content-based](collaborative-content-hybrid.md).
