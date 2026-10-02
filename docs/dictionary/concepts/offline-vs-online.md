# Offline vs online evaluation

## Why it matters

recbench is an **offline** benchmark: it replays logged history. Real products are judged **online**, by how
users react to the recommendations they actually see. The two can disagree, and knowing why prevents
over-trusting a leaderboard.

## Intuition

Offline evaluation asks: "would the model have predicted what users did next?" But what users did next was
shaped by the *old* system: they could only click what they were shown. A new model that recommends
genuinely better items users never saw gets no credit offline, because those items never appear in the logs.

## The main methods

| Method | How it works | Pros | Cons |
|---|---|---|---|
| Offline replay (recbench) | train on the past, predict logged future interactions | cheap, fast, repeatable | biased towards the logging system; cannot measure satisfaction |
| A/B test | randomly split live traffic between the old and new model; compare click, purchase, or retention rates | measures real impact | slow, needs traffic, can hurt users |
| Interleaving | merge two models' lists into one and see which model's items get clicked | needs far less traffic than A/B | only compares rankings, not long-term effects |
| Off-policy evaluation | re-weight logged data by how likely the old system was to show each item (inverse propensity scoring) | estimates online value from logs | needs logged probabilities; high variance |

## A small example: exposure bias

The old system showed item X to 1,000 users (100 clicked) and item Y to 10 users (5 clicked). Offline, X has
100 positive interactions and Y has 5, so a model learns "X is better". But Y's click rate (50%) is five times
X's (10%). Only a system that shows Y more, or an off-policy estimate that corrects for exposure, would discover that.

## In recbench

- Everything is offline, with the protocol in [evaluation protocols](evaluation-protocols.md).
- The serving API and load test show *operational* readiness (latency, throughput), not user satisfaction.
- Online-style evaluation (off-policy estimators, simulators, A/B routing) is on the [roadmap](../../results/roadmap.md).
  The evaluator is designed to accept such plugins.

## Pitfalls

- **Treating a small offline gain as a guaranteed online win.** Validate with an A/B test.
- **Optimising offline accuracy at the cost of diversity:** users may enjoy more variety than the logs suggest.

## Check your understanding

??? question "Why can't offline replay measure the value of recommending a never-shown item?"
    The logs contain no reaction to it, so a correct recommendation of it counts as a miss.

??? question "Which online method needs the least traffic to compare two rankers?"
    Interleaving.

## Further reading

- Gilotte et al. (2018), [Offline A/B testing for Recommender Systems](https://arxiv.org/abs/1801.07030) (WSDM 2018).
- [Popularity bias](popularity-bias.md).
