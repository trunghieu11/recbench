# Calibration and fairness

## Calibration (KL divergence)

**Question:** does the list reflect the *mix* of a user's interests? A user who watches 75% action and 25%
drama should not get a 100% action list, even if action is their main taste.

$$
\text{KL}(p_u \,\|\, \tilde q_u) = \sum_{c} p_u(c)\,\log_2\frac{p_u(c)}{\tilde q_u(c)},
\qquad
\tilde q_u = (1-\alpha)\,q_u + \alpha\,p_u
$$

| Symbol | Meaning |
|---|---|
| $p_u(c)$ | share of category $c$ in user $u$'s pre-test history (multi-category items split their weight) |
| $q_u(c)$ | share of category $c$ in the top-10 list |
| $\alpha$ | smoothing, 0.01, so a missing category does not give infinity |

Worked example: $p$ = {Action 0.75, Drama 0.25}.

- A 100% action list: $\tilde q$ = {Action 0.9975, Drama 0.0025}, so KL = 0.75·log₂(0.75/0.9975) +
  0.25·log₂(0.25/0.0025) = **1.352** bits.
- A list with 70% action and 30% drama: KL = **0.009** bits, almost perfectly calibrated.

Lower is better. Not computed without categories. Code: `src/recbench/metrics/catalog.py::_calibration`.

## User-group gap (fairness across users)

**Question:** does the method serve light users as well as heavy users?

recbench puts warm users into three groups by their number of pre-test interactions (terciles: light, medium,
heavy; `src/recbench/pipeline/materialize.py::activity_groups`). It reports the difference between the best and
worst group's mean NDCG@10:

$$
\text{gap} = \max_g \overline{\text{NDCG@10}}_g - \min_g \overline{\text{NDCG@10}}_g
$$

Example: light 0.02, medium 0.04, heavy 0.08 → gap = **0.06**. Lower means more even service. Code:
`src/recbench/metrics/catalog.py::_group_gap`.

## Item-side fairness

Exposure fairness for items (small sellers, niche artists) is measured by the
[long-tail share and Gini](coverage-and-popularity.md).

## When these mislead

- Calibration rewards *proportional* lists; a user who wants to explore beyond their history may prefer less calibration.
- The group gap can shrink because heavy users are served *worse*, not because light users are served better.
  Always read it with the overall NDCG.
- recbench's datasets have no protected attributes (age, gender), so demographic fairness is out of scope.

## Check your understanding

??? question "Why does the 100% action list get a high KL, even though it is mostly right?"
    It leaves out a quarter of the user's interests. KL heavily penalises categories the user likes that the
    list (almost) never shows.

??? question "Why use activity groups for user fairness here?"
    They are available for every dataset and capture a real equity issue: models often serve heavy users best.

## Further reading

- Steck (2018), [Calibrated Recommendations](https://dl.acm.org/doi/10.1145/3240323.3240372) (RecSys 2018).
- [Popularity bias](../concepts/popularity-bias.md).
