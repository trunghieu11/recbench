# Lab 10 · V-SKNN: what you did just now matters most

!!! abstract "In plain words"
    The methods so far looked at a user's whole history as a set: order did not matter. V-SKNN looks at the user's
    **last session**, the burst of activity that ended most recently, and weighs the latest items most. It finds past
    sessions, by anyone, that share items with it, and recommends what those sessions contained. There is no
    training: it searches past sessions at the moment it scores. This week you will learn how "sessions" are cut,
    what each knob does, and how to make old sessions count less, an idea from STAN.

## Your starting point

--8<-- "generated/lab/vsknn.md"

## Before you start

- Read [sequential and session-based recommendation](../dictionary/concepts/sequential-and-session.md) and sections
  1 to 5 of the [V-SKNN page](../dictionary/algorithms/vsknn.md).
- Open `labs/10-vsknn/vsknn.ipynb` and `labs/10-vsknn/experiments.yaml`; `git switch -c lab/10-vsknn`.
- Plan about 5 hours, plus box runs.

## Level 1: reproduce and read

### 1.1 Reproduce the baseline's best setting

```bash
python -m recbench.lab once --method vsknn --dataset movielens-25m --best
```

### 1.2 Read the code

Open `src/recbench/methods/neighbourhood.py`: `sessions_from_events` and `VSKNN`.

**a.** How are sessions cut, and what is `SESSION_GAP_US`?

??? success "Answer"
    Each user's events are sorted by time; a pause longer than 30 minutes (`SESSION_GAP_US`, in microseconds) starts a
    new session. The result is a session × item 0/1 matrix, each session's end time, and its owner.

**b.** In `_profile`, which items describe the user, and what do `div`, `linear` and `same` do?

??? success "Answer"
    The items of their last session, or with `vsknn_last_n` their last N items, latest first. Each gets a position
    weight: `div` gives 1, 1/2, 1/3, …; `linear` gives 1, 0.9, 0.8, … down to 0.1; `same` gives 1 to all. An item that
    appears twice keeps its most recent weight.

**c.** `score_users` picks neighbour sessions in two steps. What are they, and why two?

??? success "Answer"
    First `vsknn_sample`: among all past sessions that share an item with the profile, keep the most **recent** ones
    (cheap: only end times are compared). Then `vsknn_k`: among those, keep the most **similar** ones (cosine). The
    first step bounds the cost for popular items, which appear in a huge number of sessions.

**d.** Why is the user's own latest session excluded from the neighbours (`sims[self.last_session[user]] = 0`)?

??? success "Answer"
    It *is* the profile: it would always be the most similar session, and it would only recommend items the user
    already has.

### 1.3 Look inside

In the notebook, print `model.fit_info["sessions"]` for MovieLens and for Last.fm, and look at one user's profile with
`model._profile(user)`.

??? question "How long is a typical MovieLens 'session', and what is it in real life?"
    On MovieLens, a session is a sitting in which someone rated many films in a row, often films watched long ago, not
    a viewing session. On Last.fm a session is a listening session. The same code means different things on different
    datasets: keep that in mind when you compare.

## Level 2: understand each setting

| Setting | What it does | The baseline searches |
|---|---|---|
| `vsknn_k` | neighbour sessions kept | 50, 100, 200, 500 |
| `vsknn_sample` | recent candidate sessions considered first | 500, 1,000, 5,000 |
| `vsknn_weighting` | position weights in the profile | div, linear, same |
| `vsknn_last_n` | profile = the last N items instead of the last session | none, 10, 50 |
| `vsknn_idf` | rare items in a neighbour session count more | true, false |
| `train_window_days` | which past sessions exist at all | none, 30, 90, 365 |

**a.** Sweep `vsknn_last_n` (null, 10, 50) on MovieLens and on Last.fm. Which dataset prefers the last session, and
which a longer history?

**b.** Sweep `vsknn_weighting` (div, linear, same). Predict first: on which dataset should "the latest item matters
most" (`div`) win?

??? success "What to look for"
    Where order carries meaning, as in listening sessions where one song leads to the next, recency weighting wins. On
    MovieLens, where a "session" is a batch of ratings, order matters less and `same` may do as well.

## Level 3: data tricks

### 3.1 All of history

`no-window` stops searching `train_window_days`: neighbour sessions come from the whole history. Run it on the box and
compare.

### 3.2 The session gap

30 minutes is a convention from web shops. Add `vsknn_gap_minutes` (default 30) and search 10 minutes to 1 day.

??? tip "Hint"
    `sessions_from_events(data, gap_us)` already takes the gap as an argument. Store it on the model, because
    `_profile` cuts the user's last session with the same constant.

??? success "Solution"
    In `VSKNN.fit`:

    ```python
    self.gap_us = int(float(cfg.get("vsknn_gap_minutes", 30)) * 60_000_000)  # a longer pause starts a new session
    self.sessions, self.session_end, self.session_owner = sessions_from_events(data, self.gap_us)
    ```

    and in `_profile`: `gaps = np.flatnonzero(np.diff(times) > self.gap_us)`.

    A test: a 1-minute gap must make at least as many sessions as 30 minutes.

## Level 4: one change from the literature

### The idea: old sessions count less (STAN, Garg et al. 2019)

STAN ("sequence and time aware neighbourhood") adds three decays to session kNN: one for the position of items in the
current session (V-SKNN's `div` already does this), one for **how old a neighbour session is**, and one for item
positions inside the neighbour session. The second is the simplest and often the most useful. A session from
yesterday says more about today's tastes than one from three years ago:

$$\text{sim}'(s) = \text{sim}(s) \cdot e^{-\text{age}(s)/\lambda}$$

with the age measured in days up to the cutoff, and λ in days.

### Write the variant

Add `vsknn_recency_days` = λ (unset: off).

??? tip "Hint"
    `self.session_end` holds each session's end time in microseconds; the cutoff is `data.meta["test_start_us"]` and a
    day is `DAY_US` (from `recbench.data`). Compute one weight per session in `fit`, and apply it to the similarities
    in `score_users` **before** the top-k neighbours are chosen, so that recent sessions can also win a place.

??? success "Solution"
    In `VSKNN.fit`:

    ```python
    self.session_weight = None
    recency = cfg.get("vsknn_recency_days")
    if recency:  # STAN (Garg et al. 2019): a neighbour session counts less the older it is
        age_days = (int(data.meta["test_start_us"]) - self.session_end) / DAY_US
        self.session_weight = np.exp(-age_days / float(recency)).astype(np.float32)
    ```

    and in `score_users`, right after the cosine is computed:

    ```python
    sim = sims[touched] / (np.linalg.norm(weights) * self.session_norm[touched] + 1e-9)  # cosine
    if self.session_weight is not None:
        sim = sim * self.session_weight[touched]
    ```

    with `from recbench.data import DAY_US, HistoryBatch, TrainView` at the top.

### Test it

??? success "Solution"
    ```python
    METHOD = "vsknn"
    VARIANT = {"vsknn_recency_days": 30}
    DEFAULT = {"vsknn_gap_minutes": 30}


    def test_the_new_math_on_a_case_small_enough_to_check_by_hand(toy):
        """Neighbour sessions are weighted by exp(-age / lambda): between 0 and 1, larger for newer sessions."""
        view, _ = toy
        model = checks.fit(METHOD, view, **VARIANT)
        w = model.session_weight
        assert np.all((w > 0) & (w <= 1.0 + 1e-6))
        newest, oldest = np.argmax(model.session_end), np.argmin(model.session_end)
        assert w[newest] >= w[oldest]
    ```

### Run and compare

Uncomment `stan-recency`, run it on the box, then compare with `--segments`. Look at the **recency** segments: users
whose last event was long ago have neighbour sessions that are old too, so they are affected most.

## Record your results

| Experiment | MovieLens | Last.fm | H&M | Steam | RetailRocket | Decision |
|---|---|---|---|---|---|---|
| no-window | | | | | | |
| session-gap | | | | | | |
| stan-recency | | | | | | |

## Further reading

- Ludewig & Jannach (2018), *Evaluation of session-based recommendation algorithms*, User Modeling and User-Adapted Interaction: V-SKNN.
- Garg, Gupta, Malhotra, Vig & Shroff (2019), *Sequence and time aware neighborhood for session-based recommendations: STAN*, SIGIR 2019.
- Jannach & Ludewig (2017), *When recurrent neural networks meet the neighborhood for session-based recommendation*, RecSys 2017.
