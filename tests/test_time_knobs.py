"""Time-aware settings: a training window (`restrict`) and time decay (`weighted_matrix`)."""

from __future__ import annotations

import numpy as np

from recbench.data import TrainView


def test_restrict_keeps_the_window_plus_each_users_latest_events(toy_split):
    full = TrainView(toy_split)
    view = full.restrict(window_days=3, keep_last=4)
    age = full.event_age_days()
    for user in range(1, full.n_users + 1):
        items, times = full.user_items(user), full.user_times(user)
        kept_items, kept_times = view.user_items(user), view.user_times(user)
        recent = age[full._offsets[user] : full._offsets[user + 1]] <= 3
        expected = recent.copy()
        expected[-4:] = True  # the latest four always stay
        assert np.array_equal(kept_items, items[expected]) and np.array_equal(kept_times, times[expected])
    assert view.seen.nnz <= full.seen.nnz and view.item_pop.sum() == len(view._items)
    assert full.restrict(None) is full
    assert full.user_lengths.sum() == len(full._items)  # the original view is untouched


def test_weighted_matrix_decays_with_age(toy_split):
    data = TrainView(toy_split)
    assert (data.weighted_matrix(None) != data.seen).nnz == 0
    decayed = data.weighted_matrix(half_life_days=7)
    user = 1
    items, ages = data.user_items(user), data.event_age_days()[data._offsets[user] : data._offsets[user + 1]]
    latest_age = {int(i): a for i, a in zip(items, ages)}  # later events overwrite: the most recent age per item
    for item, age in latest_age.items():
        assert np.isclose(decayed[user, item], 2.0 ** (-age / 7), rtol=1e-5)
    counts = data.weighted_matrix(half_life_days=7, binary=False)
    assert counts.sum() <= data.interaction_counts.sum() and counts.sum() > 0


def test_before_is_a_past_view_and_views_compose(toy_split):
    full = TrainView(toy_split)
    cutoff = int(full.meta["valid_start_us"])
    past = full.before(cutoff)
    assert np.asarray(past._ts).max() < cutoff and past.meta["test_start_us"] == cutoff
    assert past.item_pop.sum() == len(past._items) < len(full._items)
    both = full.restrict(5, keep_last=3).before(cutoff)  # a view of a view still maps onto the original events
    frame = full.export_frame()
    kept = frame[both._kept_mask]
    assert len(kept) == len(both._items) and (kept["ts_us"].to_numpy() < cutoff).all()
