"""Synthetic interaction logs with known structure, for tests, tutorials, and the time-to-endpoint timer.

write_toy_clean: users have a favourite category (80% of their events) and a
habit of moving from item i to item i+1 (a sequential pattern). A method that
learns anything should beat Random; popularity alone should already help.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recbench.datasets.common import write_clean

CATEGORIES = ("Action", "Drama", "Comedy", "Horror")


def write_toy_clean(
    clean_dir: Path,
    n_users: int = 120,
    n_items: int = 40,
    events_per_user: int = 30,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    per_cat = n_items // len(CATEGORIES)
    start = pd.Timestamp("2020-01-01")
    rows = []
    for user in range(1, n_users + 1):
        favourite = user % len(CATEGORIES)
        offset_hours = rng.integers(0, 24 * 30)
        current = int(rng.integers(0, n_items))
        for step in range(events_per_user):
            roll = rng.random()
            if roll < 0.5:
                current = (current + 1) % n_items  # sequential habit
            elif roll < 0.9:
                current = favourite * per_cat + int(rng.integers(0, per_cat))  # category taste
            else:
                current = int(rng.integers(0, n_items))  # noise
            rows.append(
                {
                    "user_id": f"u{user}",
                    "item_id": f"i{current + 1}",
                    "timestamp": start + pd.Timedelta(hours=int(offset_hours) + 24 * step),
                    "session_id": "",
                    "feedback_type": "implicit",
                    "value": 1.0,
                }
            )
    interactions = pd.DataFrame(rows)
    items = pd.DataFrame(
        {
            "item_id": [f"i{i + 1}" for i in range(n_items)],
            "text": [f"title {i + 1} {CATEGORIES[min(i // per_cat, len(CATEGORIES) - 1)].lower()}" for i in range(n_items)],
            "category": [CATEGORIES[min(i // per_cat, len(CATEGORIES) - 1)] for i in range(n_items)],
            "image_path": pd.NA,
        }
    )
    users = pd.DataFrame({"user_id": [f"u{u}" for u in range(1, n_users + 1)], "attributes": "", "group_label": pd.NA})
    write_clean(interactions, items, users, clean_dir)


def write_copy_task(clean_dir: Path, n_users: int = 300, n_items: int = 30, length: int = 25, seed: int = 0) -> None:
    """Every user walks i -> i+1 -> i+2 ... A correct sequence model predicts the next item almost perfectly.

    Used to catch alignment bugs: if a model reads the wrong end of the history, it cannot solve this.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2021-01-01")
    rows = []
    for user in range(1, n_users + 1):
        first = int(rng.integers(0, n_items))
        offset = int(rng.integers(0, 24 * 10))
        for step in range(length):
            rows.append(
                {
                    "user_id": f"u{user}",
                    "item_id": f"i{(first + step) % n_items + 1}",
                    "timestamp": start + pd.Timedelta(hours=offset + 24 * step),
                    "session_id": "",
                    "feedback_type": "implicit",
                    "value": 1.0,
                }
            )
    items = pd.DataFrame(
        {"item_id": [f"i{i + 1}" for i in range(n_items)], "text": "", "category": "", "image_path": pd.NA}
    )
    users = pd.DataFrame({"user_id": [f"u{u}" for u in range(1, n_users + 1)], "attributes": "", "group_label": pd.NA})
    write_clean(pd.DataFrame(rows), items, users, clean_dir)
