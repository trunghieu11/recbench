"""Small interaction log used by tests and the time-to-endpoint timer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recbench.datasets.common import write_clean


def write_toy_clean(clean_dir: Path, n: int = 400, users: int = 24, items: int = 18, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-09-01", periods=n, freq="h")
    user_ids = rng.integers(1, users + 1, size=n)
    item_ids = rng.integers(1, items + 1, size=n)
    interactions = pd.DataFrame(
        {
            "user_id": user_ids.astype(str),
            "item_id": item_ids.astype(str),
            "timestamp": timestamps,
            "session_id": "",
            "feedback_type": "explicit",
            "value": rng.uniform(1, 5, size=n),
        }
    )
    items_frame = pd.DataFrame(
        {
            "item_id": [str(i) for i in range(1, items + 1)],
            "text": [f"title {i} action" if i % 2 == 0 else f"title {i} drama" for i in range(1, items + 1)],
            "category": ["Action" if i % 2 == 0 else "Drama" for i in range(1, items + 1)],
            "image_path": pd.NA,
        }
    )
    users_frame = pd.DataFrame(
        {
            "user_id": [str(i) for i in range(1, users + 1)],
            "attributes": "",
            "group_label": pd.NA,
        }
    )
    write_clean(interactions, items_frame, users_frame, clean_dir)
