"""Shared frame helpers for dataset adapters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_clean(interactions: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame, clean_dir: Path) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(clean_dir / "interactions.parquet", index=False)
    items.to_parquet(clean_dir / "items.parquet", index=False)
    users.to_parquet(clean_dir / "users.parquet", index=False)


def empty_users(user_ids: pd.Series) -> pd.DataFrame:
    uniq = pd.Series(user_ids.unique())
    return pd.DataFrame(
        {
            "user_id": uniq.astype(str),
            "attributes": "",
            "group_label": pd.NA,
        }
    )
