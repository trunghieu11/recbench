"""MovieLens-25M. Explicit ratings, genres as categories, title as text."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from recbench.datasets.common import empty_users, write_clean
from recbench.pipeline.download import fetch
from recbench.protocol import DatasetSpec
from recbench.registry import register_dataset

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


@register_dataset
class MovieLens25M:
    spec = DatasetSpec(
        name="movielens-25m",
        domain="video",
        feedback={"explicit"},
        official_split="quantile",
        description="MovieLens 25M ratings with genres and titles.",
    )

    def download(self, raw_dir: Path) -> None:
        marker = raw_dir / "ml-25m" / "ratings.csv"
        if marker.exists():
            return
        archive = raw_dir / "ml-25m.zip"
        fetch(URL, archive, timeout=1800)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(raw_dir)

    def to_clean(self, raw_dir: Path, clean_dir: Path) -> None:
        root = raw_dir / "ml-25m"
        ratings = pd.read_csv(root / "ratings.csv")
        movies = pd.read_csv(root / "movies.csv")
        interactions = pd.DataFrame(
            {
                "user_id": ratings["userId"].astype(str),
                "item_id": ratings["movieId"].astype(str),
                "timestamp": pd.to_datetime(ratings["timestamp"], unit="s", utc=True).dt.tz_localize(None),
                "session_id": "",
                "feedback_type": "explicit",
                "value": ratings["rating"].astype(float),
            }
        )
        items = pd.DataFrame(
            {
                "item_id": movies["movieId"].astype(str),
                "text": movies["title"].fillna("").astype(str),
                "category": movies["genres"].fillna("").astype(str),
                "image_path": pd.NA,
            }
        )
        write_clean(interactions, items, empty_users(interactions["user_id"]), clean_dir)
