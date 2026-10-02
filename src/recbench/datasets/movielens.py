"""MovieLens-25M. Explicit 0.5-5 star ratings used as implicit "this user watched it" events.

Genres are multi-valued ("Action|Comedy"); "|" separates category tokens
everywhere in recbench. Caveat: MovieLens timestamps record when a rating was
entered, which is often long after the movie was watched.
"""

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
    CLEAN_VERSION = "2"
    spec = DatasetSpec(
        name="movielens-25m",
        domain="video",
        feedback={"explicit"},
        split_rule="quantile",
        description="MovieLens 25M ratings with titles and genres.",
    )

    def download(self, raw_dir: Path) -> None:
        if (raw_dir / "ml-25m" / "ratings.csv").exists():
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
                "timestamp": pd.to_datetime(ratings["timestamp"], unit="s"),
                "session_id": "",
                "feedback_type": "explicit",
                "value": ratings["rating"].astype(float),
            }
        )
        genres = movies["genres"].fillna("").astype(str).replace("(no genres listed)", "")
        items = pd.DataFrame(
            {
                "item_id": movies["movieId"].astype(str),
                "text": movies["title"].fillna("").astype(str),
                "category": genres,
                "image_path": pd.NA,
            }
        )
        write_clean(interactions, items, empty_users(interactions["user_id"]), clean_dir)
