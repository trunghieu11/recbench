"""UCSD McAuley Steam reviews (one review = one implicit interaction) plus game metadata.

Item text and categories come from the games metadata file (title, tags,
genres), never from reviews: a review may be written after the test cutoff,
so using it as item content would leak future information.
"""

from __future__ import annotations

import ast
import gzip
import re
from pathlib import Path

import pandas as pd

from recbench.datasets.common import empty_users, write_clean
from recbench.pipeline.download import DownloadError, fetch
from recbench.protocol import DatasetSpec
from recbench.registry import register_dataset

REVIEWS_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_reviews.json.gz"
GAMES_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_games.json.gz"

# Lines are Python 2 dict literals (u'...'), not JSON. A regex pulls the four fields we need;
# ast.literal_eval is the slow fallback for the rare line the regex cannot read.
_STR = r"u?(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\")"
_USER = re.compile(r"u'username': " + _STR)
_ITEM = re.compile(r"u'product_id': u'(\d+)'")
_DATE = re.compile(r"u'date': u'(\d{4}-\d{2}-\d{2})'")
_HOURS = re.compile(r"u'hours': ([0-9.]+)")


@register_dataset
class Steam:
    CLEAN_VERSION = "2"
    spec = DatasetSpec(
        name="steam",
        domain="games",
        feedback={"implicit"},
        split_rule="quantile",
        description="Steam reviews as implicit feedback, with game titles, tags, and genres.",
    )

    def download(self, raw_dir: Path) -> None:
        for url, name in ((REVIEWS_URL, "steam_reviews.json.gz"), (GAMES_URL, "steam_games.json.gz")):
            dest = raw_dir / name
            if dest.exists():
                continue
            try:
                fetch(url, dest, timeout=1800)
            except DownloadError:
                dest.unlink(missing_ok=True)
                if name == "steam_reviews.json.gz":
                    raise
                # Metadata is optional: without it items simply have no text.

    def to_clean(self, raw_dir: Path, clean_dir: Path) -> None:
        rows = []
        with gzip.open(raw_dir / "steam_reviews.json.gz", "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parsed = _parse_review(line)
                if parsed is not None:
                    rows.append(parsed)
        frame = pd.DataFrame(rows, columns=["user_id", "item_id", "date", "hours"])
        interactions = pd.DataFrame(
            {
                "user_id": frame["user_id"].astype(str),
                "item_id": frame["item_id"].astype(str),
                "timestamp": pd.to_datetime(frame["date"], errors="coerce"),
                "session_id": "",
                "feedback_type": "implicit",
                "value": pd.to_numeric(frame["hours"], errors="coerce").fillna(0.0).astype(float),
            }
        ).dropna(subset=["timestamp"])
        items = _games(raw_dir / "steam_games.json.gz", interactions["item_id"].drop_duplicates())
        write_clean(interactions, items, empty_users(interactions["user_id"]), clean_dir)


def _parse_review(line: str) -> tuple[str, str, str, float | None] | None:
    user, item, date = _USER.search(line), _ITEM.search(line), _DATE.search(line)
    if user and item and date:
        hours = _HOURS.search(line)
        return (user.group(1) or user.group(2) or "", item.group(1), date.group(1), float(hours.group(1)) if hours else None)
    try:
        record = ast.literal_eval(line.strip())
    except (SyntaxError, ValueError):
        return None
    if not all(key in record for key in ("username", "product_id", "date")):
        return None
    return (str(record["username"]), str(record["product_id"]), str(record["date"]), record.get("hours"))


def _games(path: Path, item_ids: pd.Series) -> pd.DataFrame:
    meta: dict[str, tuple[str, str]] = {}
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    record = ast.literal_eval(line.strip())
                except (SyntaxError, ValueError):
                    continue
                game_id = str(record.get("id") or "")
                if not game_id:
                    continue
                title = str(record.get("title") or record.get("app_name") or "")
                tags = [str(tag) for tag in (record.get("tags") or [])]
                genres = [str(genre) for genre in (record.get("genres") or [])]
                text = title if not tags else f"{title}. Tags: {', '.join(tags[:15])}"
                meta[game_id] = (text, "|".join(genres))
    ids = item_ids.astype(str).tolist()
    return pd.DataFrame(
        {
            "item_id": ids,
            "text": [meta.get(i, ("", ""))[0] for i in ids],
            "category": [meta.get(i, ("", ""))[1] for i in ids],
            "image_path": pd.NA,
        }
    )
