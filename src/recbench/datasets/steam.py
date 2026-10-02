"""UCSD McAuley Steam interactions. A failed download is recorded and skipped."""

from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path

import pandas as pd

from recbench.datasets.common import empty_users, write_clean
from recbench.pipeline.download import DownloadError, fetch
from recbench.protocol import DatasetSpec
from recbench.registry import register_dataset

URLS = [
    "https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_reviews.json.gz",
]


@register_dataset
class Steam:
    spec = DatasetSpec(
        name="steam",
        domain="games",
        feedback={"implicit", "explicit"},
        official_split="quantile",
        description="Steam playtime and review text from the UCSD McAuley dump.",
    )

    def download(self, raw_dir: Path) -> None:
        if list(raw_dir.glob("steam*.json*")):
            return
        errors = []
        dest = raw_dir / "steam_reviews.json.gz"
        for url in URLS:
            try:
                fetch(url, dest, timeout=600)
                return
            except DownloadError as exc:
                errors.append(str(exc))
                if dest.exists():
                    dest.unlink()
        raise DownloadError(
            "Steam could not be downloaded. Place steam_reviews.json or steam_reviews.json.gz "
            f"in {raw_dir}. Attempts: {errors}"
        )

    def to_clean(self, raw_dir: Path, clean_dir: Path) -> None:
        files = list(raw_dir.glob("steam*.json*"))
        if not files:
            raise FileNotFoundError(f"No Steam file in {raw_dir}")
        path = files[0]
        opener = gzip.open if path.suffix == ".gz" else open
        rows = []
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    try:
                        record = ast.literal_eval(line)
                    except (SyntaxError, ValueError):
                        continue
                rows.append(record)
                if len(rows) >= 5_000_000:
                    break
        frame = pd.DataFrame(rows)
        user_col = _first(frame, ["username", "user_id", "user"])
        item_col = _first(frame, ["product_id", "item_id", "game_id"])
        time_col = _first(frame, ["date", "timestamp", "time"])
        text_col = _first(frame, ["text", "review"])
        hours_col = _first(frame, ["hours", "playtime"])
        rec_col = _first(frame, ["recommend", "rating", "score"])
        interactions = pd.DataFrame(
            {
                "user_id": frame[user_col].astype(str),
                "item_id": frame[item_col].astype(str),
                "timestamp": pd.to_datetime(frame[time_col], utc=True, errors="coerce").dt.tz_localize(None),
                "session_id": "",
                "feedback_type": "explicit" if rec_col and pd.api.types.is_numeric_dtype(frame[rec_col]) else "implicit",
                "value": frame[hours_col].astype(float) if hours_col else 1.0,
            }
        ).dropna(subset=["timestamp"])
        if rec_col and interactions["feedback_type"].iloc[0] == "explicit":
            interactions["value"] = pd.to_numeric(frame[rec_col], errors="coerce").fillna(1.0)
        names = frame[text_col].fillna("").astype(str) if text_col else ""
        items = (
            pd.DataFrame({"item_id": interactions["item_id"], "text": names if text_col else interactions["item_id"]})
            .drop_duplicates("item_id")
        )
        items["text"] = items["text"].astype(str).str.slice(0, 280)
        items["category"] = "game"
        items["image_path"] = pd.NA
        write_clean(interactions, items, empty_users(interactions["user_id"]), clean_dir)


def _first(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None
