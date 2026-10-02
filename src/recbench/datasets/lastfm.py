"""Last.fm 1K listening histories. Artist name is the item text. HetRec is not joined."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd

from recbench.datasets.common import empty_users, write_clean
from recbench.pipeline.download import DownloadError, fetch
from recbench.protocol import DatasetSpec
from recbench.registry import register_dataset

URLS = [
    "https://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz",
]


@register_dataset
class LastFM1K:
    spec = DatasetSpec(
        name="lastfm",
        domain="music",
        feedback={"implicit"},
        official_split="quantile",
        description="Last.fm 1K listening events. Item text is the artist name.",
    )

    def download(self, raw_dir: Path) -> None:
        if any(raw_dir.rglob("*.tsv")):
            return
        errors = []
        archive = raw_dir / "lastfm-1k.tar.gz"
        for url in URLS:
            try:
                fetch(url, archive, timeout=600)
                break
            except DownloadError as exc:
                errors.append(str(exc))
        else:
            raise DownloadError(
                "Last.fm 1K could not be downloaded. Place userid-timestamp-artid-artname-traid-traname.tsv "
                f"under {raw_dir}. Attempts: {errors}"
            )
        with tarfile.open(archive) as packed:
            packed.extractall(raw_dir)

    def to_clean(self, raw_dir: Path, clean_dir: Path) -> None:
        files = [path for path in raw_dir.rglob("*.tsv") if "timestamp" in path.name]
        if not files:
            raise FileNotFoundError(f"No listening TSV in {raw_dir}")
        frame = pd.read_csv(
            files[0],
            sep="\t",
            header=None,
            names=["user_id", "timestamp", "artist_id", "artist", "track_id", "track"],
            on_bad_lines="skip",
        )
        interactions = pd.DataFrame(
            {
                "user_id": frame["user_id"].astype(str),
                "item_id": frame["artist_id"].fillna(frame["artist"]).astype(str),
                "timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_localize(None),
                "session_id": "",
                "feedback_type": "implicit",
                "value": 1.0,
            }
        ).dropna(subset=["timestamp"])
        artists = frame.drop_duplicates("artist_id")
        items = pd.DataFrame(
            {
                "item_id": artists["artist_id"].fillna(artists["artist"]).astype(str),
                "text": artists["artist"].fillna("").astype(str),
                "category": "artist",
                "image_path": pd.NA,
            }
        )
        write_clean(interactions, items, empty_users(interactions["user_id"]), clean_dir)
