"""Retailrocket e-commerce clickstream: views, add-to-carts, and transactions (implicit feedback).

Item properties are hashed in the public dump; only the category id is usable.
Category values are taken from the first property snapshot, which may postdate some events.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from recbench.datasets.common import empty_users, write_clean
from recbench.pipeline.download import kaggle_download
from recbench.protocol import DatasetSpec
from recbench.registry import register_dataset

EVENT_VALUE = {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}


@register_dataset
class Retailrocket:
    CLEAN_VERSION = "2"
    spec = DatasetSpec(
        name="retailrocket",
        domain="ecommerce",
        feedback={"implicit"},
        split_rule="quantile",
        description="Retailrocket views, carts, and transactions with item category ids.",
    )

    def download(self, raw_dir: Path) -> None:
        if (raw_dir / "events.csv").exists():
            return
        kaggle_download(["datasets", "download", "-d", "retailrocket/ecommerce-dataset"], raw_dir)
        for archive in raw_dir.glob("*.zip"):
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(raw_dir)

    def to_clean(self, raw_dir: Path, clean_dir: Path) -> None:
        events = pd.read_csv(raw_dir / "events.csv")
        properties = []
        for path in sorted(raw_dir.glob("item_properties_part*.csv")):
            properties.append(pd.read_csv(path))
        props = pd.concat(properties, ignore_index=True) if properties else pd.DataFrame()
        category = pd.Series(dtype=str)
        if not props.empty and "property" in props.columns:
            cat_rows = props[props["property"].astype(str) == "categoryid"]
            category = cat_rows.drop_duplicates("itemid").set_index("itemid")["value"].astype(str)
            category.index = category.index.map(str)
        interactions = pd.DataFrame(
            {
                "user_id": events["visitorid"].astype(str),
                "item_id": events["itemid"].astype(str),
                "timestamp": pd.to_datetime(events["timestamp"], unit="ms", utc=True).dt.tz_localize(None),
                "session_id": "",
                "feedback_type": "implicit",
                "value": events["event"].map(EVENT_VALUE).fillna(1.0).astype(float),
            }
        )
        item_ids = interactions["item_id"].drop_duplicates()
        items = pd.DataFrame(
            {
                "item_id": item_ids.astype(str),
                "text": "",  # the public dump has no item names or descriptions
                "category": item_ids.map(lambda i: str(category.get(str(i), "")) if len(category) else "").astype(str),
                "image_path": pd.NA,
            }
        )
        write_clean(interactions, items, empty_users(interactions["user_id"]), clean_dir)
