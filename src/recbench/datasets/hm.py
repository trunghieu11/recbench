"""H&M fashion purchases. Full tier uses the last 7 days of the public file as test."""

from __future__ import annotations

import zipfile
import hashlib
from pathlib import Path

import pandas as pd

from recbench.datasets.common import write_clean
from recbench.pipeline.download import kaggle_download
from recbench.protocol import DatasetSpec
from recbench.registry import register_dataset


def attach_slice_images(split_dir: Path, raw_dir: Path, tier: str) -> None:
    """Point split items at real JPEGs, or at tiny generated images for the smoke slice."""
    import pyarrow.parquet as pq

    items_path = split_dir / "items.parquet"
    if not items_path.exists():
        return
    items = pq.read_table(items_path).to_pandas()
    smoke_dir = raw_dir / "images_smoke"
    paths = []
    any_real = False
    for item_id, current in zip(items["item_id"].astype(str), items["image_path"]):
        current_s = "" if current is None or (isinstance(current, float)) else str(current)
        if current_s and current_s != "None" and Path(current_s).is_file():
            paths.append(current_s)
            any_real = True
            continue
        if tier == "full":
            paths.append("")
            continue
        dest = smoke_dir / f"{item_id}.ppm"
        if not dest.exists():
            _ppm(dest, item_id)
        paths.append(str(dest))
    if tier == "full" and not any_real:
        items["image_path"] = pd.NA
    else:
        items["image_path"] = paths
    items.to_parquet(items_path, index=False)


def _ppm(path: Path, key: str) -> None:
    """8x8 image derived from the article id, used when the Kaggle archive is not on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = int(hashlib.md5(key.encode()).hexdigest()[:6], 16)
    red, green, blue = (digest >> 16) & 255, (digest >> 8) & 255, digest & 255
    header = b"P6\n8 8\n255\n"
    pixel = bytes((red, green, blue))
    path.write_bytes(header + pixel * 64)


@register_dataset
class HMFashion:
    spec = DatasetSpec(
        name="hm",
        domain="ecommerce",
        feedback={"implicit"},
        has_images=True,
        official_split="hm_last_7_days",
        description="H&M purchases, article text, customer attributes, and article images.",
    )

    def download(self, raw_dir: Path) -> None:
        if (raw_dir / "transactions_train.csv").exists() and (raw_dir / "articles.csv").exists():
            return
        # The full competition zip is mostly images. Smoke only needs the tables.
        for filename in ("transactions_train.csv", "articles.csv", "customers.csv"):
            if (raw_dir / filename).exists():
                continue
            kaggle_download(
                [
                    "competitions",
                    "download",
                    "-c",
                    "h-and-m-personalized-fashion-recommendations",
                    "-f",
                    filename,
                ],
                raw_dir,
            )
            archive = raw_dir / f"{filename}.zip"
            if archive.exists():
                with zipfile.ZipFile(archive) as zipped:
                    zipped.extractall(raw_dir)

    def to_clean(self, raw_dir: Path, clean_dir: Path) -> None:
        tx = pd.read_csv(raw_dir / "transactions_train.csv")
        articles = pd.read_csv(raw_dir / "articles.csv")
        customers = None
        if (raw_dir / "customers.csv").exists():
            customers = pd.read_csv(raw_dir / "customers.csv")
        text = (
            articles["prod_name"].fillna("").astype(str)
            + " "
            + articles.get("detail_desc", pd.Series([""] * len(articles))).fillna("").astype(str)
        )
        category = articles.get("product_type_name", pd.Series([""] * len(articles))).fillna("").astype(str)
        image_paths = []
        for article_id in articles["article_id"].astype(str):
            padded = article_id.zfill(10)
            folder = padded[:3]
            real = raw_dir / "images" / folder / padded / f"{padded}.jpg"
            image_paths.append(str(real) if real.is_file() else "")
        items = pd.DataFrame(
            {
                "item_id": articles["article_id"].astype(str),
                "text": text.str.strip(),
                "category": category,
                "image_path": image_paths,
            }
        )
        interactions = pd.DataFrame(
            {
                "user_id": tx["customer_id"].astype(str),
                "item_id": tx["article_id"].astype(str),
                "timestamp": pd.to_datetime(tx["t_dat"], utc=True).dt.tz_localize(None),
                "session_id": "",
                "feedback_type": "implicit",
                "value": tx.get("price", pd.Series([1.0] * len(tx))).astype(float),
            }
        )
        if customers is not None:
            attrs = customers.assign(
                attributes=customers.apply(
                    lambda row: " ".join(
                        f"{col}={row[col]}"
                        for col in ("age", "club_member_status", "fashion_news_frequency")
                        if col in customers.columns and pd.notna(row[col])
                    ),
                    axis=1,
                )
            )
            users = pd.DataFrame(
                {
                    "user_id": attrs["customer_id"].astype(str),
                    "attributes": attrs["attributes"].fillna("").astype(str),
                    "group_label": pd.NA,
                }
            )
        else:
            users = pd.DataFrame(
                {"user_id": interactions["user_id"].drop_duplicates().astype(str), "attributes": "", "group_label": pd.NA}
            )
        write_clean(interactions, items, users, clean_dir)
