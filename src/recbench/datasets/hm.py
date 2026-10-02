"""H&M fashion purchases. The last 7 days of the public file (2020-09-16..22) are the test window.

Article images are used only if the Kaggle image archive was unpacked under raw/hm/images;
nothing is generated when it is missing.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from recbench.datasets.common import write_clean
from recbench.pipeline.download import kaggle_download
from recbench.protocol import DatasetSpec
from recbench.registry import register_dataset


@register_dataset
class HMFashion:
    CLEAN_VERSION = "2"
    spec = DatasetSpec(
        name="hm",
        domain="ecommerce",
        feedback={"implicit"},
        has_images=True,
        split_rule="last_days",
        test_days=7,
        description="H&M purchases, article text, customer attributes, and (optional) article images.",
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
            # Kaggle layout: images/<first 3 digits>/<10-digit id>.jpg
            candidates = [raw_dir / "images" / padded[:3] / f"{padded}.jpg", raw_dir / "images" / f"{padded}.jpg"]
            real = next((path for path in candidates if path.is_file()), None)
            image_paths.append(str(real) if real else "")
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
