# H&M Personalized Fashion Recommendations

> Two years of purchases from the fashion retailer H&M: 31.8 million transactions by 1.36 million customers on
> about 105,000 articles, with product descriptions, customer attributes, and product photos.

--8<-- "generated/datasets/hm.md"

## What one row means

Customer `00000dba…` bought article 0663713001 on 2020-09-15 at a normalised price. Dates have no time of day.

## Fields used

| Source file | Field | recbench column |
|---|---|---|
| `transactions_train.csv` | t_dat, customer_id, article_id, price | timestamp, user_id, item_id, value |
| `articles.csv` | prod_name + detail_desc | item text |
| `articles.csv` | product_type_name | item category |
| `customers.csv` | age, club_member_status, fashion_news_frequency | user attributes |
| `images/<first 3 digits>/<article id>.jpg` | product photo (optional, ~30 GB) | item image path |

## How recbench prepares it

- Cleaning: `src/recbench/datasets/hm.py::HMFashion` (adapter version 2, which fixed the image path layout).
- Split rule `last_days`: the **last 7 days** of the public file (2020-09-16 to 2020-09-22) are the test window,
  like the original competition's "next week" task; the 7 days before are the validation window.
- Repeat policy: `exclude_seen`.
- **Ties:** dates only, so thousands of purchases share each timestamp; recbench orders them by their row in
  the original file.

## Pitfalls

- **Day-level timestamps** limit what sequence models can learn within a day.
- **Repeat purchases of basics** (socks, underwear) are common but removed under `exclude_seen`.
- **Images are big.** Without them, the multimodal tower is skipped ("no item images on disk"); recbench
  never generates placeholder images.
- **v0.1 bug:** a time-zone shift once left H&M's smoke split with zero test users. Fixed and tested; see the
  [review log](../../review/2026-10-02-review.md).

## Good for

Content-based and multimodal methods, realistic weekly evaluation, and methods that use customer attributes.

## How to get it

It is a Kaggle **competition** dataset:

1. Sign in to Kaggle, open
   <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations>, and accept the
   competition rules ("Join Competition" / "I understand and accept"). Downloads fail until you do.
2. Set up `~/.kaggle/kaggle.json` (see [RetailRocket](retailrocket.md#how-to-get-it)).
3. `python -m recbench.pipeline.prepare --config configs/benchmarks/smoke-cpu.yaml --datasets hm` downloads
   the three CSV files only.
4. Optional, for the multimodal tower (about 30 GB): download the full archive with
   `kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/raw/hm` and unzip it
   so that images end up under `data/raw/hm/images/`. Then delete `data/clean/hm` (so it is re-cleaned with
   image paths) and run prepare again.

Licence: Kaggle competition rules; non-commercial and academic use.
