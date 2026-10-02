# Add a dataset (or your own data)

## 1. Write an adapter

Create `src/recbench/datasets/<name>.py` with a class that downloads raw files and converts them to the
**clean contract** (`src/recbench/schema.py`):

| Table | Columns |
|---|---|
| interactions | `user_id`, `item_id`, `timestamp` (naive UTC datetime), `session_id` ("" if unknown), `feedback_type` ("implicit"/"explicit"), `value` (float) |
| items | `item_id`, `text`, `category` ("\|"-separated tokens), `image_path` (or empty) |
| users | `user_id`, `attributes` (free text), `group_label` (or empty) |

```python
from pathlib import Path

import pandas as pd

from recbench.datasets.common import empty_users, write_clean
from recbench.pipeline.download import fetch
from recbench.protocol import DatasetSpec
from recbench.registry import register_dataset


@register_dataset
class MyShop:
    CLEAN_VERSION = "1"  # bump whenever to_clean changes, so old clean files are rebuilt
    spec = DatasetSpec(
        name="myshop",
        domain="ecommerce",
        feedback={"implicit"},
        split_rule="quantile",              # or "last_days" with test_days=7
        repeat_policies=("exclude_seen",),  # add "allow_repeats" for repetitive domains
        description="My shop's orders.",
    )

    def download(self, raw_dir: Path) -> None:
        if not (raw_dir / "orders.csv").exists():
            fetch("https://example.com/orders.csv", raw_dir / "orders.csv")

    def to_clean(self, raw_dir: Path, clean_dir: Path) -> None:
        orders = pd.read_csv(raw_dir / "orders.csv")
        interactions = pd.DataFrame({
            "user_id": orders["customer"].astype(str),
            "item_id": orders["sku"].astype(str),
            "timestamp": pd.to_datetime(orders["ordered_at"], utc=True).dt.tz_localize(None),
            "session_id": "",
            "feedback_type": "implicit",
            "value": 1.0,
        })
        items = pd.DataFrame({"item_id": interactions["item_id"].unique(), "text": "", "category": "", "image_path": None})
        write_clean(interactions, items, empty_users(interactions["user_id"]), clean_dir)
```

Import it in `src/recbench/datasets/__init__.py`.

## 2. Catalog entry

Add it under `datasets:` in `dictionary/catalog.yaml` with title, domain, source, citation, licence,
commercial use, feedback, side information, and known pitfalls.

## 3. Prepare and check

```bash
python -m recbench.pipeline.prepare --config configs/benchmarks/smoke-cpu.yaml --datasets myshop
cat data/splits/myshop/smoke/meta.json
```

Check `n_eval_warm` (warm evaluation users), `repeat_share`, and the cutoff dates. If prepare stops with
`SplitError: only N warm eval users`, the sample is too small: use a bigger tier, or adjust `tier_overrides` in
the config.

## 4. Docs

Write `docs/dictionary/datasets/<name>.md` (start with `--8<-- "generated/datasets/<name>.md"`) and add it to the nav.

## Your own (private) data: a checklist

Before your company's data goes anywhere near this repository:

- [ ] **Permission.** Confirm you are allowed to use the data for this purpose (privacy policy, contracts, consent).
- [ ] **Pseudonymise ids.** Replace user ids with a salted hash, for example
  `hashlib.sha256((salt + user_id).encode()).hexdigest()[:16]`, and keep the salt secret and out of git.
- [ ] **Drop direct identifiers.** Names, e-mails, phone numbers, addresses, and IPs never enter the clean tables.
- [ ] **Minimise.** Keep only the columns the contract needs; coarsen timestamps (to the hour or day) if exact
  times are sensitive.
- [ ] **Check free text.** Item text and user attributes must not contain personal data.
- [ ] **Never commit data.** `data/` is git-ignored; keep it that way. Do not upload private data to managed
  services without approval.
- [ ] **Store securely.** Use an encrypted disk, and delete the raw exports when you are done.
- [ ] **Share only aggregates.** Report metrics, not per-user outputs (`per_user_metrics.npz` holds per-user values).

See also [security](../cloud/security.md).
