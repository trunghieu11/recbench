# Recommendation benchmark dictionary

Personal reference and runner for nine advanced recommenders. Rankings come from the full GPU tier, one task at a time. The laptop smoke run only checks that the path works.

These numbers use a temporal cutoff, histories that stop at that cutoff, and one shared file of the held-out item plus 100 negatives. Training stops at a step budget. They are not paper SOTA.

## Third-party models

BERT4Rec, S3-Rec, and DIN train through RecBole. DCN-V2 uses FuxiCTR's CrossNetV2. XSimGCL uses SELFRec's encoder (`5b022942`). HSTU uses Meta's `pytorch_hstu_mha` (`ea7b85f`). `scripts/fetch_third_party.sh` clones those two repositories into `third_party/`, which is not committed. TIGER's author repository was not available, so that method follows the paper's RQ-VAE.

## Laptop

```bash
chmod +x scripts/run_smoke_cpu.sh
./scripts/run_smoke_cpu.sh
```

CPU only. 50,000 interactions per dataset. 200 steps. Needs 40 GB free and a Kaggle token for H&M and Retailrocket.

## Ubuntu GPU machine

24 GB VRAM and 64 GB RAM is the minimum. 48 GB VRAM and 128 GB RAM keeps the larger preset.

```bash
chmod +x scripts/run_full_gpu.sh
./scripts/run_full_gpu.sh
```

The script writes `reports/full-<timestamp>/report.md` and `report.html`.

## Layout

Methods, metrics, and datasets register themselves. Add a class and a block in `dictionary/catalog.yaml`. The generated dictionary is `site/docs`.

```bash
python -m recbench.dictionary.build
python -m recbench.ttm
```

`scripts/time_to_endpoint.sh` times a raw fixture through a healthy HTTP endpoint for XSimGCL or BERT4Rec.

Cloud Run: `deploy/cloud_run.sh` serves the smoke checkpoints for XSimGCL, BERT4Rec, DCN-V2, and DIN.
