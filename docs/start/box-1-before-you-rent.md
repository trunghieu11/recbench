# 5a. Before you rent

!!! abstract "In plain words"
    Everything here is free and happens on your laptop. You check that the whole pipeline works on tiny data, you
    prepare the real data, and you create the account and the "key" you need to reach the rented box. Mistakes
    found now cost nothing; the same mistakes on the box cost money per hour.

**Time:** about 1.5 hours, mostly waiting. **Cost:** $0.

## Step 1: a free dry run (20–40 minutes)

The dry run uses the real queue and search spaces, but on the small **smoke** splits (a few hundred users per
dataset), with only 2 settings per job and at most 3 training epochs.

```bash
source .venv/bin/activate
python -m recbench.pipeline.prepare --config configs/benchmarks/quick-smoke.yaml --tier smoke,smoke-val
python -m recbench.queue run --config configs/benchmarks/quick-smoke.yaml --cpu-workers 3
python -m recbench.queue status --config configs/benchmarks/quick-smoke.yaml
```

`--cpu-workers 3` runs three jobs at once; an M4 Pro has 12 cores, so each job gets 4 threads.

!!! success "You should see"
    The last command prints one block per dataset, with 23 jobs each, for example:

    ```text
    queue: finished (the session ended normally); state written 0 min ago

    == movielens-25m: 23/23 jobs done
      rp3beta          finished     test NDCG@10=0.1090  val=0.1381
      lgbm_rerank      finished     test NDCG@10=0.1000  val=0.1807
      ...
      confirm lgbm_rerank finished     full test NDCG@10=0.0997
    ```

    Nearly every job ends `finished`. Two kinds of `unsupported` are expected:

    - the two re-rankers (`lgbm_rerank`, `dcnv2_rerank`) on H&M and Last.fm, because the smoke splits have too
      few recent users to train them ("too few recent users with a reachable next item"). The real splits are
      large enough;
    - `sansa`, if you did not install the optional `sansa` extra (see [install](install.md)).

    **Ignore the scores:** with a few hundred users and 3 epochs they mean nothing. Any `failed` job must be
    understood before you rent: its reason is printed next to it, and the
    [troubleshooting page](../how-to/troubleshooting.md) lists the known ones.

## Step 2: prepare the real splits (about 5 minutes)

```bash
python -m recbench.pipeline.prepare --config configs/benchmarks/quick.yaml --tier quick,quick-val,full,full-val
```

| Tier | What it holds | Used for |
|---|---|---|
| `quick` | about 1M events per dataset, sampled by user, with the real test window | the single test run of each job |
| `quick-val` | the same users with every test event deleted; the validation window plays the test window | the tuning trials |
| `full` | every event | the confirmation's final test |
| `full-val` | the full data's validation fold | the confirmation's size check |

A **fold** is a copy of the data cut one window earlier, so that settings can be chosen without ever looking at
the real test window.

!!! success "You should see"
    Twenty lines such as `prepared hm -> .../data/splits/hm/quick-val`, five datasets times four tiers. Then
    `du -sh data/splits/*/quick` shows 30–70 MB per dataset, and `du -sh data/splits/hm/full` about 2 GB.
    The full folds need about 22 GB of memory at their peak, so close other large apps first.

If the raw data is not downloaded yet, the first run downloads it (about 11 GB). H&M and RetailRocket need a
Kaggle token ([install, step 5](install.md#step-5-a-kaggle-token-for-hm-and-retailrocket)).

## Step 3: a vast.ai account with credit

1. Sign up at [vast.ai](https://vast.ai).
2. Add credit in the billing section. Credit is prepaid, so add only what you plan to spend: $10–15 covers all
   five datasets with a margin.

!!! info "Why vast.ai?"
    It rents other people's GPU machines by the hour, usually much cheaper than a big cloud. The trade-off: machines
    differ in quality, so you choose carefully (5b), and a machine can disappear. recbench's queue resumes after an
    interruption, so a lost machine costs only the job in progress.

## Step 4: an SSH key, your key to the box

An SSH key is a pair of files. The **private key** stays on your laptop and is never shared. The **public key**
is given to vast.ai, which installs it on every box you rent. The box then lets in whoever holds the private key,
without a password.

```bash
ls ~/.ssh/id_ed25519.pub 2>/dev/null || ssh-keygen -t ed25519 -C "vast-ai"   # Enter accepts the default file
cat ~/.ssh/id_ed25519.pub
```

!!! success "You should see"
    One line starting with `ssh-ed25519 AAAA…` and ending with `vast-ai`. Copy that whole line into vast.ai's
    account settings, under **SSH keys**. (The console's labels change from time to time; look for "Keys".)

Never paste the private key (the file without `.pub`) anywhere.

## Step 5: estimate the cost

The offer you pick in 5b shows a price per hour. A worked example at **$0.60 per hour**:

| Part | Hours | Cost |
|---|---|---|
| set up and upload the data | 0.5 | $0.30 |
| 5 datasets × 1.5–2.5 h | 7.5–12.5 | $4.50–7.50 |
| margin: checks, fetching results, mistakes | 1 | $0.60 |
| **total** | **9–14** | **about $5.40–8.40** |

At $1.00 per hour the same plan costs about $9–14. Two smaller charges come on top: disk storage, billed for as
long as the instance exists, even when it is stopped, and network transfer (about 9 GB up, a few GB down). Both
are shown on the offer. You can also run one dataset per rental (`--stop-after-dataset`, see 5c) and pay as you go.

## Checklist

- [ ] The dry run ended with every job `finished`, or `unsupported` for a reason listed above.
- [ ] `data/splits` holds `quick`, `quick-val`, `full` and `full-val` for all five datasets.
- [ ] Your vast.ai account has credit, and your public SSH key is added.
- [ ] Your latest code is on GitHub: `git status` says "up to date with 'origin/main'". The box downloads the
      code from there.

**Next:** [5b. Rent and set up](box-2-rent-and-set-up.md).
