# Security

recbench handles public datasets and runs mostly on your own machines, so the risks are modest. They are not zero:
API tokens can leak, a public endpoint can be abused, and your own data (if you add it) can contain personal
information. This page lists what the project already does and what you must do yourself.

## Secrets

| Secret | Used for | Where it should live |
|---|---|---|
| Kaggle token | downloading H&M and RetailRocket | `~/.kaggle/kaggle.json` (outside the repository) or environment variables |
| Recombee private token | the Recombee benchmark | `RECBENCH_RECOMBEE_TOKEN` in your shell or a `.env` file |
| Google credentials | `gcloud` | managed by `gcloud auth login`, outside the repository |
| identity tokens | calling the private service | printed on demand, valid for about an hour |
| SSH private key | reaching a rented GPU box | `~/.ssh/id_ed25519` on your laptop only; vast.ai gets the **public** key (`.pub`) |

Built-in protections:

- `.gitignore` ignores `.env` and `.env.*`, `data/`, `runs/`, and `bundles/`;
- `.dockerignore` ignores `.env*`, so secrets are never copied into the image;
- the scripts never print tokens (the deploy script prints a command that fetches one).

!!! danger "If a token leaks"
    Rotate it immediately in the provider's console (Recombee: database settings; Kaggle: account settings).
    Deleting the commit is not enough: anything pushed to a public repository should be considered copied.

### On a rented machine

A rented box belongs to someone else. Treat it as untrusted, and keep secrets off it:

- **Upload the prepared splits** ([5c](../start/box-3-run-and-monitor.md)) instead of preparing data there, so your
  Kaggle token never leaves your laptop. If you must prepare on the box, copy `kaggle.json` there, then delete it
  (`rm ~/.kaggle/kaggle.json ~/.kaggle/access_token`) before you destroy the box.
- Do not copy the SSH **private** key, Google credentials or any `.env` file to the box.
- The monitoring pages (MLflow, the status page) listen on the box's 127.0.0.1 only. Reach them through the SSH
  tunnel ([5c, step 6](../start/box-3-run-and-monitor.md#step-6-watch-it-level-3-two-pages-in-your-browser));
  never open those ports to the internet.
- **Destroy** the box when you are done: that also deletes its disk with your data on it.

Before every commit, check what you are adding:

```bash
git status
git diff --cached | grep -iE "token|secret|password|key" || echo "nothing suspicious"
```

## The serving endpoint

| Setting | Default | Why |
|---|---|---|
| authentication | required (`--no-allow-unauthenticated`) | only principals with the Cloud Run Invoker role can call it |
| public access | opt-in with `--public` | for short demos only |
| max instances | 1 | abuse cannot multiply costs |
| container user | unprivileged `app` | limits the damage of an application bug |
| bucket mount | read-only | the API cannot modify bundles |
| input validation | Pydantic model; `k` between 1 and 100 | oversized or malformed requests are rejected with 422 |

What the API exposes: item ids, item text (titles), and scores for user ids in the bundle. With public datasets
this is public information already. **With your own data, a public endpoint would let anyone enumerate users'
recommendations, which can reveal their history.** Keep such services private.

## Least privilege (optional hardening)

By default the Cloud Run service runs as the Compute Engine default service account, which often has broad
project permissions. For a tighter setup, create a dedicated service account that can only read the bucket:

```bash
gcloud iam service-accounts create recbench-api --display-name "recbench API"
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET \
  --member "serviceAccount:recbench-api@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role "roles/storage.objectViewer"
```

Then add `--service-account recbench-api@YOUR_PROJECT_ID.iam.gserviceaccount.com` to the `gcloud run deploy`
command in `deploy/cloud_run.sh`.

## Your own data

The [add a dataset](../how-to/add-a-dataset.md) guide has a PII (personally identifiable information) checklist.
The essentials:

- replace user ids with random ids before the data enters `data/raw/`, and keep the mapping elsewhere;
- drop free text written by users (reviews, comments): it often contains names and contact details;
- keep `data/` out of git (already ignored) and out of public buckets;
- check your organisation's rules before uploading any of it to a third-party service such as Recombee.

## Third-party code

Three research repositories are fetched by `scripts/fetch_third_party.sh` at **pinned commits**: SELFRec, Meta's
generative-recommenders, and the authors' official GRU4Rec. Libraries are installed from PyPI with version ranges.
Pinning means an upstream change cannot silently alter the code you run. Review a commit before changing a pin.

The official GRU4Rec code allows research and education only; commercial use needs its author's permission.
recbench fetches it, never copies it into this repository. Check its licence before you serve GRU4Rec in a product.

## Data licences

Licences are not a security control but a legal one: several datasets restrict commercial use. The
[datasets index](../dictionary/datasets/index.md) has the licence table. Do not deploy a public service built on
a dataset whose licence forbids it.
