# Last.fm 1K

> The complete listening histories of 992 Last.fm users from 2005 to 2013: 19.2 million plays of 176,892
> artists.

--8<-- "generated/datasets/lastfm.md"

## What one row means

User `user_000001` played a track by artist "Deep Dish" at 2009-05-04 23:08:57 UTC. recbench recommends
**artists**, not tracks.

## Fields used

| Source file | Field | recbench column |
|---|---|---|
| `userid-timestamp-artid-artname-traid-traname.tsv` | userid, timestamp | user_id, timestamp |
| same | artist MusicBrainz id, or `name:<artist name>` when the id is missing | item_id |
| same | artist name | item text |

## How recbench prepares it

- Cleaning: `src/recbench/datasets/lastfm.py::LastFM1K` (adapter version 2, which keeps artists without a
  MusicBrainz id, keyed by name).
- Split rule: the last 10% of plays by time are the test window.
- **Two repeat policies:** music is repetitive, so results are reported twice:
    - `exclude_seen` (primary): can the method find artists the user has *not* played before (discovery)?
    - `allow_repeats`: can it predict the next artist, including familiar ones? Reported as `allow_repeats/<metric>`.
- No categories: S3-Rec is skipped, and the category-based metrics (diversity, calibration) are not computed.

## Pitfalls

- **Few users, very long histories:** a user may have tens of thousands of plays. The smoke tier keeps each
  sampled user's 300 most recent pre-test plays, the quick tier 1,000 (and up to 200 test plays per user), and the
  full tier keeps everything. With fewer than 1,000 users, the quick tier keeps most of them, so its sample is
  about users' recent histories rather than a subset of users.
- **Repeats dominate:** a large share of test plays are artists the user already knows (shown as "repeat share"
  above). Discovery and repeat prediction are different tasks; read both columns.
- **Old data:** 2005–2013 listening habits.

## Good for

Repeat consumption, long sequences, and the difference between "discover" and "continue".

## How to get it

No account needed. recbench downloads the archive (about 640 MB) from the Music Technology Group
(Universitat Pompeu Fabra):

```bash
python -m recbench.pipeline.prepare --config configs/benchmarks/smoke-cpu.yaml --datasets lastfm
```

If the download fails, place `userid-timestamp-artid-artname-traid-traname.tsv` under `data/raw/lastfm/` and
run the command again. Licence: non-commercial use (Last.fm terms); cite Celma (2010), *Music Recommendation
and Discovery in the Long Tail*, Springer.
