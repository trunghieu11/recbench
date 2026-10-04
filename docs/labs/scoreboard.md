# Lab scoreboard

!!! abstract "In plain words"
    Where each lab method started, its laptop baseline, and what your experiments have changed since. Each cell is
    one method on one dataset. Once you have experiments, a cell reads *baseline → your best experiment*, with an
    arrow for the verdict of the paired test. Your best experiment is chosen by its **validation** score, never by
    its test score.

--8<-- "generated/lab/scoreboard.md"

## How to read it

- **The floor.** MostPopular recommends what is popular right now to everyone. A method below it on a dataset has
  learned less than "what is popular" there.
- **The numbers** are test NDCG@10 on the quick tier, after 10 settings were tried on the validation fold. Random
  methods are averaged over 3 seeds.
- **Datasets differ a lot in difficulty.** Compare methods within a column, never across columns.
- **Rebuild** with `python -m recbench.lab scoreboard --docs` after your experiments finish, and commit
  `docs/generated/lab/` with your pull request.

Each method's details (the baseline's best settings, intervals, training times, and every experiment) are at the top
of its lab page.
