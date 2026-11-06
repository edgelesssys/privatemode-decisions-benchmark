# A fresh held-out test of the second automation sequence

The first held-out test found that more labels can automate less
([privatemode-decisions#4](https://github.com/edgelesssys/privatemode-decisions/issues/4)).
The library's fix runs a second test sequence from a quarter of the labels
([#5](https://github.com/edgelesssys/privatemode-decisions/pull/5)). It
splits `delta`, so it could also automate less where the top only just
passes. The [plan](plan.md), committed before any of these tasks was
built, says a fix is checked on fresh tasks and fixes what decides.

**Result: the second sequence stays.** On four tasks published from July
to October 2026 (see the arXiv caveat below), with all ~500 calibration
labels:

| | one sequence | two sequences |
|---|---|---|
| automated, temperature alone | 37.6% | 37.6% |
| automated, temperature + bias | 39.4% | **46.0%** |
| draws over the 10% error bound | 0 of 400 | 0 of 400 |

The 400 draws are fewer outcomes than they look: with the temperature
alone and all labels, every draw is the same fit in another order, so
those 200 are one outcome per task. With the bias, the order moves the
folds: 0 of 200 distinct draws over.

- Two sequences never automate less than one, on any task or fit as
  frozen. Without the pre-window arXiv papers they do on one (below), so
  the cost is real, but small next to the gain.
- The gain is where a few confidently wrong labels stop the first
  sequence: `github_pr` 7% → 28% and `stackexchange_site` 94% → 100%, both
  with the bias.
- With 100 labels the rules agree on every task, as they must below the
  switch.
- The symptom of #4 is left on one task: `biorxiv_subject` automates about
  1% with 100 labels and nothing with all, under both rules. It is 65%
  accurate with 25 options, so no large test gets under 10% error; the
  small one passes by chance (1 of its 50 draws went over).
- The first plan's criteria pass here too: excess ECE 0.028 without labels
  (highest 0.087, `github_pr`), coverage 0.920, no draw with the bias over
  the bound (with the temperature alone and 100 labels, 2 of 200 went over:
  `biorxiv_subject` and `arxiv_cs`), accuracy 77.8% → 79.0% from the bias.

## The arXiv caveat

The plan asks for papers first submitted in the window. `build_arxiv_cs`
read that from the OAI record's `created` date, which isn't the first
submission: 237 of the 1,000 `arxiv_cs` papers have ids from before July
2026 (back to 2020), so the model may have seen them. The task stays as
frozen and run; rebuilding it would be a new test. The answers are the
same under both rules, so this can't favour either one, but it does move
the result. Scored again on the 763 papers from the window (not part of
the plan's decision; the older ones are dropped before the 50/50 split, so
`arxiv_cs` gets new halves, same seed;
[window-only/report.md](window-only/report.md)):

| automated, all labels | one sequence | two sequences |
|---|---|---|
| mean of the four tasks, temperature + bias | 37.0% | 43.3% |
| `arxiv_cs`, temperature + bias | 46.8% | **45.1%** |

The decision holds (+6.3 points, no draw over the bound), and `arxiv_cs`
there is the case the split costs, 1.6 points. The first set's
`arxiv_field` has the same flaw (29 papers from 2025) and reaches into
July 2026; one paper is in both sets. A future arXiv task should filter
on the id's year and month.

[report.md](report.md) has every number of the plan's decision;
`summary.json` the same as data.

## The tasks

| task | family | examples | options | accuracy |
|---|---|---|---|---|
| `biorxiv_subject` | topic | 766 | 25 | 65.0% |
| `arxiv_cs` | topic | 1,000 | 10 | 73.8% |
| `stackexchange_site` | topic | 903 | 10 | 95.8% |
| `github_pr` | intent | 965 | 4 | 76.6% |

`medrxiv_subject` was built and frozen but dropped by the plan's rule: 525
examples, under the 600 it requires. arXiv's OAI endpoint was down when the
other three were built; `arxiv_cs` was built with the same code when it
came back, and frozen before its run like the others.

## Notes on the run

- GLM-5.3-Flash through Privatemode's proxyless API (the first test used
  the proxy), raw probabilities (T = 1), 4 requests in flight, one task at
  a time; no request failed.
- Scored with the library at `df9541c`; the one-sequence rule is the same
  library with `LATE_START = 0`.
- The bioRxiv and arXiv abstracts (CC BY, CC0) go in a release of their
  own (`calibration-fresh-2026-11-06`), not published yet: until it is,
  `fetch --set fresh` fails and the texts exist only where they were built.
  The Stack Exchange questions and GitHub pull requests are fetched again
  by id.

How each builder read "published in the window" (the plan didn't say):

- bioRxiv: every 4th day from July 1. The build ran on Berlin time, which
  stepped over the end of daylight saving time: its last day was October
  28, not 29. `days()` now counts in UTC.
- medRxiv (dropped): every day, but on Berlin time October 25 twice and
  October 31 not at all.
- arXiv: every 2nd day from July 1, stopping once every category had 300
  candidates.
- Stack Exchange: the first 100 questions of each month on each site, so
  the first days of each month. Three sites had fewer than the plan's 100
  in all: history 47, computer science 78, biology 78.
- GitHub: the first 300 search results of days 1–28 of each month per
  label (`github_issue` took 200).

## Reproduce

```sh
python -m bench.holdout_data fetch --set fresh     # needs the release (see above)
python -m bench.holdout_data check --set fresh
python -m bench.holdout run --task arxiv_cs --out runs/fresh     # each of the four
python -m bench.holdout report --set fresh --run runs/fresh --out results/calibration/fresh
python -m bench.holdout report --set fresh --window-only --run runs/fresh --out results/calibration/fresh/window-only
```
