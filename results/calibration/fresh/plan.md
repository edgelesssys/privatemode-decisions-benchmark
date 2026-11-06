# A fresh held-out test of the second automation sequence: what will be measured

Written on 2026-11-06, before any of these tasks was chosen, fetched or
run. The first held-out test ([../holdout](../holdout/README.md)) found
that with the temperature alone more labels can automate less
([privatemode-decisions#4](https://github.com/edgelesssys/privatemode-decisions/issues/4)).
The library's answer is a second test sequence from a quarter of the
labels ([#5](https://github.com/edgelesssys/privatemode-decisions/pull/5),
`df9541c`). It splits `delta`, so a set whose top only just passes can now
automate less than before. The first plan says a fix is checked on a fresh
set of held-out tasks, not on the five it was found on. This is that check,
and it decides whether the second sequence stays.

## What is compared

Two rules for `calibrate(max_error=0.10)`'s automation threshold, on the
same answers and the same draws of labels:

- **one sequence:** the rule before #5 (the library at `df9541c` with
  `LATE_START = 0`, which never starts the second sequence);
- **two sequences:** the library at `df9541c` as shipped.

Everything else follows the first plan: GLM-5.3-Flash, raw probabilities
(T = 1) through Privatemode, at most 4 requests in flight per task and at
most 10 in all, each task split 50/50 with `bench.calibration.split`,
answers at the default temperature, `bias=False` and `bias=True`, 50 draws
of 100 labels and 50 of all calibration labels (in a new order), scored on
the test half.

## The decision

Means over the tasks, each weighted equally, with all calibration labels:

| question | measure | the second sequence stays if |
|---|---|---|
| does it keep the bound? | draws whose automated answers err more than 10%, pooled over tasks and both fits | at most 10% of draws |
| is it worth its cost? | automated share of the test half, two sequences minus one sequence | ≥ 0 for the temperature alone and ≥ 0 for T + bias |

If either fails, the second sequence goes (or a change to it is checked on
yet another fresh set); nothing is tuned on these tasks.

Reported, not decided on:

- per task and fit, the automated share under each rule, and every task
  where two sequences automate less than one;
- the symptom of #4: task and fit pairs where 100 labels automate more than
  all labels, under each rule;
- with 100 labels the two rules agree by construction (below the switch);
  reported as a check of the comparison;
- the first plan's criteria (excess ECE, coverage, automation bound,
  accuracy after the bias) on these tasks, with its thresholds.

## Which tasks

Five text classification tasks, none of them a dataset of the benchmark or
of the first held-out set, all published from 2026-07-01 to 2026-10-31
(after the first set's window and the model's training cutoff), with public
labels and a licence that allows this use. Built by a script without
asking any model: the examples are drawn with seed 0, each task frozen
(ids, labels, text hashes) and committed before its run. Up to 1,000
examples each, so the calibration half holds about 500 labels, above the
switch (173 at ε = 10%).

| task | family | source | options | label |
|---|---|---|---|---|
| `biorxiv_subject` | topic | bioRxiv API, posted in the window, CC BY | bioRxiv's subject categories | the category the authors chose; up to 40 per category |
| `medrxiv_subject` | topic | medRxiv API, posted in the window, CC BY | medRxiv's subject categories with at least 25 posts | the category the authors chose; 25 per category, up to 1,000 |
| `arxiv_cs` | topic | arXiv OAI-PMH, first submitted in the window, CC BY 4.0 or CC0 | cs.AI, cs.CL, cs.CR, cs.CV, cs.DC, cs.HC, cs.IR, cs.LG, cs.RO, cs.SE | the primary category; 100 per category |
| `stackexchange_site` | topic | Stack Exchange API, questions asked in the window (CC BY-SA, by id, not republished) | math, physics, stats, cs, biology, chemistry, history, philosophy, english, travel | the site it was asked on; 100 per site |
| `github_pr` | intent | GitHub pull requests opened in the window in public MIT, Apache-2.0 or BSD repositories, not by bots | bug fix, feature, documentation, dependency update | the one of bug, enhancement, documentation, dependencies on it; at most 5 per repository, 250 per label |

The input is the abstract, or the title and body (template headings,
checklists and title tags removed as in `github_issue`). A source that
can't deliver at least 600 examples is dropped and reported, not replaced;
with fewer than four tasks left the test is reported as incomplete.
