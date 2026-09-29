# Held-out test: calibration on five untouched tasks

Every choice in parts 1–3 was made on the same 28 text datasets. This is
the final test the plan fixed beforehand
([../part-3/holdout-plan.md](../part-3/holdout-plan.md)): the criteria were
written on 2026-09-26 before any task was chosen. The five tasks were
chosen, frozen and committed before the first request. They were run once
on 2026-09-29 and are reported as they came out, with nothing tuned on
them.

**Result: every criterion passes.** The zero-label default temperature
recovers 77% of the per-task gain on the new tasks, against 71% on the
known ones.

## The tasks

| task | family | options | examples | accuracy (raw) |
|---|---|---|---|---|
| `fin_topic`: financial news posts, topic | finance (new) | 20 | 1,000 | 71.0% |
| `fin_sentiment`: financial news posts, sentiment | finance (new) | 3 | 1,000 | 81.0% |
| `arxiv_field`: 2026 arXiv abstracts, field | science (new) | 8 | 1,000 | 76.4% |
| `pubmed_study`: 2026 PubMed abstracts, study type | medical (new) | 5 | 1,000 | 91.8% |
| `github_issue`: 2026 GitHub issues, bug, feature, docs or question | code (new) | 4 | 1,000 | 73.6% |

Sources, licences and selection rules are in the plan and in
`bench/holdout_data.py`. `datasets/holdout/` holds the frozen ids, labels
and text hashes. The arXiv, PubMed and GitHub tasks were published after
GLM-5.3-Flash's training cutoff, and none of the families is in the
benchmark. Each task is split 50/50 into calibration and test halves as
before; every number here is on the test halves, each dataset weighted
equally.

## Against the pre-registered criteria

| measure | pass if | result | |
|---|---|---|---|
| excess ECE, no labels (default temperature) | mean ≤ 0.06 | 0.047 | pass |
| | no dataset above 0.12 | highest 0.081 (arxiv_field, github_issue) | pass |
| 90% sets, `calibrate()` on the calibration half | mean coverage 0.87–0.93 | 0.912 | pass |
| automation at 10% error, 50 draws of 100 labels | ≤ 10% of draws over | 0.8% | pass |
| automation at 10% error, 50 draws of all labels | ≤ 10% of draws over | 0.0% | pass |
| accuracy as given → after the bias from 100 labels | no loss on average | 78.8% → 81.5% | pass |

## Without labels

| method | T (median) | confidence | accuracy | excess ECE | share of gain | on the 28 known datasets |
|---|---|---|---|---|---|---|
| raw | 1 | 95.3% | 78.8% | 0.151 | 0% | 0.129 |
| one T for all tasks (2.15) | 2.15 | 86.1% | 78.8% | 0.057 | 70% | 0.040 (63%) |
| **default: option-count formula** | 2.32 | 85.1% | 78.8% | **0.047** | **77%** | 0.032 (71%) |
| T from the task family | 2.60 | 81.9% | 78.8% | 0.051 | 74% | 0.029 (74%) |
| T per task (fitted on its calibration half) | 2.62 | 79.7% | 78.8% | 0.016 | 100% | 0.006 |

- **Raw probabilities are more overconfident here than on the known tasks**
  (95% stated against 79% right). The shipped formula removes about three
  quarters of that without a single label, as it did on the known tasks.
- **The formula beats one global T on four of five tasks.** It gives fewer
  options more softening: fin_sentiment goes from 0.018 to 0.001.
- **The family T is only as good as the family you name.** Named in the
  plan before the run, GitHub issue types went under intent. Intent's T
  (1.76) softens them far too little: 0.142 against the formula's 0.081.
  For topic-like tasks it helps. For a new kind of task, the formula is the
  safer default.
- **Where zero labels fall short:** arXiv and GitHub still carry 0.08
  excess ECE. Their own best T (2.83 and 3.46) is well above the formula's.
  Both are harder than the model's confidence suggests, which is the
  failure part 1 predicted for new families. A few labels fix it (next
  section).

## With labels: `calibrate()`

The library's `calibrate(coverage=0.9, max_error=0.10)`, run on answers at
the default temperature. Each row averages 50 draws per task:

| fit | accuracy | excess ECE | 90% sets: coverage / options | single option | automated at 10% | draws over 10% |
|---|---|---|---|---|---|---|
| T, 100 labels | 78.8% | 0.023 | 0.905 / 1.61 | 64% | 10% | 6/250 (2.4%) |
| T, all labels (~500) | 78.8% | 0.017 | 0.904 / 1.54 | 67% | 13% | 50/250, i.e. 1 of 5 tasks |
| **T + bias, 100 labels (default)** | **81.5%** | 0.007 | 0.912 / 1.48 | 70% | 9% | 2/250 (0.8%) |
| T + bias, all labels | 82.3% | 0.001 | 0.914 / 1.42 | 72% | 15% | 0/250 |

- **The bias per option gains 2.7 points from 100 labels** (part 3 on the
  known tasks: 2.0), and every task gains: fin_topic +1.3, fin_sentiment
  +3.3, arxiv_field +3.7, pubmed_study +0.9, github_issue +4.4.
- **Coverage holds** at 0.90–0.93 on every task, with 1.0–2.2 options per
  set.
- **The one bound violation is expected by design.** With the temperature
  alone and all labels, arXiv automated 66% of its test half at 11.4% error.
  Without a bias, the 50 draws of "all labels" are the same fit, so that is
  one task out of five. The guarantee allows it for 10% of tasks
  (`delta = 0.1`). The default (with the bias) stayed under the bound on
  every task.

## A finding: more labels can automate less

With the temperature alone, three tasks automate a little with 100 labels
and nothing with all ~500. The cause is the order of the test, not the
fit.

`calibrate()` tests thresholds from the most confident answers down, and
stops at the first level that fails. Its first level is 5% of the labels,
or the smallest count that could pass (22 at ε = 10%):

- With 100 labels, it starts at the top 25.
- With 500 labels, it also starts at the top 25, which is now only the
  first 5%.

On these tasks the 25 most confident answers already contain errors: 2 on
pubmed_study, 1 on fin_sentiment, 4 on github_issue. They look like label
noise, for example abstracts that read as systematic reviews but are
indexed only as "Review". So the first test fails and nothing is automated.

The bound stays safe; this only costs automation. Per the plan, any fix
(e.g. starting the sequence at a minimum share of the labels) has to be
checked on a fresh set of held-out tasks, not on these.

## Notes on the run

- GLM-5.3-Flash through the production Privatemode proxy, raw probabilities
  (T = 1), 4 requests in flight per task, with up to three tasks at once.
  The endpoint reported `glm-5.3-flash` for every row, and on average
  97.8–99.9% of the probability landed on the options.
- 32 requests were throttled (HTTP 429) and asked again, into the same run
  file.
- One resume was started by mistake with 2 requests in flight. That setting
  is part of the run key, so it re-asked three tasks into separate files.
  Those files were deleted unread, before the report existed: the plan
  allows one run.
- Jev was not run on these tasks.

## Reproduce

```sh
python -m bench.holdout_data build          # refetch; texts are cached under .cache/holdout
python -m bench.holdout_data check          # verify them against the frozen hashes
python -m bench.holdout run --task fin_topic --out runs/holdout   # and the other four
python -m bench.holdout report --run runs/holdout --out results/calibration/holdout
```

`report.md` and `summary.json` are what `bench.holdout report` wrote; this
README summarises them. The run files will join the release
`calibration-2026-09-26` (not yet published).
