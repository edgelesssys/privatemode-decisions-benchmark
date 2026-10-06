# Held-out test: calibration on five untouched tasks

Every choice in parts 1–3 was made on the same 28 text datasets. This is
the final test the plan fixed beforehand
([../part-3/holdout-plan.md](../part-3/holdout-plan.md)): the criteria were
written on 2026-09-26 before any task was chosen. On 2026-09-29 each task
was frozen and committed before its own run (commits 019fa09, 103e3b8,
9629da1; runs from 11:26Z to 11:57Z), not all of them before the first
request: the GitHub and arXiv tasks were frozen while the others ran. The
commits were pushed together with the results (12:02Z), so only their
local timestamps attest to that order. Each task was run once and is
reported as it came out, with nothing tuned on it.

**Result: every criterion passes.** The zero-label default temperature
recovers 78% of the per-task gain on the new tasks, against 71% on the
known ones (the share of the per-task temperature's reduction in ECE, as in
part 1).

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
and text hashes. The release carries the arXiv and PubMed abstracts with the
licence and source of each; the tweets and GitHub issues aren't ours to
republish, so `fetch` gets them again by id (999 of the 1,000 issues still
matched their hashes on 2026-10-01; a run leaves out the rest).
The arXiv, PubMed and GitHub texts were first published in 2026. Finance,
science, medicine and code are new domains to the benchmark; the kinds of
question are not (topic and sentiment are benchmark families). Each task is
split 50/50 into calibration and test halves as before; every number here is
on the test halves, each dataset weighted equally.

## Against the pre-registered criteria

| measure | pass if | result | |
|---|---|---|---|
| excess ECE, no labels (default temperature) | mean ≤ 0.06 | 0.047 | pass |
| | no dataset above 0.12 | highest 0.081 (github_issue) | pass |
| 90% sets, `calibrate()` on the calibration half | mean coverage 0.87–0.93 | 0.912 | pass |
| automation at 10% error, 50 draws of 100 labels | ≤ 10% of draws over | 0.8% | pass |
| automation at 10% error, 50 draws of all labels | ≤ 10% of draws over | 0.0% | pass |
| accuracy as given → after the bias from 100 labels | no loss on average | 78.8% → 81.5% | pass |

## Without labels

| method | T (median) | confidence | accuracy | ECE | excess ECE | share of gain | on the 28 known datasets: excess ECE (share) |
|---|---|---|---|---|---|---|---|
| raw | 1 | 95.3% | 78.8% | 0.165 | 0.151 | 0% | 0.129 (0%) |
| one T for all tasks (2.15) | 2.15 | 86.1% | 78.8% | 0.095 | 0.057 | 70% | 0.040 (63%) |
| **default: option-count formula** | 2.32 | 85.1% | 78.8% | 0.087 | **0.047** | **78%** | 0.032 (71%) |
| T from the task family | 2.60 | 81.9% | 78.8% | 0.095 | 0.051 | 70% | 0.029 (74%) |
| T per task (fitted on its calibration half) | 2.62 | 79.7% | 78.8% | 0.065 | 0.016 | 100% | 0.006 (100%) |

*Share of gain* is the part of the per-task temperature's reduction in ECE
that a method achieves, pooled over the tasks, as part 1 defines it. An
earlier version of this report computed it on excess ECE instead (77% for
the default, 74% for the family T), which isn't comparable with part 1's
shares; on excess ECE the known datasets give 79% for the default.

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
- **Every draw over the bound was on arXiv.** The default (T + bias, 100
  labels) went over in 2 of its 50 draws, the temperature alone in 6 of 50
  at 100 labels (12%, above `delta = 0.1` for that task, though 50 draws
  can't tell 12% from 10%) and in all 50 with all labels. Without a bias
  the 50 "all labels" draws are one fit, so that is a single observation:
  arXiv automated 66% of its test half at 11.4% error. The guarantee is a
  chance per task over the draw of labels; the other four tasks never went
  over.

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

On these tasks the 25 most confident answers already contain errors: 3 on
pubmed_study (two of the 25 tie at the boundary), 1 on fin_sentiment, 4 on
github_issue. They look like label
noise, for example abstracts that read as systematic reviews but are
indexed only as "Review". So the first test fails and nothing is automated.

The bound stays safe; this only costs automation. Per the plan, any fix
(e.g. starting the sequence at a minimum share of the labels) has to be
checked on a fresh set of held-out tasks, not on these. Filed as
[privatemode-decisions#4](https://github.com/edgelesssys/privatemode-decisions/issues/4).

## Notes on the run

- GLM-5.3-Flash through the production Privatemode proxy, raw probabilities
  (T = 1), 4 requests in flight per task, with up to three tasks at once.
  Asked with the library's calibration branch at `cf05f2c`, its head during
  the run (the run files predate recording the library); scored with the
  commit `summary.json` names, which the benchmark pins.
  The endpoint reported `glm-5.3-flash` for every row, and on average
  97.8–99.9% of the probability landed on the options.
- 32 requests were throttled (HTTP 429) and asked again, into the same run
  file.
- One resume was started by mistake with 2 requests in flight. That setting
  was then part of the run key, so it re-asked three tasks into separate
  files. Those files were deleted unread, before the report existed: the
  plan allows one run. Concurrency is no longer part of the key.
- Jev was not run on these tasks.

## Reproduce

```sh
python -m bench.holdout_data fetch          # abstracts from the release, the rest by id
python -m bench.holdout_data check          # against the frozen hashes; up to 10% edited issues reported
python -m bench.holdout report --run runs/holdout --out results/calibration/holdout
python -m bench.holdout run --task fin_topic --out runs/holdout   # asking again, for all five
```

`report.md` and `summary.json` are what `bench.holdout report` wrote from
the run files in the release `calibration-2026-09-26` (`holdout/`); this
README summarises them. `summary.json` names the library commit that
scored them. `bench.holdout_data build` chose the tasks from live sources
and refuses to replace a frozen one: rebuilding gives a different test.
