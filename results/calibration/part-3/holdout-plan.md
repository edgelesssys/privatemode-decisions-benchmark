# A final test on untouched tasks: what will be measured

Written on 2026-09-26, before any held-out task was chosen, curated or run.
Every method choice in parts 1 to 3 (the formula's form, the shrinkage, the
prefill, the bias and its strength, out-of-fold cutoffs) was made on the
same 28 text datasets. Stricter splits of those datasets can't measure how
much that choice fitted them; only tasks nobody has looked at can.

## When

After the library's calibration code and constants are final. That includes
the outcome of the accuracy plan: a prompt change moves the temperatures and
needs new constants first. The tasks are run once; the result is reported
whatever it is, and nothing is tuned on it afterwards. If it fails, a fix is
checked on a fresh set of held-out tasks.

## Which tasks

4–6 text classification datasets with public labels and a license that
allows this use, ideally published after GLM-5.3-Flash's training cutoff,
including at least one task family the benchmark doesn't have (for example
medical, finance or code). A new family is where the zero-label default is
weakest: with its whole family held out it recovered 45% of the per-task
gain, against 71% otherwise. Up to 1,000 examples each, split 50/50 with
`bench.calibration.split` as before, about 5,000 requests in all.

Candidates to check (licenses not yet verified, none downloaded):

| candidate | family | options |
|---|---|---|
| a financial news topic set (e.g. Twitter financial news topics) | finance | ~20 |
| a financial sentiment set | finance | 3 |
| a clinical or biomedical sentence classification set (e.g. PubMed RCT sentence roles) | medical | 5 |
| a programming-question or code-language classification set | code | 5–20 |
| a recent intent or topic set published after the model's cutoff | intent / topic | 10–50 |

## What counts as a pass

Measured with the shipped library on the test halves, GLM-5.3-Flash, one
run, means over the new datasets with each weighted equally:

| measure | how | pass |
|---|---|---|
| excess ECE, no labels | default temperature (option formula), ECE minus sampling floor | mean ≤ 0.06, no dataset above 0.12 |
| coverage | `calibrate(coverage=0.9)` on the calibration half, 90% sets on the test half | mean between 0.87 and 0.93 |
| automation | `calibrate(max_error=0.10)`, 50 draws of 100 and of all calibration labels | at most 10% of draws over 10% error on the test half |
| accuracy | the answer as given, and after `calibrate()`'s bias from 100 labels | reported; the bias must not lose accuracy on average |

The thresholds come from what parts 1–3 found on the known tasks (excess
ECE 0.032 by leave-one-dataset-out and 0.054 with a whole family held out;
coverage 0.898–0.909; at most 2.1% of draws over the bound), with room for
fewer datasets. Anything outside is a failure, reported as one.

## The tasks chosen (added 2026-09-29, each before its own run)

Chosen and frozen by `bench.holdout_data` without asking any model, each
committed before its own run (not all before the first request: GitHub and
arXiv were frozen while the others ran; corrected after the run). Up to 1,000 examples each; the input is
the text or abstract alone, no titles or metadata.

| task | family | source | licence | options | selection |
|---|---|---|---|---|---|
| `fin_topic` | finance | zeroshot/twitter-financial-news-topic, validation | MIT | 20 | 1,000 random rows (seed 0) |
| `fin_sentiment` | finance | zeroshot/twitter-financial-news-sentiment, validation | MIT | 3 | 1,000 random rows (seed 0) |
| `arxiv_field` | science | arXiv OAI-PMH, papers first submitted in 2026 | CC BY 4.0 or CC0 | 8 | primary category's top-level group, up to 125 per group |
| `pubmed_study` | medical | Europe PMC, first published Jan–Jun 2026, open access | CC BY | 5 | study type from the indexed publication types, only where exactly one applies (a systematic review may also be indexed as a review), 200 per type |
| `github_issue` | code | GitHub issues opened Jan–Jun 2026 in public MIT, Apache-2.0 or BSD repositories | the repository's | 4 | the one of bug, enhancement, documentation, question on it; template headings, checklists and title tags removed; at most 5 per repository, 250 per label |

SEC EDGAR 8-K items were dropped: EDGAR requires a contact address in
every request. The arXiv, Europe PMC and GitHub tasks postdate the model's
training cutoff.

Settings fixed with the tasks:

- **Zero-label methods reported** besides the pass criterion's option
  formula: raw, one global T (2.148, fitted on part 1's 28 datasets), the
  family T, and the task's own T from its calibration half (the ceiling).
  Family used: `fin_topic`, `arxiv_field` and `pubmed_study` topic,
  `fin_sentiment` sentiment, `github_issue` intent.
- **`calibrate()`** is the library's own, on answers at the default
  temperature, with `bias=True` (the default) and `bias=False`. "All labels"
  means the whole calibration half, drawn 50 times in a new order (which
  moves the folds); the coverage criterion uses the half in its stored order
  once.
- **Run:** GLM-5.3-Flash through the production proxy, raw probabilities
  (T = 1), at most 4 requests in flight per task.
