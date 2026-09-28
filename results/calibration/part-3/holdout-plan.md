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
