# Calibration of GLM-5.3-Flash one-token decisions

Can the per-option probabilities be trusted, and what makes them trustworthy
without fine-tuning? Two identical runs of the Privatemode arm on all 29
datasets (up to 1,000 examples each, `answer=` prefill), each dataset split
50/50 into calibration and test halves. Means weight every dataset equally.
ECE uses 15 equal-size bins of top-answer confidence. *Excess ECE* subtracts
the floor that sampling alone produces on a dataset's few hundred test
examples (about 0.05), so 0 means as calibrated as the sample can show.

- [Part 1](part-1/README.md): how overconfident the raw probabilities are,
  temperature without and with labels, contextual calibration, conformal
  prediction sets, and checks for label noise, contamination and off-option
  mass.
- [Part 2](part-2/README.md): Jev on the same examples, a guaranteed error
  rate for automated answers, fitting a task from few labels, per-class
  coverage, scanned documents, and position bias.
- [Part 3](part-3/README.md): a bias per option from labels, temperatures
  for several option orders, many options with few labels, order stability,
  and Kimi K2.6 and GLM-5.3.

## Results

| question | answer |
|---|---|
| Are raw probabilities calibrated? | No. They are overconfident on all 28 text datasets, by 14.6 points on average (ECE 0.146). |
| Does one number fix it? | Yes, per task: a temperature fitted on the task brings ECE to the sampling floor. The best T (1.4–3.7) follows difficulty and is stable between runs. |
| Without labels? | A temperature from the option count recovers 71% of the per-task gain on held-out datasets (excess ECE 0.129 → 0.032), and the same when related datasets or half of all tasks are held out; one from the task family 74%. It's the library's default. For a new kind of task, with its whole family held out, it recovers 45%. |
| Better than Jev? | On these 28 datasets, yes, at the same accuracy; the method was tuned on them, and the pre-registered test on untouched tasks ([part-3/holdout-plan.md](part-3/holdout-plan.md)) hasn't run yet. With no labels, excess ECE 0.032 against Jev's 0.080 as returned, and 0.042 with a default temperature fitted for Jev the same way. With ~500 labels 0.006 against 0.021, even after fixing Jev's zeros so a temperature can be fitted. Jev rounds to 0.01 and prices the right answer at exactly 0 in 4.3% of examples. |
| With a few labels? | A task temperature pulled towards the default: ECE 0.066 from 20 labels, 0.052 from 500. Isotonic regression only catches up at about 500. |
| Prediction sets? | 90% sets cover 90.8% with 1.8 options on average; 64% of answers are a single option. A cutoff per class restores a rare class from 70% to 97% coverage. |
| Automation with an error bound? | Learn then Test kept a 10% bound in every test; the naive threshold broke it about 40% of the time. The price: 19% automated at ε = 5%, 42% at 10%. |
| Removing bias without labels? | Hurts or doesn't help. Neutral inputs cost up to 16 points of accuracy and batch calibration 0.5 on average; averaging option rotations changes accuracy by −0.3 points (not significant) for 4× the requests. The bias they remove is mostly real knowledge or the real class balance. |
| Anything from option mass? | No. The model puts about 99% on the options, right or wrong. |
| Can labels also fix accuracy? | Yes: a bias per option next to the temperature gains 2.0 points from 100 labels (+1.0 from 20, +3.0 from 500), most where the model over-predicts a class (toxic_conversations +15). Cutoffs and thresholds then have to be set out-of-fold, or the error bound slips. |
| Several option orders? | They keep the one-order temperature; a formula per number of orders was worse on held-out datasets. The answer changes in 9.0% of rotations. |
| Many options, few labels? | One cutoff. Clustered conformal and grouping classes by the answers barely beat it at a few labels per class, and a cutoff per class puts 70 options in every set. |
| Other models? | Kimi K2.6 follows the same recipe (formula recovers 79% of the gain) and reads scanned documents best (81.8%). GLM-5.3's temperature doesn't depend on the option count, and it puts only 64% of its probability on the options after `answer=`. |

![Reliability](part-1/reliability.png)

**Reading the figure.** Each point is a bin of answers with about the same
stated confidence (x) and the share of them that was right (y). A calibrated
system sits on the diagonal (left) or on the 0 line (right; below 0 means it
claimed more than it delivered). Raw GLM says 90% where it is right 60% of the
time. Jev is less extreme but still 15–18 points too sure in the middle.
GLM with the default temperature stays within about 6 points everywhere, with
no labels. Jev with a default temperature of its own lands in between, within
about 10 points. Its curve zig-zags above 0.85 because Jev rounds to 0.01:
most of its answers share a few confidence values, and the equal-size bins
split them up.

**What the methods are.** Each one divides GLM's log probabilities by a
temperature T > 1, which softens them without changing the chosen answer.
They differ only in where T comes from. Every row covers the same 13,005
test examples, and every T was fitted without them:

| method | where T comes from | labels | T (median) | confidence | accuracy | excess ECE | share of gain |
|---|---|---|---|---|---|---|---|
| raw | none (T = 1) | none | 1 | 92.7% | 78.1% | 0.129 | 0% |
| one T for all tasks | one value fitted on all the other datasets | none | 2.15 | 80.0% | 78.1% | 0.040 | 63% |
| **default (option-count formula)** | `log T = 0.96 − 0.076·log(options)`, fitted on all the other datasets | none | 2.22 | 80.4% | 78.1% | **0.032** | 71% |
| T from the task family | one T per kind of task (sentiment 2.9, intent 1.8, …) | none | 2.50 | 78.9% | 78.1% | 0.029 | 74% |
| neutral-input correction + T | divide out the answer to empty input, then a T; changes answers | none | 2.33 | 76.0% | 75.0% | 0.059 | 40% |
| T per task | fitted on the task's own calibration half | ~500 | 2.03 | 79.3% | 78.1% | 0.006 | 100% |
| Jev, as returned | none; rounded to 0.01 | none | 1 | 87.5% | 77.5% | 0.080 | — |
| Jev, zeros set to 0.005 | none; makes a T fittable | none | 1 | 80.8% | 77.5% | 0.082 | — |
| **Jev, zeros set + default T** | GLM's recipe refitted on Jev: `log T = 0.62 − 0.19·log(options)`, fitted on Jev's other datasets | none | 1.25 | 80.0% | 77.5% | **0.042** | 40% |
| Jev, zeros set + T per task | fitted on the task's own calibration half | ~500 | 1.19 | 78.8% | 77.5% | 0.021 | 100% |

- **Confidence − accuracy** is the overconfidence in points.
- **Excess ECE** is the calibration error left above what sampling alone
  produces on a test half this size, so 0 is as good as the sample can show.
- **Share of gain** is how much of the per-task temperature's improvement a
  zero-label method achieves. For Jev it runs from its probabilities as
  returned to its own per-task T.
- **Jev.** The rows use Jev's published probabilities for the same examples.
  A temperature can't be fitted on raw Jev, because 4.3% of its right answers
  are priced at exactly 0, hence the "zeros set" rows. Jev's default T gets
  the same treatment as GLM's: the option-count formula fitted on Jev's own
  per-task temperatures, leaving out the dataset being scored. One T for all
  tasks does worse for Jev (excess ECE 0.089). Jev needs little softening
  (median T 1.2 against GLM's 2.0), but how much depends on the task, so a
  zero-label T helps it less: 40% of its per-task gain against GLM's 71%.
  **At zero labels GLM stays ahead, 0.032 against 0.042.**
- **Source.** The table is generated by `bench.calibrate_report` and repeated
  with per-dataset detail in [part 1](part-1/full-report.md).

## Where part 3 leaves it

**(a) Against the state after part 2.** GLM-5.3-Flash, 28 text datasets,
test halves, `calibrate()` with n random labels from the calibration half
(20 draws per dataset). Without labels nothing changed: the default
temperature is part 1's.

| | part 2: temperature only | part 3: temperature + bias, out-of-fold cutoffs |
|---|---|---|
| accuracy, 100 labels | 78.1% | **80.1%** |
| accuracy, 500 labels (23 datasets) | 76.1% | **79.1%** |
| NLL, 100 labels | 0.683 | **0.631** |
| excess ECE, 100 labels | 0.008 | **0.004** |
| 90% sets, 100 labels: coverage / options | 0.909 / 1.85 | 0.915 / **1.70** |
| automated at a 10% error bound, 100 labels | **27%** | 23% |
| draws over the bound, 100 / 500 labels (10% allowed) | 1.1% / 0.0% | 0.9% / 1.1% |
| models with a default temperature | GLM-5.3-Flash | GLM-5.3-Flash, Kimi K2.6, GLM-5.3 |

The price of the accuracy is some automation at a fixed error bound, since
out-of-fold probabilities are more cautious; `calibrate(..., bias=False)`
keeps part 2's behaviour.

**(b) Against Jev, calibrated the same way.** Jev's zeros set to half a
rounding unit so that it can be calibrated at all, then everything GLM gets:
its own zero-label default temperature (the option formula refitted on Jev),
and from labels a task temperature, a bias per option and out-of-fold
cutoffs. Same examples:

| | GLM-5.3-Flash | Jev |
|---|---|---|
| accuracy as returned | 78.1% | 77.5% |
| excess ECE, no labels (default T) | **0.032** | 0.042 |
| accuracy, 100 labels (T + bias) | **80.2%** | 79.6% |
| excess ECE, 100 labels (T + bias) | **0.005** | 0.014 |
| 90% sets, 100 labels: options | **1.71** | 1.79 |
| automated at a 10% error bound, 100 labels | 23% | 25% |
| accuracy, 500 labels (T + bias) | **79.1%** | 78.7% |
| excess ECE, 500 labels (T + bias) | **0.003** | 0.007 |
| 90% sets with more than 20 options, one cutoff | **1.60** | 3.53 |

The bias gains both systems about 2 points from 100 labels. After the same
calibration GLM stays ahead on accuracy, calibration and set size, and
automates about the same share.

## In the library

- A default temperature per measured model (GLM-5.3-Flash, Kimi K2.6,
  GLM-5.3), from the option count, or from a task family
  (`temperature="sentiment"`); `temperature=1` gives raw probabilities.
- `calibrate(answers, labels, coverage=0.9, per_class=False, max_error=None,
  bias=True)` returns a task temperature, a bias per option, conformal
  cutoffs and an automation threshold: `apply(answer)` (the corrected
  answer, whose choice can change), `predict_set(answer)` and
  `automate(answer)`.
- The prefill `answer=`, so about 99% of the probability lands on the options
  for GLM-5.3-Flash and Kimi K2.6. After `answer:` the model wanted a space
  first.

## Reproduce

The commands are in each part. The raw runs will be in the release
`calibration-2026-09-26` (not yet published), with every run of parts 1–3, including the Kimi K2.6 and GLM-5.3 runs.
