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

## Results

| question | answer |
|---|---|
| Are raw probabilities calibrated? | No. They are overconfident on all 28 text datasets, by 14.6 points on average (ECE 0.146). |
| Does one number fix it? | Yes, per task: a temperature fitted on the task brings ECE to the sampling floor. The best T (1.4–3.7) follows difficulty and is stable between runs. |
| Without labels? | A temperature from the option count recovers 71% of the per-task gain on held-out datasets (excess ECE 0.129 → 0.032), and the same when related datasets or half of all tasks are held out; one from the task family 74%. It's the library's default. For a new kind of task, with its whole family held out, it recovers 45%. |
| Better than Jev? | Yes, at the same accuracy. With no labels, excess ECE 0.032 against Jev's 0.080. With ~500 labels 0.006 against 0.021, even after fixing Jev's zeros so a temperature can be fitted. Jev rounds to 0.01 and prices the right answer at exactly 0 in 4.3% of examples. |
| With a few labels? | A task temperature pulled towards the default: ECE 0.066 from 20 labels, 0.052 from 500. Isotonic regression only catches up at about 500. |
| Prediction sets? | 90% sets cover 90.1% with 1.8 options on average; 63% of answers are a single option. A cutoff per class restores a rare class from 70% to 97% coverage. |
| Automation with an error bound? | Learn then Test kept a 10% bound in every test; the naive threshold broke it about 40% of the time. The price: 19% automated at ε = 5%, 42% at 10%. |
| Removing bias without labels? | Hurts or doesn't help. Neutral inputs cost up to 16 points of accuracy and batch calibration 0.5 on average; averaging option rotations changes accuracy by −0.3 points (not significant) for 4× the requests. The bias they remove is mostly real knowledge or the real class balance. |
| Anything from option mass? | No. The model puts about 99% on the options, right or wrong. |

![Reliability](part-1/reliability.png)

## In the library

- A default temperature for GLM Flash, from the option count, or from a task
  family (`temperature="sentiment"`); `temperature=1` gives raw
  probabilities.
- `calibrate(answers, labels, coverage=0.9, per_class=False, max_error=None)`
  returns a task temperature, conformal cutoffs and an automation threshold:
  `predict_set(answer)` and `automate(answer)`.
- The prefill `answer=`, so about 99% of the probability lands on the options.
  After `answer:` the model wanted a space first.

## Reproduce

The commands are in each part. The raw runs are in the release
`calibration-2026-09-26`.
