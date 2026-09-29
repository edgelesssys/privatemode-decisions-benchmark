# Several questions about one state: screening

GLM-5.3-Flash, 14 dev datasets, rows from the calibration halves. Every arm is compared with `B` on the same rows. *B2* is the baseline asked again: the noise floor between runs.

## Screening

Accuracy in points against the baseline, pooled over all rows with a paired bootstrap 95% interval; datasets better / worse (in brackets: significant by McNemar, p < 0.05); the worst control dataset (sst2, dbpedia_14, banking77, clinc150); NLL after a temperature fitted per arm on half the rows, on the other half; the median of those temperatures (the baseline's is 1.96); option mass; how far the mean answer moves from the baseline's (total variation; anchoring or a position prior); extra prompt tokens; the change in median latency (4 requests in flight); EUR per 1,000 decisions (baseline 0.116).

| arm | accuracy points [95% CI] | datasets + / − (sig.) | worst control | NLL after own T | own T | option mass | mean-answer shift | extra tokens | latency, ms | EUR / 1,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 | -0.30 [-0.80, +0.21] | 5 / 6 (0) | -0.8 | 0.559 (B 0.553) | 1.98 | 0.976 | 0.006 | 0 | -63 | 0.116 |
| QA | +0.89 [-0.15, +1.98] | 7 / 5 (2) | -1.2 | 0.539 (B 0.553) | 1.85 | 0.936 | 0.053 | 685 | +62 | 0.253 |
| R-Q | +2.18 [+1.21, +3.16] | 12 / 2 (3) | +0.8 | 0.499 (B 0.553) | 2.07 | 0.945 | 0.052 | 419 | +14 | 0.200 |

**Gate** (accuracy-plan Phase 2): pooled ≥ +1.0 point with the interval above 0; no control dataset losing more than the noise floor (the largest control difference between B and B2, 0.8 points, or 1 point if larger); NLL after temperature not worse; latency +30 ms at most; for filler, the placement control (F-before) must not show the same gain.

| arm | accuracy | controls | NLL | latency | placement | verdict |
|---|---|---|---|---|---|---|
| B2 | **fail** | pass | **fail** | pass | — | fails |
| QA | **fail** | **fail** | pass | **fail** | — | fails |
| R-Q | pass | pass | pass | pass | — | **passes** |

## Per dataset

Accuracy points against the baseline (McNemar p < 0.05 marked *):

| dataset | group | B | B2 | QA | R-Q |
|---|---|---|---|---|---|
| ag_news | topic | 0.864 | -1.6 | +5.2* | +2.4 |
| banking77 | intent | 0.776 | +0.0 | +3.2 | +4.0* |
| boolq | NLI / QA | 0.924 | +0.0 | -3.6 | -1.6 |
| clinc150 | intent | 0.868 | -0.4 | -1.2 | +3.2 |
| dbpedia_14 | topic | 0.976 | +0.4 | +0.4 | +0.8 |
| gnad10 | topic | 0.636 | +1.6 | -2.8 | +2.4 |
| massive_scenario_en | intent | 0.804 | +0.4 | +2.8 | +0.8 |
| mnli | NLI / QA | 0.868 | -0.8 | -2.4 | +1.2 |
| patent | topic | 0.588 | -2.0 | -2.4 | -2.0 |
| rte | NLI / QA | 0.891 | +0.7 | +0.7 | +4.3 |
| sst2 | sentiment | 0.936 | -0.8 | +0.0 | +1.2 |
| sst5 | sentiment | 0.472 | +0.0 | +9.6* | +10.0* |
| toxic_conversations | sentiment | 0.812 | -1.6 | +0.0 | +4.0* |
| xnli_de | NLI / QA | 0.808 | +0.4 | +2.8 | +0.8 |

## By task family

| arm | NLI / QA | intent | sentiment | topic |
|---|---|---|---|---|
| B2 | +0.1 | +0.0 | -0.8 | -0.4 |
| QA | -0.6 | +1.6 | +3.2 | +0.1 |
| R-Q | +1.2 | +2.7 | +5.1 | +0.9 |

## Where the gains are: by the baseline's confidence (H4)

Rows pooled over datasets, split into five equal groups by the baseline's top probability; accuracy points against the baseline in each:

| arm | least sure fifth | 2 | 3 | 4 | surest fifth |
|---|---|---|---|---|---|
| B2 | -1.3 | -0.1 | +0.0 | +0.0 | +0.0 |
| QA | +5.9 | -1.0 | -0.3 | -0.1 | +0.0 |
| R-Q | +9.7 | +0.9 | +0.1 | +0.1 | +0.0 |

## Offline ensembles

From the stored distributions, no extra requests beyond the arm's: the mean of the baseline and the arm (two reads for every decision), and the arm's answer only where the baseline's top probability is below 0.9 (the share of decisions that need the second read in brackets). Accuracy points against the baseline:

| arm | mean of B and arm | arm where B < 0.9 |
|---|---|---|
| B2 | -0.17 | -0.26 (21%) |
| QA | +1.18 | +1.11 (21%) |
| R-Q | +1.69 | +1.84 (21%) |
