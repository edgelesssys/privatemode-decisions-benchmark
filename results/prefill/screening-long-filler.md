# Longer filler, framed filler, and filler combined with R-Q: screening

GLM-5.3-Flash, 14 dev datasets, rows from the calibration halves. Every arm is compared with `B` on the same rows. *B2* is the baseline asked again: the noise floor between runs.

## Screening

Accuracy in points against the baseline, pooled over all rows with a paired bootstrap 95% interval; datasets better / worse (in brackets: significant by McNemar, p < 0.05); the worst control dataset (sst2, dbpedia_14, banking77, clinc150); NLL after a temperature fitted per arm on half the rows, on the other half; the median of those temperatures (the baseline's is 1.96); option mass; how far the mean answer moves from the baseline's (total variation; anchoring or a position prior); extra prompt tokens; the change in median latency (4 requests in flight); EUR per 1,000 decisions (baseline 0.116).

| arm | accuracy points [95% CI] | datasets + / − (sig.) | worst control | NLL after own T | own T | option mass | mean-answer shift | extra tokens | latency, ms | EUR / 1,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 | -0.30 [-0.80, +0.21] | 5 / 6 (0) | -0.8 | 0.559 (B 0.553) | 1.98 | 0.976 | 0.006 | 0 | -63 | 0.116 |
| F-before-1024 | -0.74 [-1.48, -0.06] | 4 / 6 (1) | -3.2 | 0.569 (B 0.553) | 1.91 | 0.950 | 0.020 | 1024 | +313 | 0.321 |
| F-dots | +0.06 [-0.53, +0.65] | 6 / 4 (1) | -0.4 | 0.552 (B 0.553) | 2.01 | 0.838 | 0.021 | 128 | -36 | 0.142 |
| F-dots-1024 | +0.97 [+0.27, +1.68] | 7 / 3 (1) | +0.0 | 0.545 (B 0.553) | 2.01 | 0.790 | 0.026 | 1024 | +176 | 0.321 |
| F-dots-256 | +0.56 [-0.03, +1.18] | 9 / 4 (1) | -0.8 | 0.551 (B 0.553) | 1.97 | 0.789 | 0.023 | 256 | +287 | 0.167 |
| R-Q | +2.18 [+1.21, +3.16] | 12 / 2 (3) | +0.8 | 0.499 (B 0.553) | 2.07 | 0.945 | 0.052 | 419 | +14 | 0.200 |
| R-think | +0.09 [-0.59, +0.74] | 7 / 6 (2) | -3.2 | 0.561 (B 0.553) | 1.89 | 0.952 | 0.022 | 134 | -12 | 0.143 |
| RF-1024 | +0.32 [-0.41, +1.06] | 8 / 4 (3) | -3.2 | 0.550 (B 0.553) | 1.71 | 0.980 | 0.030 | 1023 | +300 | 0.321 |
| RQ-F | +2.10 [+1.15, +3.13] | 10 / 3 (3) | +0.0 | 0.502 (B 0.553) | 2.23 | 0.763 | 0.065 | 1443 | +174 | 0.405 |
| RQ-FF | +1.83 [+0.86, +2.80] | 11 / 3 (2) | +0.4 | 0.506 (B 0.553) | 2.03 | 0.961 | 0.061 | 1442 | +114 | 0.405 |
| RQ-mid | +1.86 [+0.94, +2.83] | 10 / 3 (2) | +0.0 | 0.507 (B 0.553) | 1.81 | 0.902 | 0.057 | 1444 | +147 | 0.405 |

**Gate** (accuracy-plan Phase 2): pooled ≥ +1.0 point with the interval above 0; no control dataset losing more than the noise floor (the largest control difference between B and B2, 0.8 points, or 1 point if larger); NLL after temperature not worse; latency +30 ms at most; for filler, the placement control (F-before) must not show the same gain.

| arm | accuracy | controls | NLL | latency | placement | verdict |
|---|---|---|---|---|---|---|
| B2 | **fail** | pass | **fail** | pass | — | fails |
| F-before-1024 | **fail** | **fail** | **fail** | **fail** | **fail** | fails |
| F-dots | **fail** | pass | pass | pass | **fail** | fails |
| F-dots-1024 | **fail** | pass | pass | **fail** | pass | fails |
| F-dots-256 | **fail** | pass | pass | **fail** | pass | fails |
| R-Q | pass | pass | pass | pass | — | **passes** |
| R-think | **fail** | **fail** | **fail** | pass | — | fails |
| RF-1024 | **fail** | **fail** | pass | **fail** | — | fails |
| RQ-F | pass | pass | pass | **fail** | — | fails |
| RQ-FF | pass | pass | pass | **fail** | — | fails |
| RQ-mid | pass | pass | pass | **fail** | — | fails |

## Per dataset

Accuracy points against the baseline (McNemar p < 0.05 marked *):

| dataset | group | B | B2 | F-before-1024 | F-dots | F-dots-1024 | F-dots-256 | R-Q | R-think | RF-1024 | RQ-F | RQ-FF | RQ-mid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | topic | 0.864 | -1.6 | +0.4 | +0.0 | +2.0 | +1.6 | +2.4 | +1.6 | +0.4 | +2.8 | +2.8 | +3.2 |
| banking77 | intent | 0.776 | +0.0 | -3.2* | +0.8 | +0.4 | +0.4 | +4.0* | -1.6 | +0.4 | +4.0* | +3.2 | +6.0* |
| boolq | NLI / QA | 0.924 | +0.0 | -1.2 | -2.4* | -0.8 | -1.6 | -1.6 | +1.2 | +0.0 | -1.2 | -0.4 | -1.2 |
| clinc150 | intent | 0.868 | -0.4 | +0.8 | -0.4 | +2.0 | +1.6 | +3.2 | -3.2* | -3.2* | +3.6 | +4.0 | +5.6* |
| dbpedia_14 | topic | 0.976 | +0.4 | +0.4 | +0.4 | +0.0 | +0.0 | +0.8 | +0.0 | +0.0 | +0.0 | +0.4 | +0.0 |
| gnad10 | topic | 0.636 | +1.6 | +0.0 | +0.0 | +0.8 | -1.2 | +2.4 | +0.4 | +2.0 | +2.8 | +1.2 | +2.8 |
| massive_scenario_en | intent | 0.804 | +0.4 | -0.4 | +0.4 | +0.0 | +0.4 | +0.8 | +0.4 | +1.2 | +1.6 | +1.2 | +3.2 |
| mnli | NLI / QA | 0.868 | -0.8 | -3.6 | -1.6 | -2.0 | -1.6 | +1.2 | -1.2 | -3.6 | -3.2 | -2.0 | -2.0 |
| patent | topic | 0.588 | -2.0 | -2.0 | +0.0 | -0.8 | +0.4 | -2.0 | +1.2 | +0.8 | -2.4 | -1.2 | -3.2 |
| rte | NLI / QA | 0.891 | +0.7 | +0.0 | +0.7 | +0.0 | +2.9 | +4.3 | -1.4 | -1.4 | +3.6 | +5.1* | +2.9 |
| sst2 | sentiment | 0.936 | -0.8 | +0.0 | -0.4 | +0.0 | -0.8 | +1.2 | -0.4 | -1.2 | +1.2 | +0.8 | +0.8 |
| sst5 | sentiment | 0.472 | +0.0 | +0.0 | +1.6 | +8.0* | +4.4* | +10.0* | +1.2 | +4.8* | +8.8* | +6.4 | +4.4 |
| toxic_conversations | sentiment | 0.812 | -1.6 | -2.0 | +2.0 | +0.8 | +0.4 | +4.0* | +2.8* | +3.2* | +6.8* | +4.8* | +3.6 |
| xnli_de | NLI / QA | 0.808 | +0.4 | +0.8 | +0.0 | +2.8 | +2.0 | +0.8 | -0.4 | +0.4 | +1.6 | +0.8 | +0.4 |

## By task family

| arm | NLI / QA | intent | sentiment | topic |
|---|---|---|---|---|
| B2 | +0.1 | +0.0 | -0.8 | -0.4 |
| F-before-1024 | -1.0 | -0.9 | -0.7 | -0.3 |
| F-dots | -0.8 | +0.3 | +1.1 | +0.1 |
| F-dots-1024 | -0.0 | +0.8 | +2.9 | +0.5 |
| F-dots-256 | +0.4 | +0.8 | +1.3 | +0.2 |
| R-Q | +1.2 | +2.7 | +5.1 | +0.9 |
| R-think | -0.5 | -1.5 | +1.2 | +0.8 |
| RF-1024 | -1.2 | -0.5 | +2.3 | +0.8 |
| RQ-F | +0.2 | +3.1 | +5.6 | +0.8 |
| RQ-FF | +0.9 | +2.8 | +4.0 | +0.8 |
| RQ-mid | +0.0 | +4.9 | +2.9 | +0.7 |

## Where the gains are: by the baseline's confidence (H4)

Rows pooled over datasets, split into five equal groups by the baseline's top probability; accuracy points against the baseline in each:

| arm | least sure fifth | 2 | 3 | 4 | surest fifth |
|---|---|---|---|---|---|
| B2 | -1.3 | -0.1 | +0.0 | +0.0 | +0.0 |
| F-before-1024 | -1.8 | -1.9 | +0.0 | +0.0 | +0.0 |
| F-dots | +0.6 | -0.3 | +0.0 | +0.0 | +0.0 |
| F-dots-1024 | +4.9 | +0.0 | +0.0 | +0.0 | +0.0 |
| F-dots-256 | +2.8 | +0.0 | +0.0 | +0.0 | +0.0 |
| R-Q | +9.7 | +0.9 | +0.1 | +0.1 | +0.0 |
| R-think | +0.6 | -0.1 | +0.0 | +0.0 | +0.0 |
| RF-1024 | +2.2 | -0.6 | +0.0 | +0.0 | +0.0 |
| RQ-F | +9.6 | +1.0 | -0.1 | +0.0 | +0.0 |
| RQ-FF | +9.1 | +0.1 | -0.1 | +0.0 | +0.0 |
| RQ-mid | +8.7 | +1.0 | -0.4 | +0.0 | +0.0 |

## Offline ensembles

From the stored distributions, no extra requests beyond the arm's: the mean of the baseline and the arm (two reads for every decision), and the arm's answer only where the baseline's top probability is below 0.9 (the share of decisions that need the second read in brackets). Accuracy points against the baseline:

| arm | mean of B and arm | arm where B < 0.9 |
|---|---|---|
| B2 | -0.17 | -0.26 (21%) |
| F-before-1024 | -0.26 | -0.41 (21%) |
| F-dots | +0.06 | +0.11 (21%) |
| F-dots-1024 | +0.21 | +0.83 (21%) |
| F-dots-256 | +0.53 | +0.61 (21%) |
| R-Q | +1.69 | +1.84 (21%) |
| R-think | -0.19 | +0.01 (21%) |
| RF-1024 | +0.09 | +0.35 (21%) |
| RQ-F | +1.86 | +1.86 (21%) |
| RQ-FF | +1.69 | +1.75 (21%) |
| RQ-mid | +1.29 | +1.70 (21%) |
