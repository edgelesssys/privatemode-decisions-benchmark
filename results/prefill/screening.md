# Extra positions before `answer=`: screening

GLM-5.3-Flash, 14 dev datasets, rows from the calibration halves. Every arm is compared with `B` on the same rows. *B2* is the baseline asked again: the noise floor between runs.

## Screening

Accuracy in points against the baseline, pooled over all rows with a paired bootstrap 95% interval; datasets better / worse (in brackets: significant by McNemar, p < 0.05); the worst control dataset (sst2, dbpedia_14, banking77, clinc150); NLL after a temperature fitted per arm on half the rows, on the other half; the median of those temperatures (the baseline's is 1.96); option mass; how far the mean answer moves from the baseline's (total variation; anchoring or a position prior); extra prompt tokens; the change in median latency (4 requests in flight); EUR per 1,000 decisions (baseline 0.116).

| arm | accuracy points [95% CI] | datasets + / − (sig.) | worst control | NLL after own T | own T | option mass | mean-answer shift | extra tokens | latency, ms | EUR / 1,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 | -0.30 [-0.80, +0.21] | 5 / 6 (0) | -0.8 | 0.559 (B 0.553) | 1.98 | 0.976 | 0.006 | 0 | -63 | 0.116 |
| F-alpha | +0.41 [-0.21, +1.06] | 5 / 6 (1) | -0.4 | 0.552 (B 0.553) | 1.98 | 0.796 | 0.018 | 128 | -34 | 0.142 |
| F-before | -0.91 [-1.68, -0.21] | 3 / 11 (3) | -1.2 | 0.568 (B 0.553) | 1.92 | 0.968 | 0.018 | 128 | -10 | 0.142 |
| F-count | +0.53 [-0.12, +1.18] | 6 / 6 (1) | -0.8 | 0.548 (B 0.553) | 1.80 | 0.823 | 0.022 | 129 | -48 | 0.142 |
| F-dots | +0.06 [-0.53, +0.65] | 6 / 4 (1) | -0.4 | 0.552 (B 0.553) | 2.01 | 0.838 | 0.021 | 128 | -36 | 0.142 |
| F-scrambled | -0.18 [-0.80, +0.41] | 3 / 8 (0) | -1.6 | 0.557 (B 0.553) | 1.94 | 0.847 | 0.017 | 128 | -18 | 0.142 |
| F-words | +0.00 [-0.65, +0.62] | 5 / 6 (0) | -1.2 | 0.556 (B 0.553) | 1.89 | 0.807 | 0.019 | 129 | -49 | 0.142 |
| G | -0.30 [-1.00, +0.35] | 3 / 8 (0) | -1.2 | 0.566 (B 0.553) | 1.98 | 0.980 | 0.020 | 119 | -41 | 0.140 |
| H-128 | +2.30 [+1.36, +3.31] | 11 / 3 (2) | -0.4 | 0.603 (B 0.553) | 3.87 | 0.965 | 0.058 | 80 | +1723 | 0.188 |
| H-32 | +0.15 [-0.71, +0.97] | 7 / 6 (0) | -0.8 | 0.582 (B 0.553) | 2.27 | 0.919 | 0.032 | 32 | +804 | 0.143 |
| R-Q | +2.18 [+1.21, +3.16] | 12 / 2 (3) | +0.8 | 0.499 (B 0.553) | 2.07 | 0.945 | 0.052 | 419 | +14 | 0.200 |
| R-QSQS | +2.01 [+1.03, +3.01] | 11 / 3 (3) | +0.4 | 0.500 (B 0.553) | 2.14 | 0.900 | 0.057 | 928 | +258 | 0.304 |
| R-Qi | +0.00 [-0.77, +0.77] | 5 / 7 (1) | -2.4 | 0.553 (B 0.553) | 1.99 | 0.980 | 0.022 | 15 | +199 | 0.119 |
| R-full | +0.89 [+0.09, +1.65] | 9 / 2 (1) | +0.4 | 0.515 (B 0.553) | 2.30 | 0.917 | 0.033 | 508 | +12 | 0.221 |
| R-think | +0.09 [-0.59, +0.74] | 7 / 6 (2) | -3.2 | 0.561 (B 0.553) | 1.89 | 0.952 | 0.022 | 134 | -12 | 0.143 |

**Gate** (fixed before the screening): pooled ≥ +1.0 point with the interval above 0; no control dataset losing more than the noise floor (the largest control difference between B and B2, 0.8 points, or 1 point if larger); NLL after temperature not worse; latency +30 ms at most; for filler, the placement control (F-before) must not show the same gain.

| arm | accuracy | controls | NLL | latency | placement | verdict |
|---|---|---|---|---|---|---|
| B2 | **fail** | pass | **fail** | pass | — | fails |
| F-alpha | **fail** | pass | pass | pass | pass | fails |
| F-before | **fail** | **fail** | **fail** | pass | — | fails |
| F-count | **fail** | pass | pass | pass | pass | fails |
| F-dots | **fail** | pass | pass | pass | pass | fails |
| F-scrambled | **fail** | **fail** | **fail** | pass | pass | fails |
| F-words | **fail** | **fail** | **fail** | pass | pass | fails |
| G | **fail** | **fail** | **fail** | pass | pass | fails |
| H-128 | pass | pass | **fail** | pass | — | fails |
| H-32 | **fail** | pass | **fail** | pass | — | fails |
| R-Q | pass | pass | pass | pass | — | **passes** |
| R-QSQS | pass | pass | pass | **fail** | — | fails |
| R-Qi | **fail** | **fail** | pass | **fail** | — | fails |
| R-full | **fail** | pass | pass | pass | — | fails |
| R-think | **fail** | **fail** | **fail** | pass | — | fails |

## Per dataset

Accuracy points against the baseline (McNemar p < 0.05 marked *):

| dataset | group | B | B2 | F-alpha | F-before | F-count | F-dots | F-scrambled | F-words | G | H-128 | H-32 | R-Q | R-QSQS | R-Qi | R-full | R-think |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | topic | 0.864 | -1.6 | +0.8 | -0.4 | -0.8 | +0.0 | -0.4 | +0.8 | -0.8 | +2.0 | +0.8 | +2.4 | +2.4 | -0.4 | +0.8 | +1.6 |
| banking77 | intent | 0.776 | +0.0 | +1.2 | -0.4 | -0.4 | +0.8 | -0.4 | -1.2 | +0.0 | +2.8 | +1.2 | +4.0* | +5.6* | +0.0 | +4.0* | -1.6 |
| boolq | NLI / QA | 0.924 | +0.0 | +0.0 | -0.8 | -0.8 | -2.4* | -0.4 | -0.8 | -0.4 | -0.8 | -1.6 | -1.6 | -0.4 | -1.6 | +0.0 | +1.2 |
| clinc150 | intent | 0.868 | -0.4 | -0.4 | -1.2 | +1.2 | -0.4 | -1.6 | +0.8 | +0.0 | +2.0 | -0.8 | +3.2 | +3.2 | -2.4 | +2.0 | -3.2* |
| dbpedia_14 | topic | 0.976 | +0.4 | +0.0 | +0.4 | +0.0 | +0.4 | +0.0 | +0.0 | +0.8 | +0.8 | -0.4 | +0.8 | +0.4 | +0.0 | +0.4 | +0.0 |
| gnad10 | topic | 0.636 | +1.6 | +0.0 | +1.2 | +0.0 | +0.0 | -0.4 | +0.0 | -2.0 | +2.4 | +0.4 | +2.4 | +3.2 | +2.4 | +0.8 | +0.4 |
| massive_scenario_en | intent | 0.804 | +0.4 | -0.4 | -1.2 | +0.8 | +0.4 | +0.0 | +0.8 | +2.0 | +2.4 | -1.2 | +0.8 | +1.2 | -1.6 | -0.8 | +0.4 |
| mnli | NLI / QA | 0.868 | -0.8 | -1.6 | -3.6* | -0.4 | -1.6 | -2.4 | -1.6 | -2.0 | -2.0 | +0.0 | +1.2 | -2.8 | +3.6* | -3.6 | -1.2 |
| patent | topic | 0.588 | -2.0 | -0.8 | -1.6 | -1.6 | +0.0 | +1.2 | -0.4 | -1.2 | +6.0* | -0.4 | -2.0 | -0.8 | -2.0 | +0.0 | +1.2 |
| rte | NLI / QA | 0.891 | +0.7 | +0.7 | -4.3* | +1.4 | +0.7 | -0.7 | +0.0 | -3.6 | +4.3 | +0.7 | +4.3 | +4.3* | +2.2 | +2.2 | -1.4 |
| sst2 | sentiment | 0.936 | -0.8 | -0.4 | -0.4 | -0.8 | -0.4 | -0.8 | -0.8 | -1.2 | -0.4 | -0.4 | +1.2 | +0.4 | -0.4 | +0.4 | -0.4 |
| sst5 | sentiment | 0.472 | +0.0 | +5.6* | +3.2 | +5.2* | +1.6 | +2.8 | +1.2 | +3.6 | +8.8* | +0.8 | +10.0* | +8.8* | +0.8 | +3.6 | +1.2 |
| toxic_conversations | sentiment | 0.812 | -1.6 | -0.4 | -4.8* | +1.2 | +2.0 | +0.0 | -0.4 | -0.8 | +2.8 | +0.8 | +4.0* | +2.4 | +1.2 | +3.2 | +2.8* |
| xnli_de | NLI / QA | 0.808 | +0.4 | +1.6 | -0.4 | +2.8 | +0.0 | +0.4 | +1.6 | +0.0 | +2.0 | +2.4 | +0.8 | +1.2 | -0.8 | +0.0 | -0.4 |

## Against Jev and chain of thought

Accuracy on the dev rows: the baseline, the best screened arm (`R-Q`), and the published Jev and `glm-cot` runs on the same examples (rows they answered in brackets). `glm-cot` lets GLM-5.3-Flash reason before answering, with its own prompt.

| dataset | B | R-Q | Jev | glm-cot |
|---|---|---|---|---|
| ag_news | 0.864 | 0.888 | 0.880 (250) | 0.900 (250) |
| banking77 | 0.776 | 0.816 | 0.780 (250) | 0.788 (250) |
| boolq | 0.924 | 0.908 | 0.932 (250) | 0.928 (250) |
| clinc150 | 0.868 | 0.900 | 0.756 (250) | 0.856 (250) |
| dbpedia_14 | 0.976 | 0.984 | 0.988 (250) | 0.988 (250) |
| gnad10 | 0.636 | 0.660 | 0.592 (250) | 0.672 (250) |
| massive_scenario_en | 0.804 | 0.812 | 0.800 (250) | 0.856 (250) |
| mnli | 0.868 | 0.880 | 0.868 (250) | 0.844 (250) |
| patent | 0.588 | 0.568 | 0.484 (250) | 0.720 (250) |
| rte | 0.891 | 0.935 | 0.870 (138) | 0.920 (138) |
| sst2 | 0.936 | 0.948 | 0.940 (250) | 0.936 (250) |
| sst5 | 0.472 | 0.572 | 0.532 (250) | 0.580 (250) |
| toxic_conversations | 0.812 | 0.852 | 0.776 (250) | 0.876 (250) |
| xnli_de | 0.808 | 0.816 | 0.808 (250) | 0.868 (250) |

Headroom closed by `R-Q`: removed. As first generated, it set glm-cot on the rows it answered against B on all rows; the confirmation's figure (57%) replaces it.


## By task family

| arm | NLI / QA | intent | sentiment | topic |
|---|---|---|---|---|
| B2 | +0.1 | +0.0 | -0.8 | -0.4 |
| F-alpha | +0.2 | +0.1 | +1.6 | +0.0 |
| F-before | -2.3 | -0.9 | -0.7 | -0.1 |
| F-count | +0.8 | +0.5 | +1.9 | -0.6 |
| F-dots | -0.8 | +0.3 | +1.1 | +0.1 |
| F-scrambled | -0.8 | -0.7 | +0.7 | +0.1 |
| F-words | -0.2 | +0.1 | +0.0 | +0.1 |
| G | -1.5 | +0.7 | +0.5 | -0.8 |
| H-128 | +0.9 | +2.4 | +3.7 | +2.8 |
| H-32 | +0.4 | -0.3 | +0.4 | +0.1 |
| R-Q | +1.2 | +2.7 | +5.1 | +0.9 |
| R-QSQS | +0.6 | +3.3 | +3.9 | +1.3 |
| R-Qi | +0.8 | -1.3 | +0.5 | +0.0 |
| R-full | -0.4 | +1.7 | +2.4 | +0.5 |
| R-think | -0.5 | -1.5 | +1.2 | +0.8 |

## Where the gains are: by the baseline's confidence

Rows pooled over datasets, split into five equal groups by the baseline's top probability; accuracy points against the baseline in each:

| arm | least sure fifth | 2 | 3 | 4 | surest fifth |
|---|---|---|---|---|---|
| B2 | -1.3 | -0.1 | +0.0 | +0.0 | +0.0 |
| F-alpha | +2.1 | +0.0 | +0.0 | +0.0 | +0.0 |
| F-before | -2.8 | -1.8 | +0.0 | +0.0 | +0.0 |
| F-count | +2.5 | +0.1 | +0.0 | +0.0 | +0.0 |
| F-dots | +0.6 | -0.3 | +0.0 | +0.0 | +0.0 |
| F-scrambled | -0.6 | -0.3 | +0.0 | +0.0 | +0.0 |
| F-words | +0.3 | -0.3 | +0.0 | +0.0 | +0.0 |
| G | -0.9 | -0.6 | +0.0 | +0.0 | +0.0 |
| H-128 | +8.7 | +2.5 | +0.0 | +0.3 | +0.0 |
| H-32 | +1.5 | -0.9 | +0.1 | +0.0 | +0.0 |
| R-Q | +9.7 | +0.9 | +0.1 | +0.1 | +0.0 |
| R-QSQS | +8.4 | +1.2 | +0.3 | +0.1 | +0.0 |
| R-Qi | +0.4 | -0.6 | +0.1 | +0.0 | +0.0 |
| R-full | +4.7 | -0.4 | +0.1 | +0.0 | +0.0 |
| R-think | +0.6 | -0.1 | +0.0 | +0.0 | +0.0 |

## Offline ensembles

From the stored distributions, no extra requests beyond the arm's: the mean of the baseline and the arm (two reads for every decision), and the arm's answer only where the baseline's top probability is below 0.9 (the share of decisions that need the second read in brackets). Accuracy points against the baseline:

| arm | mean of B and arm | arm where B < 0.9 |
|---|---|---|
| B2 | -0.17 | -0.26 (21%) |
| F-alpha | +0.03 | +0.37 (21%) |
| F-before | -0.65 | -0.72 (21%) |
| F-count | +0.25 | +0.48 (21%) |
| F-dots | +0.06 | +0.11 (21%) |
| F-scrambled | -0.02 | -0.14 (21%) |
| F-words | -0.17 | -0.02 (21%) |
| G | -0.24 | -0.34 (21%) |
| H-128 | +2.25 | +1.86 (21%) |
| H-32 | +0.38 | +0.26 (21%) |
| R-Q | +1.69 | +1.84 (21%) |
| R-QSQS | +1.38 | +1.66 (21%) |
| R-Qi | +0.27 | +0.05 (21%) |
| R-full | +0.61 | +0.87 (21%) |
| R-think | -0.19 | +0.01 (21%) |
