# Screening on the MMLU-Pro dev questions

GLM-5.3-Flash, 250 MMLU-Pro test questions outside the 1,000 used for comparison (`bench.prefill.mmlu_pro_rows`). Every arm is compared with `B` on the same rows. *B2* is the baseline asked again: the noise floor between runs.

## Screening

Accuracy in points against the baseline, pooled over all rows with a paired bootstrap 95% interval; datasets better / worse (in brackets: significant by McNemar, p < 0.05); the worst control dataset (sst2, dbpedia_14, banking77, clinc150); NLL after a temperature fitted per arm on half the rows, on the other half; the median of those temperatures (the baseline's is 1.93); option mass; how far the mean answer moves from the baseline's (total variation; anchoring or a position prior); extra prompt tokens; the change in median latency (4 requests in flight); EUR per 1,000 decisions (baseline 0.079).

| arm | accuracy points [95% CI] | datasets + / − (sig.) | worst control | NLL after own T | own T | option mass | mean-answer shift | extra tokens | latency, ms | EUR / 1,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 | -0.40 [-3.20, +2.80] | 0 / 1 (0) | +nan | 1.289 (B 1.322) | 1.96 | 0.973 | 0.011 | 0 | +57 | 0.079 |
| F-before | +0.00 [-2.41, +2.80] | 0 / 0 (0) | +nan | 1.293 (B 1.322) | 1.88 | 0.979 | 0.023 | 128 | +14 | 0.104 |
| F-dots | -1.60 [-5.60, +2.00] | 0 / 1 (0) | +nan | 1.336 (B 1.322) | 2.07 | 0.849 | 0.031 | 128 | +22 | 0.104 |
| G | -0.80 [-4.40, +2.80] | 0 / 1 (0) | +nan | 1.288 (B 1.322) | 2.04 | 0.987 | 0.043 | 119 | +27 | 0.103 |
| H-1024 | +12.00 [+6.40, +18.00] | 1 / 0 (1) | +nan | 0.979 (B 1.322) | 2.64 | 0.955 | 0.066 | 546 | +35763 | 0.585 |
| H-128 | +12.40 [+7.20, +17.60] | 1 / 0 (1) | +nan | 0.972 (B 1.322) | 2.31 | 0.989 | 0.043 | 128 | +1949 | 0.174 |
| H-32 | +4.40 [+1.20, +8.00] | 1 / 0 (1) | +nan | 1.317 (B 1.322) | 2.06 | 0.982 | 0.030 | 32 | +958 | 0.106 |
| R-Q | -3.20 [-8.40, +2.00] | 0 / 1 (0) | +nan | 1.291 (B 1.322) | 1.87 | 0.964 | 0.021 | 256 | +240 | 0.134 |
| R-QSQS | -2.00 [-7.20, +3.60] | 0 / 1 (0) | +nan | 1.248 (B 1.322) | 1.72 | 0.949 | 0.022 | 570 | +253 | 0.201 |
| R-Qi | -0.80 [-4.00, +2.80] | 0 / 1 (0) | +nan | 1.315 (B 1.322) | 1.93 | 0.982 | 0.010 | 18 | +254 | 0.082 |
| R-full | +0.40 [-4.00, +4.40] | 1 / 0 (0) | +nan | 1.199 (B 1.322) | 1.79 | 0.959 | 0.022 | 309 | +258 | 0.146 |
| R-think | -2.80 [-6.40, +0.80] | 0 / 1 (0) | +nan | 1.346 (B 1.322) | 1.68 | 0.949 | 0.043 | 58 | +274 | 0.090 |

**Gate** (fixed before the screening): pooled ≥ +1.0 point with the interval above 0; no control dataset losing more than the noise floor (the largest control difference between B and B2, 0.0 points, or 1 point if larger); NLL after temperature not worse; latency +30 ms at most; for filler, the placement control (F-before) must not show the same gain.

| arm | accuracy | controls | NLL | latency | placement | verdict |
|---|---|---|---|---|---|---|
| B2 | **fail** | **fail** | pass | **fail** | — | fails |
| F-before | **fail** | **fail** | pass | pass | — | fails |
| F-dots | **fail** | **fail** | **fail** | pass | **fail** | fails |
| G | **fail** | **fail** | pass | pass | **fail** | fails |
| H-1024 | pass | **fail** | pass | pass | — | fails |
| H-128 | pass | **fail** | pass | pass | — | fails |
| H-32 | pass | **fail** | pass | pass | — | fails |
| R-Q | **fail** | **fail** | pass | **fail** | — | fails |
| R-QSQS | **fail** | **fail** | pass | **fail** | — | fails |
| R-Qi | **fail** | **fail** | pass | **fail** | — | fails |
| R-full | **fail** | **fail** | pass | **fail** | — | fails |
| R-think | **fail** | **fail** | **fail** | **fail** | — | fails |

## Per dataset

Accuracy points against the baseline (McNemar p < 0.05 marked *):

| dataset | group | B | B2 | F-before | F-dots | G | H-1024 | H-128 | H-32 | R-Q | R-QSQS | R-Qi | R-full | R-think |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | knowledge | 0.584 | -0.4 | +0.0 | -1.6 | -0.8 | +12.0* | +12.4* | +4.4* | -3.2 | -2.0 | -0.8 | +0.4 | -2.8 |

## By task family

| arm | knowledge |
|---|---|
| B2 | -0.4 |
| F-before | +0.0 |
| F-dots | -1.6 |
| G | -0.8 |
| H-1024 | +12.0 |
| H-128 | +12.4 |
| H-32 | +4.4 |
| R-Q | -3.2 |
| R-QSQS | -2.0 |
| R-Qi | -0.8 |
| R-full | +0.4 |
| R-think | -2.8 |

## Where the gains are: by the baseline's confidence

Rows pooled over datasets, split into five equal groups by the baseline's top probability; accuracy points against the baseline in each:

| arm | least sure fifth | 2 | 3 | 4 | surest fifth |
|---|---|---|---|---|---|
| B2 | +6.0 | -6.0 | -2.0 | +0.0 | +0.0 |
| F-before | +2.0 | -4.0 | +2.0 | +0.0 | +0.0 |
| F-dots | +6.0 | -10.0 | -4.0 | +0.0 | +0.0 |
| G | +0.0 | -2.0 | -2.0 | +0.0 | +0.0 |
| H-1024 | +34.0 | +14.0 | +10.0 | +2.0 | +0.0 |
| H-128 | +34.0 | +12.0 | +8.0 | +8.0 | +0.0 |
| H-32 | +8.0 | +8.0 | +4.0 | +2.0 | +0.0 |
| R-Q | +2.0 | -12.0 | -2.0 | -4.0 | +0.0 |
| R-QSQS | +6.0 | -10.0 | -2.0 | -4.0 | +0.0 |
| R-Qi | -4.0 | -2.0 | +2.0 | +0.0 | +0.0 |
| R-full | +10.0 | -4.0 | -6.0 | +2.0 | +0.0 |
| R-think | -4.0 | -8.0 | -2.0 | +0.0 | +0.0 |

## Offline ensembles

From the stored distributions, no extra requests beyond the arm's: the mean of the baseline and the arm (two reads for every decision), and the arm's answer only where the baseline's top probability is below 0.9 (the share of decisions that need the second read in brackets). Accuracy points against the baseline:

| arm | mean of B and arm | arm where B < 0.9 |
|---|---|---|
| B2 | +0.40 | -0.40 (53%) |
| F-before | +1.20 | +0.00 (53%) |
| F-dots | +0.80 | -1.60 (53%) |
| G | +0.40 | -0.80 (53%) |
| H-1024 | +13.60 | +11.20 (53%) |
| H-128 | +12.40 | +10.00 (53%) |
| H-32 | +0.80 | +3.60 (53%) |
| R-Q | -2.40 | -2.00 (53%) |
| R-QSQS | +0.00 | -0.80 (53%) |
| R-Qi | +0.40 | -0.80 (53%) |
| R-full | +1.60 | +0.00 (53%) |
| R-think | +0.00 | -2.80 (53%) |
