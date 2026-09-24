# Results

2026-09-22. Recomputed from the JSONL next to it with
`python -m bench.report results/*.jsonl --write results/report.md`, so a new
question about these runs costs no requests.

The first three sections are the three-arm runs and are the accuracy source.
The next three are hosted-only runs of the same samples and are the latency
source -- Laya runs on the machine that measures the other two, so its
forward passes perturb their timings. The last two are the labelled variants
(`--laya-shortlist 20`, `--permutations 3`), which are not like-for-like.

## banking77 — 300 examples, 77 options, seed 0

`legacy-datasets/banking77` / test. Models: privatemode = glm-5.3-flash, jev = jev-latest, laya = english. Concurrency 1, permutations 1, laya shortlist off, 1 USD = 0.92 EUR.

| metric | jev | laya | privatemode |
|---|---|---|---|
| examples scored | 300 | 300 | 300 |
| errors dropped | 0 | 0 | 0 |
| **accuracy** | 0.790 | 0.373 | 0.753 |
| macro F1 | 0.753 | 0.322 | 0.713 |
| ECE (top prob) | 0.103 | 0.512 | 0.181 |
| Brier | 0.314 | 1.102 | 0.401 |
| AUROC (vendor confidence) | 0.866 | 0.710 | 0.842 |
| acc @ 90% coverage | 0.841 | 0.404 | 0.785 |
| acc @ 50% coverage | 0.953 | 0.507 | 0.947 |
| latency p50 (ms) | 239 | 126* | 385 |
| latency p95 (ms) | 322 | 173* | 429 |
| mean input tokens | 1023 | 333 | 1620 |
| **EUR / 1000 decisions** | 0.0395 | local | 0.2921 |
| retries | 0 | 0 | 0 |

\* Measured on the machine that ran the benchmark, not in a datacentre. It is not comparable with the hosted arms and is printed only so the run is fully recorded.

**jev vs laya** — right where the other was wrong on 134 / 9 examples; McNemar p = 0.0000. Accuracy difference +0.417 [95% CI +0.353, +0.480].

**jev vs privatemode** — right where the other was wrong on 22 / 11 examples; McNemar p = 0.0801. Accuracy difference +0.037 [95% CI +0.000, +0.073]. Cost 0.14×, latency p50 0.62×.

**laya vs privatemode** — right where the other was wrong on 11 / 125 examples; McNemar p = 0.0000. Accuracy difference -0.380 [95% CI -0.443, -0.317].

Wall clock for the run: 220.0s.

## ag_news — 200 examples, 4 options, seed 0

`fancyzhx/ag_news` / test. Models: privatemode = glm-5.3-flash, jev = jev-latest, laya = english. Concurrency 1, permutations 1, laya shortlist off, 1 USD = 0.92 EUR.

| metric | jev | laya | privatemode |
|---|---|---|---|
| examples scored | 200 | 200 | 200 |
| errors dropped | 0 | 0 | 0 |
| **accuracy** | 0.895 | 0.960 | 0.880 |
| macro F1 | 0.892 | 0.959 | 0.877 |
| ECE (top prob) | 0.063 | 0.022 | 0.092 |
| Brier | 0.164 | 0.065 | 0.198 |
| AUROC (vendor confidence) | 0.793 | 0.874 | 0.806 |
| acc @ 90% coverage | 0.939 | 0.983 | 0.922 |
| acc @ 50% coverage | 0.960 | 0.990 | 0.970 |
| latency p50 (ms) | 251 | 56* | 151 |
| latency p95 (ms) | 311 | 89* | 195 |
| mean input tokens | 359 | 79 | 187 |
| **EUR / 1000 decisions** | 0.0139 | local | 0.0343 |
| retries | 0 | 0 | 0 |

\* Measured on the machine that ran the benchmark, not in a datacentre. It is not comparable with the hosted arms and is printed only so the run is fully recorded.

**jev vs laya** — right where the other was wrong on 4 / 17 examples; McNemar p = 0.0072. Accuracy difference -0.065 [95% CI -0.110, -0.020].

**jev vs privatemode** — right where the other was wrong on 6 / 3 examples; McNemar p = 0.5078. Accuracy difference +0.015 [95% CI -0.015, +0.045]. Cost 0.40×, latency p50 1.67×.

**laya vs privatemode** — right where the other was wrong on 20 / 4 examples; McNemar p = 0.0015. Accuracy difference +0.080 [95% CI +0.035, +0.130].

Wall clock for the run: 96.4s.

## boolq — 200 examples, 2 options, seed 0

`google/boolq` / validation. Models: privatemode = glm-5.3-flash, jev = jev-latest, laya = english. Concurrency 1, permutations 1, laya shortlist off, 1 USD = 0.92 EUR.

| metric | jev | laya | privatemode |
|---|---|---|---|
| examples scored | 200 | 200 | 200 |
| errors dropped | 0 | 0 | 0 |
| **accuracy** | 0.890 | 0.855 | 0.885 |
| macro F1 | 0.883 | 0.846 | 0.878 |
| ECE (top prob) | 0.057 | 0.073 | 0.053 |
| Brier | 0.141 | 0.226 | 0.158 |
| AUROC (vendor confidence) | 0.875 | 0.810 | 0.821 |
| acc @ 90% coverage | 0.944 | 0.883 | 0.939 |
| acc @ 50% coverage | 0.980 | 0.970 | 0.950 |
| latency p50 (ms) | 242 | 74* | 189 |
| latency p95 (ms) | 317 | 107* | 198 |
| mean input tokens | 424 | 147 | 228 |
| **EUR / 1000 decisions** | 0.0164 | local | 0.0417 |
| retries | 0 | 0 | 0 |

\* Measured on the machine that ran the benchmark, not in a datacentre. It is not comparable with the hosted arms and is printed only so the run is fully recorded.

**jev vs laya** — right where the other was wrong on 17 / 10 examples; McNemar p = 0.2478. Accuracy difference +0.035 [95% CI -0.015, +0.085].

**jev vs privatemode** — right where the other was wrong on 7 / 6 examples; McNemar p = 1.0000. Accuracy difference +0.005 [95% CI -0.030, +0.040]. Cost 0.39×, latency p50 1.28×.

**laya vs privatemode** — right where the other was wrong on 13 / 19 examples; McNemar p = 0.3771. Accuracy difference -0.030 [95% CI -0.085, +0.025].

Wall clock for the run: 102.5s.

## banking77 — 300 examples, 77 options, seed 0

`legacy-datasets/banking77` / test. Models: privatemode = glm-5.3-flash, jev = jev-latest. Concurrency 1, permutations 1, laya shortlist off, 1 USD = 0.92 EUR.

| metric | jev | privatemode |
|---|---|---|
| examples scored | 300 | 300 |
| errors dropped | 0 | 0 |
| **accuracy** | 0.793 | 0.753 |
| macro F1 | 0.755 | 0.719 |
| ECE (top prob) | 0.107 | 0.183 |
| Brier | 0.321 | 0.409 |
| AUROC (vendor confidence) | 0.855 | 0.845 |
| acc @ 90% coverage | 0.833 | 0.793 |
| acc @ 50% coverage | 0.967 | 0.947 |
| latency p50 (ms) | 244 | 277 |
| latency p95 (ms) | 314 | 394 |
| mean input tokens | 1023 | 1620 |
| **EUR / 1000 decisions** | 0.0395 | 0.2921 |
| retries | 0 | 0 |

**jev vs privatemode** — right where the other was wrong on 22 / 10 examples; McNemar p = 0.0501. Accuracy difference +0.040 [95% CI +0.003, +0.077]. Cost 0.14×, latency p50 0.88×.

Wall clock for the run: 163.5s.

## ag_news — 200 examples, 4 options, seed 0

`fancyzhx/ag_news` / test. Models: privatemode = glm-5.3-flash, jev = jev-latest. Concurrency 1, permutations 1, laya shortlist off, 1 USD = 0.92 EUR.

| metric | jev | privatemode |
|---|---|---|
| examples scored | 200 | 200 |
| errors dropped | 0 | 0 |
| **accuracy** | 0.895 | 0.885 |
| macro F1 | 0.892 | 0.882 |
| ECE (top prob) | 0.078 | 0.096 |
| Brier | 0.164 | 0.195 |
| AUROC (vendor confidence) | 0.792 | 0.795 |
| acc @ 90% coverage | 0.933 | 0.928 |
| acc @ 50% coverage | 0.960 | 0.970 |
| latency p50 (ms) | 238 | 150 |
| latency p95 (ms) | 309 | 174 |
| mean input tokens | 359 | 187 |
| **EUR / 1000 decisions** | 0.0139 | 0.0343 |
| retries | 0 | 0 |

**jev vs privatemode** — right where the other was wrong on 6 / 4 examples; McNemar p = 0.7539. Accuracy difference +0.010 [95% CI -0.020, +0.040]. Cost 0.40×, latency p50 1.59×.

Wall clock for the run: 80.2s.

## boolq — 200 examples, 2 options, seed 0

`google/boolq` / validation. Models: privatemode = glm-5.3-flash, jev = jev-latest. Concurrency 1, permutations 1, laya shortlist off, 1 USD = 0.92 EUR.

| metric | jev | privatemode |
|---|---|---|
| examples scored | 200 | 200 |
| errors dropped | 0 | 0 |
| **accuracy** | 0.900 | 0.880 |
| macro F1 | 0.894 | 0.871 |
| ECE (top prob) | 0.047 | 0.048 |
| Brier | 0.138 | 0.161 |
| AUROC (vendor confidence) | 0.856 | 0.814 |
| acc @ 90% coverage | 0.950 | 0.933 |
| acc @ 50% coverage | 0.980 | 0.960 |
| latency p50 (ms) | 240 | 150 |
| latency p95 (ms) | 327 | 171 |
| mean input tokens | 424 | 228 |
| **EUR / 1000 decisions** | 0.0164 | 0.0417 |
| retries | 0 | 0 |

**jev vs privatemode** — right where the other was wrong on 8 / 4 examples; McNemar p = 0.3877. Accuracy difference +0.020 [95% CI -0.015, +0.055]. Cost 0.39×, latency p50 1.60×.

Wall clock for the run: 81.6s.

## banking77 — 300 examples, 77 options, seed 0

`legacy-datasets/banking77` / test. Models: laya = english. Concurrency 1, permutations 1, laya shortlist 20, 1 USD = 0.92 EUR.

| metric | laya |
|---|---|
| examples scored | 300 |
| errors dropped | 0 |
| **accuracy** | 0.213 |
| macro F1 | 0.169 |
| ECE (top prob) | 0.619 |
| Brier | 1.342 |
| AUROC (vendor confidence) | 0.854 |
| acc @ 90% coverage | 0.237 |
| acc @ 50% coverage | 0.367 |
| latency p50 (ms) | 464* |
| latency p95 (ms) | 641* |
| mean input tokens | 191 |
| **EUR / 1000 decisions** | local |
| retries | 0 |

\* Measured on the machine that ran the benchmark, not in a datacentre. It is not comparable with the hosted arms and is printed only so the run is fully recorded.

Wall clock for the run: 146.0s.

## banking77 — 300 examples, 77 options, seed 0

`legacy-datasets/banking77` / test. Models: privatemode = glm-5.3-flash. Concurrency 1, permutations 3, laya shortlist off, 1 USD = 0.92 EUR.

| metric | privatemode |
|---|---|
| examples scored | 300 |
| errors dropped | 0 |
| **accuracy** | 0.770 |
| macro F1 | 0.735 |
| ECE (top prob) | 0.123 |
| Brier | 0.372 |
| AUROC (vendor confidence) | 0.813 |
| acc @ 90% coverage | 0.796 |
| acc @ 50% coverage | 0.947 |
| latency p50 (ms) | 299 |
| latency p95 (ms) | 430 |
| mean input tokens | 4859 |
| **EUR / 1000 decisions** | 0.8763 |
| retries | 0 |

Wall clock for the run: 280.6s.

