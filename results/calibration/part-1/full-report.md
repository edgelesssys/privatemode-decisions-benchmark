# Calibration report

Run: `eq-r1`, 28 text datasets plus rvl_cdip (scanned documents), reported separately. Calibration/test halves are a fixed random split, seed 0. ECE uses 15 equal-mass bins of top-label confidence; means over datasets weight every dataset equally.

## 1. Raw probabilities

Raw probabilities are **overconfident by 14.6 points** on average (mean confidence minus accuracy, text datasets, test halves); pooled ECE 0.146. Overconfident on 28 of 28 text datasets.

| dataset | options | test n | accuracy | confidence | overconfidence | ECE [95% CI] | NLL | Brier |
|---|---|---|---|---|---|---|---|---|
| boolq | 2 | 500 | 0.91 | 0.95 | 0.043 | 0.045 [0.028, 0.070] | 0.256 | 0.130 |
| rotten_tomatoes | 2 | 500 | 0.94 | 0.98 | 0.037 | 0.037 [0.022, 0.059] | 0.247 | 0.103 |
| rte (small) | 2 | 139 | 0.89 | 0.93 | 0.040 | 0.079 [0.041, 0.137] | 0.379 | 0.195 |
| sst2 | 2 | 436 | 0.96 | 0.99 | 0.029 | 0.029 [0.013, 0.047] | 0.171 | 0.074 |
| toxic_conversations | 2 | 500 | 0.78 | 0.92 | 0.138 | 0.140 [0.114, 0.178] | 0.691 | 0.360 |
| tweet_offensive | 2 | 430 | 0.80 | 0.93 | 0.135 | 0.141 [0.107, 0.174] | 0.635 | 0.329 |
| mnli | 3 | 500 | 0.88 | 0.94 | 0.069 | 0.079 [0.058, 0.110] | 0.475 | 0.217 |
| tweet_sentiment | 3 | 500 | 0.68 | 0.91 | 0.230 | 0.230 [0.192, 0.272] | 1.331 | 0.545 |
| xnli_de | 3 | 500 | 0.81 | 0.92 | 0.115 | 0.115 [0.090, 0.149] | 0.662 | 0.312 |
| ag_news | 4 | 500 | 0.90 | 0.98 | 0.081 | 0.081 [0.058, 0.107] | 0.551 | 0.187 |
| amazon_reviews_de | 5 | 500 | 0.59 | 0.84 | 0.254 | 0.259 [0.219, 0.300] | 1.492 | 0.644 |
| sst5 | 5 | 500 | 0.48 | 0.79 | 0.316 | 0.316 [0.274, 0.361] | 1.578 | 0.775 |
| emotion | 6 | 500 | 0.60 | 0.93 | 0.329 | 0.329 [0.283, 0.369] | 2.435 | 0.714 |
| trec_coarse (small) | 6 | 250 | 0.91 | 0.96 | 0.054 | 0.055 [0.031, 0.094] | 0.334 | 0.146 |
| gnad10 | 9 | 500 | 0.67 | 0.94 | 0.270 | 0.270 [0.231, 0.305] | 2.209 | 0.584 |
| patent | 9 | 500 | 0.55 | 0.89 | 0.341 | 0.341 [0.299, 0.386] | 2.540 | 0.756 |
| yahoo_topics | 10 | 500 | 0.76 | 0.95 | 0.188 | 0.188 [0.155, 0.222] | 1.612 | 0.427 |
| scotus | 13 | 500 | 0.68 | 0.93 | 0.252 | 0.252 [0.215, 0.290] | 2.189 | 0.561 |
| dbpedia_14 | 14 | 500 | 0.98 | 1.00 | 0.019 | 0.019 [0.009, 0.032] | 0.142 | 0.037 |
| massive_scenario_de | 18 | 500 | 0.74 | 0.90 | 0.158 | 0.158 [0.126, 0.193] | 0.937 | 0.374 |
| massive_scenario_en | 18 | 500 | 0.76 | 0.89 | 0.134 | 0.134 [0.109, 0.168] | 0.943 | 0.364 |
| newsgroups20 | 20 | 500 | 0.73 | 0.92 | 0.186 | 0.186 [0.155, 0.221] | 1.367 | 0.430 |
| trec_fine (small) | 42 | 250 | 0.80 | 0.91 | 0.106 | 0.107 [0.077, 0.162] | 1.006 | 0.319 |
| massive_intent_de | 59 | 500 | 0.80 | 0.92 | 0.120 | 0.120 [0.092, 0.149] | 0.980 | 0.314 |
| massive_intent_en | 59 | 500 | 0.82 | 0.93 | 0.106 | 0.106 [0.081, 0.130] | 0.993 | 0.269 |
| banking77 | 77 | 500 | 0.81 | 0.95 | 0.138 | 0.138 [0.109, 0.169] | 1.420 | 0.337 |
| ledgar | 100 | 500 | 0.77 | 0.93 | 0.155 | 0.155 [0.128, 0.195] | 1.612 | 0.408 |
| clinc150 | 151 | 500 | 0.88 | 0.94 | 0.060 | 0.060 [0.043, 0.087] | 0.562 | 0.193 |
| rvl_cdip | 16 | 500 | 0.67 | 0.91 | 0.248 | 0.248 [0.210, 0.287] | 1.738 | 0.544 |

## 2. Temperature

Best temperature per dataset (fitted on the calibration half): median 2.03, range 1.36–3.74, all above 1 (= overconfident): True. One global T fitted on all 28 text calibration halves: **T = 2.15**. Formula over all datasets: log T = 0.986 + -0.086·log(options).

How each zero-label method does on datasets left out of its fit. **Excess ECE** is a dataset's ECE minus its floor, the ECE a perfectly calibrated model shows on the same number of examples, averaged over datasets: 0 means as calibrated as the sample can show. (A pooled ECE over all datasets is not used: it sits at the floor for every method and hides the differences.) The ECE share is the total reduction over the per-task reduction; the NLL share is the median per dataset.

| method | mean excess ECE | ECE share | NLL share (median) |
|---|---|---|---|
| raw | 0.129 | 0 | 0 |
| global T (LODO) | 0.040 | 0.63 | 0.92 |
| formula (LODO) | 0.032 | 0.71 | 0.94 |
| same family (LODO) | 0.029 | 0.74 | 0.99 |
| leave family out | 0.062 | 0.39 | 0.91 |
| per task (oracle) | 0.006 | 1 | 1 |

`floor` is the ECE a perfectly calibrated model would show on this many examples (labels drawn from its own probabilities); values near it are as good as the sample can show.

| dataset | options | oracle T | raw ECE | oracle ECE | floor | global T ECE | formula ECE | same-family ECE | leave-family-out ECE |
|---|---|---|---|---|---|---|---|---|---|
| boolq | 2 | 1.85 | 0.045 | 0.036 | 0.032 | 0.049 (T=2.15) | 0.075 (T=2.60) | 0.049 (T=2.15) | 0.049 (T=2.15) |
| rotten_tomatoes | 2 | 1.98 | 0.037 | 0.018 | 0.026 | 0.018 (T=2.15) | 0.032 (T=2.58) | 0.058 (T=3.06) | 0.018 (T=2.06) |
| rte | 2 | 1.88 | 0.079 | 0.073 | 0.073 | 0.077 (T=2.15) | 0.096 (T=2.59) | 0.073 (T=1.87) | 0.078 (T=2.16) |
| sst2 | 2 | 2.08 | 0.029 | 0.013 | 0.026 | 0.013 (T=2.15) | 0.028 (T=2.57) | 0.053 (T=3.05) | 0.012 (T=2.06) |
| toxic_conversations | 2 | 2.71 | 0.140 | 0.056 | 0.051 | 0.066 (T=2.14) | 0.058 (T=2.51) | 0.063 (T=3.40) | 0.066 (T=2.13) |
| tweet_offensive | 2 | 3.40 | 0.141 | 0.092 | 0.055 | 0.093 (T=2.14) | 0.089 (T=2.46) | 0.087 (T=2.71) | 0.093 (T=2.13) |
| mnli | 3 | 1.71 | 0.079 | 0.049 | 0.037 | 0.061 (T=2.16) | 0.088 (T=2.50) | 0.054 (T=1.98) | 0.061 (T=2.16) |
| tweet_sentiment | 3 | 3.39 | 0.230 | 0.047 | 0.062 | 0.112 (T=2.13) | 0.087 (T=2.39) | 0.055 (T=2.90) | 0.119 (T=2.06) |
| xnli_de | 3 | 2.03 | 0.115 | 0.043 | 0.050 | 0.045 (T=2.15) | 0.062 (T=2.47) | 0.043 (T=1.77) | 0.045 (T=2.16) |
| ag_news | 4 | 2.44 | 0.081 | 0.040 | 0.033 | 0.051 (T=2.14) | 0.041 (T=2.38) | 0.044 (T=2.63) | 0.055 (T=2.03) |
| amazon_reviews_de | 5 | 2.69 | 0.259 | 0.115 | 0.060 | 0.155 (T=2.14) | 0.136 (T=2.32) | 0.097 (T=3.03) | 0.166 (T=2.06) |
| sst5 | 5 | 2.73 | 0.316 | 0.089 | 0.068 | 0.147 (T=2.14) | 0.123 (T=2.32) | 0.071 (T=3.01) | 0.156 (T=2.06) |
| emotion | 6 | 3.74 | 0.329 | 0.050 | 0.065 | 0.200 (T=2.11) | 0.181 (T=2.25) | 0.147 (T=2.66) | 0.206 (T=2.06) |
| trec_coarse | 6 | 1.83 | 0.055 | 0.037 | 0.050 | 0.064 (T=2.16) | 0.073 (T=2.32) | 0.035 (T=1.74) | 0.098 (T=2.51) |
| gnad10 | 9 | 3.27 | 0.270 | 0.073 | 0.060 | 0.171 (T=2.11) | 0.162 (T=2.19) | 0.124 (T=2.48) | 0.180 (T=2.03) |
| patent | 9 | 3.12 | 0.341 | 0.064 | 0.065 | 0.177 (T=2.11) | 0.165 (T=2.19) | 0.114 (T=2.53) | 0.191 (T=2.03) |
| yahoo_topics | 10 | 2.87 | 0.188 | 0.050 | 0.055 | 0.089 (T=2.12) | 0.082 (T=2.18) | 0.048 (T=2.56) | 0.100 (T=2.03) |
| scotus | 13 | 2.65 | 0.252 | 0.061 | 0.055 | 0.133 (T=2.13) | 0.132 (T=2.13) | 0.148 (T=2.02) | 0.131 (T=2.14) |
| dbpedia_14 | 14 | 1.59 | 0.019 | 0.017 | 0.009 | 0.027 (T=2.16) | 0.027 (T=2.16) | 0.094 (T=2.76) | 0.021 (T=2.03) |
| massive_scenario_de | 18 | 1.81 | 0.158 | 0.073 | 0.043 | 0.055 (T=2.16) | 0.049 (T=2.10) | 0.080 (T=1.74) | 0.091 (T=2.51) |
| massive_scenario_en | 18 | 1.77 | 0.134 | 0.065 | 0.042 | 0.050 (T=2.16) | 0.049 (T=2.11) | 0.069 (T=1.75) | 0.098 (T=2.51) |
| newsgroups20 | 20 | 2.16 | 0.186 | 0.056 | 0.046 | 0.057 (T=2.15) | 0.061 (T=2.07) | 0.109 (T=2.75) | 0.065 (T=2.03) |
| trec_fine | 42 | 1.59 | 0.107 | 0.083 | 0.056 | 0.124 (T=2.18) | 0.089 (T=1.98) | 0.082 (T=1.77) | 0.202 (T=2.51) |
| massive_intent_de | 59 | 1.88 | 0.120 | 0.034 | 0.042 | 0.068 (T=2.17) | 0.034 (T=1.89) | 0.040 (T=1.72) | 0.159 (T=2.51) |
| massive_intent_en | 59 | 1.80 | 0.106 | 0.025 | 0.038 | 0.075 (T=2.17) | 0.040 (T=1.90) | 0.032 (T=1.74) | 0.164 (T=2.51) |
| banking77 | 77 | 1.93 | 0.138 | 0.049 | 0.043 | 0.056 (T=2.17) | 0.057 (T=1.83) | 0.066 (T=1.71) | 0.155 (T=2.51) |
| ledgar | 100 | 2.02 | 0.155 | 0.061 | 0.049 | 0.072 (T=2.16) | 0.085 (T=1.77) | 0.202 (T=2.65) | 0.071 (T=2.14) |
| clinc150 | 151 | 1.36 | 0.060 | 0.029 | 0.026 | 0.171 (T=2.21) | 0.066 (T=1.85) | 0.062 (T=1.81) | 0.282 (T=2.51) |

Oracle T by family (geometric mean): intent 1.74, qa 1.85, nli 1.87, legal 2.31, topic 2.50, sentiment 2.70, moderation 3.04.


**Shipped in the library** (fitted on all examples): log T = 0.962 + -0.076·log(options); per family: intent 1.76, legal 2.24, moderation 2.96, nli 1.87, qa 1.76, sentiment 2.93, topic 2.60.


### Run jitter

T fitted on run 1's calibration half, evaluated on run 2's test half. Median |log(T1/T2)| = 0.006 (a factor of 1.006).

| dataset | T run 1 | T run 2 | ECE run 2, T from run 1 | ECE run 2, own T | same answer in both runs |
|---|---|---|---|---|---|
| boolq | 1.85 | 1.85 | 0.030 | 0.030 | 0.985 |
| rotten_tomatoes | 1.98 | 1.96 | 0.019 | 0.018 | 0.995 |
| rte | 1.88 | 1.86 | 0.045 | 0.045 | 0.993 |
| sst2 | 2.08 | 2.10 | 0.020 | 0.020 | 0.997 |
| toxic_conversations | 2.71 | 2.74 | 0.052 | 0.052 | 0.981 |
| tweet_offensive | 3.40 | 3.44 | 0.078 | 0.079 | 0.987 |
| mnli | 1.71 | 1.71 | 0.041 | 0.041 | 0.970 |
| tweet_sentiment | 3.39 | 3.42 | 0.073 | 0.074 | 0.961 |
| xnli_de | 2.03 | 2.04 | 0.038 | 0.038 | 0.958 |
| ag_news | 2.44 | 2.42 | 0.035 | 0.035 | 0.990 |
| amazon_reviews_de | 2.69 | 2.70 | 0.112 | 0.111 | 0.932 |
| sst5 | 2.73 | 2.74 | 0.091 | 0.090 | 0.905 |
| emotion | 3.74 | 3.76 | 0.060 | 0.060 | 0.980 |
| trec_coarse | 1.83 | 1.82 | 0.036 | 0.035 | 0.988 |
| gnad10 | 3.27 | 3.25 | 0.078 | 0.079 | 0.969 |
| patent | 3.12 | 3.10 | 0.071 | 0.075 | 0.946 |
| yahoo_topics | 2.87 | 2.85 | 0.062 | 0.061 | 0.976 |
| scotus | 2.65 | 2.66 | 0.061 | 0.055 | 0.975 |
| dbpedia_14 | 1.59 | 1.59 | 0.015 | 0.015 | 0.999 |
| massive_scenario_de | 1.81 | 1.81 | 0.064 | 0.064 | 0.950 |
| massive_scenario_en | 1.77 | 1.77 | 0.063 | 0.063 | 0.966 |
| newsgroups20 | 2.16 | 2.15 | 0.056 | 0.057 | 0.958 |
| trec_fine | 1.59 | 1.59 | 0.078 | 0.078 | 0.944 |
| massive_intent_de | 1.88 | 1.88 | 0.044 | 0.044 | 0.950 |
| massive_intent_en | 1.80 | 1.79 | 0.035 | 0.035 | 0.968 |
| banking77 | 1.93 | 1.94 | 0.048 | 0.048 | 0.971 |
| ledgar | 2.02 | 2.00 | 0.051 | 0.043 | 0.970 |
| clinc150 | 1.36 | 1.38 | 0.032 | 0.032 | 0.967 |

## 3. Contextual calibration

Priors from neutral content (`N/A`, empty, `[MASK]`, and `k. A.` for German sets), divided out, then T refitted on the corrected probabilities (global T = 2.33); compared with the formula T alone, both leave-one-dataset-out. Unlike temperature, this changes answers, so accuracy moves too. Verdict by test NLL (±0.01): neutral 3, hurts 24, helps 1.

| dataset | largest prior | ECE, T only | ECE, context + T | Δ NLL | Δ accuracy (points) | verdict |
|---|---|---|---|---|---|---|
| boolq | 0.64 | 0.075 | 0.047 | +0.000 | +0.6 | neutral |
| rotten_tomatoes | 0.76 | 0.032 | 0.030 | +0.008 | -1.0 | neutral |
| rte | 0.81 | 0.096 | 0.066 | +0.024 | -2.9 | hurts |
| sst2 | 0.65 | 0.028 | 0.025 | -0.011 | -0.2 | helps |
| toxic_conversations | 0.92 | 0.058 | 0.153 | +0.331 | -15.6 | hurts |
| tweet_offensive | 0.92 | 0.089 | 0.082 | +0.043 | -3.7 | hurts |
| mnli | 0.97 | 0.088 | 0.044 | +0.108 | -6.4 | hurts |
| tweet_sentiment | 0.98 | 0.087 | 0.235 | +0.634 | -12.6 | hurts |
| xnli_de | 0.97 | 0.062 | 0.087 | +0.093 | -2.6 | hurts |
| ag_news | 0.56 | 0.041 | 0.042 | +0.036 | -0.8 | hurts |
| amazon_reviews_de | 0.45 | 0.136 | 0.093 | +0.041 | -1.2 | hurts |
| sst5 | 0.87 | 0.123 | 0.350 | +0.465 | -12.4 | hurts |
| emotion | 0.43 | 0.181 | 0.168 | +0.058 | -1.8 | hurts |
| trec_coarse | 0.82 | 0.073 | 0.119 | +0.067 | -0.4 | hurts |
| gnad10 | 0.59 | 0.162 | 0.196 | +0.276 | -4.4 | hurts |
| patent | 0.96 | 0.165 | 0.144 | +0.315 | -0.2 | hurts |
| yahoo_topics | 0.77 | 0.082 | 0.067 | +0.021 | -0.4 | hurts |
| scotus | 0.96 | 0.132 | 0.106 | +0.018 | -1.6 | hurts |
| dbpedia_14 | 0.64 | 0.027 | 0.049 | +0.018 | +0.0 | hurts |
| massive_scenario_de | 0.97 | 0.049 | 0.076 | +0.123 | +4.0 | hurts |
| massive_scenario_en | 0.94 | 0.049 | 0.070 | +0.179 | -1.0 | hurts |
| newsgroups20 | 0.88 | 0.061 | 0.067 | +0.095 | -0.8 | hurts |
| trec_fine | 0.38 | 0.089 | 0.131 | +0.424 | -9.6 | hurts |
| massive_intent_de | 0.36 | 0.034 | 0.120 | +0.164 | -1.4 | hurts |
| massive_intent_en | 0.25 | 0.040 | 0.132 | +0.171 | -1.4 | hurts |
| banking77 | 0.62 | 0.057 | 0.122 | +0.007 | +0.6 | neutral |
| ledgar | 0.46 | 0.085 | 0.106 | +0.147 | -1.8 | hurts |
| clinc150 | 0.88 | 0.066 | 0.174 | +0.645 | -8.4 | hurts |

## 4. Conformal prediction sets

Split conformal on each dataset's calibration half, evaluated on its test half. LAC scores 1 − p(gold); APS the mass of options at least as likely as the gold one. Probabilities: raw, and after the formula temperature fitted without the dataset (the library's default).

| target | probabilities | score | mean coverage | mean set size | single-option share |
|---|---|---|---|---|---|
| 90% | raw | LAC | 0.901 | 1.83 | 0.65 |
| 90% | raw | APS | 0.965 | 7.26 | 0.25 |
| 90% | T | LAC | 0.901 | 1.77 | 0.63 |
| 90% | T | APS | 0.956 | 7.04 | 0.25 |
| 95% | raw | LAC | 0.955 | 3.24 | 0.48 |
| 95% | raw | APS | 0.983 | 8.92 | 0.07 |
| 95% | T | LAC | 0.954 | 3.13 | 0.45 |
| 95% | T | APS | 0.977 | 8.57 | 0.11 |

Per dataset, LAC after the formula T:

| dataset | options | coverage 90% | size 90% | single 90% | coverage 95% | size 95% | single 95% |
|---|---|---|---|---|---|---|---|
| boolq | 2 | 0.898 | 0.96 | 0.96 | 0.958 | 1.11 | 0.89 |
| rotten_tomatoes | 2 | 0.916 | 0.96 | 0.96 | 0.958 | 1.05 | 0.95 |
| rte | 2 | 0.928 | 1.14 | 0.86 | 0.957 | 1.29 | 0.71 |
| sst2 | 2 | 0.911 | 0.93 | 0.93 | 0.966 | 1.02 | 0.98 |
| toxic_conversations | 2 | 0.872 | 1.24 | 0.76 | 0.960 | 1.54 | 0.46 |
| tweet_offensive | 2 | 0.909 | 1.30 | 0.70 | 0.977 | 1.50 | 0.50 |
| mnli | 3 | 0.908 | 1.11 | 0.89 | 0.936 | 1.25 | 0.76 |
| tweet_sentiment | 3 | 0.898 | 1.94 | 0.42 | 0.962 | 2.38 | 0.24 |
| xnli_de | 3 | 0.902 | 1.29 | 0.72 | 0.952 | 1.68 | 0.49 |
| ag_news | 4 | 0.892 | 1.00 | 0.99 | 0.944 | 1.13 | 0.88 |
| amazon_reviews_de | 5 | 0.902 | 2.17 | 0.30 | 0.954 | 2.79 | 0.17 |
| sst5 | 5 | 0.918 | 2.37 | 0.03 | 0.966 | 2.79 | 0.01 |
| emotion | 6 | 0.910 | 3.68 | 0.12 | 0.950 | 4.30 | 0.06 |
| trec_coarse | 6 | 0.916 | 1.02 | 0.98 | 0.976 | 1.24 | 0.77 |
| gnad10 | 9 | 0.914 | 3.50 | 0.25 | 0.960 | 4.95 | 0.11 |
| patent | 9 | 0.888 | 4.31 | 0.06 | 0.946 | 6.21 | 0.00 |
| yahoo_topics | 10 | 0.926 | 2.89 | 0.43 | 0.970 | 5.05 | 0.10 |
| scotus | 13 | 0.872 | 2.40 | 0.29 | 0.922 | 4.61 | 0.07 |
| dbpedia_14 | 14 | 0.904 | 0.91 | 0.91 | 0.956 | 0.97 | 0.97 |
| massive_scenario_de | 18 | 0.900 | 1.40 | 0.67 | 0.948 | 1.98 | 0.50 |
| massive_scenario_en | 18 | 0.888 | 1.36 | 0.67 | 0.936 | 1.89 | 0.51 |
| newsgroups20 | 20 | 0.906 | 2.49 | 0.57 | 0.962 | 4.54 | 0.44 |
| trec_fine | 42 | 0.876 | 1.28 | 0.74 | 0.940 | 2.09 | 0.44 |
| massive_intent_de | 59 | 0.916 | 1.82 | 0.61 | 0.974 | 7.15 | 0.26 |
| massive_intent_en | 59 | 0.898 | 1.35 | 0.74 | 0.948 | 3.91 | 0.45 |
| banking77 | 77 | 0.884 | 1.79 | 0.61 | 0.956 | 8.58 | 0.13 |
| ledgar | 100 | 0.894 | 1.98 | 0.48 | 0.950 | 9.35 | 0.10 |
| clinc150 | 151 | 0.894 | 1.05 | 0.92 | 0.934 | 1.21 | 0.80 |
| rvl_cdip | 16 | 0.886 | 2.39 | 0.32 | 0.938 | 5.70 | 0.04 |

**Global cutoff (zero labels, heuristic, no guarantee):** LAC cutoff pooled over the other datasets' calibration halves, applied to the held-out dataset.

| target | mean coverage | worst | best | datasets > 2 points short |
|---|---|---|---|---|
| 90% | 0.902 | 0.718 | 0.988 | 9 of 28 |
| 95% | 0.950 | 0.836 | 1.000 | 9 of 28 |

**Labels needed** (90% target, 200 random calibration draws per dataset):

| labels | mean coverage | 5th percentile | 95th percentile | share of draws below 88% |
|---|---|---|---|---|
| 50 | 0.904 | 0.832 | 0.966 | 0.25 |
| 100 | 0.902 | 0.852 | 0.948 | 0.20 |
| 250 | 0.900 | 0.868 | 0.928 | 0.14 |
| 500 | 0.900 | 0.872 | 0.918 | 0.09 |

## 5. Label noise, contamination, language

Examples where at least three of the four other published arms (Jev, Laya, reasoning GLM, embeddings) agree on an answer that differs from the label are removed. This is an **upper bound** on the effect of label noise: the removed examples include ones where the label is right and GLM is confidently wrong together with the other systems, and dropping those flatters any model. After the formula T:

| dataset | removed share | overconfidence, all | overconfidence, cleaned | ECE, all | ECE, cleaned | accuracy, cleaned |
|---|---|---|---|---|---|---|
| boolq | 0.038 | -0.054 | -0.075 | 0.075 | 0.094 | 0.933 |
| rotten_tomatoes | 0.044 | -0.031 | -0.060 | 0.032 | 0.060 | 0.973 |
| rte | 0.058 | -0.071 | -0.082 | 0.096 | 0.109 | 0.916 |
| sst2 | 0.023 | -0.028 | -0.046 | 0.028 | 0.046 | 0.977 |
| toxic_conversations | 0.060 | 0.024 | -0.018 | 0.058 | 0.063 | 0.826 |
| tweet_offensive | 0.114 | 0.038 | -0.031 | 0.089 | 0.050 | 0.879 |
| mnli | 0.040 | -0.080 | -0.102 | 0.088 | 0.102 | 0.902 |
| tweet_sentiment | 0.136 | 0.081 | 0.002 | 0.087 | 0.051 | 0.759 |
| xnli_de | 0.084 | -0.057 | -0.102 | 0.062 | 0.102 | 0.865 |
| ag_news | 0.062 | 0.016 | -0.034 | 0.041 | 0.034 | 0.951 |
| amazon_reviews_de | 0.176 | 0.081 | -0.009 | 0.136 | 0.087 | 0.689 |
| sst5 | 0.168 | 0.121 | 0.066 | 0.123 | 0.094 | 0.529 |
| emotion | 0.252 | 0.180 | -0.002 | 0.181 | 0.044 | 0.794 |
| trec_coarse | 0.020 | -0.072 | -0.086 | 0.073 | 0.093 | 0.922 |
| gnad10 | 0.088 | 0.159 | 0.104 | 0.162 | 0.106 | 0.730 |
| patent | 0.116 | 0.165 | 0.097 | 0.165 | 0.097 | 0.611 |
| yahoo_topics | 0.128 | 0.081 | -0.014 | 0.082 | 0.053 | 0.867 |
| scotus | 0.108 | 0.132 | 0.063 | 0.132 | 0.072 | 0.749 |
| dbpedia_14 | 0.012 | -0.020 | -0.031 | 0.027 | 0.031 | 0.992 |
| massive_scenario_de | 0.062 | 0.004 | -0.028 | 0.049 | 0.055 | 0.778 |
| massive_scenario_en | 0.088 | -0.017 | -0.071 | 0.049 | 0.074 | 0.822 |
| newsgroups20 | 0.104 | 0.048 | -0.002 | 0.061 | 0.052 | 0.797 |
| trec_fine | 0.076 | -0.056 | -0.089 | 0.089 | 0.116 | 0.840 |
| massive_intent_de | 0.060 | -0.005 | -0.037 | 0.034 | 0.040 | 0.838 |
| massive_intent_en | 0.062 | -0.013 | -0.057 | 0.040 | 0.069 | 0.876 |
| banking77 | 0.082 | 0.032 | -0.026 | 0.057 | 0.035 | 0.871 |
| ledgar | 0.076 | 0.056 | 0.005 | 0.085 | 0.063 | 0.827 |
| clinc150 | 0.028 | -0.060 | -0.078 | 0.066 | 0.083 | 0.899 |

**Checking the flags on banking77.** All 80 flagged examples were read against both labels, by Claude, an LLM that is not one of the four voting systems, reading each text against both labels (not yet checked by a person). Label right 15, label wrong 25, both defensible 40. So only 31% of flags are clear label errors. Correcting just those, on all examples after the formula T:

| banking77 | examples | accuracy | overconfidence | ECE |
|---|---|---|---|---|
| as labelled | 1000 | 0.790 | 0.038 | 0.049 |
| all flagged removed | 920 | 0.851 | -0.015 | 0.029 |
| confirmed errors relabelled | 1000 | 0.814 | 0.014 | 0.037 |

Oracle T and held-out ECE by contamination tier (how likely the set was in training) and by language:

| grouping | group | datasets | oracle T (geo. mean) | ECE after formula T |
|---|---|---|---|---|
| tier | likely | 10 | 2.20 | 0.069 |
| tier | seen | 13 | 2.06 | 0.075 |
| tier | unclear | 5 | 2.72 | 0.120 |
| language | de | 5 | 2.27 | 0.089 |
| language | en | 23 | 2.20 | 0.079 |

Within families that span more than one tier:

| family | tier | dataset | accuracy | oracle T |
|---|---|---|---|---|
| intent | likely | massive_scenario_de | 0.76 | 1.81 |
| intent | likely | massive_scenario_en | 0.77 | 1.77 |
| intent | likely | massive_intent_de | 0.79 | 1.88 |
| intent | likely | massive_intent_en | 0.83 | 1.80 |
| intent | likely | clinc150 | 0.88 | 1.36 |
| intent | seen | trec_coarse | 0.90 | 1.83 |
| intent | seen | trec_fine | 0.81 | 1.59 |
| intent | seen | banking77 | 0.79 | 1.93 |
| moderation | likely | tweet_offensive | 0.78 | 3.40 |
| moderation | unclear | toxic_conversations | 0.79 | 2.71 |
| nli | likely | xnli_de | 0.80 | 2.03 |
| nli | seen | rte | 0.89 | 1.88 |
| nli | seen | mnli | 0.87 | 1.71 |
| sentiment | likely | tweet_sentiment | 0.68 | 3.39 |
| sentiment | likely | amazon_reviews_de | 0.60 | 2.69 |
| sentiment | seen | rotten_tomatoes | 0.93 | 1.98 |
| sentiment | seen | sst2 | 0.95 | 2.08 |
| sentiment | seen | sst5 | 0.48 | 2.73 |
| sentiment | seen | emotion | 0.60 | 3.74 |
| topic | likely | yahoo_topics | 0.75 | 2.87 |
| topic | seen | ag_news | 0.90 | 2.44 |
| topic | seen | dbpedia_14 | 0.98 | 1.59 |
| topic | seen | newsgroups20 | 0.74 | 2.16 |
| topic | unclear | gnad10 | 0.67 | 3.27 |
| topic | unclear | patent | 0.55 | 3.12 |

Regression over the 28 text datasets, log T = a + b·accuracy + tier: accuracy -1.39 ± 0.35, 'likely' -0.02 ± 0.09, 'unclear' +0.09 ± 0.12 (log T, against 'seen'; ± one standard error). A tier coefficient within about two standard errors of 0 means the tier adds nothing once accuracy is known.


## 6. Off-option mass

Probability the model put on the option tokens before the mask. AUROC for flagging wrong answers from low confidence and from low mass, with 95% bootstrap intervals: 0.5 is chance, below 0.5 means the signal points the other way (low mass on *right* answers). Intervals are wide where there are few wrong answers.

| dataset | wrong answers | mass when right | mass when wrong | AUROC confidence | AUROC mass |
|---|---|---|---|---|---|
| boolq | 89 | 0.999 | 0.999 | 0.83 [0.78, 0.87] | 0.44 [0.38, 0.51] |
| rotten_tomatoes | 68 | 0.940 | 0.966 | 0.86 [0.82, 0.90] | 0.33 [0.28, 0.40] |
| rte | 31 | 0.999 | 0.999 | 0.77 [0.69, 0.84] | 0.49 [0.38, 0.60] |
| sst2 | 42 | 0.958 | 0.986 | 0.88 [0.83, 0.93] | 0.30 [0.23, 0.38] |
| toxic_conversations | 208 | 0.984 | 0.965 | 0.72 [0.68, 0.75] | 0.63 [0.59, 0.68] |
| tweet_offensive | 193 | 0.992 | 0.988 | 0.79 [0.75, 0.82] | 0.57 [0.52, 0.61] |
| mnli | 132 | 0.997 | 0.996 | 0.80 [0.77, 0.84] | 0.50 [0.45, 0.56] |
| tweet_sentiment | 320 | 0.993 | 0.995 | 0.65 [0.62, 0.69] | 0.45 [0.41, 0.48] |
| xnli_de | 204 | 0.999 | 1.000 | 0.77 [0.74, 0.80] | 0.42 [0.38, 0.47] |
| ag_news | 104 | 0.998 | 0.999 | 0.84 [0.80, 0.88] | 0.36 [0.31, 0.43] |
| amazon_reviews_de | 403 | 0.998 | 0.998 | 0.68 [0.64, 0.71] | 0.41 [0.37, 0.44] |
| sst5 | 520 | 0.993 | 0.993 | 0.61 [0.57, 0.65] | 0.51 [0.47, 0.54] |
| emotion | 402 | 0.970 | 0.981 | 0.70 [0.66, 0.73] | 0.40 [0.37, 0.43] |
| trec_coarse | 48 | 0.989 | 0.997 | 0.79 [0.73, 0.85] | 0.24 [0.17, 0.31] |
| gnad10 | 332 | 0.994 | 0.997 | 0.71 [0.67, 0.74] | 0.39 [0.35, 0.42] |
| patent | 446 | 0.993 | 0.995 | 0.66 [0.62, 0.69] | 0.45 [0.42, 0.49] |
| yahoo_topics | 253 | 0.991 | 0.997 | 0.79 [0.75, 0.82] | 0.35 [0.31, 0.39] |
| scotus | 299 | 1.000 | 1.000 | 0.75 [0.72, 0.78] | 0.39 [0.35, 0.42] |
| dbpedia_14 | 20 | 0.977 | 0.998 | 0.95 [0.88, 0.99] | 0.16 [0.09, 0.25] |
| massive_scenario_de | 244 | 0.985 | 0.995 | 0.86 [0.84, 0.89] | 0.31 [0.28, 0.35] |
| massive_scenario_en | 230 | 0.984 | 0.996 | 0.86 [0.83, 0.88] | 0.30 [0.26, 0.33] |
| newsgroups20 | 263 | 0.990 | 0.994 | 0.88 [0.86, 0.91] | 0.45 [0.41, 0.50] |
| trec_fine | 96 | 0.984 | 0.981 | 0.80 [0.75, 0.85] | 0.52 [0.45, 0.57] |
| massive_intent_de | 212 | 0.989 | 0.990 | 0.86 [0.83, 0.89] | 0.49 [0.44, 0.53] |
| massive_intent_en | 172 | 0.989 | 0.991 | 0.87 [0.83, 0.90] | 0.47 [0.42, 0.51] |
| banking77 | 210 | 0.992 | 0.992 | 0.84 [0.81, 0.87] | 0.52 [0.48, 0.57] |
| ledgar | 248 | 0.996 | 0.998 | 0.79 [0.75, 0.82] | 0.41 [0.37, 0.45] |
| clinc150 | 120 | 0.982 | 0.989 | 0.87 [0.84, 0.90] | 0.44 [0.38, 0.49] |
| **pooled** (ranks within dataset) | 5909 |  |  | 0.74 [0.74, 0.75] | 0.44 [0.43, 0.45] |

Mass is significantly *inverted* (interval below 0.5) on: rotten_tomatoes, sst2, tweet_sentiment, xnli_de, ag_news, amazon_reviews_de, emotion, trec_coarse, gnad10, patent, yahoo_topics, scotus, dbpedia_14, massive_scenario_de, massive_scenario_en, newsgroups20, ledgar, clinc150. There, answers the model is right about carry slightly less probability on the options, so low mass is not a usable error signal in either direction.


## Figures

![Reliability](reliability.png)
![Per-dataset temperature](temperatures.png)
![ECE per dataset](ece.png)
![Conformal set sizes](set_sizes.png)
![Labels needed](labels_needed.png)
![Off-option mass](option_mass.png)
