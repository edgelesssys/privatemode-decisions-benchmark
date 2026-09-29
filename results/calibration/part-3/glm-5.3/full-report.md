# Calibration report

Run: `glm-5.3`, 28 text datasets. Calibration/test halves are a fixed random split, seed 0. ECE uses 15 equal-mass bins of top-label confidence; means over datasets weight every dataset equally.

## 1. Raw probabilities

Raw probabilities are **overconfident by 14.5 points** on average (mean confidence minus accuracy, text datasets, test halves); pooled ECE 0.145. Overconfident on 26 of 28 text datasets.

| dataset | options | test n | accuracy | confidence | overconfidence | ECE [95% CI] | NLL | Brier |
|---|---|---|---|---|---|---|---|---|
| boolq | 2 | 500 | 0.90 | 0.89 | -0.007 | 0.035 [0.026, 0.060] | 0.228 | 0.135 |
| rotten_tomatoes | 2 | 500 | 0.95 | 0.97 | 0.024 | 0.041 [0.021, 0.060] | 0.219 | 0.095 |
| rte (small) | 2 | 139 | 0.86 | 0.84 | -0.019 | 0.047 [0.045, 0.117] | 0.295 | 0.190 |
| sst2 | 2 | 436 | 0.97 | 0.98 | 0.010 | 0.017 [0.008, 0.031] | 0.114 | 0.048 |
| toxic_conversations | 2 | 500 | 0.69 | 0.83 | 0.143 | 0.163 [0.134, 0.209] | 0.746 | 0.457 |
| tweet_offensive | 2 | 430 | 0.79 | 0.86 | 0.071 | 0.079 [0.059, 0.123] | 0.463 | 0.295 |
| mnli | 3 | 500 | 0.82 | 0.90 | 0.087 | 0.087 [0.065, 0.122] | 0.508 | 0.269 |
| tweet_sentiment | 3 | 500 | 0.64 | 0.91 | 0.274 | 0.274 [0.232, 0.313] | 1.444 | 0.591 |
| xnli_de | 3 | 500 | 0.72 | 0.87 | 0.153 | 0.155 [0.125, 0.193] | 0.788 | 0.427 |
| ag_news | 4 | 500 | 0.89 | 0.97 | 0.080 | 0.080 [0.058, 0.106] | 0.558 | 0.199 |
| amazon_reviews_de | 5 | 500 | 0.60 | 0.87 | 0.272 | 0.272 [0.233, 0.313] | 1.494 | 0.644 |
| sst5 | 5 | 500 | 0.48 | 0.85 | 0.377 | 0.377 [0.337, 0.420] | 2.202 | 0.873 |
| emotion | 6 | 500 | 0.61 | 0.93 | 0.318 | 0.318 [0.279, 0.360] | 2.306 | 0.695 |
| trec_coarse (small) | 6 | 250 | 0.91 | 0.96 | 0.049 | 0.049 [0.026, 0.082] | 0.275 | 0.130 |
| gnad10 | 9 | 500 | 0.63 | 0.90 | 0.272 | 0.272 [0.236, 0.311] | 1.909 | 0.631 |
| patent | 9 | 500 | 0.52 | 0.86 | 0.337 | 0.337 [0.292, 0.381] | 2.387 | 0.771 |
| yahoo_topics | 10 | 500 | 0.75 | 0.93 | 0.176 | 0.177 [0.146, 0.217] | 1.637 | 0.434 |
| scotus | 13 | 500 | 0.66 | 0.91 | 0.256 | 0.256 [0.219, 0.291] | 1.993 | 0.571 |
| dbpedia_14 | 14 | 500 | 0.98 | 0.99 | 0.010 | 0.010 [0.001, 0.020] | 0.078 | 0.028 |
| massive_scenario_de | 18 | 500 | 0.74 | 0.89 | 0.153 | 0.156 [0.128, 0.195] | 1.066 | 0.413 |
| massive_scenario_en | 18 | 500 | 0.74 | 0.91 | 0.171 | 0.171 [0.138, 0.205] | 1.120 | 0.414 |
| newsgroups20 | 20 | 500 | 0.74 | 0.89 | 0.151 | 0.151 [0.122, 0.183] | 1.235 | 0.392 |
| trec_fine (small) | 42 | 250 | 0.81 | 0.91 | 0.095 | 0.096 [0.067, 0.144] | 0.759 | 0.293 |
| massive_intent_de | 59 | 500 | 0.83 | 0.94 | 0.103 | 0.103 [0.078, 0.131] | 0.843 | 0.267 |
| massive_intent_en | 59 | 500 | 0.85 | 0.96 | 0.109 | 0.109 [0.082, 0.139] | 0.936 | 0.254 |
| banking77 | 77 | 500 | 0.80 | 0.95 | 0.142 | 0.142 [0.112, 0.173] | 1.252 | 0.329 |
| ledgar | 100 | 500 | 0.75 | 0.93 | 0.180 | 0.180 [0.150, 0.210] | 1.727 | 0.417 |
| clinc150 | 151 | 500 | 0.86 | 0.93 | 0.069 | 0.073 [0.054, 0.101] | 0.729 | 0.228 |

## 2. Temperature

Best temperature per dataset (fitted on the calibration half): median 1.96, range 0.68–3.66, all above 1 (= overconfident): False. One global T fitted on all 28 text calibration halves: **T = 2.09**. Formula over all datasets: log T = 0.728 + -0.015·log(options).

How each zero-label method does on datasets left out of its fit. **Excess ECE** is a dataset's ECE minus its floor, the ECE a perfectly calibrated model shows on the same number of examples, averaged over datasets: 0 means as calibrated as the sample can show. (A pooled ECE over all datasets is not used: it sits at the floor for every method and hides the differences.) The ECE share is the total reduction over the per-task reduction; the NLL share is the median per dataset.

| method | mean excess ECE | ECE share | NLL share (median) |
|---|---|---|---|
| raw | 0.127 | 0 | 0 |
| global T (LODO) | 0.045 | 0.60 | 0.93 |
| formula (LODO) | 0.049 | 0.58 | 0.91 |
| same family (LODO) | 0.034 | 0.71 | 0.97 |
| leave family out | 0.059 | 0.43 | 0.87 |
| formula, related datasets held out | 0.048 | 0.58 | 0.91 |
| formula, whole family held out | 0.079 | 0.22 | 0.87 |
| formula, fitted on half the tasks, tested on the other half (50 splits) | 0.054 | 0.52 (0.40–0.65) |  |
| per task (oracle) | 0.011 | 1 | 1 |

Stricter hold-outs for the formula: leaving out related datasets together (the four MASSIVE sets, both TREC sets, MNLI and XNLI, the SST family, the two TweetEval tasks) gives excess ECE 0.048, fitting on half of the tasks 0.054, against 0.049 leaving out one dataset; with the whole task family held out, 0.079, the realistic worst case for a new kind of task. The method itself (the formula's form, the shrinkage, the prefill) was chosen on these datasets, which no split can undo; only datasets kept out of the whole study can measure that.


`floor` is the ECE a perfectly calibrated model would show on this many examples (labels drawn from its own probabilities); values near it are as good as the sample can show.

| dataset | options | oracle T | raw ECE | oracle ECE | floor | global T ECE | formula ECE | same-family ECE | leave-family-out ECE |
|---|---|---|---|---|---|---|---|---|---|
| boolq | 2 | 1.02 | 0.035 | 0.036 | 0.033 | 0.123 (T=2.10) | 0.130 (T=2.18) | 0.123 (T=2.10) | 0.123 (T=2.10) |
| rotten_tomatoes | 2 | 2.00 | 0.041 | 0.038 | 0.028 | 0.039 (T=2.09) | 0.039 (T=2.05) | 0.089 (T=3.24) | 0.037 (T=1.97) |
| rte | 2 | 0.68 | 0.047 | 0.040 | 0.055 | 0.122 (T=2.10) | 0.134 (T=2.27) | 0.093 (T=1.69) | 0.123 (T=2.11) |
| sst2 | 2 | 1.84 | 0.017 | 0.029 | 0.025 | 0.037 (T=2.09) | 0.036 (T=2.07) | 0.095 (T=3.25) | 0.033 (T=1.97) |
| toxic_conversations | 2 | 2.49 | 0.163 | 0.081 | 0.060 | 0.086 (T=2.09) | 0.089 (T=2.01) | 0.084 (T=2.17) | 0.086 (T=2.09) |
| tweet_offensive | 2 | 2.17 | 0.079 | 0.064 | 0.058 | 0.061 (T=2.09) | 0.060 (T=2.04) | 0.076 (T=2.49) | 0.061 (T=2.09) |
| mnli | 3 | 1.62 | 0.087 | 0.048 | 0.042 | 0.061 (T=2.10) | 0.059 (T=2.07) | 0.047 (T=1.39) | 0.062 (T=2.11) |
| tweet_sentiment | 3 | 3.57 | 0.274 | 0.057 | 0.061 | 0.165 (T=2.07) | 0.176 (T=1.96) | 0.084 (T=3.06) | 0.175 (T=1.97) |
| xnli_de | 3 | 1.77 | 0.155 | 0.067 | 0.055 | 0.048 (T=2.10) | 0.047 (T=2.06) | 0.117 (T=1.27) | 0.048 (T=2.11) |
| ag_news | 4 | 2.23 | 0.080 | 0.044 | 0.035 | 0.049 (T=2.09) | 0.048 (T=2.02) | 0.039 (T=2.43) | 0.048 (T=2.01) |
| amazon_reviews_de | 5 | 2.84 | 0.272 | 0.066 | 0.062 | 0.130 (T=2.08) | 0.139 (T=1.99) | 0.057 (T=3.20) | 0.142 (T=1.97) |
| sst5 | 5 | 3.55 | 0.377 | 0.092 | 0.066 | 0.224 (T=2.06) | 0.239 (T=1.97) | 0.124 (T=3.03) | 0.239 (T=1.97) |
| emotion | 6 | 3.66 | 0.318 | 0.059 | 0.063 | 0.208 (T=2.05) | 0.218 (T=1.97) | 0.103 (T=2.92) | 0.218 (T=1.97) |
| trec_coarse | 6 | 1.77 | 0.049 | 0.029 | 0.044 | 0.048 (T=2.10) | 0.046 (T=2.03) | 0.027 (T=1.74) | 0.082 (T=2.40) |
| gnad10 | 9 | 2.72 | 0.272 | 0.087 | 0.059 | 0.147 (T=2.07) | 0.158 (T=1.98) | 0.115 (T=2.35) | 0.154 (T=2.01) |
| patent | 9 | 2.91 | 0.337 | 0.070 | 0.063 | 0.165 (T=2.06) | 0.180 (T=1.98) | 0.123 (T=2.32) | 0.174 (T=2.01) |
| yahoo_topics | 10 | 2.83 | 0.177 | 0.063 | 0.054 | 0.094 (T=2.06) | 0.107 (T=1.97) | 0.064 (T=2.32) | 0.100 (T=2.01) |
| scotus | 13 | 2.39 | 0.256 | 0.088 | 0.055 | 0.133 (T=2.08) | 0.145 (T=1.98) | 0.133 (T=2.08) | 0.134 (T=2.08) |
| dbpedia_14 | 14 | 1.16 | 0.010 | 0.009 | 0.004 | 0.036 (T=2.10) | 0.029 (T=2.03) | 0.091 (T=2.54) | 0.027 (T=2.01) |
| massive_scenario_de | 18 | 1.92 | 0.156 | 0.065 | 0.044 | 0.059 (T=2.10) | 0.060 (T=1.99) | 0.089 (T=1.73) | 0.075 (T=2.40) |
| massive_scenario_en | 18 | 1.88 | 0.171 | 0.087 | 0.040 | 0.064 (T=2.10) | 0.076 (T=1.99) | 0.095 (T=1.73) | 0.064 (T=2.40) |
| newsgroups20 | 20 | 2.06 | 0.151 | 0.054 | 0.044 | 0.054 (T=2.09) | 0.060 (T=1.98) | 0.073 (T=2.51) | 0.059 (T=2.01) |
| trec_fine | 42 | 1.68 | 0.096 | 0.098 | 0.057 | 0.105 (T=2.12) | 0.109 (T=1.98) | 0.100 (T=1.75) | 0.154 (T=2.40) |
| massive_intent_de | 59 | 1.80 | 0.103 | 0.047 | 0.039 | 0.081 (T=2.11) | 0.060 (T=1.97) | 0.047 (T=1.74) | 0.149 (T=2.40) |
| massive_intent_en | 59 | 1.81 | 0.109 | 0.034 | 0.034 | 0.049 (T=2.11) | 0.029 (T=1.96) | 0.036 (T=1.74) | 0.114 (T=2.40) |
| banking77 | 77 | 1.88 | 0.142 | 0.036 | 0.041 | 0.052 (T=2.11) | 0.033 (T=1.95) | 0.060 (T=1.72) | 0.123 (T=2.40) |
| ledgar | 100 | 2.08 | 0.180 | 0.037 | 0.047 | 0.036 (T=2.09) | 0.053 (T=1.91) | 0.090 (T=2.39) | 0.037 (T=2.08) |
| clinc150 | 151 | 1.37 | 0.073 | 0.062 | 0.026 | 0.176 (T=2.16) | 0.153 (T=2.09) | 0.085 (T=1.82) | 0.261 (T=2.40) |

Oracle T by family (geometric mean): qa 1.02, nli 1.25, intent 1.76, topic 2.22, legal 2.23, moderation 2.33, sentiment 2.80.


**Shipped in the library** (fitted on all examples): log T = 0.712 + -0.008·log(options); per family: intent 1.75, legal 2.18, moderation 2.23, nli 1.54, qa 0.96, sentiment 3.09, topic 2.41.


## 4. Conformal prediction sets

Split conformal on each dataset's calibration half, evaluated on its test half. LAC scores 1 − p(gold); APS the mass of options at least as likely as the gold one. Probabilities: raw, and after the formula temperature fitted without the dataset (the library's default).

| target | probabilities | score | mean coverage | mean set size | single-option share |
|---|---|---|---|---|---|
| 90% | raw | LAC | 0.909 | 1.99 | 0.65 |
| 90% | raw | APS | 0.969 | 8.74 | 0.23 |
| 90% | T | LAC | 0.909 | 1.85 | 0.62 |
| 90% | T | APS | 0.959 | 7.92 | 0.26 |
| 95% | raw | LAC | 0.951 | 3.45 | 0.49 |
| 95% | raw | APS | 0.983 | 9.93 | 0.09 |
| 95% | T | LAC | 0.952 | 3.28 | 0.46 |
| 95% | T | APS | 0.979 | 9.39 | 0.11 |

Per dataset, LAC after the formula T:

| dataset | options | coverage 90% | size 90% | single 90% | coverage 95% | size 95% | single 95% |
|---|---|---|---|---|---|---|---|
| boolq | 2 | 0.932 | 1.05 | 0.95 | 0.970 | 1.19 | 0.81 |
| rotten_tomatoes | 2 | 0.948 | 1.00 | 1.00 | 0.962 | 1.06 | 0.94 |
| rte | 2 | 0.871 | 1.04 | 0.96 | 0.950 | 1.22 | 0.78 |
| sst2 | 2 | 0.970 | 1.00 | 1.00 | 0.970 | 1.00 | 1.00 |
| toxic_conversations | 2 | 0.880 | 1.44 | 0.56 | 0.932 | 1.59 | 0.41 |
| tweet_offensive | 2 | 0.933 | 1.36 | 0.64 | 0.970 | 1.53 | 0.47 |
| mnli | 3 | 0.908 | 1.20 | 0.80 | 0.950 | 1.46 | 0.59 |
| tweet_sentiment | 3 | 0.894 | 1.87 | 0.42 | 0.936 | 2.18 | 0.29 |
| xnli_de | 3 | 0.886 | 1.46 | 0.59 | 0.944 | 1.80 | 0.36 |
| ag_news | 4 | 0.892 | 1.01 | 0.99 | 0.942 | 1.19 | 0.84 |
| amazon_reviews_de | 5 | 0.914 | 2.25 | 0.28 | 0.954 | 2.64 | 0.17 |
| sst5 | 5 | 0.910 | 2.86 | 0.15 | 0.926 | 3.27 | 0.10 |
| emotion | 6 | 0.930 | 3.36 | 0.19 | 0.962 | 4.31 | 0.06 |
| trec_coarse | 6 | 0.944 | 1.04 | 0.96 | 0.984 | 1.30 | 0.77 |
| gnad10 | 9 | 0.908 | 2.49 | 0.35 | 0.946 | 3.67 | 0.21 |
| patent | 9 | 0.894 | 4.26 | 0.03 | 0.932 | 6.03 | 0.00 |
| yahoo_topics | 10 | 0.908 | 2.82 | 0.37 | 0.954 | 4.58 | 0.12 |
| scotus | 13 | 0.876 | 2.63 | 0.32 | 0.944 | 5.14 | 0.07 |
| dbpedia_14 | 14 | 0.984 | 1.00 | 1.00 | 0.984 | 1.00 | 1.00 |
| massive_scenario_de | 18 | 0.902 | 1.67 | 0.58 | 0.960 | 2.66 | 0.46 |
| massive_scenario_en | 18 | 0.862 | 1.41 | 0.70 | 0.924 | 1.79 | 0.56 |
| newsgroups20 | 20 | 0.880 | 2.24 | 0.59 | 0.956 | 3.82 | 0.50 |
| trec_fine | 42 | 0.920 | 1.55 | 0.62 | 0.964 | 2.42 | 0.41 |
| massive_intent_de | 59 | 0.918 | 1.43 | 0.72 | 0.972 | 5.71 | 0.35 |
| massive_intent_en | 59 | 0.878 | 1.11 | 0.90 | 0.948 | 2.19 | 0.61 |
| banking77 | 77 | 0.904 | 1.75 | 0.62 | 0.940 | 7.41 | 0.24 |
| ledgar | 100 | 0.914 | 4.30 | 0.34 | 0.954 | 18.16 | 0.03 |
| clinc150 | 151 | 0.896 | 1.16 | 0.85 | 0.936 | 1.57 | 0.65 |

**Global cutoff (zero labels, heuristic, no guarantee):** LAC cutoff pooled over the other datasets' calibration halves, applied to the held-out dataset.

| target | mean coverage | worst | best | datasets > 2 points short |
|---|---|---|---|---|
| 90% | 0.898 | 0.730 | 1.000 | 11 of 28 |
| 95% | 0.949 | 0.830 | 1.000 | 9 of 28 |

**Labels needed** (90% target, 200 random calibration draws per dataset):

| labels | mean coverage | 5th percentile | 95th percentile | share of draws below 88% |
|---|---|---|---|---|
| 50 | 0.911 | 0.834 | 0.977 | 0.23 |
| 100 | 0.909 | 0.852 | 0.970 | 0.22 |
| 250 | 0.910 | 0.862 | 0.970 | 0.15 |
| 500 | 0.905 | 0.876 | 0.948 | 0.13 |

## 5. Label noise, contamination, language


Oracle T and held-out ECE by contamination tier (how likely the set was in training) and by language:

| grouping | group | datasets | oracle T (geo. mean) | ECE after formula T |
|---|---|---|---|---|
| tier | likely | 10 | 2.11 | 0.091 |
| tier | seen | 13 | 1.76 | 0.091 |
| tier | unclear | 5 | 2.50 | 0.125 |
| language | de | 5 | 2.16 | 0.093 |
| language | en | 23 | 1.97 | 0.098 |

Within families that span more than one tier:

| family | tier | dataset | accuracy | oracle T |
|---|---|---|---|---|
| intent | likely | massive_scenario_de | 0.75 | 1.92 |
| intent | likely | massive_scenario_en | 0.77 | 1.88 |
| intent | likely | massive_intent_de | 0.83 | 1.80 |
| intent | likely | massive_intent_en | 0.85 | 1.81 |
| intent | likely | clinc150 | 0.85 | 1.37 |
| intent | seen | trec_coarse | 0.91 | 1.77 |
| intent | seen | trec_fine | 0.82 | 1.68 |
| intent | seen | banking77 | 0.80 | 1.88 |
| moderation | likely | tweet_offensive | 0.77 | 2.17 |
| moderation | unclear | toxic_conversations | 0.69 | 2.49 |
| nli | likely | xnli_de | 0.73 | 1.77 |
| nli | seen | rte | 0.88 | 0.68 |
| nli | seen | mnli | 0.82 | 1.62 |
| sentiment | likely | tweet_sentiment | 0.64 | 3.57 |
| sentiment | likely | amazon_reviews_de | 0.59 | 2.84 |
| sentiment | seen | rotten_tomatoes | 0.94 | 2.00 |
| sentiment | seen | sst2 | 0.96 | 1.84 |
| sentiment | seen | sst5 | 0.48 | 3.55 |
| sentiment | seen | emotion | 0.60 | 3.66 |
| topic | likely | yahoo_topics | 0.73 | 2.83 |
| topic | seen | ag_news | 0.89 | 2.23 |
| topic | seen | dbpedia_14 | 0.99 | 1.16 |
| topic | seen | newsgroups20 | 0.75 | 2.06 |
| topic | unclear | gnad10 | 0.63 | 2.72 |
| topic | unclear | patent | 0.53 | 2.91 |

Regression over the 28 text datasets, log T = a + b·accuracy + tier: accuracy -2.21 ± 0.45, 'likely' +0.01 ± 0.12, 'unclear' -0.03 ± 0.16 (log T, against 'seen'; ± one standard error). A tier coefficient within about two standard errors of 0 means the tier adds nothing once accuracy is known.


## 6. Off-option mass

Probability the model put on the option tokens before the mask. AUROC for flagging wrong answers from low confidence and from low mass, with 95% bootstrap intervals: 0.5 is chance, below 0.5 means the signal points the other way (low mass on *right* answers). Intervals are wide where there are few wrong answers.

| dataset | wrong answers | mass when right | mass when wrong | AUROC confidence | AUROC mass |
|---|---|---|---|---|---|
| boolq | 106 | 0.483 | 0.531 | 0.86 [0.83, 0.89] | 0.48 [0.43, 0.53] |
| rotten_tomatoes | 63 | 0.621 | 0.606 | 0.82 [0.77, 0.87] | 0.52 [0.44, 0.60] |
| rte | 32 | 0.706 | 0.754 | 0.85 [0.79, 0.90] | 0.41 [0.31, 0.52] |
| sst2 | 33 | 0.562 | 0.603 | 0.85 [0.78, 0.90] | 0.51 [0.41, 0.61] |
| toxic_conversations | 306 | 0.915 | 0.834 | 0.65 [0.62, 0.69] | 0.61 [0.57, 0.64] |
| tweet_offensive | 197 | 0.886 | 0.837 | 0.75 [0.71, 0.78] | 0.58 [0.53, 0.62] |
| mnli | 178 | 0.679 | 0.616 | 0.82 [0.79, 0.85] | 0.53 [0.49, 0.58] |
| tweet_sentiment | 359 | 0.862 | 0.878 | 0.67 [0.63, 0.70] | 0.46 [0.42, 0.49] |
| xnli_de | 269 | 0.669 | 0.712 | 0.73 [0.69, 0.76] | 0.46 [0.43, 0.50] |
| ag_news | 106 | 0.438 | 0.351 | 0.82 [0.78, 0.86] | 0.55 [0.49, 0.61] |
| amazon_reviews_de | 413 | 0.570 | 0.611 | 0.70 [0.67, 0.74] | 0.47 [0.43, 0.50] |
| sst5 | 520 | 0.393 | 0.415 | 0.56 [0.52, 0.59] | 0.47 [0.44, 0.51] |
| emotion | 396 | 0.438 | 0.388 | 0.71 [0.67, 0.74] | 0.53 [0.49, 0.57] |
| trec_coarse | 47 | 0.406 | 0.394 | 0.86 [0.80, 0.90] | 0.49 [0.42, 0.57] |
| gnad10 | 374 | 0.469 | 0.371 | 0.78 [0.75, 0.80] | 0.58 [0.54, 0.61] |
| patent | 471 | 0.562 | 0.563 | 0.64 [0.61, 0.67] | 0.49 [0.46, 0.53] |
| yahoo_topics | 265 | 0.678 | 0.590 | 0.78 [0.75, 0.81] | 0.57 [0.53, 0.61] |
| scotus | 317 | 0.651 | 0.630 | 0.77 [0.74, 0.81] | 0.51 [0.48, 0.55] |
| dbpedia_14 | 14 | 0.245 | 0.241 | 0.98 [0.96, 0.99] | 0.58 [0.43, 0.72] |
| massive_scenario_de | 252 | 0.741 | 0.683 | 0.84 [0.81, 0.86] | 0.54 [0.50, 0.59] |
| massive_scenario_en | 234 | 0.682 | 0.620 | 0.84 [0.81, 0.86] | 0.55 [0.51, 0.59] |
| newsgroups20 | 247 | 0.796 | 0.768 | 0.89 [0.87, 0.91] | 0.55 [0.51, 0.59] |
| trec_fine | 89 | 0.852 | 0.810 | 0.85 [0.81, 0.89] | 0.54 [0.47, 0.60] |
| massive_intent_de | 172 | 0.941 | 0.922 | 0.84 [0.80, 0.87] | 0.59 [0.54, 0.63] |
| massive_intent_en | 146 | 0.952 | 0.940 | 0.84 [0.80, 0.88] | 0.54 [0.49, 0.59] |
| banking77 | 196 | 0.915 | 0.885 | 0.84 [0.81, 0.87] | 0.57 [0.53, 0.62] |
| ledgar | 261 | 0.967 | 0.955 | 0.81 [0.78, 0.84] | 0.54 [0.50, 0.58] |
| clinc150 | 152 | 0.971 | 0.966 | 0.87 [0.84, 0.90] | 0.49 [0.44, 0.54] |
| **pooled** (ranks within dataset) | 6215 |  |  | 0.74 [0.73, 0.75] | 0.52 [0.52, 0.53] |

Mass is significantly *inverted* (interval below 0.5) on: tweet_sentiment. There, answers the model is right about carry slightly less probability on the options.


## Figures

![Reliability](reliability.png)
![Per-dataset temperature](temperatures.png)
![ECE per dataset](ece.png)
![Conformal set sizes](set_sizes.png)
![Labels needed](labels_needed.png)
![Off-option mass](option_mass.png)
