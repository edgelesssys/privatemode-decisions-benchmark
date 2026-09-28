# Calibration report, part 3

Same runs, halves and temperatures as parts 1 and 2 (`eq-r1`). GLM probabilities start from the library's default temperature (the option formula fitted without the dataset in question); Jev's come from the published runs on the same examples. Means weight every dataset equally.

## 1. A bias per option

`softmax(log p / T + b)`: a temperature and one bias per option, fitted together on n random labels of the calibration half (20 draws per dataset), with the temperature pulled towards the default (worth 5 examples, as before) and the bias towards 0 (worth 2 examples: `2 / n · |b|²`). Then, as `calibrate()` does, the 90% cutoff and the automation threshold (ε = 10%) on the same labels and the corrected probabilities. Evaluated on the test half. *T* is the temperature alone (what part 2 shipped), *T + b* adds the bias and sets the cutoff and the threshold on out-of-fold probabilities (5 folds: each label's answer corrected by a fit on the other folds), then fits the final correction on all labels. Excess ECE is ECE minus the sampling floor of the same probabilities.

**GLM-5.3-Flash**, mean over datasets (datasets whose calibration half has at least n examples):

| labels | datasets | accuracy, T | accuracy, T + b | points | datasets better / worse | worst dataset | NLL | excess ECE | 90% set coverage | 90% set size | automated at ε = 10% | draws over ε |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 28 | 78.1% | 79.1% | +1.0 | 16 / 6 | -1.9 | 0.691 → 0.665 | 0.017 → 0.013 | 0.906 → 0.913 | 2.61 → 2.99 | 0% → 0% | 0.0% → 0.0% |
| 50 | 28 | 78.1% | 79.6% | +1.5 | 17 / 7 | -1.2 | 0.686 → 0.648 | 0.013 → 0.007 | 0.900 → 0.908 | 1.98 → 1.93 | 25% → 20% | 0.9% → 0.7% |
| 100 | 28 | 78.1% | 80.1% | +2.0 | 18 / 6 | -0.9 | 0.683 → 0.631 | 0.008 → 0.004 | 0.902 → 0.908 | 1.84 → 1.70 | 27% → 23% | 1.1% → 1.2% |
| 250 | 27 | 77.7% | 80.2% | +2.5 | 20 / 4 | -1.3 | 0.694 → 0.625 | 0.006 → 0.005 | 0.901 → 0.906 | 1.80 → 1.59 | 38% → 32% | 0.6% → 1.3% |
| 500 | 23 | 76.1% | 79.1% | +3.0 | 19 / 3 | -1.4 | 0.745 → 0.657 | 0.006 → 0.003 | 0.898 → 0.903 | 1.88 → 1.63 | 39% → 35% | 0.0% → 0.4% |

**Why out-of-fold.** With a bias per option, setting the cutoff and the threshold on the same labels the bias was fitted on makes the answers look better than they are, and the error bound starts to slip as the fit gets more room: at 500 labels it was broken on 2 of 23 datasets. Automated share / share of draws over ε / 90% coverage:

| labels | T, same labels (part 2) | T + b, same labels | T + b, out-of-fold (shipped) |
|---|---|---|---|
| 20 | 0% / 0.0% / 0.906 | 0% / 0.0% / 0.893 | 0% / 0.0% / 0.913 |
| 50 | 25% / 0.9% / 0.900 | 29% / 1.8% / 0.893 | 20% / 0.7% / 0.908 |
| 100 | 27% / 1.1% / 0.902 | 33% / 2.3% / 0.898 | 23% / 1.2% / 0.908 |
| 250 | 38% / 0.6% / 0.901 | 43% / 4.4% / 0.899 | 32% / 1.3% / 0.906 |
| 500 | 39% / 0.0% / 0.898 | 46% / 8.7% / 0.899 | 35% / 0.4% / 0.903 |

By option count, 100 labels (accuracy points, T + b minus T):

| options | datasets | mean | worst | largest gains |
|---|---|---|---|---|
| 2 | 6 | +2.3 | -0.9 | toxic_conversations +14.6, boolq +0.7 |
| 3–6 | 8 | +2.2 | -0.6 | sst5 +9.5, tweet_sentiment +4.7 |
| 7–20 | 8 | +2.5 | -0.9 | massive_scenario_de +6.4, massive_scenario_en +6.0 |
| 21+ | 6 | +0.7 | -0.3 | massive_intent_en +1.5, trec_fine +1.5 |

**Binary tasks.** With two options, temperature plus a bias is Platt scaling, which openjev's server applies to yes/no questions. Accuracy (NLL), temperature alone → Platt:

| dataset | 100 labels | 500 labels |
|---|---|---|
| boolq | 90.8% → 91.5% (NLL 0.226 → 0.224) | 90.8% → 91.6% (NLL 0.222 → 0.216) |
| rotten_tomatoes | 94.0% → 94.2% (NLL 0.183 → 0.184) | 94.0% → 94.4% (NLL 0.179 → 0.179) |
| rte | 89.2% → 88.3% (NLL 0.313 → 0.313) | — |
| sst2 | 95.6% → 95.6% (NLL 0.130 → 0.133) | — |
| toxic_conversations | 78.2% → 92.8% (NLL 0.469 → 0.232) | 78.2% → 93.8% (NLL 0.468 → 0.212) |
| tweet_offensive | 80.0% → 79.5% (NLL 0.419 → 0.424) | — |

**How strong a pull.** 100 labels, 10 draws per dataset, accuracy points against the temperature alone and the worst dataset:

| strength (examples) | accuracy points | worst dataset | NLL |
|---|---|---|---|
| 0.5 | +2.29 | -0.9 | 0.621 |
| 1 | +2.12 | -1.2 | 0.625 |
| 2 (chosen) | +2.05 | -1.0 | 0.630 |
| 5 | +1.58 | -1.0 | 0.643 |
| 10 | +1.28 | -1.2 | 0.652 |
| 20 | +0.84 | -0.7 | 0.662 |

**Jev with the same method**, on the 28 text datasets both answer, same examples. Jev's zeros are set to half a rounding unit first, the fairest fix from part 2, so that a temperature and a bias can be fitted, and Jev starts from its own zero-label default temperature (the option formula fitted on Jev's per-task temperatures, leaving the dataset out), as GLM starts from its own:

| labels | system | accuracy, T | accuracy, T + b | excess ECE, T | excess ECE, T + b | 90% coverage, T + b | 90% set, T + b | automated, T | automated, T + b | draws over ε, T + b |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | GLM-5.3-Flash | 78.1% | 80.2% | 0.008 | 0.005 | 0.906 | 1.70 | 28% | 23% | 0.9% |
| 100 | Jev, zeros fixed | 77.5% | 79.6% | 0.023 | 0.014 | 0.910 | 1.79 | 32% | 24% | 2.1% |
| 500 | GLM-5.3-Flash | 76.1% | 79.1% | 0.006 | 0.003 | 0.903 | 1.62 | 39% | 36% | 0.4% |
| 500 | Jev, zeros fixed | 75.1% | 78.7% | 0.022 | 0.007 | 0.906 | 1.67 | 40% | 35% | 0.9% |

## 2. Several option orders: their own temperature

`SystemOne(permutations=k)` averages k rotated option orders and then applied the one-order default temperature. From the rotation runs (28 text datasets, 100 rows each, 4 orders), the orders the library would ask for k = 2 and 4. Temperatures fitted per dataset (*own T*) or by the option formula leaving the dataset out (*LODO*). Averaging softens, so the best temperature falls: the median ratio to one order's is 0.95 for 2 orders and 0.91 for 4. *Automatable* is the share of answers that can be automated at an observed error of at most 10%, most confident first, knowing the labels: how well the confidence ranks answers.

| orders | temperature | accuracy | NLL | excess ECE | automatable |
|---|---|---|---|---|---|
| 1 | one-order formula | 78.96% | 0.703 | 0.016 | 61% |
| 1 | own formula (LODO) | 78.96% | 0.703 | 0.016 | 61% |
| 1 | own T per dataset | 78.96% | 0.672 | 0.003 | 61% |
| 2 | one-order formula | 78.93% | 0.690 | 0.016 | 63% |
| 2 | own formula (LODO) | 78.93% | 0.712 | 0.032 | 63% |
| 2 | one-order formula × ratio | 78.93% | 0.692 | 0.019 | 63% |
| 2 | own T per dataset | 78.93% | 0.658 | 0.004 | 63% |
| 4 | one-order formula | 78.64% | 0.685 | 0.014 | 64% |
| 4 | own formula (LODO) | 78.64% | 0.706 | 0.031 | 65% |
| 4 | one-order formula × ratio | 78.64% | 0.686 | 0.021 | 65% |
| 4 | own T per dataset | 78.64% | 0.649 | 0.000 | 65% |

Formulas `log T = a + b · log(options)` fitted on all datasets: 1 order: a = 0.986, b = -0.086; 2 orders: a = 0.871, b = -0.112; 4 orders: a = 0.887, b = -0.131.


**Re-reading only uncertain answers.** razorback16/openjev asks a question three more times when the entropy of its answer is above 0.1 nats and averages the four. The same with rotations: the first order's raw distribution decides, the answers above the entropy threshold are asked in all 4 orders and averaged, and each kind gets its own temperature (the LODO formulas above).

| re-read when entropy | answers re-read | extra requests | accuracy | NLL | excess ECE | automatable |
|---|---|---|---|---|---|---|
| never (one order) | 0% | +0% | 78.96% | 0.703 | 0.016 | 61% |
| > 0.5 nats | 16% | +48% | 78.61% | 0.703 | 0.015 | 62% |
| > 0.3 nats | 23% | +69% | 78.68% | 0.698 | 0.020 | 62% |
| > 0.1 nats | 34% | +102% | 78.61% | 0.699 | 0.019 | 63% |
| > 0.05 nats | 41% | +123% | 78.61% | 0.697 | 0.022 | 63% |
| always (4 orders) | 100% | +300% | 78.64% | 0.706 | 0.031 | 65% |

**Order stability.** How often the answer changes when the options are rotated: the share of the other orders whose answer differs from the first order's. GLM Flash changes its answer in 9.0% of rotations (median 8.2%). openjev reports 18.5% for its untuned base and 2.3% after fine-tuning, measured by shuffling on its own questions, so the numbers are indicative only.

| dataset | options | answer changes | accuracy, one order |
|---|---|---|---|
| sst5 | 5 | 23.0% | 0.51 |
| newsgroups20 | 20 | 21.0% | 0.69 |
| amazon_reviews_de | 5 | 18.0% | 0.58 |
| patent | 9 | 16.0% | 0.55 |
| ledgar | 100 | 14.3% | 0.81 |
| clinc150 | 151 | 12.3% | 0.89 |
| massive_scenario_de | 18 | 12.0% | 0.79 |
| massive_scenario_en | 18 | 12.0% | 0.80 |
| tweet_sentiment | 3 | 11.5% | 0.76 |
| trec_fine | 42 | 11.0% | 0.78 |
| banking77 | 77 | 11.0% | 0.76 |
| xnli_de | 3 | 8.5% | 0.78 |
| emotion | 6 | 8.3% | 0.61 |
| scotus | 13 | 8.3% | 0.70 |
| rte | 2 | 8.0% | 0.83 |
| massive_intent_de | 59 | 7.7% | 0.83 |
| massive_intent_en | 59 | 7.7% | 0.91 |
| mnli | 3 | 7.0% | 0.85 |
| yahoo_topics | 10 | 7.0% | 0.74 |
| gnad10 | 9 | 6.0% | 0.72 |
| trec_coarse | 6 | 5.7% | 0.93 |
| boolq | 2 | 4.0% | 0.92 |
| toxic_conversations | 2 | 3.0% | 0.85 |
| tweet_offensive | 2 | 3.0% | 0.74 |
| ag_news | 4 | 2.3% | 0.92 |
| rotten_tomatoes | 2 | 1.0% | 0.94 |
| dbpedia_14 | 14 | 1.0% | 0.99 |
| sst2 | 2 | 0.0% | 0.93 |

## 3. Many options, few labels

Prediction sets at 90% on the text datasets with more than 20 options, 20 random 50/50 splits each (the calibration half, about 500 labels, is also what fits the task temperature). *Per class* is `per_class=True`: an option with fewer than 9 labels is in every set. *Clustered (Ding)* is Ding et al. (2023): half of the labels describe each class by quantiles of its scores, k-means groups classes that behave alike (one group per 50 remaining labels), and the other half sets a cutoff per group; classes with too few labels share the marginal cutoff. *Grouped by answers* groups classes the same way but from the unlabelled answers (quantiles of the probability a class gets where it is the answer), so every label goes into the cutoffs. **Class gap** is the mean distance of a class's coverage from 90%, over classes seen at least 10 times across the splits; **under 80%** is the share of those classes covered less than 80% of the time.

| dataset | options | cutoffs | coverage | set size | class gap | classes under 80% |
|---|---|---|---|---|---|---|
| trec_fine | 42 | one cutoff | 0.899 | 1.38 | 0.161 | 26% |
|  |  | per class | 0.964 | 36.84 | 0.093 | 0% |
|  |  | clustered (Ding) | 0.884 | 1.44 | 0.142 | 26% |
|  |  | grouped by answers | 0.899 | 2.74 | 0.140 | 26% |
| massive_intent_de | 59 | one cutoff | 0.900 | 1.65 | 0.078 | 5% |
|  |  | per class | 0.952 | 40.88 | 0.078 | 0% |
|  |  | clustered (Ding) | 0.920 | 2.04 | 0.076 | 4% |
|  |  | grouped by answers | 0.911 | 2.67 | 0.064 | 5% |
| massive_intent_en | 59 | one cutoff | 0.896 | 1.37 | 0.107 | 10% |
|  |  | per class | 0.960 | 40.71 | 0.081 | 0% |
|  |  | clustered (Ding) | 0.916 | 1.75 | 0.099 | 10% |
|  |  | grouped by answers | 0.920 | 2.32 | 0.090 | 9% |
| banking77 | 77 | one cutoff | 0.902 | 2.02 | 0.114 | 16% |
|  |  | per class | 0.987 | 63.54 | 0.089 | 0% |
|  |  | clustered (Ding) | 0.905 | 2.25 | 0.112 | 16% |
|  |  | grouped by answers | 0.902 | 3.93 | 0.099 | 18% |
| ledgar | 100 | one cutoff | 0.898 | 2.12 | 0.139 | 18% |
|  |  | per class | 0.984 | 88.10 | 0.094 | 0% |
|  |  | clustered (Ding) | 0.888 | 2.08 | 0.139 | 19% |
|  |  | grouped by answers | 0.876 | 3.07 | 0.129 | 19% |
| clinc150 | 151 | one cutoff | 0.903 | 1.05 | 0.123 | 14% |
|  |  | per class | 0.985 | 150.15 | 0.099 | 0% |
|  |  | clustered (Ding) | 0.932 | 1.14 | 0.121 | 12% |
|  |  | grouped by answers | 0.924 | 1.36 | 0.120 | 14% |

Mean over these datasets:

| cutoffs | coverage | set size | class gap | classes under 80% |
|---|---|---|---|---|
| one cutoff | 0.900 | 1.60 | 0.120 | 15% |
| per class | 0.972 | 70.04 | 0.089 | 0% |
| clustered (Ding) | 0.907 | 1.78 | 0.115 | 14% |
| grouped by answers | 0.905 | 2.68 | 0.107 | 15% |

**Jev with the same cutoffs** (trec_fine, massive_intent_de, massive_intent_en, banking77, ledgar, clinc150; zeros set to half a rounding unit, then its own task temperature). GLM on all its rows, Jev on the rows it answered:

| cutoffs | GLM set size | GLM class gap | Jev set size | Jev class gap |
|---|---|---|---|---|
| one cutoff | 1.60 | 0.120 | 3.53 | 0.120 |
| per class | 70.04 | 0.089 | 70.81 | 0.089 |
| clustered (Ding) | 1.78 | 0.115 | 3.66 | 0.113 |
| grouped by answers | 2.68 | 0.107 | 4.14 | 0.111 |

**Known class rates, from logs only.** `p′ ∝ p · π / π̂`: π are the class rates someone knows from logs, π̂ the mean answer over unlabelled traffic (the test half). The rates are the dataset's true ones, each multiplied by a random factor of up to ±25% or up to 2× either way, then renormalized; 20 draws. Text datasets with up to 20 options, accuracy points against the default temperature, and the bias from 50 and 100 labels (section 1) on the same datasets:

| correction | accuracy points | worst dataset | datasets better / worse |
|---|---|---|---|
| exact rates | +2.6 | -2.2 | 16 / 3 |
| within 25% | +2.5 | -1.6 | 14 / 3 |
| up to 2× off | +1.8 | -3.1 | 13 / 8 |
| bias from 50 labels | +1.8 | -1.2 | 14 / 6 |
| bias from 100 labels | +2.3 | -0.9 | 15 / 5 |

## 4. Other models

One run of each model on all 29 datasets (up to 1,000 examples, 4 in flight), the same halves. *Served* is the model the endpoint reported answering, recorded in every row. The formula is fitted on all text datasets; excess ECE after it is leave-one-dataset-out, as for GLM Flash in part 1.

| run | served | text datasets | accuracy | option mass | task T, median (range) | formula a, b | excess ECE raw | excess ECE, formula (LODO) | excess ECE, task T | rvl_cdip: accuracy, own vs formula T |
|---|---|---|---|---|---|---|---|---|---|---|
| glm-5.3-flash (part 1 run) | — | 28 | 78.0% | 0.973 | 2.03 (1.36–3.74) | 0.986, -0.086 (r = -0.43) | 0.129 | 0.032 | 0.006 | 69.6%, T 2.14 vs 2.11 |
| kimi-k2.6 | kimi-k2.6 | 28 | 78.9% | 1.000 | 2.17 (1.43–4.95) | 1.080, -0.099 (r = -0.45) | 0.121 | 0.025 | 0.009 | 81.8%, T 1.96 vs 2.24 |
| glm-5.3 | glm-5.3 | 28 | 76.9% | 0.644 | 1.96 (0.68–3.66) | 0.728, -0.015 (r = -0.05) | 0.127 | 0.049 | 0.011 | — |

## Figures

![bias](bias.png)
![order_stability](order_stability.png)
