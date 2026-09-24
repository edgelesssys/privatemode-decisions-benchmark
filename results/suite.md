# Suite result

Products: jev, laya, privatemode. Controls (not products, and not part of the like-for-like comparison): embed-nn, glm-cot.

29 datasets, arms: embed-nn, glm-cot, jev, laya, privatemode. Per-dataset figures are averaged over replicates and carry the spread between them; there are deliberately no per-dataset p-values.

## Per dataset

| dataset (options) | lang | majority | embed-nn | glm-cot | jev | laya | privatemode |
|---|---|---|---|---|---|---|---|
| boolq (2) | en | 0.62 | 0.596 ±0.000 | 0.930 ±0.000 | 0.925 ±0.003 | 0.843 ±0.000 | 0.905 ±0.001 |
| rotten_tomatoes (2) | en | 0.50 | 0.721 ±0.000 | 0.945 ±0.000 | 0.929 ±0.001 | 0.874 ±0.000 | 0.916 ±0.004 |
| rte (2) | en | 0.53 | 0.639 ±0.000 | 0.906 ±0.000 | 0.886 ±0.004 | 0.783 ±0.000 | 0.773 ±0.007 |
| sst2 (2) | en | 0.51 | 0.850 ±0.000 | 0.958 ±0.000 | 0.959 ±0.001 | 0.925 ±0.000 | 0.951 ±0.000 |
| toxic_conversations (2) | en | 0.93 | 0.897 ±0.000 | 0.863 ±0.000 | 0.756 ±0.004 | 0.922 ±0.000 | 0.800 ±0.003 |
| tweet_offensive (2) | en | 0.72 | 0.663 ±0.000 | 0.790 ±0.000 | 0.770 ±0.007 | 0.792 ±0.000 | 0.785 ±0.013 |
| mnli (3) | en | 0.35 | 0.572 ±0.000 | 0.864 ±0.000 | 0.855 ±0.003 | 0.868 ±0.000 | 0.841 ±0.005 |
| tweet_sentiment (3) | en | 0.48 | 0.578 ±0.000 | 0.694 ±0.000 | 0.671 ±0.003 | 0.588 ±0.000 | 0.684 ±0.007 |
| xnli_de (3) | de | 0.34 | 0.476 ±0.000 | 0.843 ±0.000 | 0.800 ±0.009 | 0.635 ±0.000 | 0.770 ±0.013 |
| ag_news (4) | en | 0.27 | 0.683 ±0.000 | 0.925 ±0.000 | 0.903 ±0.001 | 0.938 ±0.000 | 0.894 ±0.004 |
| amazon_reviews_de (5) | de | 0.21 | 0.305 ±0.000 | 0.597 ±0.000 | 0.582 ±0.006 | 0.327 ±0.000 | 0.603 ±0.010 |
| sst5 (5) | en | 0.29 | 0.299 ±0.000 | 0.585 ±0.000 | 0.567 ±0.007 | 0.512 ±0.000 | 0.463 ±0.007 |
| emotion (6) | en | 0.34 | 0.621 ±0.000 | 0.612 ±0.000 | 0.591 ±0.003 | 0.597 ±0.000 | 0.592 ±0.002 |
| trec_coarse (6) | en | 0.28 | 0.366 ±0.000 | 0.930 ±0.000 | 0.921 ±0.002 | 0.884 ±0.000 | 0.912 ±0.012 |
| gnad10 (9) | de | 0.16 | 0.262 ±0.000 | 0.711 ±0.000 | 0.618 ±0.002 | 0.433 ±0.000 | 0.653 ±0.003 |
| patent (9) | en | 0.21 | 0.154 ±0.000 | 0.699 ±0.000 | 0.486 ±0.003 | 0.222 ±0.000 | 0.545 ±0.012 |
| yahoo_topics (10) | en | 0.12 | 0.579 ±0.000 | 0.756 ±0.000 | 0.741 ±0.002 | 0.644 ±0.000 | 0.736 ±0.003 |
| scotus (13) | en | 0.27 | 0.238 ±0.000 | 0.713 ±0.000 | 0.734 ±0.001 | 0.351 ±0.000 | 0.693 ±0.004 |
| dbpedia_14 (14) | en | 0.09 | 0.879 ±0.000 | 0.984 ±0.000 | 0.987 ±0.000 | 0.833 ±0.000 | 0.982 ±0.000 |
| rvl_cdip (16) | en | 0.07 | — | 0.696 ±0.000 | — | — | 0.702 ±0.006 |
| massive_scenario_de (18) | de | 0.14 | 0.798 ±0.000 | 0.830 ±0.000 | 0.738 ±0.006 | 0.327 ±0.000 | 0.771 ±0.000 |
| massive_scenario_en (18) | en | 0.14 | 0.812 ±0.000 | 0.815 ±0.000 | 0.752 ±0.002 | 0.567 ±0.000 | 0.764 ±0.003 |
| newsgroups20 (20) | en | 0.06 | 0.645 ±0.000 | 0.757 ±0.000 | 0.718 ±0.007 | 0.416 ±0.000 | 0.736 ±0.001 |
| trec_fine (42) | en | 0.25 | 0.506 ±0.000 | 0.768 ±0.000 | 0.856 ±0.000 | 0.512 ±0.000 | 0.796 ±0.000 |
| massive_intent_de (59) | de | 0.07 | 0.767 ±0.000 | 0.842 ±0.000 | 0.792 ±0.003 | 0.220 ±0.000 | 0.779 ±0.000 |
| massive_intent_en (59) | en | 0.07 | 0.822 ±0.000 | 0.865 ±0.000 | 0.830 ±0.004 | 0.446 ±0.000 | 0.824 ±0.004 |
| banking77 (77) | en | 0.02 | 0.794 ±0.000 | 0.806 ±0.000 | 0.793 ±0.000 | 0.379 ±0.000 | 0.769 ±0.000 |
| ledgar (100) | en | 0.06 | 0.459 ±0.000 | 0.760 ±0.000 | 0.749 ±0.002 | 0.210 ±0.000 | 0.742 ±0.004 |
| clinc150 (151) | en | 0.16 | 0.628 ±0.000 | 0.879 ±0.000 | 0.784 ±0.001 | — | 0.875 ±0.011 |

## What the labels allow

Where most arms return the same answer and the gold label disagrees, the likeliest explanation is an ambiguous label rather than a shared failure. `consensus wrong` is a floor estimate for that; `ceiling` is the share at least one arm got right, which is the most any of them could have scored. Context for the accuracy column, never subtracted from it.

| dataset (options) | arms agreed | consensus wrong | ceiling | best arm | headroom to ceiling |
|---|---|---|---|---|---|
| boolq (2) | 0.80 | 0.053 | 0.969 | 0.930 | +0.039 |
| rotten_tomatoes (2) | 1.00 | 0.073 | 0.966 | 0.945 | +0.021 |
| rte (2) | 1.00 | 0.117 | 0.968 | 0.906 | +0.061 |
| sst2 (2) | 0.93 | 0.033 | 0.979 | 0.959 | +0.020 |
| toxic_conversations (2) | 0.93 | 0.109 | 0.956 | 0.922 | +0.034 |
| tweet_offensive (2) | 0.82 | 0.151 | 0.915 | 0.792 | +0.123 |
| mnli (3) | 0.79 | 0.083 | 0.936 | 0.868 | +0.068 |
| tweet_sentiment (3) | 0.71 | 0.175 | 0.888 | 0.694 | +0.194 |
| xnli_de (3) | 0.72 | 0.116 | 0.911 | 0.843 | +0.068 |
| ag_news (4) | 1.00 | 0.084 | 0.970 | 0.938 | +0.032 |
| amazon_reviews_de (5) | 0.65 | 0.245 | 0.704 | 0.603 | +0.101 |
| sst5 (5) | 0.91 | 0.385 | 0.790 | 0.585 | +0.206 |
| emotion (6) | 0.82 | 0.267 | 0.728 | 0.621 | +0.107 |
| trec_coarse (6) | 0.69 | 0.048 | 0.957 | 0.930 | +0.027 |
| gnad10 (9) | 0.90 | 0.294 | 0.730 | 0.711 | +0.019 |
| patent (9) | 0.88 | 0.422 | 0.624 | 0.699 | -0.075 |
| yahoo_topics (10) | 0.96 | 0.231 | 0.802 | 0.756 | +0.046 |
| scotus (13) | 0.59 | 0.142 | 0.778 | 0.734 | +0.044 |
| dbpedia_14 (14) | 1.00 | 0.014 | 0.992 | 0.987 | +0.005 |
| massive_scenario_de (18) | 0.82 | 0.116 | 0.879 | 0.830 | +0.049 |
| massive_scenario_en (18) | 0.85 | 0.123 | 0.881 | 0.815 | +0.067 |
| newsgroups20 (20) | 0.89 | 0.192 | 0.821 | 0.757 | +0.064 |
| trec_fine (42) | 0.94 | 0.126 | 0.907 | 0.856 | +0.051 |
| massive_intent_de (59) | 0.88 | 0.130 | 0.847 | 0.842 | +0.005 |
| massive_intent_en (59) | 0.88 | 0.085 | 0.897 | 0.865 | +0.032 |
| banking77 (77) | 0.92 | 0.166 | 0.851 | 0.806 | +0.044 |
| ledgar (100) | 0.70 | 0.135 | 0.798 | 0.760 | +0.038 |

## Overall

| arm | datasets | mean normalised accuracy | sets lost to majority baseline | median replicate spread | mean coverage at 95% accuracy |
|---|---|---|---|---|---|
| embed-nn | 28 | 0.355 | 5 | 0.000 | 0.160 |
| glm-cot | 29 | 0.665 | 1 | 0.000 | 0.478 |
| jev | 28 | 0.574 | 1 | 0.003 | 0.494 |
| laya | 27 | 0.422 | 1 | 0.000 | 0.279 |
| privatemode | 29 | 0.585 | 1 | 0.004 | 0.431 |

## Pairwise, across datasets

**embed-nn vs glm-cot** over 28 datasets: 1 win / 2 tie / 25 loss (ties within ±0.01, the measured run-to-run noise). Median difference -0.233; Wilcoxon signed-rank p = 0.0000 over 28 non-zero.

**embed-nn vs jev** over 28 datasets: 4 win / 2 tie / 22 loss (ties within ±0.01, the measured run-to-run noise). Median difference -0.185; Wilcoxon signed-rank p = 0.0001 over 28 non-zero.

**embed-nn vs laya** over 27 datasets: 9 win / 1 tie / 17 loss (ties within ±0.01, the measured run-to-run noise). Median difference -0.065; Wilcoxon signed-rank p = 0.5561 over 27 non-zero.

**embed-nn vs privatemode** over 28 datasets: 5 win / 1 tie / 22 loss (ties within ±0.01, the measured run-to-run noise). Median difference -0.161; Wilcoxon signed-rank p = 0.0001 over 28 non-zero.

**glm-cot vs jev** over 28 datasets: 21 win / 5 tie / 2 loss (ties within ±0.01, the measured run-to-run noise). Median difference +0.020; Wilcoxon signed-rank p = 0.0002 over 28 non-zero.

**glm-cot vs laya** over 27 datasets: 23 win / 2 tie / 2 loss (ties within ±0.01, the measured run-to-run noise). Median difference +0.151; Wilcoxon signed-rank p = 0.0000 over 27 non-zero.

**glm-cot vs privatemode** over 29 datasets: 21 win / 7 tie / 1 loss (ties within ±0.01, the measured run-to-run noise). Median difference +0.023; Wilcoxon signed-rank p = 0.0000 over 29 non-zero.

**jev vs laya** over 27 datasets: 22 win / 1 tie / 4 loss (ties within ±0.01, the measured run-to-run noise). Median difference +0.154; Wilcoxon signed-rank p = 0.0001 over 27 non-zero.

**jev vs privatemode** over 28 datasets: 10 win / 8 tie / 10 loss (ties within ±0.01, the measured run-to-run noise). Median difference +0.007; Wilcoxon signed-rank p = 0.6406 over 28 non-zero.

**laya vs privatemode** over 27 datasets: 5 win / 2 tie / 20 loss (ties within ±0.01, the measured run-to-run noise). Median difference -0.135; Wilcoxon signed-rank p = 0.0004 over 27 non-zero.

## By option count

Paired: a band averages only the datasets on which every product arm answered, so each column in a row covers the same sets. Averaging each arm over whatever it could answer instead would credit an arm with the sets its rivals cannot run -- clinc150 for Jev at the wide end, rvl_cdip for Privatemode in the middle -- and compare different questions under one heading. A control arm is shown only where it ran on all of a band's sets.

| options | sets | embed-nn | glm-cot | jev | laya | privatemode |
|---|---|---|---|---|---|---|
| 2 | 6 | 0.728 | 0.899 | 0.871 | 0.857 | 0.855 |
| 3–6 | 8 | 0.487 | 0.756 | 0.736 | 0.669 | 0.720 |
| 7–20 | 8 | 0.546 | 0.783 | 0.722 | 0.474 | 0.735 |
| 21–80 | 4 | 0.722 | 0.820 | 0.818 | 0.389 | 0.792 |
| 81+ | 1 | 0.459 | 0.760 | 0.749 | 0.210 | 0.742 |

## Ladders — the same examples at two option counts

The only place the option-count effect is measured rather than inferred: identical examples, identical arms, one variable.

| ladder | arm | options | narrow | wide | change |
|---|---|---|---|---|---|
| trec | embed-nn | 6 → 42 | 0.366 | 0.506 | +0.140 |
| trec | glm-cot | 6 → 42 | 0.930 | 0.768 | -0.162 |
| trec | jev | 6 → 42 | 0.921 | 0.856 | -0.065 |
| trec | laya | 6 → 42 | 0.884 | 0.512 | -0.372 |
| trec | privatemode | 6 → 42 | 0.912 | 0.796 | -0.116 |
| massive_en | embed-nn | 18 → 59 | 0.812 | 0.822 | +0.010 |
| massive_en | glm-cot | 18 → 59 | 0.815 | 0.865 | +0.050 |
| massive_en | jev | 18 → 59 | 0.752 | 0.830 | +0.078 |
| massive_en | laya | 18 → 59 | 0.567 | 0.446 | -0.121 |
| massive_en | privatemode | 18 → 59 | 0.764 | 0.824 | +0.059 |
| massive_de | embed-nn | 18 → 59 | 0.798 | 0.767 | -0.031 |
| massive_de | glm-cot | 18 → 59 | 0.830 | 0.842 | +0.012 |
| massive_de | jev | 18 → 59 | 0.738 | 0.792 | +0.054 |
| massive_de | laya | 18 → 59 | 0.327 | 0.220 | -0.107 |
| massive_de | privatemode | 18 → 59 | 0.771 | 0.779 | +0.008 |

## Memorisation: the same task with renamed labels

Every option swapped for a frozen synonym, nothing else changed. An arm that reads the state and reasons about the options should barely move; one that has learned the label string should fall.

A renaming is never perfectly neutral -- some synonyms are simply harder words -- so every arm loses a little. That common loss is the perturbation, not memorisation. `excess` is the drop beyond the median arm's on the same dataset, and it is the column to read. It needs at least three arms: the median of two is their mean, so the excess would be symmetric by construction and would say nothing at all. With fewer it is left blank and only the raw drop is reported.

| dataset (options) | arm | original | renamed | drop | excess |
|---|---|---|---|---|---|
| boolq (2) | jev | 0.925 | 0.895 | -0.030 | -0.000 |
| boolq (2) | laya | 0.843 | 0.837 | -0.006 | +0.024 |
| boolq (2) | privatemode | 0.905 | 0.702 | -0.203 | -0.172 |
| rotten_tomatoes (2) | jev | 0.929 | 0.939 | +0.010 | +0.008 |
| rotten_tomatoes (2) | laya | 0.874 | 0.876 | +0.002 | -0.000 |
| rotten_tomatoes (2) | privatemode | 0.916 | 0.890 | -0.026 | -0.028 |
| rte (2) | jev | 0.886 | 0.903 | +0.016 | -0.000 |
| rte (2) | laya | 0.783 | 0.765 | -0.018 | -0.034 |
| rte (2) | privatemode | 0.773 | 0.801 | +0.029 | +0.013 |
| sst2 (2) | jev | 0.959 | 0.958 | -0.002 | -0.000 |
| sst2 (2) | laya | 0.925 | 0.916 | -0.009 | -0.007 |
| sst2 (2) | privatemode | 0.951 | 0.956 | +0.006 | +0.007 |
| toxic_conversations (2) | jev | 0.756 | 0.756 | -0.000 | +0.010 |
| toxic_conversations (2) | laya | 0.922 | 0.912 | -0.010 | -0.000 |
| toxic_conversations (2) | privatemode | 0.800 | 0.786 | -0.014 | -0.004 |
| tweet_offensive (2) | jev | 0.770 | 0.783 | +0.013 | -0.011 |
| tweet_offensive (2) | laya | 0.792 | 0.828 | +0.036 | +0.012 |
| tweet_offensive (2) | privatemode | 0.785 | 0.809 | +0.024 | -0.000 |
| mnli (3) | jev | 0.855 | 0.867 | +0.012 | +0.030 |
| mnli (3) | laya | 0.868 | 0.843 | -0.025 | -0.007 |
| mnli (3) | privatemode | 0.841 | 0.823 | -0.018 | -0.000 |
| tweet_sentiment (3) | jev | 0.671 | 0.653 | -0.018 | -0.000 |
| tweet_sentiment (3) | laya | 0.588 | 0.589 | +0.001 | +0.019 |
| tweet_sentiment (3) | privatemode | 0.684 | 0.609 | -0.076 | -0.058 |
| xnli_de (3) | jev | 0.800 | 0.812 | +0.012 | +0.018 |
| xnli_de (3) | laya | 0.635 | 0.629 | -0.006 | -0.000 |
| xnli_de (3) | privatemode | 0.770 | 0.740 | -0.030 | -0.024 |
| ag_news (4) | jev | 0.903 | 0.793 | -0.110 | -0.042 |
| ag_news (4) | laya | 0.938 | 0.870 | -0.068 | -0.000 |
| ag_news (4) | privatemode | 0.894 | 0.843 | -0.051 | +0.017 |
| amazon_reviews_de (5) | jev | 0.582 | 0.574 | -0.008 | -0.018 |
| amazon_reviews_de (5) | laya | 0.327 | 0.363 | +0.036 | +0.026 |
| amazon_reviews_de (5) | privatemode | 0.603 | 0.613 | +0.010 | -0.000 |
| sst5 (5) | jev | 0.567 | 0.564 | -0.003 | -0.000 |
| sst5 (5) | laya | 0.512 | 0.498 | -0.014 | -0.011 |
| sst5 (5) | privatemode | 0.463 | 0.474 | +0.011 | +0.014 |
| emotion (6) | jev | 0.591 | 0.527 | -0.064 | -0.000 |
| emotion (6) | laya | 0.597 | 0.460 | -0.137 | -0.073 |
| emotion (6) | privatemode | 0.592 | 0.573 | -0.019 | +0.045 |
| trec_coarse (6) | jev | 0.921 | 0.856 | -0.065 | +0.049 |
| trec_coarse (6) | laya | 0.884 | 0.770 | -0.114 | -0.000 |
| trec_coarse (6) | privatemode | 0.912 | 0.772 | -0.140 | -0.026 |
| gnad10 (9) | jev | 0.618 | 0.644 | +0.026 | -0.000 |
| gnad10 (9) | laya | 0.433 | 0.467 | +0.034 | +0.008 |
| gnad10 (9) | privatemode | 0.653 | 0.581 | -0.073 | -0.099 |
| patent (9) | jev | 0.486 | 0.241 | -0.245 | -0.000 |
| patent (9) | laya | 0.222 | 0.171 | -0.051 | +0.195 |
| patent (9) | privatemode | 0.545 | 0.281 | -0.264 | -0.019 |
| yahoo_topics (10) | jev | 0.741 | 0.668 | -0.073 | -0.000 |
| yahoo_topics (10) | laya | 0.644 | 0.516 | -0.128 | -0.055 |
| yahoo_topics (10) | privatemode | 0.736 | 0.680 | -0.056 | +0.017 |
| scotus (13) | jev | 0.734 | 0.569 | -0.165 | -0.000 |
| scotus (13) | laya | 0.351 | 0.336 | -0.015 | +0.151 |
| scotus (13) | privatemode | 0.693 | 0.446 | -0.247 | -0.081 |
| dbpedia_14 (14) | jev | 0.987 | 0.976 | -0.011 | -0.000 |
| dbpedia_14 (14) | laya | 0.833 | 0.777 | -0.056 | -0.045 |
| dbpedia_14 (14) | privatemode | 0.982 | 0.971 | -0.011 | -0.000 |
| massive_scenario_de (18) | jev | 0.738 | 0.613 | -0.125 | -0.009 |
| massive_scenario_de (18) | laya | 0.327 | 0.225 | -0.102 | +0.014 |
| massive_scenario_de (18) | privatemode | 0.771 | 0.655 | -0.116 | -0.000 |
| massive_scenario_en (18) | jev | 0.752 | 0.565 | -0.187 | -0.008 |
| massive_scenario_en (18) | laya | 0.567 | 0.388 | -0.179 | -0.000 |
| massive_scenario_en (18) | privatemode | 0.764 | 0.622 | -0.142 | +0.036 |
| newsgroups20 (20) | jev | 0.718 | 0.640 | -0.078 | -0.000 |
| newsgroups20 (20) | laya | 0.416 | 0.293 | -0.123 | -0.045 |
| newsgroups20 (20) | privatemode | 0.736 | 0.677 | -0.058 | +0.020 |
| trec_fine (42) | jev | 0.856 | 0.672 | -0.184 | -0.034 |
| trec_fine (42) | laya | 0.512 | 0.464 | -0.048 | +0.102 |
| trec_fine (42) | privatemode | 0.796 | 0.646 | -0.150 | -0.000 |
| massive_intent_de (59) | jev | 0.792 | 0.661 | -0.131 | -0.000 |
| massive_intent_de (59) | laya | 0.220 | 0.080 | -0.140 | -0.009 |
| massive_intent_de (59) | privatemode | 0.779 | 0.649 | -0.130 | +0.001 |
| massive_intent_en (59) | jev | 0.830 | 0.721 | -0.109 | +0.023 |
| massive_intent_en (59) | laya | 0.446 | 0.249 | -0.197 | -0.065 |
| massive_intent_en (59) | privatemode | 0.824 | 0.692 | -0.132 | -0.000 |
| banking77 (77) | jev | 0.793 | 0.605 | -0.188 | -0.000 |
| banking77 (77) | laya | 0.379 | 0.258 | -0.121 | +0.067 |
| banking77 (77) | privatemode | 0.769 | 0.559 | -0.210 | -0.022 |
| ledgar (100) | jev | 0.749 | 0.486 | -0.263 | -0.101 |
| ledgar (100) | laya | 0.210 | 0.072 | -0.138 | +0.024 |
| ledgar (100) | privatemode | 0.742 | 0.580 | -0.162 | -0.000 |
| clinc150 (151) | jev | 0.784 | 0.581 | -0.203 | — |
| clinc150 (151) | privatemode | 0.875 | 0.600 | -0.275 | — |

## Cost per decision

Billed from each vendor's own `usage` block at the rates in `bench/pricing.py`, never from a token count of ours. The arms serialize state and options differently, so the token counts are reported next to the prices.

| dataset (options) | embed-nn | glm-cot | jev | laya | privatemode |
|---|---|---|---|---|---|
| boolq (2) | 141 tok · 0.0184 | 201 tok · 0.1823 | 433 tok · 0.0167 | local | 236 tok · 0.0478 |
| rotten_tomatoes (2) | 26 tok · 0.0034 | 91 tok · 0.1731 | 316 tok · 0.0122 | local | 125 tok · 0.0256 |
| rte (2) | 75 tok · 0.0097 | 142 tok · 0.3660 | 371 tok · 0.0143 | local | 178 tok · 0.0363 |
| sst2 (2) | 24 tok · 0.0031 | 93 tok · 0.1818 | 317 tok · 0.0123 | local | 127 tok · 0.0260 |
| toxic_conversations (2) | 71 tok · 0.0093 | 134 tok · 0.3682 | 361 tok · 0.0139 | local | 169 tok · 0.0345 |
| tweet_offensive (2) | 39 tok · 0.0051 | 103 tok · 0.3611 | 332 tok · 0.0128 | local | 137 tok · 0.0280 |
| mnli (3) | 44 tok · 0.0058 | 118 tok · 0.3341 | 350 tok · 0.0135 | local | 170 tok · 0.0346 |
| tweet_sentiment (3) | 25 tok · 0.0032 | 93 tok · 0.3245 | 321 tok · 0.0124 | local | 140 tok · 0.0286 |
| xnli_de (3) | 57 tok · 0.0074 | 140 tok · 0.5241 | 371 tok · 0.0143 | local | 192 tok · 0.0390 |
| ag_news (4) | 53 tok · 0.0069 | 128 tok · 0.1781 | 360 tok · 0.0139 | local | 189 tok · 0.0384 |
| amazon_reviews_de (5) | 63 tok · 0.0082 | 149 tok · 0.3686 | 370 tok · 0.0143 | local | 220 tok · 0.0447 |
| sst5 (5) | 24 tok · 0.0031 | 104 tok · 0.2862 | 337 tok · 0.0130 | local | 180 tok · 0.0366 |
| emotion (6) | 21 tok · 0.0027 | 98 tok · 0.2391 | 335 tok · 0.0129 | local | 190 tok · 0.0386 |
| trec_coarse (6) | 9 tok · 0.0012 | 93 tok · 0.3379 | 332 tok · 0.0128 | local | 182 tok · 0.0371 |
| gnad10 (9) | 719 tok · 0.0935 | 743 tok · 0.4690 | 1025 tok · 0.0396 | local | 876 tok · 0.1759 |
| patent (9) | 127 tok · 0.0165 | 245 tok · 0.5411 | 506 tok · 0.0196 | local | 381 tok · 0.0769 |
| yahoo_topics (10) | 134 tok · 0.0174 | 238 tok · 0.2548 | 496 tok · 0.0192 | local | 391 tok · 0.0789 |
| scotus (13) | 1915 tok · 0.2490 | 1928 tok · 0.8027 | 2351 tok · 0.0909 | local | 2151 tok · 0.4309 |
| dbpedia_14 (14) | 79 tok · 0.0103 | 187 tok · 0.1598 | 467 tok · 0.0180 | local | 391 tok · 0.0789 |
| rvl_cdip (16) | — | 1116 tok · 0.5131 | — | — | 1348 tok · 0.2702 |
| massive_scenario_de (18) | 12 tok · 0.0015 | 132 tok · 0.4299 | 407 tok · 0.0157 | local | 391 tok · 0.0788 |
| massive_scenario_en (18) | 8 tok · 0.0011 | 124 tok · 0.3735 | 400 tok · 0.0155 | local | 383 tok · 0.0773 |
| newsgroups20 (20) | 225 tok · 0.0293 | 396 tok · 0.4503 | 707 tok · 0.0273 | local | 701 tok · 0.1408 |
| trec_fine (42) | 9 tok · 0.0012 | 237 tok · 0.4559 | 597 tok · 0.0231 | local | 840 tok · 0.1687 |
| massive_intent_de (59) | 12 tok · 0.0015 | 369 tok · 0.3273 | 781 tok · 0.0302 | local | 1197 tok · 0.2400 |
| massive_intent_en (59) | 8 tok · 0.0011 | 359 tok · 0.2854 | 771 tok · 0.0298 | local | 1187 tok · 0.2381 |
| banking77 (77) | 13 tok · 0.0018 | 525 tok · 0.3568 | 1024 tok · 0.0395 | local | 1620 tok · 0.3246 |
| ledgar (100) | 135 tok · 0.0176 | 607 tok · 0.4058 | 1207 tok · 0.0466 | local | 2040 tok · 0.4087 |
| clinc150 (151) | 10 tok · 0.0013 | 697 tok · 0.3406 | 1418 tok · 0.0548 | — | 5656 tok · 1.1325 |

### Cost per 1000 correct decisions

Medians over the 28 datasets every priced arm answered.

| arm | median EUR / 1000 answers | median EUR / 1000 correct |
|---|---|---|
| embed-nn | 0.0054 | 0.0101 |
| glm-cot | 0.3487 | 0.4347 |
| jev | 0.0156 | 0.0216 |
| laya | local | local |
| privatemode | 0.0624 | 0.0797 |

## Latency

| arm | datasets | median p10 (unqueued floor) | median p50 | median p95 | throttled |
|---|---|---|---|---|---|
| embed-nn | no quiet run | — | — | — | 0 |
| glm-cot | no quiet run | — | — | — | 0 |
| jev | 28 | 216 ms | 251 ms | 330 ms | 0 |
| laya | not measured | not measured | — | — | — |
| privatemode | 29 | 147 ms | 152 ms | 198 ms | 0 |

From concurrency-1 runs only, whatever their size; `no quiet run` means there is none yet and nothing is reported rather than a number taken under load. A p50 well above the p10 floor, or any throttled response, means an arm was queueing rather than computing. Laya is not speed-measured: it runs on a laptop while the others run in a datacentre.

### Latency per dataset

| dataset (options) | embed-nn | glm-cot | jev | laya | privatemode |
|---|---|---|---|---|---|
| boolq (2) | — | — | 255 / 219 | — | 148 / 145 |
| rotten_tomatoes (2) | — | — | 272 / 233 | — | 148 / 145 |
| rte (2) | — | — | 252 / 218 | — | 147 / 144 |
| sst2 (2) | — | — | 271 / 228 | — | 151 / 145 |
| toxic_conversations (2) | — | — | 251 / 216 | — | 174 / 147 |
| tweet_offensive (2) | — | — | 245 / 214 | — | 150 / 145 |
| mnli (3) | — | — | 270 / 218 | — | 149 / 145 |
| tweet_sentiment (3) | — | — | 255 / 219 | — | 149 / 146 |
| xnli_de (3) | — | — | 276 / 230 | — | 151 / 146 |
| ag_news (4) | — | — | 253 / 216 | — | 150 / 146 |
| amazon_reviews_de (5) | — | — | 259 / 221 | — | 150 / 146 |
| sst5 (5) | — | — | 237 / 210 | — | 152 / 147 |
| emotion (6) | — | — | 284 / 235 | — | 157 / 146 |
| trec_coarse (6) | — | — | 248 / 208 | — | 148 / 144 |
| gnad10 (9) | — | — | 243 / 209 | — | 189 / 151 |
| patent (9) | — | — | 252 / 212 | — | 151 / 147 |
| yahoo_topics (10) | — | — | 254 / 227 | — | 153 / 148 |
| scotus (13) | — | — | 279 / 230 | — | 387 / 275 |
| dbpedia_14 (14) | — | — | 265 / 223 | — | 189 / 156 |
| rvl_cdip (16) | — | — | — | — | 602 / 443 |
| massive_scenario_de (18) | — | — | 248 / 217 | — | 151 / 148 |
| massive_scenario_en (18) | — | — | 235 / 209 | — | 151 / 148 |
| newsgroups20 (20) | — | — | 243 / 216 | — | 156 / 151 |
| trec_fine (42) | — | — | 242 / 211 | — | 161 / 154 |
| massive_intent_de (59) | — | — | 235 / 204 | — | 275 / 267 |
| massive_intent_en (59) | — | — | 236 / 214 | — | 298 / 286 |
| banking77 (77) | — | — | 247 / 216 | — | 302 / 291 |
| ledgar (100) | — | — | 240 / 209 | — | 320 / 294 |
| clinc150 (151) | — | — | 249 / 215 | — | 719 / 649 |

p50 / p10 in ms.

