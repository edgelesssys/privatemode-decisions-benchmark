# `R-Q` against the baseline on the test halves

GLM-5.3-Flash, the test halves of 29 datasets, two replicates of each arm; accuracy is the mean of the replicates.

**15 wins, 12 ties, 2 losses** (ties within ±0.01); median difference +1.2 points, mean +1.64; Wilcoxon signed-rank p = 0.00211.

| dataset | options | rows | B | R-Q | points | replicate spread | Jev | Laya | glm-cot |
|---|---|---|---|---|---|---|---|---|---|
| ag_news | 4 | 500 | 0.896 | 0.921 | +2.5 | 0.4 | 0.904 | 0.926 | 0.926 |
| amazon_reviews_de | 5 | 500 | 0.603 | 0.602 | -0.1 | 1.0 | 0.566 | 0.340 | 0.598 |
| banking77 | 77 | 500 | 0.802 | 0.818 | +1.6 | 1.6 | 0.802 | 0.370 | 0.808 |
| boolq | 2 | 500 | 0.918 | 0.930 | +1.2 | 1.2 | 0.930 | 0.854 | 0.938 |
| clinc150 | 151 | 500 | 0.875 | 0.908 | +3.3 | 0.2 | 0.796 | — | 0.888 |
| dbpedia_14 | 14 | 500 | 0.982 | 0.976 | -0.6 | 0.4 | 0.986 | 0.840 | 0.984 |
| emotion | 6 | 500 | 0.606 | 0.597 | -0.9 | 0.8 | 0.580 | 0.598 | 0.617 |
| gnad10 | 9 | 500 | 0.666 | 0.681 | +1.5 | 0.4 | 0.614 | 0.422 | 0.718 |
| ledgar | 100 | 500 | 0.756 | 0.764 | +0.8 | 0.4 | 0.780 | 0.236 | 0.784 |
| massive_intent_de | 59 | 500 | 0.799 | 0.843 | +4.4 | 0.2 | 0.786 | 0.214 | 0.852 |
| massive_intent_en | 59 | 500 | 0.829 | 0.848 | +1.9 | 0.6 | 0.832 | 0.440 | 0.856 |
| massive_scenario_de | 18 | 500 | 0.742 | 0.785 | +4.3 | 0.8 | 0.730 | 0.314 | 0.826 |
| massive_scenario_en | 18 | 500 | 0.756 | 0.787 | +3.1 | 1.6 | 0.744 | 0.564 | 0.808 |
| mnli | 3 | 500 | 0.864 | 0.861 | -0.3 | 0.8 | 0.844 | 0.866 | 0.878 |
| newsgroups20 | 20 | 500 | 0.723 | 0.722 | -0.1 | 0.2 | 0.716 | 0.400 | 0.784 |
| patent | 9 | 500 | 0.549 | 0.541 | -0.8 | 1.0 | 0.482 | 0.224 | 0.697 |
| rotten_tomatoes | 2 | 500 | 0.937 | 0.952 | +1.5 | 0.4 | 0.934 | 0.856 | 0.950 |
| rte | 2 | 139 | 0.878 | 0.899 | +2.2 | 1.4 | 0.899 | 0.777 | 0.935 |
| rvl_cdip | 16 | 500 | 0.666 | 0.644 | -2.2 | 1.2 | — | — | 0.688 |
| scotus | 13 | 500 | 0.680 | 0.688 | +0.8 | 0.4 | 0.708 | 0.326 | 0.680 |
| sst2 | 2 | 436 | 0.959 | 0.967 | +0.8 | 0.2 | 0.968 | 0.945 | 0.956 |
| sst5 | 5 | 500 | 0.475 | 0.578 | +10.3 | 0.2 | 0.570 | 0.512 | 0.578 |
| toxic_conversations | 2 | 500 | 0.787 | 0.850 | +6.3 | 1.0 | 0.756 | 0.920 | 0.860 |
| trec_coarse | 6 | 250 | 0.902 | 0.936 | +3.4 | 0.8 | 0.948 | 0.876 | 0.940 |
| trec_fine | 42 | 250 | 0.810 | 0.846 | +3.6 | 1.2 | 0.828 | 0.492 | 0.756 |
| tweet_offensive | 2 | 430 | 0.805 | 0.802 | -0.2 | 0.5 | 0.793 | 0.784 | 0.800 |
| tweet_sentiment | 3 | 500 | 0.681 | 0.686 | +0.5 | 1.0 | 0.652 | 0.584 | 0.710 |
| xnli_de | 3 | 500 | 0.808 | 0.795 | -1.3 | 0.8 | 0.790 | 0.640 | 0.830 |
| yahoo_topics | 10 | 500 | 0.756 | 0.756 | +0.0 | 0.8 | 0.754 | 0.672 | 0.764 |

By option count (Jev where it answered every dataset in the band):

| options | datasets | B | R-Q | Jev |
|---|---|---|---|---|
| 2 | 6 | 0.881 | 0.900 | 0.880 |
| 3–6 | 8 | 0.729 | 0.747 | 0.732 |
| 7–20 | 9 | 0.724 | 0.731 | — |
| 21–80 | 4 | 0.810 | 0.839 | 0.812 |
| 81+ | 2 | 0.816 | 0.836 | 0.788 |

**Headroom closed** on the 16 datasets where glm-cot is more than 2 points ahead of the baseline: median 44%.


**Against Jev** on the 28 datasets both answer, same examples: the baseline 13–8–7 (wins–ties–losses), `R-Q` 16–8–3; mean accuracy 0.780 and 0.798 against Jev's 0.775.


**Against Jev and Laya** on the same examples. Normalised accuracy is 0 for always answering the majority class of the rows scored and 1 for all right, averaged over the datasets an arm answers; the mean accuracy and the wins–ties–losses are over the datasets Jev answers, and so is the median price (prompt tokens at list price):

| arm | datasets | normalised accuracy | mean accuracy, Jev's datasets | against Jev | Wilcoxon p | EUR / 1000, median |
|---|---|---|---|---|---|---|
| B | 29 | 0.585 | 0.780 | 13–8–7 | 0.29 | 0.0641 |
| R-Q | 29 | 0.638 | 0.798 | 16–8–3 | 0.00084 | 0.0948 |
| Jev | 28 | 0.561 | 0.775 |  |  |  |
| Laya | 27 | 0.414 | — |  |  |  |

**Cost and latency.** Prompt tokens 672 → 1113 on average, EUR 0.135 → 0.223 per 1,000 decisions (glm-cot: about 0.35). Median latency 237 → 491 ms at 4 requests in flight, with both replicates running at once: that is load, not the prompt. The concurrency-1 measurement below is the one to use.


**Latency at concurrency 1** (one request at a time, 80 test rows per dataset; p50 / p95 in ms, prompt tokens in brackets; B2 is the baseline again, for the noise):

| dataset | options | B | B2 | R-Q |
|---|---|---|---|---|
| sst2 | 2 | 171 / 180 (138) | 152 / 181 (138) | 178 / 226 (192) |
| ag_news | 4 | 163 / 364 (198) | 148 / 158 (198) | 149 / 159 (287) |
| trec_coarse | 6 | 147 / 155 (193) | 143 / 162 (193) | 145 / 164 (318) |
| massive_scenario_en | 18 | 148 / 164 (395) | 152 / 163 (395) | 152 / 193 (721) |
| banking77 | 77 | 273 / 337 (1630) | 270 / 333 (1630) | 386 / 441 (3188) |
| clinc150 | 151 | 569 / 627 (2839) | 337 / 497 (2839) | 900 / 1103 (5608) |

## Does the default temperature still fit?

Each dataset's test rows split in two: temperatures fitted on one half, scored on the other, excess ECE (ECE minus the sampling floor). *Shipped* is the library's formula (a = 0.962, b = -0.076), fitted on the baseline prompt; *refitted* is the formula fitted on this arm, leaving the dataset out.

| arm | task T, median | formula a, b | excess ECE raw | with the shipped formula | with a refitted formula |
|---|---|---|---|---|---|
| B | 2.03 | 0.895, -0.053 | 0.124 | 0.023 | 0.027 |
| R-Q | 2.08 | 0.813, -0.037 | 0.116 | 0.016 | 0.026 |

## Label-renaming control

The same test rows with every option renamed to a synonym (the suite's contamination control): accuracy with renamed options minus with the originals. A variant that leans more on memorised label strings loses more.

| arm | datasets | points, renamed − original |
|---|---|---|
| B | 29 | -7.70 |
| R-Q | 29 | -8.10 |

## MMLU-Pro

The 1000 test questions openjev-sglang sampled (seed 42, revision b189ec76), asked the same way: the question as the state, the options as letters with their texts as descriptions, zero-shot. Jev's and openjev's numbers are theirs, not paired with ours.

| system | accuracy | source |
|---|---|---|
| B | 63.1% | this run |
| R-Q | 61.9% | this run |
| H-1024-r1 | 75.5% | this run, 1000 questions |
| Jev (jev-1.13-20260917) | 82.9% | published, same questions |
| openjev-sglang (Qwen3.6-35B-A3B) | 58.8% | published, same questions |

## JevBench, public items

The 231 public items of [JevBench](https://github.com/fstandhartinger/jevbench) (MIT), used for comparison only: nothing was chosen, fitted or tuned on them. Yes/no questions become the options no/yes described by the false/true criteria, score questions one option per level, as JevBench's own open-model adapter maps them. *Hard* is the 111 public hard items, ECE JevBench's own (10 equal-width bins); the published hard tier also has 109 held-out items. Latency at several requests in flight.

| system | reasoning | output tokens | public items | public hard | ECE, public hard | latency p50 |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash, one pass | no | 1 | 0.885 | 0.761 | 0.087 | 0.23 s |
| GLM-5.3-Flash, one pass, R-Q | no | 1 | 0.894 | 0.779 | 0.099 | 0.32 s |
| GLM-5.3-Flash, 1024 thinking tokens, then the read | yes | 447 | 0.931 | 0.856 | 0.085 (raw) | 14.85 s |
| glm-5.3, structured output + thinking (JevBench's runner) | yes | 501 | 0.987 | 0.982 | 0.044 | 5.58 s |
| glm-5.3-flash, structured output + thinking (JevBench's runner) | yes | 518 | 0.983 | 0.964 | 0.057 | 2.82 s |
| *Jev 1.13.0, published* |  |  | 0.866 | *0.741 (220)* |  |  |
| *GPT-6 Luna, medium effort, published* |  |  | 0.996 | *0.986 (220)* |  |  |
| *DeepSeek V4.1 Flash, thinking, published* |  |  | 0.978 | *0.950 (220)* |  |  |
| *JevK5 v0.2, published* |  |  | 0.853 | *0.700 (220)* |  |  |
| *openjev-sglang (Qwen3.6-35B-A3B), published* |  |  | 0.853 | *0.714 (220)* |  |  |
