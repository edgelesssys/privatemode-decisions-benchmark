# Privatemode Decisions vs. TypeSafe Jev vs. Laya

Speed, accuracy, calibration and cost of three System One implementations
on labelled public data. Every arm gets the identical state, option names
in the same order and instruction; only what is behind the call differs.

* **Privatemode**: [`decisions`](https://github.com/edgelesssys/privatemode-decisions)
  against GLM-5.3-Flash through a `privatemode-proxy`, one masked forward
  pass of a general-purpose LLM in an attested enclave.
* **Jev**: TypeSafe's hosted decision model (`POST /v1/systemone`, `jev-latest`).
* **Laya**: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya),
  a 421M ModernBERT-large encoder with a decision head, run locally.

## JevBench

JevBench's 231 public items ([MIT](https://github.com/fstandhartinger/jevbench),
commit `1bcc55e`) against Jev's published scores, for comparison only:

| system | output tokens | public items (231) | public hard items (111) |
|---|---:|---:|---:|
| Privatemode (GLM-5.3-Flash, one pass) | 1 | 0.894 | 0.779 |
| Jev 1.13.0, published | 45 | 0.866 | not published (0.741 on all 220 hard items, 109 of them sealed) |
| GLM-5.3-Flash, structured output + thinking (JevBench's runner) | 518 | 0.983 | 0.964 |

Among the one-pass systems Privatemode is ahead on the public items;
thinking first, at seconds per decision, is ahead of both.

On MMLU-Pro one pass scores 61.9% against Jev's published 82.9% on the
same 1,000 questions; the gap is in questions that need calculation. ECE
and more systems: [`results/prefill/`](results/prefill/README.md#jevbench).

## Results

### The library as it ships

Test halves of the 29 datasets (up to 500 examples each); Jev and Laya on
the same examples from the published suite. Each row says what it was
measured with:

| | Privatemode | Jev | Laya | measured with |
|---|---|---|---|---|
| datasets it can answer | 29 | 28 | 27 | |
| normalised accuracy | **0.638** | 0.560 | 0.414 | current prompt, [confirmation](results/prefill/confirmation.md) |
| mean accuracy, the 28 datasets Jev answers | **0.798** | 0.775 | | current prompt, confirmation |
| against Jev (wins–ties–losses) | 16–8–3, Wilcoxon p = 0.001 | | | current prompt, confirmation |
| latency p50, concurrency 1, from Germany | about 150 ms | 251 ms | runs locally | current prompt on six datasets; Jev from the suite |
| EUR per 1,000 decisions, median | 0.095 | 0.016 | runs locally | current prompt's tokens at list price |
| excess ECE without labels | **0.032** | 0.080 (0.040 with a default T) | | state-first prompt, [calibration](results/calibration/README.md) |
| accuracy with 100 labels (`calibrate()`) | **80.2%** | 79.6% | | state-first prompt, calibration |

Normalised accuracy is 0 for always answering the majority class and 1 for
all right, over the datasets each system answers. The prompt layout was
chosen on the other halves. Latency is flat up to 18 options; long option
lists are sent twice (386 ms at 77, 900 ms at 151). On the current prompt
the default temperature fits at least as well as on the state-first one
(0.016 against 0.023).

### The published suite (state-first prompt)

29 datasets, up to 1,000 examples, two replicates, seed 0, with the
library's earlier prompt, which today's library no longer builds (reproducing
the runs needs a library version from before the question-first change). Recomputed by
`bench.aggregate` into [`results/suite.md`](results/suite.md), with the
replicate spread on every figure; raw runs in the release
[`runs-2026-09-24`](https://github.com/edgelesssys/privatemode-decisions-benchmark/releases/tag/runs-2026-09-24) ([how to rebuild](results/README.md)).

**Accuracy.** On the 28 datasets both answer, Jev and Privatemode were
indistinguishable (10–8–10, Wilcoxon p = 0.64); normalised means 0.574
(Jev), 0.585 (Privatemode) and 0.422 (Laya, behind both at p < 0.001). By
option count, on the datasets all three answer:

| options | datasets | jev | privatemode | laya |
|---|---|---|---|---|
| 2 | 6 | 0.871 | 0.855 | 0.857 |
| 3–6 | 8 | 0.736 | 0.720 | 0.669 |
| 7–20 | 8 | 0.722 | 0.735 | 0.474 |
| 21–80 | 4 | 0.818 | 0.792 | 0.389 |
| 81+ | 1 | 0.749 | 0.742 | 0.210 |

**Latency and cost** (concurrency 1, medians over the 28 datasets both
answer; cost from billed usage):

| arm | p10 | p50 | p95 | EUR / 1000 | EUR / 1000 correct |
|---|---|---|---|---|---|
| jev | 216 ms | 251 ms | 330 ms | 0.0156 | 0.0216 |
| privatemode | 147 ms | 152 ms | 198 ms | 0.0624 | 0.0797 |

Location matters: from a US runner Jev's p50 was 164 ms and Privatemode's
299 ms, from Germany 264 and 180 ms (`bench/latency_probe.py`,
`results/latency-probe/`). Prices are list prices (`bench/pricing.py`:
Privatemode read 2026-09-24, Jev 2026-09-21 at EUR 0.92 per USD).

**Capability limits.** Only Privatemode reads scanned documents
(rvl_cdip: 0.702). On clinc150's 151 options it scores 0.875 against Jev's
0.784, where Laya can't fit the options; it takes two requests, since
Privatemode reports at most 128 log probabilities per response (719 ms
against Jev's 249 ms). The model's single-token numbers cap options at 191.

**Controls.** `glm-cot` (the same model reasoning first) and `embed-nn`
(Qwen3-Embedding-4B nearest option, the floor), paired on the same sets:

| options | embed-nn | privatemode | glm-cot |
|---|---|---|---|
| 2 | 0.728 | 0.855 | 0.899 |
| 3–6 | 0.487 | 0.720 | 0.756 |
| 7–20 | 0.546 | 0.735 | 0.783 |
| 21–80 | 0.722 | 0.792 | 0.820 |
| 81+ | 0.459 | 0.742 | 0.760 |

Reasoning leads in every band at EUR 0.35 per 1,000 and seconds per
decision. The label-renaming control's drops are in `results/suite.md`
(a synonym can change the question, e.g. boolq's true/false).

### Further analyses

- **[Calibration](results/calibration/README.md):** raw GLM-5.3-Flash is
  14.6 points overconfident; the library's default temperature leaves
  0.032 excess ECE without labels, `calibrate()` adds 2 points of accuracy
  from 100 labels, and all pre-registered criteria held on five untouched
  tasks. The suite's ECE and Brier columns describe the raw model.
- **[A longer prompt](results/prefill/README.md):** asking the question
  before the state too gains 1.6 points (15 better, 12 tied, 2 worse), now
  the library's default; filler gains at most a point; with five
  questions per state the cacheable all-questions-first layout keeps +0.9.

Not measured here: confidential computing and model choice.

## The four runs

The same 29 datasets, four times, because each would contaminate the others:

| run | command | why separate |
|---|---|---|
| A, accuracy | `bench.suite -n 1000 --replicates 2 --arms all --concurrency 16` | hosted arms flip up to 3.5% of answers between identical runs, so two replicates; its latency is discarded |
| B, latency | `bench.suite -n 100 --replicates 1 --arms hosted --concurrency 1` | one request at a time measures the model, not the queue (ag_news p50: Privatemode 189 ms, Jev 235 ms alone; both 249 ms at 12 in flight) |
| C, controls | `bench.suite -n 1000 --replicates 1 --arms controls --concurrency 16` | `glm-cot` takes 7.2 s per decision |
| D, memorisation | `bench.suite -n 1000 --replicates 1 --arms all --perturb rename --concurrency 16` | the same examples as A with every option renamed to a synonym; must match A's `n`, since smaller samples aren't prefixes |

## Run it

```sh
docker run -d -p 127.0.0.1:8080:8080 ghcr.io/edgelesssys/privatemode/privatemode-proxy:latest \
  --apiKey <privatemode-api-key>

python3.14 -m venv .venv
.venv/bin/pip install "privatemode-decisions[images] @ git+https://github.com/edgelesssys/privatemode-decisions@b15d84b3228cc966947239c90c5fcdbba147a70a"
.venv/bin/pip install -e '.[dev,laya]'               # laya pulls torch + transformers
cp .env.example .env                                 # proxy URL, Jev key, HF token
```

The first install is the library under test (`pip install -e
../privatemode-decisions` to work on both). Then, in order:

```sh
# 1. Freeze each dataset's option set and validate it. Once, checked in.
.venv/bin/python -m bench.curate                     # all 29; --force to re-do one

# 2. Run. The suite forecasts its cost and refuses to start over --budget-eur.
.venv/bin/python -m bench.suite --dry-run -n 1000 --replicates 2 --arms all
.venv/bin/python -m bench.suite -n 1000 --replicates 2 --arms all --concurrency 16
#    (--optimize cost for the library's cacheable layout; the same prompt with one question)

# 3. Aggregate. Recomputed from the JSONL, so a new question costs nothing.
.venv/bin/python -m bench.aggregate results --write results/suite.md
.venv/bin/python -m bench.report results/banking77/*.jsonl   # one run in detail

.venv/bin/python -m pytest -q tests                  # the metrics and the aggregation
```

One dataset: `python -m bench.run --dataset banking77 -n 300`. Runs resume:
everything that makes two runs incomparable (the frozen spec's hash,
sample, seed, arms and models, the library's prefill, temperature and
version) is hashed into the file name. Filters: `--only`, `--skip`,
`--family`, `--language`, `--max-options`. Concurrency defaults to 1 for
honest latency; the proxy sustains 16 for accuracy runs. Laya downloads
about 1.7 GB of weights on first use.

## Datasets

29 sets filling a grid of option count, task and language
([`METHODOLOGY.md`](METHODOLOGY.md)), fetched over the Hugging Face
datasets-server (set `HF_TOKEN`):

| option count | sets |
|---|---|
| 2 | boolq · sst2 · rotten_tomatoes · rte · tweet_offensive · toxic_conversations |
| 3–6 | tweet_sentiment · mnli · xnli_de · ag_news · sst5 · amazon_reviews_de · emotion · trec_coarse |
| 9–20 | gnad10 · patent · yahoo_topics · scotus · dbpedia_14 · massive_scenario_en · massive_scenario_de · newsgroups20 · rvl_cdip |
| 42–77 | trec_fine · massive_intent_en · massive_intent_de · banking77 |
| 100+ | ledgar · clinc150 |

24 English and 5 German, across intent, sentiment, topic, moderation,
NLI, QA, legal and scanned documents (`rvl_cdip`, Privatemode only). TREC
and MASSIVE are ladders: the same examples at two granularities, so
option count changes alone. `bench.curate` freezes each option set with
its statistics into `datasets/<name>.json` once; runs read that file, so
what a run asks is pinned and reviewable. Adding a set is a `Spec` in
`bench/specs.py` and one curation run.

## What is measured

Accuracy and macro F1 (over the gold classes), ECE and Brier from each
arm's probabilities, AUROC and selective accuracy from each vendor's own
`confidence`, latency of the successful attempt, cost from each vendor's
`usage`, and paired statistics (exact McNemar, paired bootstrap). The
runner uses one connection-pooled client and retry policy for both hosted
arms, interleaves examples and rotates arm order, discards warmup, and
scores the same examples for every arm: infrastructure errors are retried
or dropped for all, an arm's own failure counts as wrong.

## Contributing

Written by one of the parties it compares, so everything is public:
methodology, frozen specs, harness and aggregation. Jev and Laya ran with
their defaults, and the Privatemode prompt is the library's. Pull requests
for better settings, fixes, datasets or systems are welcome
([CONTRIBUTING.md](CONTRIBUTING.md)).

## Layout

```
bench/specs.py      the registry: what is in the suite and why
bench/hub.py        the datasets-server client, cache and rate-limit backoff
bench/curate.py     scan a dataset once, freeze its option set, validate it
bench/datasets.py   a frozen spec -> tasks, with truncation as a parameter
bench/adapters.py   one arm per product, behind ask(task) -> Answer
bench/pricing.py    the rates, the EUR/USD judgement call, the budget estimator
bench/metrics.py    accuracy, calibration, selective accuracy, paired tests
bench/run.py        one dataset through every arm -> resumable JSONL
bench/suite.py      the registry under filters, with a budget guard
bench/report.py     one run -> markdown
bench/aggregate.py  a suite -> one result
bench/latency_probe.py  the latency runs from another place, for comparison
bench/calibrat*.py  the calibration analysis (results/calibration/)
bench/prefill*.py   prompt variants in one read, and their confirmation (results/prefill/)
.github/workflows/  CI, and the latency probe from a US runner
datasets/           the frozen option sets and validation statistics
results/            the aggregated results; the raw runs are in the release
METHODOLOGY.md      why these datasets, and how each design choice was made
CONTRIBUTING.md     how to propose a change, a dataset, or a system
```
