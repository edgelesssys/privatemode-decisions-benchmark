# Privatemode Decisions vs. TypeSafe Jev vs. Laya

Speed, accuracy and cost of three System One implementations on labelled
public data.

All three expose the same abstraction — a piece of state, a set of named
options, one typed answer with a probability per option — so the comparison
can be structural instead of a prompt-engineering contest. Each arm receives
the identical state string, the identical option names in the identical
order, and the identical instruction line; the only thing that differs is
what is behind the call.

* **Privatemode** — [`decisions`](https://github.com/edgelesssys/privatemode-decisions)
  against GLM-5.3-Flash through a `privatemode-proxy`. A general-purpose LLM
  in an attested enclave, answering in one masked forward pass read out of a
  single logit row.
* **Jev** — `POST https://api.typesafe.ai/v1/systemone`, `jev-latest`. A
  hosted, purpose-built decision model that returns no token stream.
* **Laya** — [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya),
  421M parameters, Apache 2.0, running on the laptop. A ModernBERT-large
  encoder with a decision head, so "one forward pass" is literal.

## Results

29 labelled datasets, 1000 examples each where the split allows, two
replicates, seed 0. Everything below is recomputed from the runs by
`python -m bench.aggregate`; the full tables, with the spread between
replicates on every figure, are in [`results/suite.md`](results/suite.md).
The raw runs are in the release
[`runs-2026-09-24`](https://github.com/edgelesssys/privatemode-decisions-benchmark/releases/tag/runs-2026-09-24);
[`results/README.md`](results/README.md) shows how to rebuild the report
from them.
The first three-dataset pilot is kept in [`results/pilot/`](results/pilot/).

### Accuracy

On the 28 datasets both can answer, Jev and Privatemode are
indistinguishable: 10 wins, 8 ties and 10 losses, median difference +0.007,
Wilcoxon signed-rank p = 0.64. Normalised against each dataset's
majority-class baseline, the means are 0.574 (Jev, 28 datasets), 0.585
(Privatemode, 29) and 0.422 (Laya, 27); Laya is behind both at p < 0.001.

Each band averages only the datasets on which all three products
answered, so every column covers the same questions:

| options | datasets | jev | privatemode | laya |
|---|---|---|---|---|
| 2 | 6 | 0.871 | 0.855 | 0.857 |
| 3–6 | 8 | 0.736 | 0.720 | 0.669 |
| 7–20 | 8 | 0.722 | 0.735 | 0.474 |
| 21–80 | 4 | 0.818 | 0.792 | 0.389 |
| 81+ | 1 | 0.749 | 0.742 | 0.210 |

### Latency and cost

Latency from concurrency-1 runs only, cost from the billed usage of run A;
medians over the 28 datasets both answer:

| arm | p10 | p50 | p95 | throttled | EUR / 1000 | EUR / 1000 correct |
|---|---|---|---|---|---|---|
| jev | 216 ms | 251 ms | 330 ms | 0 | 0.0156 | 0.0216 |
| privatemode | 147 ms | 152 ms | 198 ms | 0 | 0.0624 | 0.0797 |

These are the model calls as a user sees them, network included, from one
machine in Germany. To see how much the place matters,
`bench/latency_probe.py` ran four datasets (sst2, ag_news, trec_coarse,
banking77; 150 decisions per system, one at a time) from Germany and from
a GitHub Actions runner in the US, in the same minutes. Medians over the
four:

| measured from | privatemode p50 | jev p50 |
|---|---|---|
| Germany (home broadband) | 180 ms | 264 ms |
| US (GitHub Actions runner on Azure) | 299 ms | 164 ms |

Privatemode serves from Europe and Jev apparently from the US, so each is
faster close to home. Both places were measured under the same conditions;
the values are higher than the table above, which was taken on another day
and over all datasets. The summaries are in `results/latency-probe/`, and
the workflow `.github/workflows/latency-probe.yml` repeats the US side.

Laya runs on a laptop and has neither a comparable latency nor a price per
decision. The rates in `bench/pricing.py` are list prices: Privatemode's
from [privatemode.ai/pricing](https://privatemode.ai/pricing), read
2026-09-24 (GLM-5.3-Flash: EUR 0.20 / 0.65 / 0.05 per million input /
output / cached tokens), and Jev's from
[docs.typesafe.ai/models](https://docs.typesafe.ai/models), read 2026-09-21,
converted at EUR 0.92 per USD.

### Capability limits

| | privatemode | jev | laya |
|---|---|---|---|
| scanned documents (rvl_cdip) | 0.702 | cannot | cannot |
| clinc150, 151 options | 0.875, in two requests | 0.784 | cannot (192-token option budget) |

Privatemode reports at most 128 entries in `logprob_token_ids`, while
`allowed_token_ids` takes every option. So a 151-way question goes out as
the same request twice, with the whole mask in both, reading the
probabilities of 128 options from the first response and of the other 23
from the second; the library merges the two reads before renormalizing.
Both are the same forward pass at temperature 0, so the result is the
distribution one request would return. It costs a second request: 719 ms at
the median on clinc150 against Jev's 249 ms, and twice the input tokens.
The two go out in parallel; one after the other took 846 ms in a separate
check. The limit is then the model's 191 single-token indexes.

### Controls and perturbations

Two controls ran on all 29 datasets, zero-shot like the products:
`glm-cot`, the same GLM-5.3-Flash asked normally and allowed to reason
before it answers, and `embed-nn`, Qwen3-Embedding-4B picking the option
closest to the state with no decision model at all, the floor. Paired with
the products on the same sets:

| options | embed-nn | privatemode | glm-cot |
|---|---|---|---|
| 2 | 0.728 | 0.855 | 0.899 |
| 3–6 | 0.487 | 0.720 | 0.756 |
| 7–20 | 0.546 | 0.735 | 0.783 |
| 21–80 | 0.722 | 0.792 | 0.820 |
| 81+ | 0.459 | 0.742 | 0.760 |

Reasoning is ahead in every band and costs EUR 0.35 per 1000 decisions
against Privatemode's 0.062, with hundreds of output tokens per decision
instead of one. 62 of its 35,686 decisions ran out of the 4,000-token
reasoning budget without answering, 53 of them on newsgroups20; they count
as wrong answers rather than being dropped, since dropping them would take
the hard examples out of its score. Infrastructure errors (HTTP 500/502)
were retried until every example had an answer.

The label-renaming control is complete for all three products. Its
per-dataset drops are in `results/suite.md`, with one caution for reading
them: a synonym can change what a question asks (boolq's `true`/`false`
became `correct`/`wrong`), in which case the drop does not isolate
memorisation.

### What is outside these columns

Confidential computing. Privatemode runs the model inside an attested
enclave whose memory stays encrypted during processing, and model choice is
a parameter rather than fixed. Neither is quantified here. This benchmark
reports accuracy, calibration, latency and billed cost, and does not weigh
them against each other or against anything it did not measure.

## The four runs, and why they are four

The same 29 datasets are run four times. They are separate because each
would contaminate the others.

### A — accuracy

```sh
python -m bench.suite -n 1000 --replicates 2 --arms all --concurrency 16
```

1000 examples, two replicates, the three products. This is where the
accuracy numbers come from. Two replicates because both hosted arms flip up
to 3.5% of their answers between identical runs at temperature 0: without a
repeat there is no way to tell a one-point difference from the servers'
mood, and every per-dataset figure carries the spread between them.

Latency from this run is discarded — sixteen requests are in flight at
once.

### B — latency

```sh
python -m bench.suite -n 100 --replicates 1 --arms hosted --concurrency 1
```

One request at a time, nothing else in the pipe. That is the only condition
under which a timing measures the model rather than the queue, and 100
examples are enough for p10, p50 and p95. Accuracy from this run does not
feed the headline: the sample is a tenth of A's.

The split is not precautionary, it is measured. On ag_news at concurrency 1
Privatemode's p50 is 189 ms against Jev's 235 ms; at concurrency 12 both sit
at 249 ms.

### C — control arms

```sh
python -m bench.suite -n 1000 --replicates 1 --arms controls --concurrency 16
```

Two arms that are not products. `glm-cot` is the same GLM-5.3-Flash
Privatemode's arm runs, asked normally and allowed to reason — it separates
*the technique*, one masked forward pass with no room to think, from *the
model*. `embed-nn` embeds the state and every option with
Qwen3-Embedding-4B and picks the closest option, with no decision model at
all: the floor, and the answer to how hard the task is.

Separate from A because `glm-cot` averages 7.2 s per decision against the
products' fractions of a second. Folding it into A would have turned two
hours into six.

### D — memorisation

```sh
python -m bench.suite -n 1000 --replicates 1 --arms all --perturb rename \
  --concurrency 16
```

The same sample as A — same seed, same `n`, so the same examples — with
every option replaced by a frozen synonym and nothing else changed. An arm
that reads the state and reasons about the options should barely move; one
that has learned the label string should fall. The difference from A is the
memorisation delta.

The sample has to match exactly. `sample(range(total), 300)` is not a
prefix of `sample(range(total), 1000)`, so running this at a different size
would confound the perturbation with a change of examples — which is why
the control arms were re-run at 1000 rather than compared across sizes.

## Run it

```sh
docker run -d -p 127.0.0.1:8080:8080 ghcr.io/edgelesssys/privatemode/privatemode-proxy:latest \
  --apiKey <privatemode-api-key>

python3.14 -m venv .venv
.venv/bin/pip install "privatemode-decisions[images] @ git+https://github.com/edgelesssys/privatemode-decisions"
.venv/bin/pip install -e '.[dev,laya]'               # laya pulls torch + transformers
cp .env.example .env                                 # proxy URL, Jev key, HF token
```

The first install is the library under test; to work on it alongside, use
`pip install -e ../privatemode-decisions` with a local checkout instead.

Three commands, in the order they have to happen.

```sh
# 1. Freeze each dataset's option set and validate it. Once, checked in.
.venv/bin/python -m bench.curate                     # all 29; --force to re-do one

# 2. Run. The suite forecasts its cost and refuses to start over --budget-eur.
.venv/bin/python -m bench.suite --dry-run -n 1000 --replicates 2 --arms all
.venv/bin/python -m bench.suite -n 1000 --replicates 2 --arms all --concurrency 16

# 3. Aggregate. Recomputed from the JSONL, so a new question costs nothing.
.venv/bin/python -m bench.aggregate results --write results/suite.md
.venv/bin/python -m bench.report results/banking77/*.jsonl   # one run in detail

.venv/bin/python -m pytest -q tests                  # the metrics and the aggregation
```

A single dataset, for iterating: `python -m bench.run --dataset banking77 -n 300`.

**Runs resume.** A run's identity — dataset, the content hash of its frozen
spec, sample, seed, replicate, truncation budget, image resolution, each
arm's model, the variant knobs — is hashed into its filename, and rows are
appended as they land. Re-running the same command answers only what is
missing; a re-curation that changes an option set invalidates the runs taken
against it, rather than quietly appending to them.

Selection: `--only`, `--skip`, `--family`, `--language`, `--max-options`.
Knobs: `--seed`, `--concurrency`, `--permutations`, `--laya-shortlist`,
`--max-chars`, `--image-max-side`, `--eur-per-usd`. Concurrency is 1 by
default because latency is only honest when nothing else is in flight; the
Privatemode proxy sustains 16 without a single 429, so accuracy runs should
use it and latency runs should not.

Laya downloads ~1.7 GB of weights on first use and needs no key.

## Datasets

29 sets, chosen to fill a grid rather than to make a pile — the axes and the
reasoning are in [`METHODOLOGY.md`](METHODOLOGY.md). Fetched over the Hugging
Face datasets-server: no `datasets` dependency, pages cached under `.cache/`,
and an `HF_TOKEN` is required in practice because the anonymous rate limit
does not survive curating two dozen sets.

| option count | sets |
|---|---|
| 2 | boolq · sst2 · rotten_tomatoes · rte · tweet_offensive · toxic_conversations |
| 3–6 | tweet_sentiment · mnli · xnli_de · ag_news · sst5 · amazon_reviews_de · emotion · trec_coarse |
| 9–20 | gnad10 · patent · yahoo_topics · scotus · dbpedia_14 · massive_scenario_en · massive_scenario_de · newsgroups20 · rvl_cdip |
| 42–77 | trec_fine · massive_intent_en · massive_intent_de · banking77 |
| 100+ | ledgar · clinc150 |

24 English and 5 German; intent, sentiment, topic, moderation, NLI, QA,
legal and document. `rvl_cdip` is scanned business documents rather than
text — the one axis where Jev and Laya cannot compete at all, and the
runner leaves them out of it and says so rather than producing a column of
errors.

**Three are ladders**: the same examples labelled at two granularities.
TREC carries a 6-way and a 42-way label for every question; MASSIVE carries
an 18-way scenario and a 59-way intent for every utterance, in English and
German. Every cross-dataset option-count comparison confounds option count
with task, domain and text at once; a ladder changes one variable.

**Option sets are frozen, not derived at runtime.** `bench/curate.py` scans
each dataset once and writes `datasets/<name>.json` — the option vocabulary,
the chance level, the majority share, the class coverage, the state-length
distribution, duplicate states — and that file is checked in. A run reads
it. The point is that what a run asks is pinned and reviewable rather than
a side effect of which rows happened to be sampled.

Curation earned that design immediately. `amazon_reviews_de` found four of
five stars because the split is ordered by label; `scotus` and
`amazon_reviews_de` ship their classes as bare integers, so a frozen rename
map gives them names a model can answer from; `toxic_conversations` is 92%
one class and `tweet_offensive` 72%, both flagged because accuracy there has
to be read against the majority share rather than `1/options`; `trec_fine`
and `massive_intent` are 42- and 59-way in these splits rather than 50 and
60, because the missing classes never occur.

Adding a set is a `Spec` in `bench/specs.py` and one curation run.

## What is measured, and how

Accuracy metrics score the argmax; calibration metrics score the
probabilities, and those are the reason to run this at all — a distribution
you can threshold is worth more than a label you cannot.

* **accuracy**, **macro F1** — macro F1 is averaged over the classes present
  in the gold labels, so an arm that invents a class loses precision rather
  than collecting a free zero.
* **ECE**, **Brier** — from `probabilities`, identically on all three sides,
  using the probability each arm put on the option it picked. A distribution
  that omits the gold label entirely is charged for it, which is what makes
  the shortlist variant's damage visible.
* **AUROC**, **selective accuracy** — from each vendor's own `confidence`
  field, because that is the number each product tells you to gate on, and
  the three derive it differently. `acc @ 50% coverage` is the abstention
  story: route automatically above the threshold, send the rest to a human.
* **latency** p50/p95 of the successful attempt, retries counted separately.
* **cost** from each vendor's own `usage` block, never from a token count of
  our own. The three serialize differently and that is part of what is being
  compared.
* **paired statistics** — McNemar (exact, two-sided) and a paired bootstrap
  CI, for every pair of arms. They see the same examples; the unpaired
  comparison throws away exactly the information that makes a 2-point
  difference readable.

Fairness rules the runner enforces: the same connection-pooled HTTP client
for both hosted arms (a TLS handshake costs more than either forward pass),
the same retry policy, examples interleaved rather than batched per arm, the
arm order rotating, warmup discarded, and the same examples scored for
every arm: infrastructure errors are retried, or dropped for all arms
alike, while an arm's own failure to answer counts as a wrong answer.

## Contributing

This benchmark was written by one of the parties it compares, so it is
public in full: the methodology, the frozen dataset specifications, the
harness, and the aggregation. Jev and Laya ran with their default
settings, and the Privatemode prompt is the library's, not tuned on these
datasets.

Pull requests are welcome: a setting that serves one of the systems
better, a mistake in a dataset or the aggregation, another dataset, or
another system. [CONTRIBUTING.md](CONTRIBUTING.md) says how to do each.

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
.github/workflows/  CI, and the latency probe from a US runner
datasets/           the frozen option sets and validation statistics
results/            the aggregated results; the raw runs are in the release
METHODOLOGY.md      why these datasets, and how each design choice was made
CONTRIBUTING.md     how to propose a change, a dataset, or a system
```
