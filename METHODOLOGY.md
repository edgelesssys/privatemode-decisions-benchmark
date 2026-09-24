# Methodology

The design for the full benchmark, written before the runs so that it can be
held against them. This document is the pre-registration: what is measured,
on what, how it is aggregated, and which results would count as evidence
against our own product.

Status: the three-set pilot in [`results/report.md`](results/report.md) is
done. This describes the build-out to ~24 datasets.

## 0. The noise floor, measured first

Every arm runs at `temperature=0`. Neither hosted arm is deterministic
anyway — servers batch, kernels reassociate, cache state differs. Two
independent runs of the identical sample, same seed, same prompts:

| set | arm | answers that flipped | Δ accuracy |
|---|---|---|---|
| banking77 | jev | 2.3% | 0.003 |
| banking77 | privatemode | 2.7% | 0.000 |
| ag_news | jev | 0.0% | 0.000 |
| ag_news | privatemode | 0.5% | 0.005 |
| boolq | jev | 1.0% | 0.010 |
| boolq | privatemode | 3.5% | 0.005 |

**A single run resolves accuracy to about ±1 point.** Every design decision
below follows from that number, and any claim smaller than it is noise
wearing a table.

Consequences, in order of importance:

1. Two replicate runs per (arm, dataset), always. The cost is trivial; the
   alternative is publishing run-to-run variance as a finding.
2. A difference of 3 points at n=300 is roughly three times the noise. That
   is suggestive, not settled — which is exactly how the pilot's banking77
   result is worded.
3. The replicate spread goes into the reported interval. A CI that only
   covers sampling error understates what a reader would see on re-running.

## 1. Datasets: axes, not a pile

The pilot's finding is that **the option count reverses the ranking**: a
421M local model is the best arm at 4 options and the worst by far at 77.
A suite that averages over that hides the only thing worth knowing. So the
sets are chosen to fill a grid, declared here before they are run.

| axis | levels |
|---|---|
| option count | 2 · 3–6 · 10–20 · 50–80 · 100+ |
| task family | intent/routing · sentiment/rating · topic · moderation · NLI · QA over a passage |
| state length | short (~20 tok) · paragraph (~130) · document (800+) |
| label names | self-explanatory · opaque codes needing a description |
| language | EN · DE |

The headline artefact is an **accuracy-over-option-count curve per arm**,
not a grand mean. Where a mean is unavoidable, it is over
`(accuracy − chance) / (1 − chance)`: boolq's chance level is 50% and
banking77's is 1.3%, and averaging them raw is meaningless.

### Availability

Sets are fetched from the Hugging Face datasets-server. Script-backed
datasets are no longer served, so the canonical home of a dataset is often
dead and a mirror has to be used — `PolyAI/banking77` fails,
`legacy-datasets/banking77` works. Confirmed reachable:

`google/boolq` · `stanfordnlp/sst2` · `cornell-movie-review-data/rotten_tomatoes` ·
`nyu-mll/glue` (rte, mrpc, mnli) · `cardiffnlp/tweet_eval` (irony,
offensive, sentiment, emotion) · `fancyzhx/ag_news` · `dair-ai/emotion` ·
`SetFit/sst5` · `legacy-datasets/banking77` · `clinc/clinc_oos` (151
intents) · `coastalcph/lex_glue` (ledgar, 100 classes; scotus, 13 classes
and up to 68k characters) · `community-datasets/gnad10` (German, 9 classes)
· `facebook/xnli` (de) · `google/civil_comments` ·
`mteb/amazon_massive_intent` (en and de, 60 intents)

An `HF_TOKEN` is required in practice: the anonymous rate limit does not
survive curating two dozen sets, let alone running them.

### The document axis

One set, `rvl_cdip` (1600 scanned business documents, 16 types), is images
rather than text. It is in the suite because it is the one axis on which two
of the three arms cannot compete: Jev is text-only by its own documentation
and Laya is an encoder.

That is reported as what it is. Text-only arms are left out of an image run,
the omission is recorded in the run's meta block and printed in the report
header, and the resulting single-column table is never dressed up as a
comparison. Two of three arms being *unable* rather than *worse* is a
capability finding, and it needs no further construction to be one --
building an OCR pipeline to give them something to run would be measuring a
system nobody ships.

Image resolution is a declared parameter (`--image-max-side`, default
1024 px, recorded), not a detail: images are downloaded and scaled locally
before they are sent, and at 512 px the same page costs 632 input tokens
against 1345 at 1024 px. The token count *is* the price of a document
decision, and it is set by this knob.

### Truncation policy

`scotus` reaches 68k characters and Laya's English checkpoint holds 512
tokens. Truncation is therefore a declared parameter, not an accident: every
state is cut to the same character budget for every arm, the budget is
recorded in the run's meta block, and the fraction of truncated examples is
reported per set. An arm that cannot see the evidence should lose on it
visibly, not silently.

## 2. Contamination is the main threat

Jev and Laya are trained *for* this task family. Laya's own model card
separates "in-task 0.753" from "zero-shot 0.651", so the distinction is not
ours to invent. banking77, ag_news and sst2 are in essentially every
training mix. A benchmark that ignores this measures recall of the training
set and calls it accuracy.

Sets are stratified into three tiers — *certainly seen*, *probably seen*,
*published after the plausible cutoff* — and aggregated separately.

### The label-renaming control

Built. Every option is swapped for a frozen synonym and nothing else
changes: same examples, same order, same instructions. An arm that reads the
state and reasons about the options should barely move; one that has learned
`card_arrival` as the answer to "where is my card" should fall. The gap is a
**memorisation delta**, reported per arm per dataset.

A renaming is never perfectly neutral — some synonyms are simply harder
words — so every arm loses something. That common loss is the perturbation,
not memorisation, and the reported figure is therefore the **excess** drop
beyond the median arm's on the same dataset.

The first measurement, on ag_news at n=150, is why this control is not
optional:

| arm | original | renamed | drop | excess |
|---|---|---|---|---|
| glm-cot | 0.913 | 0.880 | −0.033 | +0.047 |
| jev | 0.907 | 0.827 | −0.080 | 0.000 |
| privatemode | 0.913 | 0.833 | −0.080 | 0.000 |
| embed-nn | 0.687 | 0.607 | −0.080 | 0.000 |
| **laya** | **0.953** | **0.807** | **−0.147** | **−0.067** |

Laya wins ag_news outright at 0.953 and gives back 14.7 points when the same
four classes are called `Global`, `Athletics`, `Commerce` and `STEM`. The
chain-of-thought control, which actually reads, is the most robust arm on
the board. Reporting the first table without the second would have published
a win that is substantially label-string recall.

Three properties make the control evidence rather than decoration.

*The renaming is generated once and checked in.* `perturbations/*.json` is
reviewable: a reader can see what every option became and object to any of
it. Regenerating per run would let the perturbation drift towards whatever
produced the nicest number.

*It is validated mechanically.* Replacements must be unique, non-empty,
different from the original, and must share no distinctive word with it —
otherwise `card_arrival` becomes `card_arrival_status` and the memorised
string is still there to be recognised.

*Its generator is named, and it cuts against us.* GLM-5.3-Flash writes the
synonyms, which is the same family Privatemode's arm runs. If the synonyms
happen to suit GLM's priors, the delta understates Privatemode's
memorisation and nobody else's. That is the safe direction for a benchmark
we publish, and it is recorded rather than hidden.

### Still to build

**State paraphrase.** Inputs paraphrased once by a third model and frozen,
on the same logic. More expensive than the renaming and noisier, so it is
planned for the *certainly seen* tier rather than the whole suite.

## 3. Statistics for two dozen sets

* **n = 1000** per set, **2 replicates** per (arm, set).
* **No per-dataset significance stars.** With 24 sets, "significant"
  somewhere is guaranteed by chance. The headline test is a **Wilcoxon
  signed-rank over the 24 paired per-set differences**; per-set results get
  confidence intervals and no p-values.
* Paired McNemar and a paired bootstrap stay, per set, as detail.
* Win / tie / loss counts alongside the means, because they survive a single
  pathological set.

## 3b. What the labels allow

A 40-example probe with five arms produced a result worth building on.
Three independent arms — Jev, Privatemode and the chain-of-thought control —
agreed on 36 of 40 predictions and on all 40 correctness outcomes. Their
shared errors were not hard examples; they were `get_physical_card` against
`order_physical_card`, `declined_transfer` against `failed_transfer`,
`extra_charge_on_statement` against `request_refund`. Those are label pairs
no system can separate, because the gold choice between them is the
annotator's.

So every dataset also carries two numbers that bound what is winnable:

**`consensus_wrong`** — the share of examples where the modal prediction,
held by at least half the arms, disagrees with the gold label. A floor
estimate for label noise. On banking77 it is 22.5%.

**`ceiling`** — the share at least one arm got right, which is the most any
of them could have scored. On banking77, 0.875 against a best arm of 0.750.

That reframes the pilot's headline. A 3.7-point gap between two arms sits in
12.5 points of contested ground, not in the 25 points the distance from 0.75
to 1.0 suggests. Reporting accuracy without it invites a reader to treat a
quarter of the dataset as capability difference when it is annotation.

Agreement is measured on the modal answer, not on unanimity. Unanimity reads
as the stricter test and is the useless one: adding a single weak arm that
disagrees with everything drives it to zero and takes the signal with it. On
the same probe, unanimity across all five arms found nothing while the modal
rule found the eleven.

The estimate leans one way and the report says so: arms can agree on a wrong
answer for reasons that are not label noise, a shared pretraining corpus
among them. It sits beside accuracy as context and is never subtracted from
it.

## 3c. What the latency clock is allowed to cover

Latency is a headline number, so what the clock measures is decided rather
than inherited. Three things could contaminate it and each is handled.

**Our own queuing.** The `decisions` client starts its clock before acquiring
the shared in-flight semaphore, so a request waiting on *our* gate reported
that wait as vendor latency. The benchmark's transport starts the clock
after the gate and records the wait separately, so "the gate was never
contended" is a measurement rather than an assumption. At concurrency 12 on
ag_news that wait is 118 ms for Jev and 149 ms for Privatemode — none of
which is either vendor's.

**Vendor throttling.** A 429 or 503 is a closed door, not a slow model.
Those are counted per arm, the retry's time is excluded from the reported
latency, and any arm that saw one gets its latency flagged: it was not
measured under the same conditions as one that did not. Across every run so
far the count is zero — no throttling has yet occurred on either account,
which is worth stating precisely because it cannot be assumed to hold.

**Load from the benchmark itself.** Concurrency is part of a run's identity,
so a throughput run cannot resume a latency run's file. That is not
cosmetic. On ag_news at concurrency 1, Privatemode's p50 is 189 ms against
Jev's 235 ms; at concurrency 12 both sit at 249 ms. Without the split, one
resumed file would have turned a 1.25× difference into a tie under a meta
block naming whichever load condition happened to be written first.
Aggregation takes latency **only** from concurrency-1 runs and reports
nothing rather than something misleading when there are none.

Every report prints **p10 beside p50**. p10 is the arm's unqueued floor; a
p50 well above it is the signature of something waiting — a vendor queue, a
throttle, or another process on the machine — rather than a slower model.
That is how the pilot's inflated Privatemode figure was caught: 274 → 385 ms
with Laya computing locally on the same laptop, against 273 → 277 ms in the
hosted-only rerun of the identical sample.

**Laya is not speed-benchmarked at all.** It runs on a laptop while the
others run in a datacentre. Its timings are recorded so the run is complete,
marked in every table, and excluded from every comparison.

## 4. The primary metric is coverage, not accuracy

These are routing products. The decision they support is "how much can be
automated before the error rate costs more than the automation saves". So
the headline is **coverage at a fixed target accuracy** — what fraction of
traffic can each arm take while staying at 95% correct — with plain
accuracy as a secondary. This folds calibration into the main result instead
of exiling it to its own column, which is right: a distribution you can
threshold is the product.

## 5. Control arms

Two arms exist to make the comparison interpretable rather than to win it.
Both are built, both run under `--arms full`, and every report labels them
so that no table reads as a five-way product comparison.

**`glm-cot` — chain-of-thought on the same model.** GLM-5.3-Flash asked
normally, reasoning allowed, answer parsed. The Privatemode Decisions arm
runs the same model, so this separates *the technique* — one masked forward
pass, no room to think — from *the model*. Without it nobody can tell which
of the two a Privatemode number is about, and that is the first question a
sceptical reader asks.

It is asked for a confidence as well as an answer. Scoring a bare label
against arms that return a distribution would be a straw man, and a
practitioner would ask for the number anyway; the self-report goes on the
chosen option with the remainder spread evenly, which is a generous reading
of a figure the model simply asserts. The calibration metrics then say what
it is worth. A reply with no resolvable option is a recorded error, never a
guess: a parser that quietly fell back to the first option would show up as
an accuracy difference and be read as a model difference.

Cost is the point as much as accuracy. Reasoning is output tokens, and
output is priced at 3.25× input on this model.

**`embed-nn` — the nearest option by embedding.** `Qwen3-Embedding-4B`,
which Privatemode already serves at €0.13 / M input. Zero-shot, no decision
model: embed the state, embed the options, take the closest. If a
purpose-built decision model cannot beat it on a set, that is worth knowing
before anyone argues about decision models. Option embeddings are computed
once per option set and reused, which is how it would be deployed.

Its softmax temperature is declared, not fitted. Fitting it on the same data
the arm is scored on would flatter the baseline, and a fitted floor is not a
floor.

## 6. What the runner does

Built, and named here so the claims above are checkable against code.

* **Resumability.** A run's identity -- dataset, the content hash of its
  frozen spec, sample, seed, replicate, truncation budget, image
  resolution, each arm's model, the variant knobs -- is hashed into its
  filename. Rows are appended as they land and re-read on resume, so a run
  that dies at example 700 of 1000 resumes at 700. A file whose stored
  identity disagrees with the current one is refused rather than appended
  to, so resuming cannot merge two different experiments, and a re-curation
  that changes an option set invalidates the runs taken against it. Error
  rows do not count as done.
* **Suite runner** (`python -m bench.suite`). Filters by name, family,
  language or option count, runs widest-options-first so an expensive
  failure happens early, and writes an index of the runs it produced.
* **Budget guard.** The cost is forecast from each dataset's frozen
  statistics before any request goes out, and a suite over `--budget-eur`
  refuses to start. The estimator is fitted to the pilot's measured token
  counts and lands within 3% on all six of them: the right accuracy for a
  guard, the wrong accuracy for a result, so it never reaches a report.
* **Validation per set** (`python -m bench.curate`). Option vocabulary,
  chance level, majority share, class coverage, state-length distribution
  and duplicate states, run once and checked in.
* **Aggregation** (`python -m bench.aggregate`). Per-dataset accuracy with
  its replicate spread, normalised accuracy overall, the pairwise Wilcoxon,
  the option-count bands, the ladders, and cost and latency. No per-dataset
  p-values, by construction.

Still missing, and honestly so: the provenance block does not yet record
the code revision, dtype or device, and the perturbation controls of §2 are
designed but not built.

## 7. Budget

24 sets × 1000 examples × 2 replicates = 48k decisions per arm.

| arm | cost | wall clock |
|---|---|---|
| jev | ~€1.20 | ~25 min at concurrency 8 |
| privatemode | €10–20, depending on option width | ~25 min |
| laya | €0 | ~80 min on a laptop |

Under €25 and an afternoon. **The budget is not the constraint**, which is
the main strategic point: effort belongs in dataset curation and
contamination controls, not in scale.

## 8. Hardware, and what a GPU does and does not buy

For **accuracy, a GPU changes nothing.** Same weights, same eval mode, same
argmax. The one caveat is dtype: fp16 on a GPU against fp32 on a CPU can
flip a close decision, so the dtype is pinned and recorded rather than
inherited.

For **throughput** it is a convenience at this scale — 48k local forward
passes are about 80 minutes on an M-series laptop. It starts to matter an
order of magnitude further out, where Laya's ability to batch (which the
hosted APIs, one request per question, do not have) becomes its own study.

For **latency it is not enough.** The hosted arms' numbers include network
round trips; a local forward pass is a different quantity on any hardware.
There is exactly one honest way to compare: put Laya behind the same HTTP
interface on a cloud GPU in a comparable region and measure it identically.
Until that is done, Laya gets no latency claim — which is why the pilot
marks its timings and keeps them out of the comparison.

German is an arm, not a translation: `laya-multilingual` (322M) is a
different checkpoint, and running the English one on German data would be a
straw man.

## 9. What would count as evidence against us

Stated in advance, because a benchmark published by one of the vendors is
worth what its falsifiers are worth.

* Jev or Laya beating Privatemode at equal or lower cost across a majority
  of sets, holding at every option count — the pilot already shows this for
  cost, and for accuracy at the narrow end.
* The memorisation delta being smaller for a competitor than for us.
* The chain-of-thought control beating all three System One arms by enough
  to matter, which would mean the single-forward-pass constraint costs more
  than it saves at this task.

None of these would be edited out. The one thing this benchmark structurally
cannot measure is the reason Privatemode exists — the model runs inside an
attested enclave and the prompt is never readable by the operator — and no
accuracy table prices that. Saying so is not a hedge; leaving it implicit
would be the dishonest move.
