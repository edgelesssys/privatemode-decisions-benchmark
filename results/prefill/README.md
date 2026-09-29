# Extra positions before `answer=`: accuracy without generation

Can GLM-5.3-Flash answer more accurately in one masked read if the prompt
gets longer before `answer=`? Repetition, filler and short real thinking,
screened on a dev set, the winner confirmed on the test halves of all 29
datasets, then compared on MMLU-Pro and JevBench with published numbers.
The plan is `accuracy-plan.md` and `filler-tokens-plan.md` in the library
repo; the tools are `bench/prefill.py` (runs), `bench/prefill_report.py`
(screening) and `bench/prefill_confirm.py` (confirmation).

## Results

| question | answer |
|---|---|
| What does the model see? | Today's prompt already ends `<|assistant|><think></think>answer=`, with no whitespace: we run GLM-5.3-Flash in "no thinking" mode. Text put in `reasoning_content` or in `<think>…</think>` in the prefill renders the same way, so thinking content can be injected in distribution. |
| Does filler help? | **A little, only when long, and not on top of repetition.** At 128 tokens every kind of filler (dots, letters, number words, digits, a generic opening in GLM's style) is within ±0.5 points. Dots gain with length: +0.56 at 256 tokens, **+0.97 [+0.27, +1.68] at 1,024**, all of it in the least sure fifth of answers. The same 1,024 dots before the question cost −0.74, so it is computation after the question, as in the literature, but weak on a Flash-class model. Combined with R-Q it adds nothing (below). |
| Does repetition help? | **Yes: the question and its options also before the state (R-Q).** Dev +2.2 points [+1.2, +3.2], 12 datasets better and 2 worse. On the test halves +1.6 points (median +1.2), 15 wins, 12 ties, 2 losses, Wilcoxon p = 0.002, confirmed in two replicates. |
| Why that shape? | Under the causal mask the state is read before the model knows what is asked. Repeating only the instruction before the state does nothing (+0.0), so it's the options, read first, that help. A third copy (R-QSQS) or repeating the state too (R-full) gains no more. |
| Where? | Where the baseline is unsure: +9.7 points in the least sure fifth, about 0 elsewhere. Most on sentiment (sst5 +10.3, toxic_conversations +6.3) and intent (massive +3 to +4); nothing on topic sets that were already easy, and −2.2 on scanned documents (rvl_cdip). |
| What does it cost? | Prompt tokens ×1.6 on average (EUR 0.135 → 0.223 per 1,000). At concurrency 1, nothing measurable up to about 20 options; with long option lists the list is sent twice: +110 ms on banking77 (77 options), +330 ms on clinc150 (151). With several questions per state the layout matters (next row). |
| Several questions per state? | Leading every request with **all** of the call's questions keeps a shared, cacheable prefix, but keeps less of the gain: +0.9 points [−0.15, +1.98] with five questions per state, against +2.2 when each request leads with its own question. On 2,000-token states it was the fastest layout (p50 670 ms, 59% cached), because Privatemode only caches prefixes of about 2,300 tokens. It is the library's default; the own question first is `question_first="own"`. |
| Calibration? | Better: NLL after a temperature 0.553 → 0.499 on the dev set, and the shipped default temperature still fits (excess ECE 0.016 with R-Q against 0.023 for the baseline, both on the test halves), so no new constants are needed. |
| Label strings? | No more dependence: renaming every option costs −7.7 points for the baseline and −8.1 with R-Q. |
| And real thinking? | The ceiling, and a different trade. 128 thinking tokens and then our read: +2.3 points on the dev set at +1.7 s, but worse calibrated (its own T is 3.9); 32 tokens do nothing. On MMLU-Pro it is +12 points where no prompt variant helps. Thinking only where the one-pass answer is below 0.9 keeps most of it: +10.0 on MMLU-Pro for 53% of questions, +1.9 on the classification sets for 21%. |

## Follow-up: long filler, framed filler, and filler with R-Q

Same 3,388 dev rows and baseline as the screening; the full tables are in
[screening-long-filler.md](screening-long-filler.md).

| arm | extra tokens | accuracy points [95% CI] | datasets + / − | option mass |
|---|---:|---|---|---:|
| dots, 128 tokens | 128 | +0.06 [−0.53, +0.65] | 6 / 4 | 0.84 |
| dots, 256 tokens | 256 | +0.56 [−0.03, +1.18] | 9 / 4 | 0.79 |
| dots, 1,024 tokens | 1,024 | **+0.97 [+0.27, +1.68]** | 7 / 3 | 0.79 |
| the same 1,024 dots before the question (placement control) | 1,024 | −0.74 [−1.48, −0.06] | 4 / 6 | 0.95 |
| "Let me think." + question and options restated + dots to 1,024 + "Ok, now let me answer." (RF) | 1,023 | +0.32 [−0.41, +1.06] | 8 / 4 | 0.98 |
| **R-Q** | 419 | **+2.18 [+1.21, +3.16]** | 12 / 2 | 0.95 |
| R-Q + 1,024 dots in the think block (RQ-F) | 1,443 | +2.10 [+1.15, +3.13] | 10 / 3 | 0.76 |
| R-Q + the same framed, "Let me think. … Ok, now let me answer." (RQ-FF) | 1,442 | +1.83 [+0.86, +2.80] | 11 / 3 | 0.96 |
| question, state, 1,024 dots, question (RQ-mid) | 1,444 | +1.86 [+0.94, +2.83] | 10 / 3 | 0.90 |

- **Long filler is real but weak.** The gain grows with length and
  disappears when the filler comes before the question, the signature of
  extra computation after the question. At 1,024 tokens it is just under the
  plan's +1-point gate, from one dev run among many arms, and costs 1,024
  prefill tokens per decision.
- **Framing keeps the model on the answer, not the accuracy.** Saying that
  the thinking is over ("Ok, now let me answer.") keeps the probability on
  the options at 0.96–0.98 where bare dots let it fall to 0.79, but gains no
  accuracy. Restating the question in the think block doesn't help, padded
  (+0.32) or not (+0.09): the state has already been read by then.
- **Filler and R-Q don't add up.** Every combination lands at or below R-Q
  alone, for 3.4× R-Q's extra tokens. The stored runs say why: of the 93
  answers the dots fix, R-Q fixes 65 too (24 if they were independent), and
  the 28 the dots fix alone are fewer than asking the baseline again flips
  (35). Averaging the two arms' answers (81.6%) is below R-Q alone (82.1%).
- Stopped here: no combination beat R-Q, so none went to the test halves.

## Several questions about one state

The benchmark asks one question per state. To see what the layouts do when
a call has several, the same 3,388 dev rows were asked as if the call had
five questions: the row's own question at a random position among four
generic ones (language, tone, whether a person is named, length), all with
their options before the state, then the state and the row's question
(arm `QA` in `bench.prefill`; full tables in
[screening-multi-question.md](screening-multi-question.md)).

| layout | accuracy vs state first [95% CI] | datasets + / − | least sure fifth |
|---|---|---|---:|
| own question first (R-Q) | **+2.18 [+1.21, +3.16]** | 12 / 2 | +9.7 |
| all five questions first (QA) | +0.89 [−0.15, +1.98] | 7 / 5 | +5.9 |

With four other questions in front, about 40% of the gain remains: +3.2
points on sentiment and +1.6 on intent, nothing on NLI and topic tasks. The
row's position in the block makes no clear difference (each position
within ±2 points).

What the layouts cost, five questions per call with `mode="staged"`, one
call at a time (`bench.prefill_cache`, [cache-multi-question.json](cache-multi-question.json)):

| layout | short states (ag_news): p50, cached | 2,000-token states (scotus): p50, cached |
|---|---|---|
| state first | 403 ms, 0% | 809 ms, 3% |
| own question first | 409 ms, 0% | 935 ms, 0% |
| all questions first | 530 ms, 0% | **670 ms, 59%** |

Privatemode only caches prefixes of about 2,300 tokens. The state-first
requests stay below that (about 1,950 tokens each on scotus), so even the
state they share isn't reused; the question block lifts every request of
the call over it, and the block itself is shared by every call with the
same questions. Short states are never cached, so there the extra tokens
only cost time.

**What the library does with it:** all of the call's questions first is the
default (`question_first=True`), for the cache with long states;
`question_first="own"` leads each request with its own question, more
accurate with several questions per call; `False` is the old layout. With
one question per call the first two are identical, which is what the
confirmation below measured.

## (a) Against the current state

GLM-5.3-Flash, test halves, mean of two replicates per arm:

| | today (state first) | question also first (R-Q) |
|---|---|---|
| mean accuracy, 29 datasets | 0.776 | **0.793** |
| wins / ties / losses, R-Q against today | | 15 / 12 / 2 |
| by options: 2 / 3–6 / 7–20 / 21–80 / 81+ | 0.881 / 0.729 / 0.724 / 0.810 / 0.816 | 0.900 / 0.747 / 0.731 / 0.839 / 0.836 |
| share of glm-cot's lead closed (14 datasets where it is > 2 points ahead) | | 57% (median) |
| excess ECE with the shipped default temperature | 0.023 | 0.016 |
| prompt tokens, EUR per 1,000 | 672, 0.135 | 1,113, 0.223 |
| latency p50 at concurrency 1, up to 18 options / 77 / 151 | ~150 / 273 / 569 ms | ~150 / 386 / 900 ms |
| JevBench public items (231) | 0.885 | 0.894 |
| MMLU-Pro (1,000) | 63.1% | 61.9% |

## (b) Against Jev

| | Jev | today | R-Q | GLM-5.3-Flash with thinking |
|---|---|---|---|---|
| 28 suite datasets, same examples (mean accuracy) | 0.775 | 0.780 (13–8–7) | **0.798 (16–8–3)** | |
| JevBench public items (231) | 0.866 | 0.885 | 0.894 | **0.931** (1,024 tokens, then our read); **0.983** (structured output, JevBench's runner) |
| JevBench public hard items | 0.741 (220 items) | 0.761 (111) | 0.779 (111) | 0.856 / 0.964 (111) |
| MMLU-Pro, 1,000 questions of openjev-sglang's sample | **82.9%** | 63.1% | 61.9% | 75.5% (1,024 tokens, then our read) |

Wins–ties–losses against Jev in brackets. Jev's MMLU-Pro and JevBench
numbers are published (openjev-sglang's `evals/results/mmlu-pro-2026-09-18`,
JevBench v1.4.2), on the same questions but not paired with ours. On the
classification suite R-Q puts GLM clearly ahead of Jev; on MMLU-Pro, a
knowledge-and-reasoning test, one pass is 20 points behind and even 1,024
thinking tokens leave 7. The other published MMLU-Pro number,
openjev-sglang's Qwen3.6-35B-A3B, is 58.8%.

## JevBench

The 231 public items of [JevBench](https://github.com/fstandhartinger/jevbench)
(MIT, commit `1bcc55e`), **for comparison only**: nothing was chosen,
fitted or tuned on them, and R-Q was chosen before they were run. Yes/no
items become the options no/yes described by the false/true criteria, score
items one option per level, as JevBench's own open-model adapter maps them.
ECE is JevBench's own (10 equal-width bins) on the 111 public hard items.

| system | output tokens | public items | public hard | ECE, public hard |
|---|---:|---:|---:|---:|
| GLM-5.3-Flash, one pass | 1 | 0.885 | 0.761 | 0.087 |
| GLM-5.3-Flash, one pass, R-Q | 1 | 0.894 | 0.779 | 0.099 |
| GLM-5.3, one pass | 1 | 0.853 | 0.721 | 0.060 |
| GLM-5.3-Flash, 1,024 thinking tokens, then our read | 447 | 0.931 | 0.856 | 0.085 (raw) |
| GLM-5.3-Flash, structured output + thinking | 518 | 0.983 | 0.964 | 0.057 |
| GLM-5.3, structured output + thinking | 501 | **0.987** | **0.982** | 0.044 |
| *Jev 1.13.0, published* | 45 | 0.866 | *0.741 (220)* | *0.061 (220)* |
| *GPT-6 Luna, medium effort, published* | 122 | 0.996 | *0.986 (220)* | |
| *DeepSeek V4.1 Flash, thinking, published* | 1,753 | 0.978 | *0.950 (220)* | *0.033 (220)* |

The structured-output rows are JevBench's own runner (`jevbench.cli run
--adapter openai_compat`, the board's LLM-baseline protocol) against the
Privatemode proxy, one run each; 4 and 3 empty answers count as wrong. They
cost about $0.45 (Flash) and $4.98 (GLM-5.3) per 1,000 at our list prices,
against Jev's $0.040. The sealed items and the chance-corrected score need
the maintainer's run.

## What changes in the library

The library now asks the questions before the state as well, by default:
all of the call's questions first (`question_first=True`), each request's
own question first as an option (`"own"`), or the old layout (`False`).
The default temperatures apply unchanged: the calibration check above found
them at least as good on the new prompt (excess ECE 0.016 against 0.023).

The benchmark's Privatemode arm follows the library's default and records
it in the run identity; `--state-first` reproduces the layout the published
suite (`results/suite.md`) and the calibration runs used.

## Checked and dropped

- Filler: nothing at 128 tokens of any kind, +1 point at 1,024 dots, and
  nothing on top of R-Q in three combinations (follow-up above).
- A content-free opening in GLM's style (G): GLM starts thinking on the
  content at once, so only 53 of 698 opening sentences said nothing about
  the example; the text built from its recurring openings gains nothing.
- The question restated in the think block (R-think), the question
  instruction alone first (R-Qi), three copies (R-QSQS) and the whole prompt
  twice (R-full): no more than R-Q, or less.
- 32 thinking tokens (nothing); 128 help but are a different product
  (seconds, not milliseconds) and need their own temperature.

## Reproduce

```sh
# Phase 0/1: G's text from the model's own openings
python -m bench.generic_thought --run runs/r1 --out runs/prefill/generic
# Phase 2 and 3: screening on 250 calibration-half rows of 14 datasets, and 250 MMLU-Pro questions
python -m bench.prefill --run runs/r1 --out runs/prefill/screen --generic runs/prefill/generic/generic.txt \
    --arms B,B2,R-Q,R-full,F-dots,F-before,R-think,F-alpha,F-words,F-scrambled,F-count,G,H-32,H-128,R-Qi,R-QSQS
python -m bench.prefill --run runs/r1 --out runs/prefill/screen --only mmlu_pro --generic runs/prefill/generic/generic.txt \
    --arms B,B2,R-Q,R-Qi,R-QSQS,R-full,R-think,F-dots,F-before,G,H-32,H-128,H-1024
python -m bench.prefill_report --runs runs/prefill/screen --published <published runs>/results --out report/
# follow-up: longer filler, framed filler, and filler with R-Q (1,024 tokens unless noted)
python -m bench.prefill --run runs/r1 --out runs/prefill/long --arms F-dots,RF,F-before --tokens 1024 --suffix=-1024
python -m bench.prefill --run runs/r1 --out runs/prefill/combo --arms RQ-F,RQ-FF,RQ-mid --tokens 1024
python -m bench.prefill --run runs/r1 --out runs/prefill/long256 --arms F-dots --tokens 256 --suffix=-256
# Phase 4: test halves, two replicates, plus MMLU-Pro, JevBench, renaming and latency
for r in 1 2; do python -m bench.prefill --run runs/r1 --out runs/prefill/confirm --split test --rows 100000 \
    --only jevbench,mmlu_pro,<the 29 datasets> --jevbench <jevbench checkout> --arms B,R-Q --suffix=-r$r; done
python -m bench.prefill --run runs/r1 --out runs/prefill/confirm --split test --rows 100000 \
    --only jevbench,mmlu_pro --jevbench <jevbench checkout> --arms H-1024 --suffix=-r1 --threads 12
python -m bench.prefill --run runs/r1 --out runs/prefill/renamed --split test --rows 250 --perturb rename --arms B,R-Q
python -m bench.prefill --run runs/r1 --out runs/prefill/latency --split test --rows 80 --threads 1 --arms B,R-Q,B2 \
    --only sst2,ag_news,trec_coarse,massive_scenario_en,banking77,clinc150
python -m bench.prefill --run runs/r1 --out runs/prefill/screen --arms QA
python -m bench.prefill_cache --out cache-ag.json -n 40
python -m bench.prefill_cache --out cache-scotus.json -n 30 --dataset scotus
# JevBench's own runner, per tier (original, easy, hard) and model
python -m jevbench.cli run --tasks datasets/public/<tier>.jsonl --adapter openai_compat \
    --endpoint $DECISIONS_BASE_URL --model glm-5.3-flash --key-env DECISIONS_API_KEY --results <model>/<tier>/results.jsonl
python -m bench.prefill_confirm --runs runs/prefill/confirm --variant R-Q --published <published runs>/results \
    --renamed runs/prefill/renamed --latency runs/prefill/latency --mmlu-extra H-1024-r1 \
    --jevbench <jevbench checkout> --jevbench-runs <jevbench runs> --out report/
```

`runs/r1` is the calibration run of part 1 (it defines the halves). Every
run keeps at most 4 requests in flight unless `--threads` says otherwise,
and at most `--per-minute` requests a minute, since the API key's limit of
1,000 a minute is shared.
