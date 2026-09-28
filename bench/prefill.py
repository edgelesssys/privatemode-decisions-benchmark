"""Extra positions before ``answer=``: prompt variants, one read each.

    python -m bench.prefill --run runs/r1 --out runs/prefill --arms B,B2,R-Q \\
        [--rows 250] [--only sst5,rte] [--generic generic.txt] [--split calibration]

Every arm still ends its prompt in ``answer=`` and reads the answer from one
masked forward pass (two for the ``H`` arms, which first generate a short
thought). What changes is the prompt before it:

* ``R-*`` repeat the question (and the state) so that the state is read
  after the question; ``R-think`` restates the question inside the think
  block the chat template already renders empty before ``answer=``. ``R-Q``
  puts the question and options before the state as well as after it;
  ``R-Qi`` only the question's instruction, without the options (cheaper
  with many options); ``R-QSQS`` is ``R-Q`` followed by the state and the
  question again (three copies of the question, two of the state). With
  images, the images lead and only the text is repeated. ``RF`` combines
  restatement and filler, in words the model can follow: "Let me think. The
  question: … Options: 0 …, 1 …." then dots up to about ``--tokens`` tokens in
  all, then "Ok, now let me answer." before ``</think>answer=``. ``RQ-F``
  is ``R-Q`` with dots in the think block, ``RQ-FF`` the same framed as
  "Let me think. … Ok, now let me answer.", ``RQ-mid`` puts the dots between
  the state and the final question: ``[question] [state] [dots] [question]``.
* ``F-*`` put content-free filler of about ``--tokens`` tokens in the think
  block (``F-before``: before the state and question instead, the placement
  control). ``F-count`` counts in digits and is a diagnostic only, since the
  answers are digit tokens.
* ``G`` puts a content-free opening in the model's own style in the think
  block, from ``--generic``.
* ``H-32`` and ``H-128`` generate that many thinking tokens and then read the
  answer after them: real thinking, the reference.

Rows come from the calibration halves of ``--run`` (``bench.calibration.split``),
``--rows`` per dataset, the same rows for every arm (``--split test`` for the
confirmation; ``--perturb rename`` renames the options, the suite's
contamination control). One JSONL per arm and
dataset under ``--out``, resumed on the next run. Four requests in flight.
The library is used unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from decisions import Choice, SystemOne
from decisions.client import APIError, OpenAIClient, set_max_in_flight
from decisions.images import to_data_url
from decisions.inference import PREAMBLE, PREFIX, batches

from . import calibration as c
from . import hub
from .datasets import Task, load

DEV = ("patent", "rte", "sst5", "xnli_de", "toxic_conversations", "gnad10", "massive_scenario_en",
       "ag_news", "boolq", "mnli", "sst2", "dbpedia_14", "banking77", "clinc150")
CONTROLS = ("sst2", "dbpedia_14", "banking77", "clinc150")
ARMS = ("B", "B2", "R-Q", "R-full", "R-think", "F-dots", "F-alpha", "F-words", "F-scrambled",
        "F-before", "F-count", "G", "H-32", "H-128",
        # Phase 3, shapes of the question sandwich:
        "R-Qi", "R-QSQS",
        # The question restated in the think block, padded with dots to --tokens:
        "RF",
        # R-Q combined with filler of --tokens:
        "RQ-F", "RQ-FF", "RQ-mid")
MODEL = os.environ.get("DECISIONS_MODEL", "glm-5.3-flash")
#: As the suite sends scanned pages (``bench.run`` default).
IMAGE_MAX_SIDE = 1024

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen " \
       "fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))


#: Filler made of ``m`` units; the unit count for a token length is measured.
UNITS = {
    "dots": lambda m: " ".join(["."] * m),
    "alpha": lambda m: " ".join(chr(ord("a") + i % 26) for i in range(m)),
    "words": lambda m: " ".join(words(i + 1) for i in range(m)),
    "count": lambda m: " ".join(str(i + 1) for i in range(m)),
}


class Limited(OpenAIClient):
    """At most ``per_minute`` requests, spaced evenly, and a backoff on 429:
    the API key's limit (1,000 a minute) is shared with other runs."""

    def __init__(self, *args, per_minute: float = 500, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.interval, self.next, self.lock = 60.0 / per_minute, 0.0, threading.Lock()

    def post(self, path: str, payload: dict):
        for attempt in range(6):
            with self.lock:
                now = time.monotonic()
                wait, self.next = max(0.0, self.next - now), max(now, self.next) + self.interval
            time.sleep(wait)
            try:
                return super().post(path, payload)
            except APIError as error:
                if error.status != 429 or attempt == 5:
                    raise
                time.sleep(2 ** attempt + random.random())
        raise RuntimeError("unreachable")


class Tokens:
    """Token counts of text, from the serving tokenizer (``/completions``
    with ``echo``), cached."""

    def __init__(self, client: OpenAIClient, model: str) -> None:
        self.client, self.model, self.cache = client, model, {}

    def count(self, text: str) -> int:
        if text not in self.cache:
            body, _ = self.client.post("/completions", {
                "model": self.model, "prompt": text, "echo": True, "max_tokens": 0,
                "logprobs": 0, "return_tokens_as_token_ids": True})
            self.cache[text] = len(body["choices"][0]["logprobs"]["tokens"])
        return self.cache[text]

    def units_for(self, kind: str, tokens: int) -> int:
        """The unit count whose filler is closest to ``tokens`` tokens."""
        base = self.count("\n")    # a BOS token, if the tokenizer adds one
        m = tokens
        for _ in range(8):
            got = self.count("\n" + UNITS[kind](m)) - base
            if got == tokens:
                break
            m = max(1, round(m * tokens / max(got, 1)))
        return m


def dev_rows(run: dict[str, c.Dataset], name: str, rows: int, half: str) -> list[int]:
    """``rows`` random example ids from one half of the dataset, the same for
    every arm; all of the half if it has fewer."""
    cal, test = c.split(run[name])
    pool = np.sort((cal if half == "calibration" else test).index)
    if rows >= len(pool):
        return [int(i) for i in pool]
    rng = np.random.default_rng(c.SEED + 1)
    return sorted(int(i) for i in rng.choice(pool, rows, replace=False))


#: MMLU-Pro as openjev-sglang asked Jev (evals/mmlu_pro.py there): the
#: question is the state, the options are letters whose descriptions are the
#: option texts, and the 1,000 test questions are ``random.Random(42)``'s
#: sample of the 12,032, at revision b189ec76 (the current one, MIT).
MMLU_PRO = ("TIGER-Lab/MMLU-Pro", "default", "test")
MMLU_PRO_ROWS = 12032
MMLU_PRO_INSTRUCTIONS = "Choose the correct answer to the multiple-choice question in the state."


def mmlu_pro_rows(half: str, rows: int) -> list[int]:
    """The comparison sample (``test``), or ``rows`` other questions for
    screening (``calibration``), so the two never share a question."""
    comparison = random.Random(42).sample(range(MMLU_PRO_ROWS), 1000)
    if half == "test":
        return sorted(comparison[:rows])
    taken = set(comparison)
    rest = [i for i in range(MMLU_PRO_ROWS) if i not in taken]
    return sorted(random.Random(c.SEED + 1).sample(rest, rows))


def mmlu_pro_tasks(indexes: list[int]) -> list[Task]:
    found = hub.pages(*MMLU_PRO, indexes)
    tasks = []
    for i in indexes:
        row = found[i]
        letters = [chr(65 + k) for k in range(len(row["options"]))]
        tasks.append(Task(state=row["question"], instructions=MMLU_PRO_INSTRUCTIONS,
                          criteria=dict(zip(letters, row["options"])),
                          gold=letters[row["answer_index"]], index=i))
    return tasks


#: JevBench's public items (github.com/fstandhartinger/jevbench, MIT), for
#: comparison with its published numbers only: never used to choose, fit or
#: tune anything. Read from a checkout (``--jevbench``), tiers in this order;
#: the example index is the position in that order.
JEVBENCH_TIERS = ("original", "easy", "hard")


def jevbench_tasks(root: Path) -> list[Task]:
    """Every public item as a choice question, mapped as JevBench's own
    open-model adapter does: ``noul`` becomes the options no/yes described by
    the false/true criteria, ``score`` the level numbers described by the
    level texts."""
    tasks, index = [], 0
    for tier in JEVBENCH_TIERS:
        for line in (root / "datasets" / "public" / f"{tier}.jsonl").open():
            item = json.loads(line)
            q, crit = item["question"], item["question"].get("criteria")
            if q["type"] == "noul":
                crit = crit or {}
                criteria = {"no": crit.get("false"), "yes": crit.get("true")}
            elif q["type"] == "score":
                criteria = {str(i): text for i, text in enumerate(crit)}
            elif isinstance(crit, dict):
                criteria = dict(crit)
            else:
                criteria = {label: None for label in item["labels"]}
            tasks.append(Task(state=item["state"], instructions=q["instructions"],
                              criteria=criteria, gold=str(item["expected"]), index=index))
            index += 1
    return tasks


class Prefill(SystemOne):
    """Builds each arm's request around the library's own."""

    def __init__(self, *args, fillers: dict[str, str], generic: str | None,
                 tokens: "Tokens | None" = None, length: int = 128, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fillers, self.generic = fillers, generic
        self.tokens, self.length, self.pads = tokens, length, {}

    def restated(self, question: Choice) -> str:
        options = ", ".join(f"{i} {name}" for i, name in enumerate(question.criteria))
        return f"The question: {question.instructions} Options: {options}."

    def padded_thought(self, question: Choice, restate: bool = True) -> str:
        """RF: the restated question, dots, and a line that says the thinking
        is over, about ``length`` tokens in all (without the restatement
        for RQ-FF, whose question already came before the state)."""
        head = "Let me think. " + (self.restated(question) + " " if restate else "")
        tail = " Ok, now let me answer."
        used = self.tokens.count("\n" + head + tail) - self.tokens.count("\n")
        need = max(0, self.length - used)
        if need not in self.pads:
            self.pads[need] = UNITS["dots"](self.tokens.units_for("dots", need)) if need else ""
        return head + self.pads[need] + tail

    def question_text(self, question: Choice) -> str:
        return json.dumps(self._question(question), ensure_ascii=False)

    def payload(self, arm: str, state, question: Choice, allowed, read, index: int,
                thought: str | None = None, images: tuple[str, ...] = ()) -> dict:
        body = self._request(state, question, allowed, read, images)
        content = body["messages"][0]["content"]
        user = content if isinstance(content, str) else content[-1]["text"]
        whole = json.dumps({"state": state, **self._question(question)}, ensure_ascii=False)
        think = None
        if arm == "R-Q":
            user = PREAMBLE + self.question_text(question) + "\n" + whole
        elif arm in ("RQ-F", "RQ-FF"):
            user = PREAMBLE + self.question_text(question) + "\n" + whole
            think = (self.fillers["dots"] if arm == "RQ-F"
                     else self.padded_thought(question, restate=False))
        elif arm == "RQ-mid":
            user = (PREAMBLE + self.question_text(question) + "\n"
                    + json.dumps({"state": state}, ensure_ascii=False) + "\n"
                    + self.fillers["dots"] + "\n" + self.question_text(question))
        elif arm == "R-Qi":
            user = PREAMBLE + json.dumps({"question": question.instructions}, ensure_ascii=False) + "\n" + whole
        elif arm == "R-QSQS":
            user = PREAMBLE + self.question_text(question) + "\n" + whole + "\n" + whole
        elif arm == "R-full":
            user = PREAMBLE + whole + "\n" + whole
        elif arm == "F-before":
            user = PREAMBLE + self.fillers["dots"] + "\n" + whole
        elif arm == "R-think":
            think = self.restated(question)
        elif arm == "RF":
            think = self.padded_thought(question)
        elif arm in ("F-dots", "F-alpha", "F-words", "F-count"):
            think = self.fillers[arm[2:]]
        elif arm == "F-scrambled":
            units = self.fillers["alpha"].split(" ")
            random.Random(index).shuffle(units)
            think = " ".join(units)
        elif arm == "G":
            think = self.generic
        elif arm.startswith("H-"):
            think = thought
        if isinstance(content, str):
            body["messages"][0]["content"] = user
        else:
            content[-1] = {"type": "text", "text": user}
        if think is not None:
            body["messages"][1]["content"] = f"<think>{think}</think>{PREFIX}"
        return body

    def think(self, state, question: Choice, tokens: int) -> tuple[str, float, int]:
        """A real thought of at most ``tokens`` tokens: the text, the time and
        the tokens it took."""
        body, elapsed = self.client.post("/chat/completions", {
            "model": self.model, "max_tokens": tokens, "temperature": 0,
            "messages": [{"role": "user", "content": self._text(state, question)}]})
        message = body["choices"][0]["message"]
        text = message.get("reasoning_content") or message.get("reasoning") or ""
        if not text and "</think>" in (message.get("content") or ""):
            text = message["content"].split("</think>")[0]
        return text.strip(), elapsed, (body.get("usage") or {}).get("completion_tokens", 0)

    def ask(self, arm: str, task, ids: list[int]) -> dict:
        question = Choice(task.criteria, instructions=task.instructions)
        allowed = ids[:len(task.criteria)]
        images = tuple(to_data_url(path, max_side=IMAGE_MAX_SIDE) for path in task.images)
        latency, thought, generated = 0.0, None, 0
        if arm.startswith("H-"):
            thought, latency, generated = self.think(task.state, question, int(arm[2:]))
        reads, prompt_tokens, cached = [], 0, 0
        slowest = 0.0
        for read in batches(allowed, self.max_logprob_ids):
            body, elapsed = self.client.post(
                "/chat/completions", self.payload(arm, task.state, question, allowed, read,
                                                  task.index, thought, images))
            reads.append((read, body))
            usage = body.get("usage") or {}
            prompt_tokens = max(prompt_tokens, usage.get("prompt_tokens", 0))
            cached = max(cached, (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            slowest += elapsed
        weights = self._answer(reads, question, allowed)
        total = math.fsum(weights.values())
        row = {"index": task.index, "gold": task.gold,
               "probabilities": {k: v / total for k, v in weights.items()},
               "option_mass": min(1.0, total), "prompt_tokens": prompt_tokens,
               "cached_tokens": cached, "latency_s": latency + slowest}
        if thought is not None:
            row.update(thought=thought, generated_tokens=generated)
        return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="the calibration run the halves come from")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="B,B2,R-Q,R-full,F-dots,F-before")
    parser.add_argument("--only", default=",".join(DEV))
    parser.add_argument("--rows", type=int, default=250)
    parser.add_argument("--split", default="calibration", choices=("calibration", "test"))
    parser.add_argument("--tokens", type=int, default=128, help="filler length in tokens")
    parser.add_argument("--generic", help="file with the G arm's text")
    parser.add_argument("--suffix", default="", help="appended to the arm's directory name")
    parser.add_argument("--per-minute", type=float, default=500, help="request rate limit")
    parser.add_argument("--perturb", default=None, help="'rename': the label-renaming control")
    parser.add_argument("--threads", type=int, default=4, help="requests in flight")
    parser.add_argument("--jevbench", help="a checkout of fstandhartinger/jevbench (public items)")
    args = parser.parse_args()
    set_max_in_flight(max(4, args.threads))
    client = Limited(os.environ["DECISIONS_BASE_URL"], os.environ.get("DECISIONS_API_KEY") or None,
                     per_minute=args.per_minute)
    tokens = Tokens(client, MODEL)
    fillers = {kind: make(tokens.units_for(kind, args.tokens)) for kind, make in UNITS.items()}
    generic = Path(args.generic).read_text().strip() if args.generic else None
    engine = Prefill(client, MODEL, temperature=1.0, max_workers=4, fillers=fillers, generic=generic,
                     tokens=tokens, length=args.tokens)
    run = c.load_run(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fillers.json").write_text(json.dumps(
        {k: {"tokens": tokens.count("\n" + v) - tokens.count("\n"), "text": v[:200]}
         for k, v in fillers.items()}, indent=1))
    ids = engine.oracle.single_token_indexes(PREFIX, limit=191)
    lock = threading.Lock()
    for name in args.only.split(","):
        if name == "mmlu_pro":
            tasks = mmlu_pro_tasks(mmlu_pro_rows(args.split, args.rows))
        elif name == "jevbench":
            if args.split != "test" or not args.jevbench:
                raise SystemExit("jevbench is for the confirmation only: --split test --jevbench <checkout>")
            tasks = jevbench_tasks(Path(args.jevbench))
        else:
            wanted = set(dev_rows(run, name, args.rows, args.split))
            tasks = [t for t in load(name, 1000, perturbation=args.perturb) if t.index in wanted]
        for arm in args.arms.split(","):
            if arm == "G" and not generic:
                raise SystemExit("the G arm needs --generic")
            path = out / f"{arm}{args.suffix}" / f"{name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            done = {json.loads(line)["index"] for line in path.open()} if path.exists() else set()
            todo = [t for t in tasks if t.index not in done]
            if not todo:
                continue
            started = perf_counter()

            def one(task, arm=arm):
                for attempt in range(3):
                    try:
                        return engine.ask(arm, task, ids)
                    except Exception as failure:   # noqa: BLE001 - retried, then left for the next run
                        error = f"{type(failure).__name__}: {failure}"[:200]
                return {"index": task.index, "error": error}

            failed, errors = 0, set()
            with ThreadPoolExecutor(args.threads) as pool, path.open("a") as handle:
                for row in pool.map(one, todo):
                    if "error" in row:
                        failed += 1
                        errors.add(row["error"])
                        continue
                    with lock:
                        handle.write(json.dumps(row) + "\n")
            print(f"{arm:12} {name:22} {len(todo) - failed} rows in {perf_counter() - started:.0f}s"
                  + (f", {failed} failed ({'; '.join(sorted(errors)[:2])})" if failed else ""),
                  flush=True)


if __name__ == "__main__":
    main()
