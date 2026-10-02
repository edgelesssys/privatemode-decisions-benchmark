"""The G arm's text: a content-free opening in GLM's own thinking style.

    python -m bench.generic_thought --run runs/r1 --out runs/prefill/generic

Samples ``--samples`` real thinking traces per dev dataset with the library's
prompt (rows from the calibration halves that the screening doesn't use) and
looks at the first three sentences of each. GLM starts on the content at
once: almost no opening sentence is free of the example (``sentences.json``
counts them). What recurs across datasets is how the sentences *start*
("The question asks which …", "The state is …"). So the text is built from
the openings found in at least ``--datasets`` datasets, each finished with
content-free words (:data:`COMPLETIONS`; openings that state an answer are
left out), then padded with :data:`PADDING` to about ``--tokens`` tokens.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from decisions import Choice
from decisions.client import OpenAIClient, set_max_in_flight

from . import calibration as c
from .datasets import load
from .prefill import DEV, MODEL, Prefill, Tokens, dev_rows

SENTENCE = re.compile(r"(?<=[.!?])\s+")
#: A content-free ending for each recurring opening. Openings that give an
#: answer ("So the answer is", "The answer is") are not continued.
COMPLETIONS = {
    "The state is": "The state is the material to judge.",
    "The question asks": "The question asks which option applies.",
    "The question is": "The question is which option fits best.",
    "This is a": "This is a question with numbered options.",
    "This is about": "This is about the state given above.",
    "Let me think": "Let me think about which option this maps to.",
}
PADDING = ("Let me read the state carefully.", "Let me go through the options one by one.",
           "Let me consider what each option means.", "Let me weigh each option against the state.",
           "Then I pick the number of the option that fits best.")


def content_free(sentence: str, task) -> bool:
    lowered = sentence.lower()
    if any(ch in sentence for ch in "\"'“”‘’`") or re.search(r"\d", sentence):
        return False
    if any(label.lower() in lowered for label in task.criteria):
        return False
    state_words = {w for w in re.findall(r"[a-zäöüß]{4,}", str(task.state).lower())}
    generic = {"question", "options", "option", "state", "text", "answer", "which", "about",
               "this", "that", "with", "from", "have", "given", "choose", "best", "need"}
    return not (set(re.findall(r"[a-zäöüß]{4,}", lowered)) & (state_words - generic))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--think-tokens", type=int, default=256)
    parser.add_argument("--datasets", type=int, default=3,
                        help="an opening counts if it starts sentences in this many datasets")
    args = parser.parse_args()
    set_max_in_flight(4)
    client = OpenAIClient.from_env()
    engine = Prefill(client, MODEL, temperature=1.0, fillers={}, generic=None)
    run = c.load_run(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    traces_path = out / "traces.jsonl"
    traces = [json.loads(line) for line in traces_path.open()] if traces_path.exists() else []
    have = {(t["dataset"], t["index"]) for t in traces}
    tasks = []
    for name in DEV:
        used = set(dev_rows(run, name, 250, "calibration"))
        cal, _ = c.split(run[name])
        spare = [int(i) for i in np.sort(cal.index) if int(i) not in used][: args.samples]
        tasks += [(name, t) for t in load(name, 1000) if t.index in spare]

    def trace(item):
        name, task = item
        if (name, task.index) in have:
            return None
        text, _, _ = engine.think(task.state, Choice(task.criteria, instructions=task.instructions),
                                  args.think_tokens)
        return {"dataset": name, "index": task.index, "trace": text}

    with ThreadPoolExecutor(4) as pool, traces_path.open("a") as handle:
        for row in pool.map(trace, tasks):
            if row:
                traces.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_key = {(name, t.index): t for name, t in tasks}
    openings, total, free = {}, 0, Counter()
    for t in traces:
        task = by_key.get((t["dataset"], t["index"]))
        if task is None:
            continue
        for sentence in SENTENCE.split(t["trace"])[:3]:
            words_ = sentence.split()
            if len(words_) < 3:
                continue
            total += 1
            openings.setdefault(" ".join(words_[:3]), set()).add(t["dataset"])
            if content_free(sentence, task):
                free[sentence.strip()] += 1
    ranked = sorted(openings.items(), key=lambda kv: -len(kv[1]))
    chosen = [COMPLETIONS[o] for o, sets in ranked if len(sets) >= args.datasets and o in COMPLETIONS]
    tokens = Tokens(client, MODEL)
    size = lambda text: tokens.count("\n" + text) - tokens.count("\n")
    text = " ".join(chosen)
    for sentence in PADDING * 4:
        if size(text + " " + sentence) > args.tokens:
            break
        text = (text + " " + sentence).strip()
    (out / "sentences.json").write_text(json.dumps(
        {"opening sentences": total, "free of the example": sum(free.values()),
         "free sentences": free.most_common(30),
         "openings in most datasets": [(o, len(d)) for o, d in ranked[:20]],
         "used": chosen}, indent=1, ensure_ascii=False))
    (out / "generic.txt").write_text(text + "\n")
    print(f"{len(traces)} traces, {total} opening sentences, {sum(free.values())} free of the "
          f"example; generic text: {size(text)} tokens")
    print(text)


if __name__ == "__main__":
    main()
