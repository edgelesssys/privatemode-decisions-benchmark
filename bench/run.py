"""Run one dataset through every arm and write one JSONL per run.

Design decisions the numbers depend on:

*Interleaved, not batched.* Each example goes to every arm back to back, and
the arm order rotates. Running all of one arm and then all of the other
measures the two halves of the afternoon as much as it measures the
products.

*Concurrency 1 by default.* Latency is one of the three headline numbers and
it is only honest when nothing else is in flight. ``--concurrency N`` exists
for measuring throughput instead; the report says which was used.

*Warmup is discarded.* The Privatemode arm pays two extra requests the first
time it sees a model (the token oracle asks the server's tokenizer which
indexes are single tokens), Laya pays for a cold graph, and the hosted arms
benefit from a warm connection. None of that is the steady state.

*A run has an identity, and its file is named after it.* Everything that
changes the question or the answer -- the dataset and the exact bytes of its
frozen spec, the sample, the truncation budget, the image resolution, each
arm's model, the variant knobs -- is hashed into the filename. Resuming
therefore cannot silently mix two different experiments: a changed setting
is a different file, and a file whose stored identity disagrees with the
current one is refused rather than appended to.

*Rows are appended as they land and re-read on resume*, so a run that dies
at example 700 of 1000 resumes at 700 rather than restarting. At suite
scale that is the difference between an afternoon and a lost afternoon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from system_one.client import APIError, set_max_in_flight

from .adapters import (ChainOfThoughtArm, EmbeddingArm, JevArm, LayaArm,
                       PrivatemodeArm)
from .datasets import DEFAULT_MAX_CHARS, NAMES, Task, frozen_dir, has_images, load
from .pricing import DEFAULT_EUR_PER_USD
from .report import summarize
from .specs import BY_NAME

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
#: ``products`` is the like-for-like comparison. ``full`` adds the two
#: control arms, which exist to make that comparison interpretable rather
#: than to take part in it; the report labels them.
PRODUCTS = ("privatemode", "jev", "laya")
CONTROLS = ("glm-cot", "embed-nn")
ARM_SETS = {"hosted": ("privatemode", "jev"),
            "all": PRODUCTS,
            "full": PRODUCTS + CONTROLS,
            "controls": CONTROLS,
            **{name: (name,) for name in PRODUCTS + CONTROLS}}


def load_env(path: Path) -> None:
    """A three-line .env reader, so the repo needs no dependency for it."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def ask_with_retry(arm, task: Task, attempts: int = 4):
    """Exponential backoff on the statuses the vendors ask you to retry.

    Applied identically to every arm: a retry policy that favours one shows
    up as a latency difference that is really a policy difference. Only the
    successful attempt's latency is reported, with the retry count beside it.
    """
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            return arm.ask(task), attempt
        except APIError as error:
            if error.status not in RETRY_STATUS or attempt == attempts:
                raise
        except (OSError, RuntimeError):
            if attempt == attempts:
                raise
        time.sleep(delay + random.random() * 0.25)
        delay *= 2
    raise RuntimeError("unreachable")


def build_arms(args) -> list:
    wanted = ARM_SETS[args.arms]
    arms = []
    if "privatemode" in wanted:
        arms.append(PrivatemodeArm(permutations=args.permutations,
                                   image_max_side=args.image_max_side))
    if "jev" in wanted:
        arms.append(JevArm())
    if "laya" in wanted:
        arms.append(LayaArm(shortlist=args.laya_shortlist))
    if "glm-cot" in wanted:
        arms.append(ChainOfThoughtArm(image_max_side=args.image_max_side,
                                      max_tokens=args.cot_max_tokens))
    if "embed-nn" in wanted:
        arms.append(EmbeddingArm())
    return arms


def identity(args, arms) -> dict:
    """Everything that would make two runs incomparable, in one dict.

    The frozen spec is included by content hash rather than by name: a
    re-curation that changes an option set has to invalidate the runs taken
    against the old one, and nothing else would catch that.

    Concurrency is in here too, which is not obvious: it cannot change an
    answer, only how long it took. But latency is a headline number, and
    without it a throughput run would happily resume a latency run's file
    and leave one set of timings measured against an empty pipe and the
    next against eleven competitors, under a meta block naming whichever
    came first. Two load conditions are two experiments.
    """
    spec = BY_NAME[args.dataset]
    frozen_bytes = (frozen_dir() / f"{args.dataset}.json").read_bytes()
    return {
        "dataset": spec.name, "hf": spec.hf, "config": spec.config,
        "split": spec.split,
        "frozen_sha1": hashlib.sha1(frozen_bytes).hexdigest()[:12],
        "n": args.n, "seed": args.seed, "replicate": args.replicate,
        "concurrency": args.concurrency,
        "max_chars": args.max_chars,
        "image_max_side": args.image_max_side if has_images(spec) else None,
        "arms": {arm.name: arm.model for arm in arms},
        "permutations": args.permutations,
        "laya_shortlist": args.laya_shortlist,
        "cot_max_tokens": args.cot_max_tokens,
        "perturb": args.perturb,
    }


def run_key(ident: dict) -> str:
    canonical = json.dumps(ident, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def already_done(path: Path, ident: dict) -> set[tuple[str, int]]:
    """(arm, index) pairs this file already holds, after checking identity.

    An error row does not count as done: a run that failed on the vendor's
    503 should retry it next time round rather than bank the failure.
    """
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    stored = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") == "meta":
            stored = record.get("identity")
        elif record.get("kind") == "row" and "error" not in record:
            done.add((record["arm"], record["index"]))
    if stored is not None and stored != ident:
        raise SystemExit(
            f"{path} was written with a different configuration; refusing to "
            "append. Delete it or change --seed/--replicate.")
    return done


def execute(args) -> Path:
    spec = BY_NAME[args.dataset]
    tasks = load(args.dataset, args.n, args.seed, args.max_chars,
                 perturbation=args.perturb)
    options = len(tasks[0].criteria)
    cut = sum(task.truncated for task in tasks)

    arms = build_arms(args)
    # An arm that cannot answer this kind of question at all would produce a
    # column of errors, and paired scoring would then drop every example for
    # the arms that could answer. Ask each one first, leave out the ones that
    # say no, and record why. The reason is a finding, not a failure.
    skipped: dict[str, str] = {}
    for arm in list(arms):
        reason = arm.unsupported(tasks[0])
        if reason:
            skipped[arm.name] = reason
            arms.remove(arm)
            arm.close()
    if not arms:
        raise SystemExit(f"{spec.name}: no selected arm can answer it "
                         f"({'; '.join(f'{k}: {v}' for k, v in skipped.items())})")

    ident = identity(args, arms)
    key = run_key(ident)
    out_dir = Path(args.out) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key}-r{args.replicate}.jsonl"
    done = already_done(path, ident) if args.resume else set()

    print(f"{spec.name}: {len(tasks)} examples, {options} options, seed "
          f"{args.seed}, replicate {args.replicate}, {cut} truncated at "
          f"{args.max_chars} chars"
          + (f", images at {args.image_max_side}px" if has_images(spec) else "")
          + (f", skipping {', '.join(skipped)}" if skipped else "")
          + (f", resuming with {len(done)} answers already in hand" if done else ""),
          file=sys.stderr)

    fresh = not path.exists()
    handle = path.open("a")
    if fresh:
        handle.write(json.dumps({
            "kind": "meta", "identity": ident,
            "dataset": spec.name, "hf": spec.hf, "split": spec.split,
            "n": len(tasks), "options": options,
            "max_chars": args.max_chars, "truncated": cut,
            "images": has_images(spec),
            "image_max_side": ident["image_max_side"],
            "skipped_arms": skipped,
            "seed": args.seed, "replicate": args.replicate,
            "concurrency": args.concurrency,
            "permutations": args.permutations,
            "laya_shortlist": args.laya_shortlist,
            "perturb": args.perturb,
            "eur_per_usd": args.eur_per_usd,
            "started": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "arms": ident["arms"],
        }) + "\n")
        handle.flush()

    lock = Lock()
    progress = [0]
    todo = [(position, task) for position, task in enumerate(tasks)
            if any((arm.name, task.index) not in done for arm in arms)]

    for task in tasks[:args.warmup]:
        for arm in arms:
            try:
                ask_with_retry(arm, task)
            except Exception as error:  # a cold arm may simply be down
                print(f"warmup {arm.name}: {error}", file=sys.stderr)

    def run(item: tuple[int, Task]) -> None:
        position, task = item
        # Rotate who goes first, so no arm always pays for the one before it
        # having just warmed the network path.
        shift = position % len(arms)
        for arm in arms[shift:] + arms[:shift]:
            if (arm.name, task.index) in done:
                continue
            row = {"kind": "row", "arm": arm.name, "index": task.index,
                   "gold": task.gold}
            try:
                answer, attempts = ask_with_retry(arm, task)
                row.update(choice=answer.choice,
                           correct=answer.choice == task.gold,
                           probabilities=answer.probabilities,
                           confidence=answer.confidence,
                           latency_s=answer.latency_s,
                           input_tokens=answer.input_tokens,
                           output_tokens=answer.output_tokens,
                           cached_tokens=answer.cached_tokens,
                           gate_wait_s=answer.gate_wait_s,
                           throttled=answer.throttled,
                           concurrency=args.concurrency,
                           attempts=attempts)
            except Exception as error:
                row.update(error=f"{type(error).__name__}: {error}"[:300])
            with lock:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
        with lock:
            progress[0] += 1
            if progress[0] % 100 == 0 or progress[0] == len(todo):
                print(f"  {progress[0]}/{len(todo)}", file=sys.stderr)

    started = time.perf_counter()
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(run, todo))
    else:
        for item in todo:
            run(item)
    wall = time.perf_counter() - started

    handle.write(json.dumps({"kind": "end", "wall_s": wall,
                             "answered": len(todo)}) + "\n")
    handle.close()
    for arm in arms:
        arm.close()
    print(f"wrote {path} ({wall:.1f}s)", file=sys.stderr)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="banking77", choices=sorted(NAMES))
    parser.add_argument("-n", "--n", type=int, default=200,
                        help="examples to sample from the split (default: 200)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicate", type=int, default=0,
                        help="replicate number; part of the run identity, so "
                             "two replicates are two files. The measured "
                             "run-to-run flip rate is up to 3.5%% at "
                             "temperature 0, which is why there are two.")
    parser.add_argument("--arms", default="hosted", choices=sorted(ARM_SETS))
    parser.add_argument("--concurrency", type=int, default=1,
                        help="1 keeps the latency numbers honest (default)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="examples run and discarded before measuring")
    parser.add_argument("--permutations", type=int, default=1,
                        help="Privatemode only: option orders averaged per "
                             "question. >1 is no longer a like-for-like "
                             "comparison; the report says so.")
    parser.add_argument("--laya-shortlist", type=int, default=0, metavar="K",
                        help="Laya only: keep the K options closest to the "
                             "state before deciding. Like --permutations, it "
                             "is no longer a like-for-like comparison.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help="state truncation budget, applied identically to "
                             "every arm and recorded in the run")
    parser.add_argument("--image-max-side", type=int, default=1024,
                        help="longest edge a document image is scaled to "
                             "before it is sent; image tokens are the whole "
                             "cost of a document decision, so it is recorded")
    parser.add_argument("--cot-max-tokens", type=int, default=4000,
                        help="reasoning budget for the chain-of-thought "
                             "control. A deployment choice, not a model "
                             "property, so it is recorded; at 3.3x the input "
                             "price it is also most of that arm's cost.")
    parser.add_argument("--perturb", default="none",
                        choices=["none", "rename"],
                        help="rename: swap every option for a frozen synonym "
                             "and change nothing else. The accuracy an arm "
                             "loses is its memorisation delta. Not part of "
                             "the like-for-like suite.")
    parser.add_argument("--eur-per-usd", type=float, default=DEFAULT_EUR_PER_USD)
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="start the run's file from scratch")
    parser.add_argument("--out", default="results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env(Path(__file__).resolve().parent.parent / ".env")
    # Room for every arm at every worker, and for the option orders one
    # Privatemode question fans out into -- a semaphore below that turns
    # --permutations into a latency penalty that is ours, not the model's.
    set_max_in_flight(max(2, 2 * args.concurrency * max(1, args.permutations)))
    path = execute(args)
    print()
    print(summarize(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
