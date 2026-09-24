"""Turning a frozen dataset spec into tasks the arms can be asked.

The option set is not discovered here. ``bench.curate`` scans a dataset
once, writes ``datasets/<name>.json`` and that file is checked in; this
module reads it. The reason is reproducibility with teeth: a run on another
machine, or in six months, asks the identical question with the identical
options in the identical order, and the option set is a reviewable artefact
rather than a side effect of whichever rows happened to be sampled.

Sampling is a seeded shuffle of row *indexes*, never of a downloaded prefix.
banking77 is ordered by label; a prefix would be one intent repeated two
hundred times.

Truncation is a declared parameter. ``max_chars`` is applied identically to
every arm and recorded in the run's meta block, and the fraction of examples
it bit is reported. Letting each arm silently truncate at its own context
window instead would hide the difference in exactly the cell -- long legal
text -- where it matters most.
"""

from __future__ import annotations

import json
import random
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from . import hub
from .specs import BY_NAME, SPECS, Spec

#: Applied to every arm alike. About 2000 tokens: generous enough that the
#: hosted arms are not crippled to fit the smallest local context window,
#: bounded enough that one Supreme Court opinion cannot cost more than a
#: whole other dataset.
DEFAULT_MAX_CHARS = 8000


@dataclass(frozen=True)
class Task:
    state: str
    instructions: str
    criteria: dict[str, str | None]
    gold: str
    index: int
    truncated: bool = False
    #: Local paths of the images that accompany ``state``; empty for text.
    images: tuple[str, ...] = ()


def frozen_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "datasets"


def frozen(name: str) -> dict:
    path = frozen_dir() / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing -- run `python -m bench.curate {name}` first. "
            "Curation is a separate step so the option set is frozen and "
            "reviewed before any arm sees the data.")
    return json.loads(path.read_text())


def state_of(spec: Spec, row: dict) -> str:
    kind = spec.state["kind"]
    if kind == "text":
        return str(row[spec.state["field"]]).strip()
    if kind == "join":
        parts = []
        for name in spec.state["fields"]:
            value = str(row.get(name) or "").strip()
            if value:
                parts.append(f"{name.replace('_', ' ')}: {value}")
        return "\n\n".join(parts)
    if kind == "qa":
        passage = str(row[spec.state["passage"]]).strip()
        question = str(row[spec.state["question"]]).strip().rstrip("?")
        return f"{passage}\n\nQuestion: {question}?"
    if kind == "image":
        return spec.state["caption"]
    raise ValueError(f"unknown state kind {kind!r}")


def has_images(spec: Spec) -> bool:
    return spec.state["kind"] == "image"


def image_urls(spec: Spec, row: dict) -> list[str]:
    """The picture URLs a row carries, before download."""
    if not has_images(spec):
        return []
    value = row[spec.state["field"]]
    return [value["src"]] if isinstance(value, dict) else [v["src"] for v in value]


def present(spec: Spec, raw: str) -> str:
    """The name an arm sees for a raw label value; see ``Spec.rename``."""
    return spec.rename.get(raw, raw) if spec.rename else raw


def gold_of(spec: Spec, row: dict, options: list[str]) -> str:
    kind = spec.gold["kind"]
    if kind == "class_label":
        # ``options`` is already renamed, and renaming preserves order.
        return options[int(row[spec.gold["field"]])]
    if kind == "label_text":
        return present(spec, str(row[spec.gold["field"]]))
    if kind == "bool":
        return "true" if row[spec.gold["field"]] else "false"
    raise ValueError(f"unknown gold kind {kind!r}")


def load(name: str, n: int, seed: int = 0,
         max_chars: int = DEFAULT_MAX_CHARS,
         perturbation: str | None = None) -> list[Task]:
    spec = BY_NAME[name]
    meta = frozen(name)
    options: list[str] = meta["options"]
    #: original option name -> the name this run presents it under. Applied
    #: to the gold label as well as to the option list, and *after* the gold
    #: has been read, which is the only way that works for every gold kind:
    #: a ``class_label`` gold indexes the option list and so renames itself,
    #: while a ``label_text`` or ``bool`` gold returns a raw string that
    #: would otherwise no longer appear among the options at all.
    swap: dict[str, str] = {}
    if perturbation == "rename":
        from .perturb import load_rename
        mapping = load_rename(name)
        missing = [option for option in options if option not in mapping]
        if missing:
            raise ValueError(f"{name}: no renaming for {len(missing)} options "
                             f"({', '.join(missing[:3])}); re-generate it")
        swap = {option: mapping[option] for option in options}
    elif perturbation not in (None, "none"):
        raise ValueError(f"unknown perturbation {perturbation!r}")
    presented = [swap.get(option, option) for option in options]
    criteria: dict[str, str | None] = {option: None for option in presented}

    total = meta["rows"]
    chosen = sorted(random.Random(seed).sample(range(total), min(n, total)))
    rows = hub.pages(spec.hf, spec.config, spec.split, chosen)

    def images_for(index: int, row: dict, retried: bool = False) -> tuple[str, ...]:
        """Local paths for a row's pictures, refreshing stale links once.

        The asset URLs the rows endpoint hands out expire, so a page served
        from our own cache can carry links the server answers with 403. One
        forced re-fetch of the pages gets current ones; a second failure is
        a real error and is raised.
        """
        try:
            return tuple(str(hub.asset(url, f"{name}-{index}-{position}"))
                         for position, url in enumerate(image_urls(spec, row)))
        except urllib.error.HTTPError as error:
            if retried or error.code not in (403, 404, 410):
                raise
        rows.update(hub.pages(spec.hf, spec.config, spec.split, chosen,
                              refresh=True))
        return images_for(index, rows[index], retried=True)

    tasks = []
    for index in chosen:
        row = rows[index]
        state = state_of(spec, row)
        cut = len(state) > max_chars
        if cut:
            state = state[:max_chars]
        # Read against the original option set, then renamed, so every
        # gold kind travels the same path.
        gold = swap.get(g := gold_of(spec, row, options), g)
        if gold not in criteria:
            raise ValueError(
                f"{name} row {index}: gold {gold!r} is not in the frozen "
                "option set -- re-curate the dataset")
        images = images_for(index, row)
        tasks.append(Task(state=state, instructions=spec.instructions,
                          criteria=criteria, gold=gold, index=index,
                          truncated=cut, images=images))
    return tasks


#: Names the CLI accepts.
NAMES = [spec.name for spec in SPECS]
