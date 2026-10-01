"""The held-out tasks of results/calibration/part-3/holdout-plan.md.

    python -m bench.holdout_data fetch                     # the texts, from the release
    python -m bench.holdout_data check                     # verify them against the hashes
    python -m bench.holdout_data build --only name --force # choose anew (replaces a task)

Five tasks nobody has looked at, built without asking any model:

* ``fin_topic``, ``fin_sentiment``: the validation splits of
  zeroshot/twitter-financial-news-topic and -sentiment (MIT), 1,000 random
  rows each (seed 0).
* ``arxiv_field``: arXiv papers first submitted in 2026 under CC BY 4.0 or
  CC0, the abstract; the label is the top-level group of the primary
  category (8 groups), up to 125 per group.
* ``pubmed_study``: Europe PMC open-access papers first published in
  January–June 2026 under CC BY, the abstract; the label is the study type
  from the indexed publication types, kept only where exactly one of five
  applies, 200 per type.
* ``github_issue``: GitHub issues opened in January–June 2026 in public
  repositories under MIT, Apache-2.0 or BSD licences, the title and body
  without template headings; the label is the one of four default labels
  (bug, enhancement, documentation, question) a maintainer put on it, 250
  per label.

``datasets/holdout/<task>.json`` freezes each task: its question, options,
and per example the source id, label and a SHA-256 of the text. The texts
are not in the repository. ``fetch`` puts them into ``.cache/holdout/``:
the arXiv and PubMed abstracts (CC BY or CC0, ``SHIPPED``) from the
release ``calibration-2026-09-26`` (``TEXTS_URL``), with the licence and
source of each; the tweets and GitHub issues, which aren't ours to
republish, by id from their sources. ``check`` and ``load`` fail on a
missing text or one whose hash differs; ``load(strict=False)``, which
``bench.holdout run`` uses, leaves such examples out and says how many
(an issue edited since, for one). ``build`` chose the examples once,
from live sources (arXiv, Europe PMC and GitHub searches return other
results later), so it refuses to replace a frozen task without ``--force``:
rebuilding makes a different test, not a reproduction. The two Hugging
Face datasets were read at ``HF_REVISIONS``, their last commits before the
build.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import hub
from .datasets import Task

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "datasets" / "holdout"
CACHE = ROOT / ".cache" / "holdout"
SEED = 0
MAX_CHARS = 4000
#: The tasks whose texts the release carries: open-access abstracts.
SHIPPED = ("arxiv_field", "pubmed_study")
#: The released texts, one JSON line each: task, id, licence, source, text.
TEXTS_URL = ("https://github.com/edgelesssys/privatemode-decisions-benchmark/releases/download/"
             "calibration-2026-09-26/holdout-texts.jsonl")
#: The Hugging Face datasets' commits at build time (unchanged since
#: 2024-02-23). The datasets-server serves only the latest revision, so
#: ``build_hf`` checks that it still is this one.
HF_REVISIONS = {"zeroshot/twitter-financial-news-topic": "acbc8af2a35ccf0916124efcbe9e6cf25f191012",
                "zeroshot/twitter-financial-news-sentiment": "ccbe24de388e287beb92dd393a335c376b350ac3"}

FIN_TOPICS = ["Analyst Update", "Fed | Central Banks", "Company | Product News",
              "Treasuries | Corporate Debt", "Dividend", "Earnings", "Energy | Oil", "Financials",
              "Currencies", "General News | Opinion", "Gold | Metals | Materials", "IPO",
              "Legal | Regulation", "M&A | Investments", "Macro", "Markets", "Politics",
              "Personnel Change", "Stock Commentary", "Stock Movement"]
FIN_SENTIMENTS = ["Bearish", "Bullish", "Neutral"]

#: arXiv's top-level groups (https://arxiv.org/category_taxonomy): every archive
#: that isn't one of the named groups is physics.
ARXIV_GROUPS = {"cs": "Computer Science", "econ": "Economics",
                "eess": "Electrical Engineering and Systems Science", "math": "Mathematics",
                "q-bio": "Quantitative Biology", "q-fin": "Quantitative Finance",
                "stat": "Statistics"}
PHYSICS = "Physics"
OPEN_LICENSES = ("http://creativecommons.org/licenses/by/4.0/",
                 "http://creativecommons.org/publicdomain/zero/1.0/")

#: Study types from Europe PMC's publication types, each a query and the
#: types that mark it; a paper counts only if exactly one type applies.
STUDY_TYPES = {
    "Randomized controlled trial": ("randomized controlled trial", {"Randomized Controlled Trial"}),
    "Systematic review or meta-analysis": ("systematic review", {"Systematic Review", "Meta-Analysis",
                                                                 "systematic-review"}),
    "Observational study": ("observational study", {"Observational Study"}),
    "Case report": ("case reports", {"Case Reports", "case-report"}),
    "Narrative review": ("review", {"Review", "review-article"}),
}
ISSUE_LABELS = {"bug": "Bug report", "enhancement": "Feature request",
                "documentation": "Documentation", "question": "Question"}
PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"}

TASKS = {
    "fin_topic": ("Which topic is this financial news post about?", FIN_TOPICS),
    "fin_sentiment": ("What market sentiment does this financial news post express?", FIN_SENTIMENTS),
    "arxiv_field": ("Which field is this paper's primary subject?",
                    sorted(list(ARXIV_GROUPS.values()) + [PHYSICS])),
    "pubmed_study": ("What kind of study is this abstract from?", list(STUDY_TYPES)),
    "github_issue": ("What kind of issue is this?", list(ISSUE_LABELS.values())),
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def get(url: str, attempts: int = 5) -> bytes:
    """curl rather than urllib: arXiv's CDN refuses urllib's requests."""
    for attempt in range(attempts):
        done = subprocess.run(["curl", "-sfL", "--max-time", "120", "-A",
                               "privatemode-decisions-benchmark", url], capture_output=True)
        if done.returncode == 0:
            return done.stdout
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{url}: curl exit {done.returncode}")


def clean(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]


# -- the sources --------------------------------------------------------------------

def build_hf(name: str, dataset: str, n: int = 1000) -> list[dict]:
    current = json.loads(get(f"https://huggingface.co/api/datasets/{dataset}"))["sha"]
    if current != HF_REVISIONS[dataset]:
        raise SystemExit(f"{dataset} is at {current}, not the pinned {HF_REVISIONS[dataset]}")
    total = hub.head(dataset, "default", "validation")["num_rows_total"]
    indexes = sorted(random.Random(SEED).sample(range(total), n))
    rows = hub.pages(dataset, "default", "validation", indexes)
    options = TASKS[name][1]
    return [{"id": f"{dataset}/validation/{i}", "label": options[rows[i]["label"]],
             "text": rows[i]["text"]} for i in indexes]


def arxiv_records(day: str) -> list[dict]:
    """Papers whose metadata changed on ``day``, with the fields we need.
    Cached per day: arXiv's endpoint refuses requests now and then."""
    cached = CACHE / "arxiv" / f"{day}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    out = arxiv_fetch(day)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out))
    return out


def arxiv_fetch(day: str) -> list[dict]:
    out, token = [], None
    while True:
        query = ({"verb": "ListRecords", "resumptionToken": token} if token else
                 {"verb": "ListRecords", "metadataPrefix": "arXiv", "from": day, "until": day})
        xml = get("https://oaipmh.arxiv.org/oai?" + urllib.parse.urlencode(query), attempts=10).decode()
        for record in xml.split("<record>")[1:]:
            field = lambda tag: (re.findall(rf"<{tag}>(.*?)</{tag}>", record, re.S) or [""])[0]
            out.append({"id": field("id"), "created": field("created"),
                        "categories": field("categories").split(), "license": field("license"),
                        "abstract": field("abstract")})
        token = (re.findall(r"<resumptionToken[^>]*>([^<]+)<", xml) or [None])[0]
        if not token:
            return out
        time.sleep(3)


def arxiv_group(category: str) -> str:
    archive = category.split(".")[0]
    return ARXIV_GROUPS.get(archive, PHYSICS)


def build_arxiv(per_group: int = 125) -> list[dict]:
    """New 2026 papers under an open licence, walking the days of 2026 until
    every group has ``per_group`` or the days run out."""
    by_group: dict[str, list[dict]] = {g: [] for g in TASKS["arxiv_field"][1]}
    seen = set()
    day = time.strptime("2026-01-05", "%Y-%m-%d")
    for offset in range(0, 240, 3):
        date = time.strftime("%Y-%m-%d", time.localtime(time.mktime(day) + offset * 86400))
        for r in arxiv_records(date):
            if (r["id"] in seen or not r["created"].startswith("2026") or r["license"] not in OPEN_LICENSES
                    or not r["categories"]):
                continue
            seen.add(r["id"])
            group = arxiv_group(r["categories"][0])
            by_group[group].append({"id": f"arxiv:{r['id']}", "label": group, "text": clean(r["abstract"]),
                                    "license": r["license"]})
        if all(len(v) >= per_group for v in by_group.values()):
            break
        time.sleep(3)
    rng = random.Random(SEED)
    out = []
    for group, rows in sorted(by_group.items()):
        rows.sort(key=lambda r: r["id"])
        out += rng.sample(rows, min(per_group, len(rows)))
    return out


def europe_pmc(query: str, cursor: str = "*") -> dict:
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
        {"query": query, "format": "json", "pageSize": 1000, "resultType": "core", "cursorMark": cursor}))
    return json.loads(get(url))


def build_pubmed(per_type: int = 200) -> list[dict]:
    rng = random.Random(SEED)
    marks = {name: types for name, (_, types) in STUDY_TYPES.items()}
    out = []
    for name, (pub_type, _) in STUDY_TYPES.items():
        query = (f'FIRST_PDATE:[2026-01-01 TO 2026-06-30] AND OPEN_ACCESS:y AND LICENSE:"cc by" '
                 f'AND HAS_ABSTRACT:y AND PUB_TYPE:"{pub_type}"')
        candidates, cursor = [], "*"
        for _ in range(5):
            body = europe_pmc(query, cursor)
            for r in body.get("resultList", {}).get("result", []):
                types = set((r.get("pubTypeList") or {}).get("pubType") or [])
                matching = [n for n, m in marks.items() if types & m]
                # Exactly one study type; a systematic review is also indexed as a review.
                allowed = {name, "Narrative review"} if name == "Systematic review or meta-analysis" else {name}
                if name in matching and set(matching) <= allowed:
                    text = clean(r.get("abstractText") or "")
                    if len(text) > 200 and (r.get("license") or "").lower() == "cc by":
                        candidates.append({"id": f"pmc:{r.get('pmcid') or r.get('id')}", "label": name,
                                           "text": text, "license": "cc by"})
            cursor = body.get("nextCursorMark")
            if not cursor or len(candidates) >= 4 * per_type:
                break
        candidates.sort(key=lambda r: r["id"])
        out += rng.sample(candidates, min(per_type, len(candidates)))
    return out


def gh(path: str) -> dict:
    return json.loads(subprocess.run(["gh", "api", path], check=True, capture_output=True,
                                     text=True).stdout)


def issue_text(issue: dict) -> str:
    body = issue.get("body") or ""
    # Template headings and checklists say which template was used, i.e. the label.
    body = "\n".join(line for line in body.splitlines()
                     if not re.match(r"^\s*(#+ |\*\*[^*]+\*\*\s*$|- \[[ xX]\])", line))
    title = re.sub(r"^\s*[\[(](bug|feature|feat|docs?|question|enhancement|request)[^\])]*[\])]\s*:?\s*",
                   "", issue["title"], flags=re.I)
    return clean(f"{title}\n\n{body}")


def build_github(per_label: int = 250) -> list[dict]:
    rng = random.Random(SEED)
    licenses: dict[str, str | None] = {}
    out = []
    for label, name in ISSUE_LABELS.items():
        candidates = []
        for month in range(1, 7):
            q = f'is:issue is:public label:{label} created:2026-{month:02d}-01..2026-{month:02d}-28'
            for page in (1, 2):
                result = gh("search/issues?" + urllib.parse.urlencode(
                    {"q": q, "per_page": 100, "page": page, "sort": "created"}))
                for issue in result.get("items", []):
                    labels = {l["name"].lower() for l in issue.get("labels", [])}
                    if len(labels & set(ISSUE_LABELS)) != 1 or issue.get("pull_request"):
                        continue
                    repo = issue["repository_url"].split("repos/", 1)[1]
                    if repo not in licenses:
                        try:
                            licenses[repo] = ((gh(f"repos/{repo}").get("license") or {}).get("key"))
                        except subprocess.CalledProcessError:
                            licenses[repo] = None
                    if licenses[repo] not in PERMISSIVE:
                        continue
                    text = issue_text(issue)
                    if len(text) > 80:
                        candidates.append({"id": f"github:{repo}#{issue['number']}", "label": name,
                                           "text": text, "license": licenses[repo]})
                time.sleep(2.5)       # the search API allows 30 requests a minute
        # One repository mustn't dominate a label.
        by_repo: dict[str, list[dict]] = {}
        for c in sorted(candidates, key=lambda r: r["id"]):
            by_repo.setdefault(c["id"].split("#")[0], []).append(c)
        capped = [c for rows in by_repo.values() for c in rows[:5]]
        out += rng.sample(capped, min(per_label, len(capped)))
    return out


BUILDERS = {
    "fin_topic": lambda: build_hf("fin_topic", "zeroshot/twitter-financial-news-topic"),
    "fin_sentiment": lambda: build_hf("fin_sentiment", "zeroshot/twitter-financial-news-sentiment"),
    "arxiv_field": build_arxiv,
    "pubmed_study": build_pubmed,
    "github_issue": build_github,
}


# -- freezing and loading -------------------------------------------------------------

def freeze(name: str, rows: list[dict]) -> None:
    question, options = TASKS[name]
    FROZEN.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    counts = {o: sum(r["label"] == o for r in rows) for o in options}
    (FROZEN / f"{name}.json").write_text(json.dumps({
        "task": name, "question": question, "options": options, "counts": counts,
        "built": time.strftime("%Y-%m-%d"),
        "examples": [{"index": i, "id": r["id"], "label": r["label"], "sha256": sha(r["text"]),
                      **({"license": r["license"]} if r.get("license") else {})}
                     for i, r in enumerate(rows)]}, indent=1) + "\n")
    (CACHE / f"{name}.jsonl").write_text("".join(
        json.dumps({"id": r["id"], "text": r["text"]}) + "\n" for r in rows))
    print(f"{name}: {len(rows)} examples, {counts}")


def load(name: str, strict: bool = True) -> list[Task]:
    """The frozen task, texts from the cache, each checked against its hash.
    Not ``strict``: examples without a matching text are left out."""
    frozen = json.loads((FROZEN / f"{name}.json").read_text())
    texts = {}
    cached = CACHE / f"{name}.jsonl"
    if cached.exists():
        texts = {r["id"]: r["text"] for r in map(json.loads, cached.read_text().splitlines())}
    missing = [e for e in frozen["examples"] if e["id"] not in texts]
    changed = [e for e in frozen["examples"]
               if e["id"] in texts and sha(texts[e["id"]]) != e["sha256"]]
    if strict and missing:
        raise SystemExit(f"{name}: {len(missing)} texts not cached; run `python -m bench.holdout_data "
                         f"fetch`")
    if strict and changed:
        raise SystemExit(f"{name}: the text of {changed[0]['id']} (and {len(changed) - 1} more) "
                         "changed since it was frozen")
    if missing or changed:
        print(f"{name}: {len(missing)} texts missing and {len(changed)} changed since the "
              "freeze; left out")
    skip = {e["id"] for e in missing + changed}
    tasks = []
    for e in frozen["examples"]:
        if e["id"] in skip:
            continue
        text = texts[e["id"]]
        tasks.append(Task(state=text, instructions=frozen["question"],
                          criteria={o: None for o in frozen["options"]}, gold=e["label"],
                          index=e["index"]))
    return tasks


def refetch_hf(name: str, ids: list[str]) -> dict[str, str]:
    """Tweets by row index, from the pinned revision of their dataset."""
    dataset = ids[0].rsplit("/validation/", 1)[0]
    current = json.loads(get(f"https://huggingface.co/api/datasets/{dataset}"))["sha"]
    if current != HF_REVISIONS[dataset]:
        raise SystemExit(f"{dataset} is at {current}, not the pinned {HF_REVISIONS[dataset]}")
    indexes = {i: int(i.rsplit("/", 1)[1]) for i in ids}
    rows = hub.pages(dataset, "default", "validation", sorted(indexes.values()))
    return {i: rows[n]["text"] for i, n in indexes.items()}


def refetch_github(name: str, ids: list[str]) -> dict[str, str]:
    """Issues by repository and number, through ``gh``; a deleted one is left out."""
    def one(i: str) -> tuple[str, str | None]:
        repo, number = i.removeprefix("github:").split("#")
        try:
            return i, issue_text(gh(f"repos/{repo}/issues/{number}"))
        except subprocess.CalledProcessError:
            return i, None
    with ThreadPoolExecutor(8) as pool:
        return {i: text for i, text in pool.map(one, ids) if text is not None}


#: How the tasks the release doesn't carry are fetched again, by id.
REFETCH = {"fin_topic": refetch_hf, "fin_sentiment": refetch_hf, "github_issue": refetch_github}


def fetch(url: str = TEXTS_URL) -> None:
    """Every task's texts into the cache, one file per task: the released
    ones from ``url``, the others by id from their sources."""
    with urllib.request.urlopen(url, timeout=300) as response:
        rows = [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in TASKS:
        if name in REFETCH:
            ids = [e["id"] for e in json.loads((FROZEN / f"{name}.json").read_text())["examples"]]
            texts = REFETCH[name](name, ids)
        else:
            texts = {r["id"]: r["text"] for r in rows if r["task"] == name}
        (CACHE / f"{name}.jsonl").write_text("".join(
            json.dumps({"id": i, "text": t}) + "\n" for i, t in texts.items()))


#: The licence names behind the URLs and words the sources report.
LICENCES = {"http://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
            "http://creativecommons.org/publicdomain/zero/1.0/": "CC0 1.0", "cc by": "CC BY"}


def source(i: str) -> str:
    kind, ident = i.split(":", 1)
    return {"arxiv": f"https://arxiv.org/abs/{ident}",
            "pmc": f"https://europepmc.org/article/PMC/{ident}"}[kind]


def export(path: Path) -> None:
    """The released tasks' cached texts as one file, each checked against its
    hash, with its licence and source for attribution."""
    lines = []
    for name in SHIPPED:
        frozen = json.loads((FROZEN / f"{name}.json").read_text())
        texts = {t.index: t.state for t in load(name)}
        lines += [json.dumps({"task": name, "id": e["id"], "license": LICENCES[e["license"]],
                              "source": source(e["id"]), "text": texts[e["index"]]}) + "\n"
                  for e in frozen["examples"]]
    path.write_text("".join(lines))
    print(f"wrote {len(lines)} texts to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("fetch", "check", "build", "export"))
    parser.add_argument("--only", default=",".join(TASKS))
    parser.add_argument("--force", action="store_true",
                        help="build: replace a frozen task (a new test, not a reproduction)")
    parser.add_argument("--out", type=Path, default=Path("holdout-texts.jsonl"), help="export: the file")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch()
        for name in TASKS:
            print(name, len(load(name, strict=False)), "examples match their frozen hashes")
        return
    if args.command == "export":
        export(args.out)
        return
    for name in args.only.split(","):
        if args.command == "build":
            if (FROZEN / f"{name}.json").exists() and not args.force:
                raise SystemExit(f"{name} is frozen; `build --force` would replace it with a new "
                                 "sample. To reproduce, `fetch` the released texts.")
            freeze(name, BUILDERS[name]())
        else:
            print(name, len(load(name)), "examples verified")


if __name__ == "__main__":
    main()
