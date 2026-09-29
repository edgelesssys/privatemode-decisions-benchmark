"""The held-out tasks of results/calibration/part-3/holdout-plan.md.

    python -m bench.holdout_data build [--only name,...]   # choose and freeze
    python -m bench.holdout_data check                     # refetch, verify hashes

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
themselves are cached under ``.cache/holdout/`` (not committed) and
refetched by ``check``/``load``, which fail if a text changed.
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
from pathlib import Path

from . import hub
from .datasets import Task

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "datasets" / "holdout"
CACHE = ROOT / ".cache" / "holdout"
SEED = 0
MAX_CHARS = 4000

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
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "privatemode-decisions-benchmark", "Accept": "*/*"}), timeout=120) as r:
                return r.read()
        except Exception:   # noqa: BLE001 - retried
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def clean(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]


# -- the sources --------------------------------------------------------------------

def build_hf(name: str, dataset: str, n: int = 1000) -> list[dict]:
    total = hub.head(dataset, "default", "validation")["num_rows_total"]
    indexes = sorted(random.Random(SEED).sample(range(total), n))
    rows = hub.pages(dataset, "default", "validation", indexes)
    options = TASKS[name][1]
    return [{"id": f"{dataset}/validation/{i}", "label": options[rows[i]["label"]],
             "text": rows[i]["text"]} for i in indexes]


def arxiv_records(day: str) -> list[dict]:
    """Papers whose metadata changed on ``day``, with the fields we need."""
    out, token = [], None
    while True:
        query = ({"verb": "ListRecords", "resumptionToken": token} if token else
                 {"verb": "ListRecords", "metadataPrefix": "arXiv", "from": day, "until": day})
        xml = get("https://oaipmh.arxiv.org/oai?" + urllib.parse.urlencode(query)).decode()
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


def load(name: str) -> list[Task]:
    """The frozen task, texts from the cache, each checked against its hash."""
    frozen = json.loads((FROZEN / f"{name}.json").read_text())
    texts = {}
    cached = CACHE / f"{name}.jsonl"
    if cached.exists():
        texts = {r["id"]: r["text"] for r in map(json.loads, cached.read_text().splitlines())}
    missing = [e for e in frozen["examples"] if e["id"] not in texts]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} texts not cached; run `python -m bench.holdout_data "
                         f"build --only {name}` on the same sources")
    tasks = []
    for e in frozen["examples"]:
        text = texts[e["id"]]
        if sha(text) != e["sha256"]:
            raise SystemExit(f"{name}: the text of {e['id']} changed since it was frozen")
        tasks.append(Task(state=text, instructions=frozen["question"],
                          criteria={o: None for o in frozen["options"]}, gold=e["label"],
                          index=e["index"]))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--only", default=",".join(TASKS))
    args = parser.parse_args()
    for name in args.only.split(","):
        if args.command == "build":
            freeze(name, BUILDERS[name]())
        else:
            print(name, len(load(name)), "examples verified")


if __name__ == "__main__":
    main()
