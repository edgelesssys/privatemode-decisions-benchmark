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
import calendar
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

#: The fresh set (results/calibration/fresh/plan.md): published in this window.
WINDOW = ("2026-07-01", "2026-10-31")
ARXIV_CS = {"cs.AI": "Artificial Intelligence", "cs.CL": "Computation and Language",
            "cs.CR": "Cryptography and Security", "cs.CV": "Computer Vision and Pattern Recognition",
            "cs.DC": "Distributed, Parallel, and Cluster Computing",
            "cs.HC": "Human-Computer Interaction", "cs.IR": "Information Retrieval",
            "cs.LG": "Machine Learning", "cs.RO": "Robotics", "cs.SE": "Software Engineering"}
SE_SITES = {"math": "Mathematics", "physics": "Physics", "stats": "Statistics (Cross Validated)",
            "cs": "Computer Science", "biology": "Biology", "chemistry": "Chemistry",
            "history": "History", "philosophy": "Philosophy", "english": "English Language & Usage",
            "travel": "Travel"}
PR_LABELS = {"bug": "Bug fix", "enhancement": "Feature", "documentation": "Documentation",
             "dependencies": "Dependency update"}
#: Filled from the walked posts when the task is built: the subject categories
#: themselves come from the source (``build_rxiv``), then frozen with the task.
RXIV_OPTIONS: dict[str, list[str]] = {}

TASKS = {
    "fin_topic": ("Which topic is this financial news post about?", FIN_TOPICS),
    "fin_sentiment": ("What market sentiment does this financial news post express?", FIN_SENTIMENTS),
    "arxiv_field": ("Which field is this paper's primary subject?",
                    sorted(list(ARXIV_GROUPS.values()) + [PHYSICS])),
    "pubmed_study": ("What kind of study is this abstract from?", list(STUDY_TYPES)),
    "github_issue": ("What kind of issue is this?", list(ISSUE_LABELS.values())),
    # The fresh set.
    "biorxiv_subject": ("Which subject category is this preprint in?", RXIV_OPTIONS),
    "medrxiv_subject": ("Which subject category is this preprint in?", RXIV_OPTIONS),
    "arxiv_cs": ("Which computer science area is this paper's primary subject?", list(ARXIV_CS.values())),
    "stackexchange_site": ("On which Stack Exchange site was this question asked?", list(SE_SITES.values())),
    "github_pr": ("What kind of pull request is this?", list(PR_LABELS.values())),
}
#: The held-out sets: the first, and the fresh one that checks the fix of #4.
SETS = {"first": ["fin_topic", "fin_sentiment", "arxiv_field", "pubmed_study", "github_issue"],
        "fresh": ["biorxiv_subject", "arxiv_cs", "stackexchange_site", "github_pr"]}
#: Frozen, but dropped by the fresh plan's rule (fewer than 600 examples): 525.
DROPPED = ("medrxiv_subject",)


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


# -- the fresh set's sources --------------------------------------------------------

def days(stride: int = 1) -> list[str]:
    start, end = (time.mktime(time.strptime(d, "%Y-%m-%d")) for d in WINDOW)
    return [time.strftime("%Y-%m-%d", time.localtime(start + o * 86400))
            for o in range(0, int((end - start) / 86400) + 1, stride)]


def rxiv_day(server: str, day: str) -> list[dict]:
    """bioRxiv's or medRxiv's posts of one day, cached."""
    cached = CACHE / server / f"{day}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    out, cursor = [], 0
    while True:
        body = json.loads(get(f"https://api.biorxiv.org/details/{server}/{day}/{day}/{cursor}/json"))
        page = body.get("collection") or []
        out += [{k: r.get(k) for k in ("doi", "version", "date", "license", "category", "abstract")}
                for r in page]
        cursor += len(page)
        if not page or cursor >= int(body["messages"][0].get("total") or 0):
            break
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out))
    return out


def build_rxiv(name: str, server: str, stride: int, per: int, at_least: int,
               most: int = 1000) -> list[dict]:
    """First versions under CC BY posted in the window, the label the
    authors' subject category; categories with at least ``at_least`` posts,
    up to ``per`` each, at most ``most`` in all."""
    by: dict[str, list[dict]] = {}
    seen = set()
    for day in days(stride):
        for r in rxiv_day(server, day):
            if r["doi"] in seen or r["version"] != "1" or r["license"] != "cc_by" or not r["category"]:
                continue
            seen.add(r["doi"])
            text = clean(r["abstract"] or "")
            if len(text) > 200:
                category = r["category"].strip().capitalize()
                by.setdefault(category, []).append({"id": f"{server}:{r['doi']}", "label": category,
                                                    "text": text, "license": "cc by"})
    kept = sorted(c for c, rows in by.items() if len(rows) >= at_least)
    rng = random.Random(SEED)
    out = []
    for category in kept:
        rows = sorted(by[category], key=lambda r: r["id"])
        out += rng.sample(rows, min(per, len(rows)))
    if len(out) > most:
        out = sorted(rng.sample(out, most), key=lambda r: (kept.index(r["label"]), r["id"]))
    RXIV_OPTIONS[name] = sorted({r["label"] for r in out})
    return out


def build_arxiv_cs(per: int = 100) -> list[dict]:
    by: dict[str, list[dict]] = {c: [] for c in ARXIV_CS}
    seen = set()
    for day in days(2):
        for r in arxiv_records(day):
            if (r["id"] in seen or not WINDOW[0] <= r["created"] <= WINDOW[1]
                    or r["license"] not in OPEN_LICENSES or not r["categories"]
                    or r["categories"][0] not in ARXIV_CS):
                continue
            seen.add(r["id"])
            by[r["categories"][0]].append({"id": f"arxiv:{r['id']}", "label": ARXIV_CS[r["categories"][0]],
                                           "text": clean(r["abstract"]), "license": r["license"]})
        if all(len(v) >= 3 * per for v in by.values()):
            break
        time.sleep(3)
    rng = random.Random(SEED)
    out = []
    for category, rows in sorted(by.items()):
        rows.sort(key=lambda r: r["id"])
        out += rng.sample(rows, min(per, len(rows)))
    return out


def stackexchange(path: str, **params) -> dict:
    url = f"https://api.stackexchange.com/2.3/{path}?" + urllib.parse.urlencode(params)
    body = json.loads(subprocess.run(["curl", "-sf", "--compressed", "--max-time", "120", url],
                                     check=True, capture_output=True).stdout)
    if body.get("backoff"):
        time.sleep(body["backoff"])
    return body


def question_text(q: dict) -> str:
    return clean(f"{html.unescape(q['title'])}\n\n{q.get('body') or ''}")


def build_stackexchange(per: int = 100) -> list[dict]:
    """Questions asked in the window, one page of 100 per site and month."""
    rng = random.Random(SEED)
    months = [("2026-07-01", "2026-08-01"), ("2026-08-01", "2026-09-01"),
              ("2026-09-01", "2026-10-01"), ("2026-10-01", "2026-11-01")]
    epoch = lambda d: calendar.timegm(time.strptime(d, "%Y-%m-%d"))  # noqa: E731
    out = []
    for site, name in SE_SITES.items():
        candidates = []
        for start, end in months:
            body = stackexchange("questions", site=site, fromdate=epoch(start), todate=epoch(end) - 1,
                                 pagesize=100, sort="creation", order="asc", filter="withbody")
            candidates += [{"id": f"stackexchange:{site}/{q['question_id']}", "label": name,
                            "text": question_text(q)} for q in body["items"]]
            time.sleep(1)
        candidates = sorted((c for c in candidates if len(c["text"]) > 80), key=lambda r: r["id"])
        out += rng.sample(candidates, min(per, len(candidates)))
    return out


BOTS = ("dependabot", "renovate", "github-actions", "pre-commit-ci", "snyk-bot", "mergify")


def build_github_pr(per_label: int = 250) -> list[dict]:
    rng = random.Random(SEED)
    licenses: dict[str, str | None] = {}
    out = []
    for label, name in PR_LABELS.items():
        candidates = []
        for month in range(7, 11):
            q = (f"is:pr is:public label:{label} created:2026-{month:02d}-01..2026-{month:02d}-28 "
                 + " ".join(f"-author:app/{b}" for b in BOTS[:2]))
            for page in (1, 2, 3):
                result = gh("search/issues?" + urllib.parse.urlencode(
                    {"q": q, "per_page": 100, "page": page, "sort": "created"}))
                for pr in result.get("items", []):
                    labels = {l["name"].lower() for l in pr.get("labels", [])}
                    author = (pr.get("user") or {}).get("login", "").lower()
                    if (len(labels & set(PR_LABELS)) != 1 or (pr.get("user") or {}).get("type") == "Bot"
                            or any(b in author for b in BOTS)):
                        continue
                    repo = pr["repository_url"].split("repos/", 1)[1]
                    if repo not in licenses:
                        try:
                            licenses[repo] = ((gh(f"repos/{repo}").get("license") or {}).get("key"))
                        except subprocess.CalledProcessError:
                            licenses[repo] = None
                    if licenses[repo] not in PERMISSIVE:
                        continue
                    text = issue_text(pr)
                    if len(text) > 80:
                        candidates.append({"id": f"github:{repo}#{pr['number']}", "label": name,
                                           "text": text, "license": licenses[repo]})
                time.sleep(2.5)
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
    "biorxiv_subject": lambda: build_rxiv("biorxiv_subject", "biorxiv", stride=4, per=40, at_least=1),
    "medrxiv_subject": lambda: build_rxiv("medrxiv_subject", "medrxiv", stride=1, per=25, at_least=25),
    "arxiv_cs": build_arxiv_cs,
    "stackexchange_site": build_stackexchange,
    "github_pr": build_github_pr,
}


# -- freezing and loading -------------------------------------------------------------

def freeze(name: str, rows: list[dict]) -> None:
    question, options = TASKS[name]
    if options is RXIV_OPTIONS:
        options = RXIV_OPTIONS[name]
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
    """Issues by repository and number, through ``gh``. A deleted or hidden
    one (HTTP 404 or 410) is left out; any other failure (authentication,
    rate limit, server error) is retried, then raised rather than taken for
    a deletion."""
    def one(i: str) -> tuple[str, str | None]:
        repo, number = i.removeprefix("github:").split("#")
        for attempt in range(4):
            try:
                return i, issue_text(gh(f"repos/{repo}/issues/{number}"))
            except subprocess.CalledProcessError as error:
                if re.search(r"HTTP 4(04|10)\b", error.stderr or ""):
                    return i, None
                if attempt == 3:
                    raise RuntimeError(f"{i}: {(error.stderr or '').strip()[:200]}") from error
                time.sleep(5 * (attempt + 1))
    with ThreadPoolExecutor(8) as pool:
        return {i: text for i, text in pool.map(one, ids) if text is not None}


def refetch_stackexchange(name: str, ids: list[str]) -> dict[str, str]:
    """Questions by site and id, 100 per request; a deleted one is left out."""
    by_site: dict[str, list[str]] = {}
    for i in ids:
        site, qid = i.removeprefix("stackexchange:").split("/")
        by_site.setdefault(site, []).append(qid)
    out = {}
    for site, qids in by_site.items():
        for k in range(0, len(qids), 100):
            body = stackexchange("questions/" + ";".join(qids[k:k + 100]), site=site, pagesize=100,
                                 filter="withbody")
            out |= {f"stackexchange:{site}/{q['question_id']}": question_text(q) for q in body["items"]}
            time.sleep(1)
    return out


#: Tasks whose sources change after the freeze: ``check`` reports their
#: edited or deleted texts instead of failing (a run leaves them out), as
#: long as at least ``DRIFT_FLOOR`` of them still match.
DRIFTS = ("github_issue", "github_pr", "stackexchange_site")
DRIFT_FLOOR = 0.9

#: How the tasks the release doesn't carry are fetched again, by id.
REFETCH = {"fin_topic": refetch_hf, "fin_sentiment": refetch_hf, "github_issue": refetch_github,
           "github_pr": refetch_github, "stackexchange_site": refetch_stackexchange}


def fetch(url: str = TEXTS_URL) -> None:
    """Every task's texts into the cache, one file per task: the released
    ones from ``url``, the others by id from their sources."""
    with urllib.request.urlopen(url, timeout=300) as response:
        rows = [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in SETS["first"]:
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


def verify(name: str) -> int:
    """How many of a task's cached texts match their frozen hashes; raises
    on a mismatch, or for a task in ``DRIFTS`` on a missing cache or fewer
    than ``DRIFT_FLOOR`` of its texts."""
    if name not in DRIFTS:
        return len(load(name))
    if not (CACHE / f"{name}.jsonl").exists():
        raise SystemExit(f"{name}: no texts cached; run `python -m bench.holdout_data fetch`")
    total = len(json.loads((FROZEN / f"{name}.json").read_text())["examples"])
    matched = len(load(name, strict=False))
    if matched < DRIFT_FLOOR * total:
        raise SystemExit(f"{name}: only {matched} of {total} texts match their frozen hashes")
    if matched < total:
        print(f"{name}: {total - matched} edited or deleted since the freeze, as expected")
    return matched


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
        for name in SETS["first"]:
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
            print(name, verify(name), "examples verified")


if __name__ == "__main__":
    main()
