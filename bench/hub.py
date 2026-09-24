"""The Hugging Face datasets-server, as much of it as this benchmark needs.

No ``datasets`` dependency and no local parquet: the rows endpoint serves
any public dataset as JSON, 100 rows per request, and the ``features`` block
carries ``ClassLabel`` names where a dataset has them. Pages are cached on
disk under ``.cache/`` keyed by dataset, split and offset, so curation and
every later run read the same bytes.

Two things learned the hard way and encoded here. Script-backed datasets are
no longer served at all, so the canonical home of a dataset is often dead
and a mirror has to be used -- ``PolyAI/banking77`` returns 404 while
``legacy-datasets/banking77`` works. And the anonymous rate limit is
CloudFront's, not the API's: a few dozen pages earns a 429 that lasts
minutes. ``HF_TOKEN`` raises it and is effectively required at suite scale.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # the server's maximum
RETRY_STATUS = (429, 500, 502, 503, 504)


def cache_dir() -> Path:
    return Path(os.environ.get("BENCH_CACHE")
                or Path(__file__).resolve().parent.parent / ".cache")


def get(url: str, attempts: int = 6) -> dict:
    """GET with backoff measured in minutes, because the limit is that coarse."""
    headers = {"User-Agent": "privatemode-system-one-benchmark"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUS or attempt == attempts:
                raise
        except urllib.error.URLError:
            if attempt == attempts:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def page(hf: str, config: str, split: str, offset: int, length: int,
         refresh: bool = False) -> dict:
    key = f"{hf}-{config}-{split}-{offset}-{length}".replace("/", "_")
    path = cache_dir() / f"{key}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    query = urllib.parse.urlencode({"dataset": hf, "config": config,
                                    "split": split, "offset": offset,
                                    "length": length})
    body = get(f"{ROWS_URL}?{query}")
    if "rows" not in body:
        raise RuntimeError(f"no rows for {hf}/{config}/{split}: "
                           f"{json.dumps(body)[:300]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return body


def head(hf: str, config: str, split: str) -> dict:
    return page(hf, config, split, 0, 1)


def pages(hf: str, config: str, split: str, indexes: list[int],
          refresh: bool = False) -> dict[int, dict]:
    """Rows for ``indexes``, fetching whole pages so the cache is reusable.

    Whole pages rather than exact rows on purpose: a second sample with a
    different seed or size mostly hits pages already on disk, and curation
    and the runs then read identical bytes.
    """
    total = head(hf, config, split)["num_rows_total"]
    rows: dict[int, dict] = {}
    for number in sorted({index // PAGE for index in indexes}):
        offset = number * PAGE
        body = page(hf, config, split, offset, min(PAGE, total - offset),
                    refresh=refresh)
        for row in body["rows"]:
            rows[row["row_idx"]] = row["row"]
    return rows


def scan(hf: str, config: str, split: str, limit: int | None = None):
    """Every row in the split, page by page. Curation only; runs sample."""
    total = head(hf, config, split)["num_rows_total"]
    stop = total if limit is None else min(total, limit)
    offset = 0
    while offset < stop:
        length = min(PAGE, stop - offset)
        for row in page(hf, config, split, offset, length)["rows"]:
            yield row["row_idx"], row["row"]
        offset += length


def class_label_names(features: list[dict], field: str) -> list[str] | None:
    for feature in features:
        if feature["name"] == field and "names" in feature["type"]:
            return list(feature["type"]["names"])
    return None


def fetch_bytes(url: str, attempts: int = 6) -> bytes:
    headers = {"User-Agent": "privatemode-system-one-benchmark"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUS or attempt == attempts:
                raise
        except urllib.error.URLError:
            if attempt == attempts:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def asset(url: str, key: str) -> Path:
    """An image the datasets-server serves, cached on disk under ``key``.

    The rows endpoint hands images back as URLs under ``cached-assets``, and
    those expire: a page JSON read from our own cache can carry links the
    server no longer honours, which returns 403 rather than the picture.
    Callers recover by re-fetching the page for a fresh URL.

    ``key`` is what the file is stored as, and it is the caller's stable
    identifier -- dataset, split and row -- not a hash of the URL. Keying by
    URL would mean re-downloading every image each time the links rotate,
    and would leave the old copies behind under names nothing refers to.

    Downloading at all, rather than passing the URL to the model server, is
    what makes ``--image-max-side`` govern what the model sees:
    ``to_data_url`` forwards an http URL untouched.
    """
    ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
    ext = ext if len(ext) <= 4 and ext.isalnum() else "bin"
    path = cache_dir() / "assets" / f"{key}.{ext}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch_bytes(url))
    return path
