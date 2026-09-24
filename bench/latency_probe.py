"""Measure latency from wherever this runs, so two places can be compared.

The suite's latency numbers were taken from one machine in Germany, and they
include the network. How much of a decision is distance and how much is the
service itself cannot be read off one location. This probe repeats the
latency part of the benchmark somewhere else, with the same code, datasets,
sample, and settings, so the two results sit side by side.

It does three things:

* records the country it runs in, and nothing more precise,
* times trivial requests to both APIs over a kept-alive connection, which is
  the network plus each service's gateway, with no model involved,
* runs `bench.run` on a few datasets for both hosted arms, one request at a
  time, exactly as the suite's latency runs did.

It writes one JSON summary plus the raw run files to --out. Neither contains
an API key; the workflow checks that before uploading anything.

    python -m bench.latency_probe --out probe-de
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DATASETS = ("sst2", "ag_news", "trec_coarse", "banking77")

APIS = {
    "privatemode": ("api.privatemode.ai", False),
    "jev": ("api.typesafe.ai", True),
}


def location() -> dict:
    """The country of this machine's public address.

    Only the country: city and network operator would say where the person
    who ran the probe lives, and the comparison needs neither. Describe the
    connection (home broadband, a cloud runner) when reporting the result.
    """
    try:
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=10) as r:
            info = json.load(r)
    except OSError as error:
        return {"error": str(error)}
    return {"country": info.get("country")}


def trivial_requests(samples: int = 20) -> dict:
    """GET /v1/models over one kept-alive connection, after a warm-up.

    Privatemode is asked without a key: its gateway answers 401 straight
    away, which is the round trip to it and nothing else. Jev needs its key
    for this endpoint, so its figure includes whatever the endpoint does.
    """
    out = {}
    for arm, (host, needs_key) in APIS.items():
        headers = {}
        if needs_key:
            headers["Authorization"] = f"Bearer {os.environ['JEV_API_KEY']}"
        connection = http.client.HTTPSConnection(host, timeout=20)
        times = []
        for index in range(samples + 5):
            started = time.perf_counter()
            connection.request("GET", "/v1/models", headers=headers)
            response = connection.getresponse()
            response.read()
            if index >= 5:
                times.append((time.perf_counter() - started) * 1000)
        connection.close()
        times.sort()
        out[arm] = {
            "min_ms": round(times[0], 1),
            "median_ms": round(statistics.median(times), 1),
            "max_ms": round(times[-1], 1),
            "status": response.status,
        }
    return out


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def decisions(out: Path, n: int) -> dict:
    """Run the hosted arms on each dataset and summarize their latency."""
    runs = out / "runs"
    for dataset in DATASETS:
        subprocess.run(
            [sys.executable, "-m", "bench.run", "--dataset", dataset, "-n", str(n),
             "--arms", "hosted", "--concurrency", "1", "--warmup", "5",
             "--out", str(runs)],
            check=True,
        )
    per_dataset: dict[str, dict] = {}
    for path in sorted(runs.rglob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        dataset = rows[0].get("dataset", path.parent.name)
        for arm in APIS:
            latencies = [row["latency_s"] * 1000 for row in rows
                         if row.get("kind") == "row" and row.get("arm") == arm
                         and "latency_s" in row and not row.get("error")]
            if latencies:
                per_dataset.setdefault(dataset, {})[arm] = {
                    "n": len(latencies),
                    "p10_ms": round(percentile(latencies, 0.10), 1),
                    "p50_ms": round(statistics.median(latencies), 1),
                    "p95_ms": round(percentile(latencies, 0.95), 1),
                }
    summary = {}
    for arm in APIS:
        values = [d[arm] for d in per_dataset.values() if arm in d]
        if values:
            summary[arm] = {
                key: round(statistics.median(v[key] for v in values), 1)
                for key in ("p10_ms", "p50_ms", "p95_ms")
            }
    return {"per_dataset": per_dataset, "median_over_datasets": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("-n", type=int, default=150,
                        help="examples per dataset and arm (default: 150)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {
        "location": location(),
        "trivial_requests": trivial_requests(),
        "decisions": decisions(args.out, args.n),
        "datasets": list(DATASETS),
        "n": args.n,
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
