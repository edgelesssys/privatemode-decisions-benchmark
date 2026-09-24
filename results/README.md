# Results

`suite.md` is the aggregate, regenerated with:

```sh
python -m bench.aggregate results --write results/suite.md
```

`suite.json` indexes the runs the last suite produced.

## Where the raw answers are

In the release [`runs-2026-09-24`](https://github.com/edgelesssys/privatemode-decisions-benchmark/releases/tag/runs-2026-09-24),
not in the repository: one answer per line with its full probability
distribution comes to 320 MB across the 186 runs, 52 MB compressed. Every
run the suite and the latency probes took is in it, the controls and the
renaming included, so the report can be rebuilt without asking a single
question again:

```sh
curl -LO https://github.com/edgelesssys/privatemode-decisions-benchmark/releases/download/runs-2026-09-24/runs.tar.gz
curl -LO https://github.com/edgelesssys/privatemode-decisions-benchmark/releases/download/runs-2026-09-24/runs.tar.gz.sha256
shasum -a 256 -c runs.tar.gz.sha256        # or: sha256sum -c
tar -xzf runs.tar.gz                       # into results/<dataset>/
.venv/bin/python -m bench.aggregate results --write results/suite.md
git diff --exit-code results/suite.md      # no output: the same report
```

The files hold, per answer, the chosen option, the gold label, the
probabilities, the vendor's confidence, the token counts, the latency and
any error message; no state text, no keys. They are `.gitignore`d here.

Re-running instead of downloading asks the identical questions in the
identical order, because a run's identity pins the dataset, the content
hash of its frozen option set, the sample, the seed, the truncation budget,
the image resolution, every arm's model and every variant knob:

```sh
python -m bench.suite -n 1000 --replicates 2 --arms all --concurrency 16
python -m bench.suite -n 100  --replicates 1 --arms hosted --concurrency 1
```

What re-running cannot reproduce is the answers themselves: both hosted
arms flip up to 3.5% of their answers between identical runs at temperature
0. That is why every figure carries the spread between replicates, and why
nothing in the report rests on a difference smaller than it.

`pilot/report.md` is the first three-dataset comparison, taken before the
suite existed. Its raw files are in the release as well; they predate run
identities, and the aggregation skips them.
