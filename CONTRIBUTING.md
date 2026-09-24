# Contributing

This benchmark was written by one of the parties it compares, so outside
eyes are the most useful thing it can get. Pull requests are welcome, and
especially these:

- a setting that serves Jev, Laya, or Privatemode better, with a run that
  shows it,
- a mistake in a dataset specification, a label mapping, or the
  aggregation,
- another dataset, or another system in the same shape: state and named
  options in, a choice and a probability per option out.

An issue is fine too, if you found something and would rather not fix it
yourself.

## Setup

See [Run it](README.md#run-it) in the README: the Privatemode proxy, a
virtual environment with the library and this package, and a `.env` with
your keys. Without keys you can still run the tests and the aggregation.

```sh
.venv/bin/python -m pytest -q tests      # what CI runs; no keys, no network
```

## A setting that serves a system better

Run the affected datasets with the setting and without it, on the same
sample (same `--seed` and `-n`), and put both results in the pull request.
Hosted arms flip up to 3.5% of their answers between identical runs, so
use two replicates (`--replicate 0` and `--replicate 1`) before reading
anything into a difference of one or two points.

```sh
.venv/bin/python -m bench.run --dataset banking77 -n 1000 --arms jev --concurrency 16
```

A setting for one system only is fine, but say so: the suite runs every
system at its defaults, and a knob one side gets and the other does not
changes what is being compared.

## A mistake in a dataset or the aggregation

Dataset specifications are in `bench/specs.py`, their frozen option sets
in `datasets/<name>.json`. Fix the spec, then re-freeze that one set:

```sh
.venv/bin/python -m bench.curate <name> --force
```

A changed option set invalidates the runs taken against it; the run files
are named after the spec's content hash, so old and new never mix.

For the aggregation, add a test in `tests/` that fails before your change,
then regenerate the report:

```sh
.venv/bin/python -m bench.aggregate results --write results/suite.md
```

## Another dataset

A `Spec` in `bench/specs.py` (where the data lives, how to build the state,
how to read the gold label, one instruction line) and one curation run.
Check the frozen file before running anything: the option names, the
majority share, and whether any class is missing from the split.

## Another system

An arm in `bench/adapters.py`: a class with a `name`, `ask(task) -> Answer`
and, if there are tasks it cannot take, `unsupported(task)` returning the
reason. Register it in `build_arms` and `PRODUCTS` in `bench/run.py`, and
add its prices to `bench/pricing.py` with the page they came from.

It gets the same state string, the same option names in the same order and
the same instruction line as every other arm. Cost comes from the system's
own reported usage, never from a token count of ours.

## In the pull request

- Which command produced the numbers, and the changed lines of
  `results/suite.md`.
- Never commit a `.env` or an API key. The raw run files are not checked in
  (`results/*/*.jsonl` is ignored); start from the ones in the release (see
  [`results/README.md`](results/README.md)), add yours, and say in the pull
  request where they can be downloaded.
- Keep claims to what was measured: this repository reports accuracy,
  calibration, latency, and billed cost, and does not weigh them against
  each other.
