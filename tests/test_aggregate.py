"""The aggregation decides what the suite says, so it is worth pinning."""

from bench.aggregate import TIE, coverage_at


def test_coverage_at_finds_the_largest_answerable_fraction():
    # Four right, one wrong, and the wrong one is the least confident.
    scored = {"correct": [True, True, True, True, False],
              "confidence": [0.9, 0.8, 0.7, 0.6, 0.1]}
    assert coverage_at(scored, 0.95) == 0.8
    # At a target it can meet everywhere, it takes everything.
    assert coverage_at({"correct": [True, True],
                        "confidence": [0.9, 0.8]}, 0.95) == 1.0


def test_coverage_at_is_zero_when_the_target_is_never_reached():
    scored = {"correct": [False, False, True], "confidence": [0.9, 0.8, 0.1]}
    assert coverage_at(scored, 0.95) == 0.0


def test_coverage_at_ignores_a_lucky_prefix_it_cannot_hold():
    # Confident and right at the very top, but the target is met again
    # further down: the largest qualifying coverage wins, not the first.
    scored = {"correct": [True, False, True, True, True, True, True, True,
                          True, True, True, True, True, True, True, True,
                          True, True, True, True, True],
              "confidence": [1.0 - i / 100 for i in range(21)]}
    assert coverage_at(scored, 0.95) == 1.0


def test_tie_threshold_matches_the_measured_noise_floor():
    # Pinned on purpose: the pilot measured run-to-run accuracy moves of up
    # to one point, so a "win" must be larger than that.
    assert TIE == 0.01


def test_consensus_separates_label_noise_from_capability():
    from bench.aggregate import consensus

    def row(index, choice, gold):
        return {"index": index, "choice": choice, "gold": gold,
                "correct": choice == gold}

    rows = {
        # 0: everyone right. 1: everyone agrees and is wrong -- the gold
        # label is the suspect. 2: they disagree, one is right.
        "a": [row(0, "x", "x"), row(1, "y", "z"), row(2, "p", "p")],
        "b": [row(0, "x", "x"), row(1, "y", "z"), row(2, "q", "p")],
    }
    result = consensus(rows)
    assert result["agreed"] == 2 / 3           # indexes 0 and 1
    assert result["consensus_wrong"] == 1 / 3  # index 1 only
    assert result["ceiling"] == 2 / 3          # 0 and 2 are winnable


def test_consensus_survives_one_arm_that_disagrees_with_everything():
    from bench.aggregate import consensus

    def row(index, choice, gold):
        return {"index": index, "choice": choice, "gold": gold,
                "correct": choice == gold}

    # Three arms agree on a wrong answer; a fourth, much weaker, is off on
    # its own. Requiring unanimity would score this as no evidence at all,
    # which is the failure mode the modal rule exists to avoid.
    rows = {name: [row(0, "y", "z")] for name in ("a", "b", "c")}
    rows["weak"] = [row(0, "w", "z")]
    assert consensus(rows)["consensus_wrong"] == 1.0


def test_consensus_needs_at_least_two_arms():
    from bench.aggregate import consensus
    assert consensus({"a": [{"index": 0, "choice": "x", "gold": "x",
                             "correct": True}]}) == {}


def _fake_rows(spec, options):
    """Rows shaped like the spec expects, so load() runs without the network.

    The gold cycles through the frozen option set, which is the part the
    rename has to carry across.
    """
    def row(index):
        out = {}
        kind = spec.state["kind"]
        if kind == "text":
            out[spec.state["field"]] = f"example {index}"
        elif kind == "join":
            for name in spec.state["fields"]:
                out[name] = f"{name} {index}"
        elif kind == "qa":
            out[spec.state["passage"]] = f"passage {index}"
            out[spec.state["question"]] = f"question {index}"
        gold = spec.gold
        if gold["kind"] == "class_label":
            out[gold["field"]] = index % len(options)
        elif gold["kind"] == "label_text":
            out[gold["field"]] = options[index % len(options)]
        elif gold["kind"] == "bool":
            out[gold["field"]] = index % 2 == 0
        return out
    return row


def test_rename_perturbation_moves_the_gold_label_too(monkeypatch):
    """The bug that failed eleven datasets in the memorisation phase.

    Renaming replaced the option list but not the gold, so a dataset whose
    gold is a label *string* rather than a class index answered with a name
    that no longer appeared among the options. A dataset whose gold is an
    index renamed itself by accident, which is why exactly the label_text
    and bool sets failed and the class_label ones looked fine. One of each
    kind is checked here.
    """
    from bench import hub
    from bench.datasets import frozen, load
    from bench.specs import BY_NAME

    for name in ("ag_news", "trec_coarse", "boolq"):  # class_label, label_text, bool
        spec = BY_NAME[name]
        make = _fake_rows(spec, frozen(name)["options"])
        monkeypatch.setattr(hub, "pages", lambda *a, _make=make, **k:
                            {i: _make(i) for i in a[3]})
        plain = load(name, 5, seed=0)
        renamed = load(name, 5, seed=0, perturbation="rename")
        assert [t.index for t in plain] == [t.index for t in renamed]
        for before, after in zip(plain, renamed):
            assert after.gold in after.criteria, (name, after.gold)
            assert after.gold != before.gold, (name, before.gold)
            assert set(after.criteria) != set(before.criteria)


def test_option_bands_compare_every_arm_on_the_same_sets():
    from bench.aggregate import option_bands

    def acc(v):
        return {"accuracy": v}

    averaged = {
        # Both products answered: counts for both.
        "shared": {"jev": acc(0.70), "pm": acc(0.72)},
        # Only Jev could answer: must not lift Jev's band mean.
        "jev_only": {"jev": acc(0.95)},
    }
    options = {"shared": 100, "jev_only": 151}
    bands = option_bands(averaged, list(averaged), options.get, ["jev", "pm"], ["jev", "pm"])
    assert len(bands) == 1
    label, members, means = bands[0]
    assert label == "81+" and members == ["shared"]
    assert means == {"jev": 0.70, "pm": 0.72}


def test_option_bands_leave_an_unfinished_control_blank():
    from bench.aggregate import option_bands

    averaged = {"a": {"jev": {"accuracy": 0.8}, "cot": {"accuracy": 0.9}},
                "b": {"jev": {"accuracy": 0.6}}}
    (_, members, means), = option_bands(averaged, ["a", "b"], lambda d: 2, ["jev"], ["jev", "cot"])
    assert members == ["a", "b"]
    assert means["cot"] is None and abs(means["jev"] - 0.7) < 1e-9
