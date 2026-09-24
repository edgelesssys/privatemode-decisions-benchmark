"""The dataset registry: what is in the suite, and why each one is there.

Declarative on purpose. A spec says where the rows live and how to turn one
into a state and a gold option; it does not say what the option *set* is.
That is discovered once by ``bench.curate`` and frozen into
``datasets/<name>.json``, so the option set a run sees is pinned, auditable
and identical across machines rather than rediscovered at runtime.

Two of the axes are filled by **ladders**: the same examples labelled at two
granularities. TREC carries a 6-way coarse and a 50-way fine label for every
question; MASSIVE carries an 18-way scenario and a 60-way intent for every
utterance, in English and German. Every other option-count comparison in the
suite confounds option count with task, domain and text at once -- a ladder
changes the option count and nothing else, which turns the pilot's headline
finding from a correlation across datasets into a controlled comparison
within one.

``tier`` records how likely a set is to be in the arms' training data, which
is the main threat to a benchmark of models trained for this task family.
It is a judgement, recorded so it can be argued with: ``seen`` for the
standard sets that appear in every mix, ``likely`` for well-known ones,
``unclear`` for the rest. Nothing here is safely held out; that is what the
perturbation controls are for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Spec:
    name: str
    hf: str
    config: str
    split: str
    #: How to build the state: {"kind": "text"|"join"|"qa"|"image", ...}.
    #: ``image`` puts a fixed caption in the state string and the picture
    #: itself alongside it; only arms that accept images can run the set.
    state: dict[str, Any]
    #: How to read the gold option: {"kind": "class_label"|"label_text"|"bool", ...}
    gold: dict[str, Any]
    instructions: str
    family: str          # intent · sentiment · topic · moderation · nli · qa · legal · document
    language: str        # en · de
    tier: str            # seen · likely · unclear
    note: str = ""
    ladder: tuple[str, str] | None = None   # (ladder id, rung name)
    options_hint: int = 0                   # expected count; curation checks it
    #: Raw label value -> the option name the arms actually see. Some sets
    #: ship their classes as bare integers, and asking a model to choose
    #: between "1" and "7" with nothing else to go on measures nothing at
    #: all. Order is preserved, so a ClassLabel index still resolves.
    rename: dict[str, str] | None = None


def text(field_name: str) -> dict:
    return {"kind": "text", "field": field_name}


def join(*fields: str) -> dict:
    return {"kind": "join", "fields": list(fields)}


SPECS: list[Spec] = [
    # -- 2 options ------------------------------------------------------
    Spec("boolq", "google/boolq", "default", "validation",
         {"kind": "qa", "passage": "passage", "question": "question"},
         {"kind": "bool", "field": "answer"},
         "Answer the question from the passage.",
         "qa", "en", "seen", options_hint=2,
         note="Paragraph of evidence per example; the binary case with real "
              "context rather than a sentence."),
    Spec("sst2", "stanfordnlp/sst2", "default", "validation",
         text("sentence"), {"kind": "class_label", "field": "label"},
         "Is the sentiment of this movie-review excerpt positive or negative?",
         "sentiment", "en", "seen", options_hint=2,
         note="The most-trained-on sentiment set in existence. Kept precisely "
              "for that: it is the upper bound on what memorisation buys."),
    Spec("rotten_tomatoes", "cornell-movie-review-data/rotten_tomatoes",
         "default", "test",
         text("text"), {"kind": "class_label", "field": "label"},
         "Is this movie review positive or negative?",
         "sentiment", "en", "seen", options_hint=2,
         note="SST-2's sibling on longer sentences."),
    Spec("rte", "nyu-mll/glue", "rte", "validation",
         join("sentence1", "sentence2"),
         {"kind": "class_label", "field": "label"},
         "Does the first sentence entail the second?",
         "nli", "en", "seen", options_hint=2,
         note="Entailment is a different shape of question from "
              "classification, and the arms are sold for both."),
    Spec("tweet_offensive", "cardiffnlp/tweet_eval", "offensive", "test",
         text("text"), {"kind": "class_label", "field": "label"},
         "Is this tweet offensive?",
         "moderation", "en", "likely", options_hint=2,
         note="Moderation is the guardrail use case all three vendors lead "
              "their marketing with."),
    Spec("toxic_conversations", "mteb/toxic_conversations_50k", "default", "test",
         text("text"), {"kind": "label_text", "field": "label_text"},
         "Is this comment toxic?",
         "moderation", "en", "unclear", options_hint=2,
         note="Heavily imbalanced, which is what moderation traffic actually "
              "looks like and where accuracy alone stops being informative."),

    # -- 3 to 6 options -------------------------------------------------
    Spec("tweet_sentiment", "cardiffnlp/tweet_eval", "sentiment", "test",
         text("text"), {"kind": "class_label", "field": "label"},
         "What is the sentiment of this tweet?",
         "sentiment", "en", "likely", options_hint=3),
    Spec("mnli", "nyu-mll/glue", "mnli", "validation_matched",
         join("premise", "hypothesis"),
         {"kind": "class_label", "field": "label"},
         "Does the premise entail the hypothesis, contradict it, or neither?",
         "nli", "en", "seen", options_hint=3),
    Spec("xnli_de", "facebook/xnli", "de", "validation",
         join("premise", "hypothesis"),
         {"kind": "class_label", "field": "label"},
         "Folgt die Hypothese aus der Prämisse, widerspricht sie ihr, oder "
         "keines von beidem?",
         "nli", "de", "likely", options_hint=3,
         note="German, and a direct translation of an English set -- so the "
              "language effect can be read without the task changing."),
    Spec("ag_news", "fancyzhx/ag_news", "default", "test",
         text("text"), {"kind": "class_label", "field": "label"},
         "Which section of the newspaper does this story belong to?",
         "topic", "en", "seen", options_hint=4,
         note="The pilot's surprise: the 421M local model won here."),
    Spec("sst5", "SetFit/sst5", "default", "test",
         text("text"), {"kind": "label_text", "field": "label_text"},
         "How positive is this review, on a five-point scale?",
         "sentiment", "en", "seen", options_hint=5,
         note="An ordinal scale rather than a set of categories; option order "
              "carries meaning here, which is why permutation debiasing "
              "rotates rather than shuffles."),
    Spec("amazon_reviews_de", "mteb/amazon_reviews_multi", "de", "test",
         text("text"), {"kind": "label_text", "field": "label_text"},
         "Wie viele Sterne vergibt diese Rezension?",
         "sentiment", "de", "likely", options_hint=5,
         rename={str(i): f"{i + 1} Sterne" if i else "1 Stern" for i in range(5)},
         note="German, ordinal, consumer text."),
    Spec("emotion", "dair-ai/emotion", "split", "test",
         text("text"), {"kind": "class_label", "field": "label"},
         "Which emotion does this message express?",
         "sentiment", "en", "seen", options_hint=6),
    Spec("trec_coarse", "SetFit/TREC-QC", "default", "test",
         text("text"), {"kind": "label_text", "field": "label_coarse_text"},
         "What kind of answer does this question ask for?",
         "intent", "en", "seen", options_hint=6, ladder=("trec", "coarse"),
         note="Ladder rung. Identical examples to trec_fine, six options "
              "instead of fifty."),

    # -- 9 to 20 options ------------------------------------------------
    Spec("gnad10", "community-datasets/gnad10", "default", "test",
         text("text"), {"kind": "class_label", "field": "label"},
         "In welches Ressort gehört dieser Artikel?",
         "topic", "de", "unclear", options_hint=9,
         note="German news, full articles -- long German state, which nothing "
              "else in the suite provides."),
    Spec("patent", "ccdv/patent-classification", "abstract", "test",
         text("text"), {"kind": "class_label", "field": "label"},
         "Which patent category does this abstract belong to?",
         "topic", "en", "unclear", options_hint=9,
         note="Technical domain with opaque category names -- the case where "
              "the option names do not speak for themselves."),
    Spec("yahoo_topics", "community-datasets/yahoo_answers_topics",
         "yahoo_answers_topics", "test",
         join("question_title", "question_content", "best_answer"),
         {"kind": "class_label", "field": "topic"},
         "Which topic does this question belong to?",
         "topic", "en", "likely", options_hint=10),
    Spec("scotus", "coastalcph/lex_glue", "scotus", "test",
         text("text"), {"kind": "class_label", "field": "label"},
         "Which issue area does this Supreme Court opinion concern?",
         "legal", "en", "unclear", options_hint=13,
         rename={str(i + 1): name for i, name in enumerate([
             "Criminal Procedure", "Civil Rights", "First Amendment",
             "Due Process", "Privacy", "Attorneys", "Unions",
             "Economic Activity", "Judicial Power", "Federalism",
             "Interstate Relations", "Federal Taxation", "Miscellaneous"])},
         note="Up to 68k characters. The set that forces the truncation "
              "policy to be a declared parameter rather than an accident."),
    Spec("dbpedia_14", "fancyzhx/dbpedia_14", "dbpedia_14", "test",
         join("title", "content"), {"kind": "class_label", "field": "label"},
         "Which kind of entity does this article describe?",
         "topic", "en", "seen", options_hint=14),
    Spec("massive_scenario_en", "mteb/amazon_massive_scenario", "en", "test",
         text("text"), {"kind": "label_text", "field": "label_text"},
         "Which scenario does this voice command belong to?",
         "intent", "en", "likely", options_hint=18,
         ladder=("massive_en", "scenario"),
         note="Ladder rung. Identical utterances to massive_intent_en."),
    Spec("massive_scenario_de", "mteb/amazon_massive_scenario", "de", "test",
         text("text"), {"kind": "label_text", "field": "label_text"},
         "Zu welchem Szenario gehört dieser Sprachbefehl?",
         "intent", "de", "likely", options_hint=18,
         ladder=("massive_de", "scenario"),
         note="Ladder rung, German."),
    Spec("newsgroups20", "SetFit/20_newsgroups", "default", "test",
         text("text"), {"kind": "label_text", "field": "label_text"},
         "Which newsgroup was this message posted to?",
         "topic", "en", "seen", options_hint=20),

    # -- 50 to 80 options -----------------------------------------------
    Spec("trec_fine", "SetFit/TREC-QC", "default", "test",
         text("text"), {"kind": "label_text", "field": "label_text"},
         "What kind of answer does this question ask for?",
         "intent", "en", "seen", options_hint=50, ladder=("trec", "fine"),
         note="Ladder rung. Identical examples to trec_coarse."),
    Spec("massive_intent_en", "mteb/amazon_massive_intent", "en", "test",
         text("text"), {"kind": "label_text", "field": "label_text"},
         "Which intent does this voice command express?",
         "intent", "en", "likely", options_hint=60,
         ladder=("massive_en", "intent"),
         note="Ladder rung."),
    Spec("massive_intent_de", "mteb/amazon_massive_intent", "de", "test",
         text("text"), {"kind": "label_text", "field": "label_text"},
         "Welche Absicht drückt dieser Sprachbefehl aus?",
         "intent", "de", "likely", options_hint=60,
         ladder=("massive_de", "intent"),
         note="Ladder rung, German."),
    Spec("banking77", "legacy-datasets/banking77", "default", "test",
         text("text"), {"kind": "class_label", "field": "label"},
         "Which banking-support intent does this customer message express?",
         "intent", "en", "seen", options_hint=77,
         note="The pilot's headline set."),

    # -- documents as images --------------------------------------------
    Spec("rvl_cdip", "jinhybr/rvl_cdip_400_train_val_test", "default", "test",
         {"kind": "image", "field": "image",
          "caption": "A scanned page from a business document."},
         {"kind": "class_label", "field": "label"},
         "Which kind of document is this scan?",
         "document", "en", "likely", options_hint=16,
         note="The one axis on which two of the three arms cannot compete at "
              "all: Jev is text-only by its own documentation and Laya is an "
              "encoder. They need an OCR step in front; the Privatemode arm "
              "reads the pixels. Letter vs memo and report vs publication are "
              "the classic confusions, so this is not a trivially easy set."),

    # -- 100+ options ---------------------------------------------------
    Spec("ledgar", "coastalcph/lex_glue", "ledgar", "test",
         text("text"), {"kind": "class_label", "field": "label"},
         "Which kind of contract provision is this?",
         "legal", "en", "unclear", options_hint=100,
         note="Wide and long at once, in a domain none of the arms is sold "
              "for. The hardest cell in the grid."),
    Spec("clinc150", "clinc/clinc_oos", "plus", "test",
         text("text"), {"kind": "class_label", "field": "intent"},
         "Which intent does this request express?",
         "intent", "en", "likely", options_hint=151,
         note="151 options, including an explicit out-of-scope class -- the "
              "only set here that asks an arm to say 'none of these'."),
]

BY_NAME = {spec.name: spec for spec in SPECS}


def ladders() -> dict[str, list[Spec]]:
    """Ladder id -> its rungs, narrowest option set first."""
    out: dict[str, list[Spec]] = {}
    for spec in SPECS:
        if spec.ladder:
            out.setdefault(spec.ladder[0], []).append(spec)
    for rungs in out.values():
        rungs.sort(key=lambda s: s.options_hint)
    return out
