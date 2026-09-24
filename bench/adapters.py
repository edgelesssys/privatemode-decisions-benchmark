"""One arm per product, behind one method: ``ask(task) -> Answer``.

Both arms POST through the same connection-pooled client from the
``system_one`` package. That is not laziness -- latency is one of the three
things being measured, and a fresh TLS handshake costs more than either
forward pass, so the transport has to be held constant or the numbers are
about urllib rather than about the models.

``unsupported(task)`` asks an arm whether it can answer this kind of
question at all, and returns the reason when it cannot. Two limits show up
in this suite. Jev is text-only by its own documentation and Laya is an
encoder, so neither can be shown a scanned page. And Laya's option markers
share a 192-token head budget, which 151-way clinc150 exceeds -- its own
``shortlist`` module exists because of that.

Both are capability findings, and both are reported as such: the runner
leaves the arm out of that dataset, records the reason in the run and prints
it. A column of errors is not a measurement, and it is worse than useless
here because paired scoring would then drop every example for the arms that
could answer.

Neither arm gets a knob the other does not have. In particular the
Privatemode arm runs with ``permutations=1``: option-order debiasing is a
real feature of the library and it does improve accuracy, but Jev exposes no
equivalent, so switching it on here would compare a debiased system against
a raw one. Measure it separately (``--permutations``) and say so.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from system_one import Choice, OpenAIClient, SystemOne
from system_one.images import to_data_url

from .datasets import Task
from .timing import TimedClient

QUESTION_ID = "answer"


@dataclass(frozen=True)
class Answer:
    choice: str
    probabilities: dict[str, float]
    #: What the vendor tells you to threshold on. The two are computed
    #: differently, which is why the report also derives a common one from
    #: ``probabilities``.
    confidence: float
    latency_s: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    #: Time spent waiting on our own in-flight gate, excluded from
    #: ``latency_s``. Recorded so "the gate was never contended" is a
    #: measurement rather than an assumption.
    gate_wait_s: float = 0.0
    #: 429/503/529 responses seen while producing this answer. Any at all
    #: and the arm's latency is flagged in the report.
    throttled: int = 0


class Arm:
    """What every arm has to answer, beyond ``ask``."""

    def unsupported(self, task: Task) -> str | None:
        """Why this arm cannot take this task, or ``None`` if it can."""
        return None


class PrivatemodeArm(Arm):
    """Privatemode System One: one masked forward pass through the proxy.

    The deployment reports at most 128 ids in ``logprob_token_ids``, while
    the mask in ``allowed_token_ids`` takes every option. Above 128 options
    the library sends the same request once per batch of ids to read, with
    the whole mask in each, and merges the reads before renormalizing: the
    same forward pass read in slices, so the distribution is the one a
    single request would return, at the latency of the slower request. The limit is then the model's single-token
    indexes, 191 on GLM-5.3-Flash, which the library enforces itself.
    """

    name = "privatemode"
    accepts_images = True

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, permutations: int = 1,
                 image_max_side: int | None = None) -> None:
        base_url = base_url or os.environ["SYSTEM_ONE_BASE_URL"]
        api_key = api_key or os.environ.get("SYSTEM_ONE_API_KEY") or None
        self.model = model or os.environ.get("SYSTEM_ONE_MODEL", "glm-5.3-flash")
        self.permutations = permutations
        #: Longest edge the picture is scaled to before it is sent. Image
        #: tokens are the whole cost of a document decision, so this is a
        #: declared parameter and the run records it.
        self.image_max_side = image_max_side
        self._client = TimedClient(base_url, api_key)
        self._engine = SystemOne(self._client, self.model,
                                 permutations=permutations)

    def ask(self, task: Task) -> Answer:
        response = self._engine.system_one(
            task.state,
            {QUESTION_ID: Choice(criteria=task.criteria,
                                 instructions=task.instructions)},
            images=list(task.images) or None,
            image_max_side=self.image_max_side,
            # One question is one request up to 128 options. Above that, the
            # id batches go out in parallel: measured on clinc150, 649 ms
            # against 846 ms one after another; the prefix cache does not
            # make the second request much cheaper.
            mode="parallel" if len(task.criteria) > 128 else "sequential")
        answer = response.answers[QUESTION_ID]
        usage = response.usage
        return Answer(choice=answer.choice, probabilities=dict(answer.probabilities),
                      confidence=answer.confidence,
                      # One question in one request: the call time is the
                      # request time. When it takes several, what the caller
                      # waited for is the wall.
                      latency_s=(response.timings["calls"][QUESTION_ID]
                                 if response.timings["requests"] == 1
                                 else response.timings["wall"]),
                      input_tokens=usage.input_tokens,
                      output_tokens=usage.output_tokens,
                      cached_tokens=usage.cached_tokens,
                      **self._client.last())

    def close(self) -> None:
        self._engine.close()


class JevArm(Arm):
    """TypeSafe AI Jev: POST /v1/systemone, one Choice question."""

    name = "jev"
    accepts_images = False

    def unsupported(self, task: Task) -> str | None:
        return "text-only by its documentation" if task.images else None

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None) -> None:
        base_url = base_url or os.environ.get("JEV_BASE_URL",
                                              "https://api.typesafe.ai/v1")
        api_key = api_key or os.environ["JEV_API_KEY"]
        self.model = model or os.environ.get("JEV_MODEL", "jev-latest")
        # Usage attribution happens per API key in the vendor console; the
        # request body has no label field (every unknown field is a 400).
        self._client = TimedClient(base_url, api_key)

    def _payload(self, task: Task) -> dict[str, Any]:
        return {
            "model": self.model,
            "state": task.state,
            "questions": {QUESTION_ID: {"type": "choice",
                                        "instructions": task.instructions,
                                        "criteria": task.criteria}},
        }

    def ask(self, task: Task) -> Answer:
        body, elapsed = self._client.post("/systemone", self._payload(task))
        answer = body["answers"][QUESTION_ID]
        usage = body.get("usage") or {}
        return Answer(choice=answer["choice"],
                      probabilities=dict(answer["probabilities"]),
                      confidence=float(answer.get("confidence", 0.0)),
                      latency_s=elapsed,
                      input_tokens=usage.get("input_tokens", 0),
                      output_tokens=usage.get("output_tokens", 0),
                      cached_tokens=0,
                      **self._client.last())

    def close(self) -> None:
        self._client.close()


class LayaArm(Arm):
    """Laya: a 421M encoder that does the same job on the machine you own.

    Not a causal model at all -- a ModernBERT-large backbone with a decision
    head on top, so "one forward pass" is literal rather than a prefill plus
    one sampled token. It takes the same ``{"type": "choice", "instructions",
    "criteria"}`` payload as Jev, which is why it drops into this benchmark
    without a translation layer.

    Its latency is recorded but does not belong in the comparison: it runs on
    a laptop while the other two run in a datacentre, so the number says more
    about the laptop than about the model. The report marks it.

    ``shortlist`` is the package's own answer to wide option sets: embed the
    state and the options, keep the top *k* by cosine, and let the decision
    head choose among those. It is a second model call in front of the first
    rather than one forward pass, so it is reported as a labelled variant --
    the same treatment ``--permutations`` gets on the Privatemode side, and
    for the same reason: neither belongs in the like-for-like table, and
    leaving both out would misrepresent what each product can do.

    ``Router`` holds mutable per-call state, so calls are serialized. With
    the default ``--concurrency 1`` that costs nothing.
    """

    name = "laya"
    accepts_images = False

    def __init__(self, checkpoint: str | None = None, device: str | None = None,
                 shortlist: int = 0) -> None:
        from laya import Router  # heavy, and only needed when this arm runs

        self.model = checkpoint or os.environ.get("LAYA_MODEL", "english")
        self.shortlist = shortlist
        self._router = Router(device=device or os.environ.get("LAYA_DEVICE"),
                              default=self.model, preload=True)
        self._lock = __import__("threading").Lock()
        self._embed = None
        if shortlist:
            from laya import embed_fn_from_agent
            self._embed = embed_fn_from_agent(self._router.load(self.model))

    def unsupported(self, task: Task) -> str | None:
        """Ask the package itself, rather than reimplement its budget maths.

        The option markers share a 192-token head, and how many options fit
        depends on how long their names are -- 77 banking intents fit, 151
        CLINC ones do not. Reproducing that arithmetic here would mean
        keeping a copy of the package's internals correct forever, so the
        check is a real attempt at the real call.
        """
        if task.images:
            return "an encoder; it cannot be shown an image"
        try:
            with self._lock:
                self._router.predict(task.state, {QUESTION_ID: {
                    "type": "choice", "instructions": task.instructions,
                    "criteria": task.criteria}}, model=self.model)
        except ValueError as error:
            return str(error)
        return None

    def ask(self, task: Task) -> Answer:
        payload = {QUESTION_ID: {"type": "choice",
                                 "instructions": task.instructions,
                                 "criteria": task.criteria}}
        started = time.perf_counter()
        with self._lock:
            if self._embed is not None:
                from laya import predict_shortlist
                body = predict_shortlist(self._router, task.state, payload,
                                         self._embed, k=self.shortlist,
                                         model=self.model)
            else:
                body = self._router.predict(task.state, payload, model=self.model)
        elapsed = time.perf_counter() - started
        answer = body["answers"][QUESTION_ID]
        usage = body.get("usage") or {}
        return Answer(choice=answer["choice"],
                      probabilities=dict(answer["probabilities"]),
                      confidence=float(answer.get("confidence", 0.0)),
                      latency_s=elapsed,
                      input_tokens=usage.get("input_tokens", 0),
                      output_tokens=usage.get("output_tokens", 0),
                      cached_tokens=0)

    def close(self) -> None:
        pass


# -- control arms ---------------------------------------------------------
#
# These two exist to make the comparison interpretable, not to win it. The
# three product arms answer the question "which product is better at this";
# without a control they cannot answer "better than what, and because of
# what". Both are reported alongside the products and labelled as controls.


ANSWER_LINE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
CONFIDENCE_LINE = re.compile(r"CONFIDENCE:\s*([01]?\.?\d+)", re.IGNORECASE)


class ChainOfThoughtArm(Arm):
    """The same GLM-5.3-Flash, asked normally and allowed to reason.

    The Privatemode arm runs this model. So this separates *the technique* --
    one masked forward pass, no room to think -- from *the model*. Without
    it, a reader cannot tell which of the two any Privatemode number is
    about, and that is the first question a sceptical one will ask.

    It is asked for a confidence as well as an answer, because that is what
    a practitioner would do and because scoring it against arms that return
    a real distribution would otherwise be a straw man. The self-report is
    put on the chosen option and the remainder spread evenly over the rest;
    that is a generous reading of a number the model simply asserts, and the
    calibration metrics say how much it is worth.

    Cost is not incidental here: reasoning is output tokens, and output is
    priced at 3.3x input on this model.

    GLM-5.3 reasons natively and puts that reasoning in ``reasoning_content``
    rather than ``content``, so a budget that is too small buys thinking and
    returns an empty answer with ``finish_reason: length``. At 800 tokens
    that happened often enough to show up as retries; at 2500 one
    newsgroups20 example still produced ten thousand characters of reasoning
    and no answer. The default is now 4000 and it is a recorded run
    parameter, because a reasoning budget is a deployment choice rather than
    a fact about the model -- and because at 3.3x the input price, the budget
    *is* most of this arm's cost.

    A reply that ran out mid-thought is reported as exactly that rather than
    as a parse failure. The two have different fixes, and conflating them
    would hide a configuration problem inside a model result.
    """

    name = "glm-cot"
    accepts_images = True

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, max_tokens: int = 4000,
                 image_max_side: int | None = None) -> None:
        base_url = base_url or os.environ["SYSTEM_ONE_BASE_URL"]
        api_key = api_key or os.environ.get("SYSTEM_ONE_API_KEY") or None
        self.model = model or os.environ.get("SYSTEM_ONE_MODEL", "glm-5.3-flash")
        self.max_tokens = max_tokens
        self.image_max_side = image_max_side
        self._client = TimedClient(base_url, api_key)

    def _prompt(self, task: Task) -> str:
        options = "\n".join(
            f"- {name}" + (f": {description}" if description else "")
            for name, description in task.criteria.items())
        return (f"{task.instructions}\n\n"
                f"State:\n{task.state}\n\n"
                f"Options:\n{options}\n\n"
                "Think it through, then end with exactly two lines:\n"
                "ANSWER: <one option name, copied exactly>\n"
                "CONFIDENCE: <a number between 0 and 1>")

    def _content(self, task: Task):
        text = self._prompt(task)
        if not task.images:
            return text
        urls = [to_data_url(image, max_side=self.image_max_side)
                for image in task.images]
        return [*({"type": "image_url", "image_url": {"url": url}} for url in urls),
                {"type": "text", "text": text}]

    @staticmethod
    def _resolve(raw: str, options: list[str]) -> str | None:
        """Map what the model wrote onto an option, or give up honestly.

        Three passes, narrowing: exact, case- and punctuation-insensitive,
        then the last option name to appear anywhere in the reply. Giving up
        is a recorded error rather than a guess -- a parse that silently
        picks option zero would show up as an accuracy difference and be
        read as a model difference.
        """
        stripped = raw.strip().strip("`*\"' .")
        for option in options:
            if stripped == option:
                return option
        def flat(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", value.lower())
        flattened = flat(stripped)
        for option in options:
            if flattened == flat(option):
                return option
        found = [(raw.lower().rfind(option.lower()), option)
                 for option in options if option.lower() in raw.lower()]
        return max(found)[1] if found else None

    def ask(self, task: Task) -> Answer:
        body, elapsed = self._client.post("/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": self._content(task)}],
            "max_tokens": self.max_tokens, "temperature": 0})
        choice = body["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
        options = list(task.criteria)

        if not text and choice.get("finish_reason") == "length":
            spent = len(message.get("reasoning_content") or "")
            raise RuntimeError(
                f"ran out of tokens while reasoning ({spent} characters of it) "
                f"and never answered; raise --cot-max-tokens")

        match = ANSWER_LINE.search(text)
        answer = self._resolve(match.group(1), options) if match else None
        if answer is None:
            answer = self._resolve(text, options)
        if answer is None:
            raise RuntimeError(
                f"no option in the reply (finish={choice.get('finish_reason')}): "
                f"{text[-120:]!r}")

        found = CONFIDENCE_LINE.search(text)
        confidence = float(found.group(1)) if found else 1.0
        confidence = min(1.0, max(0.0, confidence))
        rest = (1.0 - confidence) / max(1, len(options) - 1)
        usage = body.get("usage") or {}
        return Answer(
            choice=answer,
            probabilities={name: (confidence if name == answer else rest)
                           for name in options},
            confidence=confidence,
            latency_s=elapsed,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0) or 0,
            **self._client.last())

    def close(self) -> None:
        self._client.close()


class EmbeddingArm(Arm):
    """Nearest option over Qwen3-Embedding-4B: the floor.

    Zero-shot, no decision model at all -- embed the state, embed each
    option, take the closest. It is here to answer "how hard is this task
    really", and it is the cheapest thing on the list at EUR 0.13 per
    million input tokens. If a purpose-built decision model cannot beat it
    on a set, that is worth knowing before anyone argues about decision
    models.

    Option embeddings are computed once per option set and reused, which is
    how it would be deployed and why its cost per decision is almost
    entirely the state.

    ``TEMPERATURE`` turns cosines into a distribution. It is declared rather
    than fitted: fitting it on the same data the arm is scored on would flatter
    the baseline, and a fitted floor is not a floor.
    """

    name = "embed-nn"
    accepts_images = False
    TEMPERATURE = 0.05

    def unsupported(self, task: Task) -> str | None:
        return "embeddings are text-only" if task.images else None

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None) -> None:
        base_url = base_url or os.environ["SYSTEM_ONE_BASE_URL"]
        api_key = api_key or os.environ.get("SYSTEM_ONE_API_KEY") or None
        self.model = model or os.environ.get("EMBED_MODEL", "qwen3-embedding-4b")
        self._client = TimedClient(base_url, api_key)
        self._options: dict[tuple, tuple[list[list[float]], int]] = {}

    @staticmethod
    def _unit(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], int, float]:
        body, elapsed = self._client.post(
            "/embeddings", {"model": self.model, "input": texts})
        vectors = [self._unit(item["embedding"])
                   for item in sorted(body["data"], key=lambda d: d["index"])]
        usage = body.get("usage") or {}
        return vectors, usage.get("prompt_tokens", 0), elapsed

    def _option_vectors(self, task: Task) -> tuple[list[list[float]], int]:
        key = (task.instructions, tuple(task.criteria.items()))
        if key not in self._options:
            texts = [f"{name}: {description}" if description else name
                     for name, description in task.criteria.items()]
            vectors, tokens, _ = self._embed(texts)
            self._options[key] = (vectors, tokens)
        return self._options[key]

    def ask(self, task: Task) -> Answer:
        option_vectors, option_tokens = self._option_vectors(task)
        vectors, tokens, elapsed = self._embed([task.state])
        state = vectors[0]
        scores = [sum(a * b for a, b in zip(state, option))
                  for option in option_vectors]
        top = max(scores)
        weights = [math.exp((score - top) / self.TEMPERATURE) for score in scores]
        total = sum(weights)
        probabilities = [weight / total for weight in weights]
        names = list(task.criteria)
        best = max(range(len(names)), key=probabilities.__getitem__)
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        confidence = (1.0 if len(names) == 1
                      else 1 - entropy / math.log(len(names)))
        return Answer(choice=names[best],
                      probabilities=dict(zip(names, probabilities)),
                      confidence=max(0.0, min(1.0, confidence)),
                      latency_s=elapsed,
                      # The option set is embedded once and reused, so only
                      # the state is charged per decision.
                      input_tokens=tokens, output_tokens=0, cached_tokens=0,
                      **self._client.last())

    def close(self) -> None:
        self._client.close()
