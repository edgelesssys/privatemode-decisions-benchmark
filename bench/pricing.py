"""Prices, and the one judgement call in this repository.

Privatemode bills euros, TypeSafe bills dollars, so a single "cost
per 1000 decisions" number cannot be produced without a conversion rate.
The rate is a knob (``--eur-per-usd``) and the report prints it, because it
is the only figure here that is neither measured nor quoted.

Both cost models bill what the vendor's own ``usage`` block reports, not
what we think the prompt should have cost. The two arms serialize state and
options differently, and that difference is part of what is being compared.
"""

from __future__ import annotations

from dataclasses import dataclass

#: EUR / 1M tokens, as (input, output, cached input).
#:
#: From https://privatemode.ai/pricing, 2026-09-24, after GLM-5.3-Flash's
#: price was lowered. The cost column is a headline number, so it follows the
#: published page rather than any figure shared ahead of it.
PRIVATEMODE_EUR_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "glm-5.3-flash": (0.20, 0.65, 0.05),
    "glm-flash-latest": (0.20, 0.65, 0.05),
    "glm-5.3": (1.55, 7.74, 0.15),
    "glm-latest": (1.55, 7.74, 0.15),
    "gpt-oss-120b": (0.43, 1.70, 0.04),
    "kimi-k2.6": (1.55, 7.74, 0.15),
}

#: EUR / 1M input tokens. From https://privatemode.ai/pricing, 2026-09-24.
EMBEDDING_EUR_PER_MTOK = 0.13

#: Fetched 2026-09-21 from https://docs.typesafe.ai/models (USD / 1M tokens).
#: Output is free: Jev returns a typed decision, not a token stream.
JEV_USD_PER_MTOK: tuple[float, float] = (0.042, 0.0)

DEFAULT_EUR_PER_USD = 0.92


@dataclass(frozen=True)
class Cost:
    eur: float
    detail: str


def privatemode_cost(model: str, input_tokens: int, output_tokens: int,
                     cached_tokens: int = 0) -> Cost:
    """``input_tokens`` is the full prompt count, cached tokens included."""
    key = model.split("/")[-1]
    if key not in PRIVATEMODE_EUR_PER_MTOK:
        raise KeyError(f"No published price for {model!r}; add it to pricing.py")
    rate_in, rate_out, rate_cached = PRIVATEMODE_EUR_PER_MTOK[key]
    fresh = max(0, input_tokens - cached_tokens)
    eur = (fresh * rate_in + cached_tokens * rate_cached
           + output_tokens * rate_out) / 1e6
    return Cost(eur, f"{fresh} in + {cached_tokens} cached + {output_tokens} out")


def embedding_cost(input_tokens: int) -> Cost:
    """Embeddings have no output stream, so only the input is billed."""
    return Cost(input_tokens * EMBEDDING_EUR_PER_MTOK / 1e6, f"{input_tokens} in")


def jev_cost(input_tokens: int, output_tokens: int,
             eur_per_usd: float = DEFAULT_EUR_PER_USD) -> Cost:
    rate_in, rate_out = JEV_USD_PER_MTOK
    usd = (input_tokens * rate_in + output_tokens * rate_out) / 1e6
    return Cost(usd * eur_per_usd, f"{input_tokens} in + {output_tokens} out (free)")


def local_cost(*_: object) -> Cost:
    """Weights on your own hardware have no per-decision list price.

    Reporting zero would be a lie in the other direction -- the laptop, the
    electricity and the operational burden of running it are real -- but
    they are fixed costs, and there is no honest way to put them in the same
    column as a published per-token rate. The report prints "local" instead
    of a number, and this exists so the arithmetic elsewhere has something
    to call.
    """
    return Cost(0.0, "local weights, no per-decision list price")


# -- estimating a run before paying for it --------------------------------
#
# Fitted to the pilot's measured token counts, which is why the constants
# look arbitrary: the two products serialize state and options differently,
# and the shape of that difference is itself a finding. Privatemode carries
# little fixed overhead and about 20 tokens per option; Jev carries roughly
# 270 fixed and about 10 per option. Checked against the three pilot sets it
# lands within 3% on all six numbers -- close enough to guard a budget with,
# nowhere near close enough to report as a measurement.
FIXED_TOKENS = {"privatemode": 55, "jev": 270, "laya": 0,
                "glm-cot": 60, "embed-nn": 0}
PER_OPTION_TOKENS = {"privatemode": 20, "jev": 10, "laya": 3,
                     "glm-cot": 8, "embed-nn": 0}
#: Reasoning tokens the chain-of-thought control emits per decision. Output
#: is priced at 3.3x input on this model, so this dominates its cost.
COT_OUTPUT_TOKENS = 200
CHARS_PER_TOKEN = 4
#: Image tokens at ``--image-max-side`` 1024, measured on RVL-CDIP. Cost
#: scales with area, so halving the edge quarters the bill.
IMAGE_TOKENS_AT_1024 = 970


def estimate_tokens(arm: str, options: int, state_chars: float,
                    image_side: int | None = None) -> int:
    tokens = (FIXED_TOKENS.get(arm, 0)
              + PER_OPTION_TOKENS.get(arm, 0) * options
              + state_chars / CHARS_PER_TOKEN)
    if image_side:
        tokens += IMAGE_TOKENS_AT_1024 * (image_side / 1024) ** 2
    return int(tokens)


def estimate_eur(arm: str, model: str, examples: int, options: int,
                 state_chars: float, image_side: int | None = None,
                 eur_per_usd: float = DEFAULT_EUR_PER_USD) -> float:
    if arm == "laya":
        return 0.0
    tokens = estimate_tokens(arm, options, state_chars, image_side)
    if arm == "jev":
        return jev_cost(tokens * examples, 0, eur_per_usd).eur
    if arm == "embed-nn":
        return embedding_cost(tokens * examples).eur
    output = COT_OUTPUT_TOKENS if arm == "glm-cot" else 1
    return privatemode_cost(model, tokens * examples, output * examples).eur
