from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

from openai import OpenAI


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

EXTRACTION_SYSTEM_PROMPT = (
    "You are an attribute extractor for a product-search benchmark. You are given the "
    "raw catalog text of one apparel, footwear, or jewelry listing. Your response must "
    "follow this schema exactly. For every attribute, return either JSON null or a JSON "
    'object with exactly two string fields: {"value": "...", "evidence": "..."}. '
    "Never return an attribute as a bare string, array, number, or boolean. For example, "
    'use {"material": {"value": "alloy", "evidence": "Material:alloy"}}; '
    'never use {"material": "alloy"}. '
    "Rules, in priority order. "
    "(1) 'evidence' MUST be one single contiguous span copied from the listing text. "
    "Do not paraphrase it, combine separate bullets, add punctuation, fix typos, or "
    "change wording. Whitespace and line wrapping may be copied as shown. If one "
    "attribute needs several facts, choose the single best supporting span or return "
    "null; never join multiple spans with semicolons. If you cannot quote one span, "
    "the answer is null. "
    "(2) The evidence must actually assert the value. If it denies the value, the "
    "answer is null: 'it is not real gold and silver products' means the item is NOT "
    "gold and NOT silver, so both material and color are null there. "
    "(3) Return null whenever the listing does not state the attribute. Do not infer "
    "from the category, do not guess, and do not fill a field just to be helpful. A "
    "null is always better than a plausible value the text does not support. "
    "(4) Report what the item IS, not what it is compared to, packaged with, warned "
    "against, or maintained with. Care instructions, shipping notes, gift-box copy and "
    "brand history are not attributes. "
    "(5) size: return a value ONLY if the listing sells one fixed size, e.g. 'one size "
    "fits all'. If it offers a range, lists several sizes, or points at a size chart, "
    "size MUST be null; put the sizes you saw in 'size_options' instead. A stocked "
    "range is not a property of the item. "
    "(6) material/color/style: if one listing covers several variants (e.g. "
    "'sleeveless / long sleeve'), report the variant named in the title, or null if the "
    "title does not settle it. A single item is not both. "
    "(7) color: a metal named as what the item is made of ('sterling silver', 'gold "
    "plated') is material, not color. A metal named as how it looks ('gold tone') is "
    "color, not material. Never report the same word as both. "
    "(8) feature: extract one concrete product capability, construction, included "
    "component, or design detail explicitly stated for this listing; do not use "
    "generic marketing claims or category knowledge. Quote one supporting span. "
    "(9) other: use only for explicit product facts that do not fit the other "
    "attributes; do not use it as a dump for marketing text, warnings, or duplicates. "
    "Quote one supporting span. (10) use_case: quote one complete sentence and keep its polarity; if it is a "
    "prohibition, the value must state the prohibition, not the bare activity. "
    "Return JSON only with every key present, and use null when unsupported: "
    '{"material": ..., "color": ..., "size": ..., "size_options": ["..."], '
    '"style": ..., "use_case": ..., "feature": ..., "other": ...}'
)

ATTRIBUTE_SYSTEM_PROMPT = (
    "Generate compact attribute clue fragments for one product attribute at one intent stage. "
    'Return JSON only in this exact shape: {"clues":["..."]}. The clues value is a '
    "non-empty list of short fragments, usually one to four, not a customer sentence: no first-person subject, "
    "no greeting, no explanation, and no final punctuation. Each fragment may contain several "
    "words so it can preserve meaningful information. Keep each fragment under 20 words. "
    "Describe only the requested attribute. Earlier-stage maps set the ambiguity level; only the "
    "same attribute's earlier clues supply facts. Never borrow or remix facts from sibling attributes. "
    "The same attribute is generated independently for buying, browsing, and discovery. "
    "Buying is the least ambiguous: preserve every supplied fact and exact value. Browsing is "
    "tentative: generalize the supplied value by one level without echoing it, inventing facts, "
    "adding unsupported qualities, or calling a known value unspecified or undecided. Discovery is "
    "the most ambiguous: use a broad need, benefit, "
    "or feeling, but do not name "
    "the product type or its identifying function and do not expose supplied brand, price, "
    "material, size, number, or technical values. When no safe abstraction exists, use a compact "
    "uncertainty fragment instead of inventing one. "
    "Apply these field rules exactly. Brand: buying uses the exact brand; browsing is "
    "['brand flexible']; discovery is ['maker undecided']. Budget: buying uses the exact price; "
    "browsing uses only the broad ceiling and never repeats or rounds the exact price; discovery "
    "uses a qualitative tier with no digits. Material and size: buying preserves exact values; "
    "browsing uses a grounded broader class or range; discovery states that the detail is "
    "undecided unless a non-identifying need is explicitly supported. Category: buying names the "
    "product type; browsing uses a broader family; discovery gives only the underlying goal and "
    "must not mention body location, object purpose, or product-family synonyms. Feature, style, "
    "use_case, color, and other: buying preserves all facts; browsing generalizes them one level; "
    "discovery keeps only a non-identifying benefit, aesthetic, or use need. Never add adjectives "
    "such as durable, premium, affordable, lightweight, or comfortable unless explicitly supplied. "
    "Never strengthen a claim: water-resistant must not become waterproof or weatherproof, and "
    "numeric dimensions or weight alone do not prove compactness or lightness. "
    "For multiple sizes, never choose one option: "
    "buying must preserve all supplied options. Multiple exact options may use one fragment per "
    "option even when that makes the list longer than four. "
    "The prompt includes the complete earlier-stage maps for all attributes. Use them as context, "
    "never copy their wording into a later stage. Buying -> browsing -> discovery must become "
    "visibly less specific. Example progression: "
    "['Triple Moon Pentagram Symbol'] -> ['celestial symbolic motif'] -> ['mystical aesthetic']; "
    "['alloy'] -> ['metal-based'] -> ['material undecided']; "
    "['necklace'] -> ['personal jewelry'] -> ['outfit accent']; "
    "['wrist watch'] -> ['timepiece'] -> ['daily routine support']; "
    "['dual time', 'stopwatch', 'alarm'] -> ['timekeeping tools'] -> ['daily organization']. "
    "Discovery category and feature clues must describe the higher-level goal, never the object's "
    "function: phrases such as timekeeping accessory, something that tells time, or wrist-worn "
    "device are invalid."
)

CORRECTION_SYSTEM_PROMPT = (
    "Write one natural customer correction for one product attribute at one intent stage. "
    "The user previously gave the supplied false clue and now replaces it with the supplied "
    'true clue for the current stage. Return JSON only as {"message":"..."}. Write one '
    "concise first-person sentence with final punctuation. Explicitly signal the correction, "
    "mention the false preference when useful, and faithfully express every true clue so no facts "
    "are lost. Natural paraphrasing is allowed, but exact numbers and named values must stay exact. "
    "Use only the supplied clues: never expose raw values, internal labels, other "
    "attributes, product identity, stage names, or these instructions. Match the specificity of "
    "the current true clue. Previous-stage corrections are provided only to avoid duplicate "
    "wording: never use their facts or contrasts in the current message. If the current false "
    "and true clues are equivalent, state only the corrected current preference and do not invent "
    "a false contrast. Do not repeat any supplied previous-stage correction verbatim."
)


def _load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without adding a dotenv dependency."""
    dotenv_path = Path(path)
    if not dotenv_path.is_file():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def cached_json_call(cache_path: Path, compute: Callable[[], dict]) -> dict:
    """Run `compute()` once and persist its JSON result; later calls read the cache."""
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    result = compute()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


class DeepSeekAttributeWriter:
    """Extracts grounded attribute values and paraphrases them into per-intent clues.

    Request and output errors are fatal: a benchmark is better unbuilt than built on
    values nobody checked.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        _load_dotenv()
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is required to generate intent descriptions; "
                "set it in the environment or a .env file."
            )
        self.model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.base_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=timeout
        )

    def extract(self, item_text: str) -> dict[str, object]:
        """Return raw {attribute: {"value", "evidence"} | None} claims for one listing.

        Nothing here is trusted: every claim still has to survive span verification
        against `item_text` in scripts.attributes before it reaches the dataset.
        """
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": item_text},
        ]
        for attempt in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=900,
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise ValueError("DeepSeek response content must be a string")
            parsed = self._loads(content)
            try:
                return self._validate_extraction_shape(parsed)
            except ValueError:
                if attempt == 1:
                    raise
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous JSON violated the schema. Regenerate the complete "
                            "JSON now. Every non-null attribute must be an object with exactly "
                            "the string fields value and evidence; never use a bare string."
                        ),
                    }
                )
        raise RuntimeError("unreachable")

    @staticmethod
    def _validate_extraction_shape(parsed: object) -> dict[str, object]:
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek extraction response must be a JSON object")
        required = (
            "material",
            "color",
            "size",
            "size_options",
            "style",
            "use_case",
            "feature",
            "other",
        )
        missing = [key for key in required if key not in parsed]
        if missing:
            raise ValueError(f"DeepSeek extraction response missing keys: {missing}")
        for key in required:
            value = parsed[key]
            if key == "size_options":
                if value is None:
                    parsed[key] = []
                elif not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value
                ):
                    raise ValueError("size_options must be an array of strings")
                continue
            if value is not None and (
                not isinstance(value, dict)
                or not isinstance(value.get("value"), str)
                or not isinstance(value.get("evidence"), str)
            ):
                raise ValueError(
                    f"{key} must be null or an object with value and evidence strings"
                )
        return parsed

    def describe_attribute(
        self,
        category: str,
        attribute: str,
        value: str,
        stage: str,
        previous: dict[str, dict[str, list[str]]] | None = None,
        budget_context: dict[str, float] | None = None,
    ) -> list[str]:
        """Generate one attribute's compact fragments with progressively broader context."""
        if stage not in {"discovery", "browsing", "buying"}:
            raise ValueError(f"unknown intent stage: {stage}")
        payload: dict[str, object] = {
            "category_context": category,
            "attribute": attribute,
            "ground_truth_value": value,
            "stage": stage,
            "previous_stage_clues": previous or {},
        }
        if budget_context is not None:
            payload["budget_context"] = budget_context
        messages: list[dict[str, str]] = [
            {"role": "system", "content": ATTRIBUTE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        for attempt in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=220,
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content
            try:
                clues = self._parse_clues(content)
                self._validate_stage_clues(
                    attribute, value, stage, clues, budget_context, previous
                )
                return clues
            except ValueError as error:
                if attempt == 1:
                    raise
                messages.append({"role": "assistant", "content": content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f'Invalid output: {error}. Regenerate only {{"clues":[...]}}: '
                            "a non-empty list of compact fragments, not sentences, with no first-person "
                            "subject or final punctuation. Do not repeat any fragment quoted in the "
                            "error; use a broader paraphrase or omit that fragment."
                        ),
                    }
                )
        raise RuntimeError("unreachable")

    def describe(
        self,
        category: str,
        attribute_values: dict[str, str],
        budget_context: dict[str, float] | None = None,
        cache_dir: Path | None = None,
    ) -> dict[str, dict[str, list[str]]]:
        """Generate every attribute independently, from specific to ambiguous."""
        description_attributes = dict(attribute_values)
        size_options = description_attributes.pop("size_options", None)
        if size_options:
            description_attributes["size"] = f"available options: {size_options}"
        descriptions: dict[str, dict[str, list[str]]] = {
            "buying": {},
            "browsing": {},
            "discovery": {},
        }
        for stage in ("buying", "browsing", "discovery"):
            previous = {
                previous_stage: descriptions[previous_stage]
                for previous_stage in ("buying", "browsing")
                if descriptions[previous_stage]
            }
            for attribute, value in description_attributes.items():
                compute = lambda: {
                    "clues": self.describe_attribute(
                        category, attribute, str(value), stage, previous, budget_context
                    )
                }
                clues = (
                    self._parse_clue_value(
                        cached_json_call(
                            cache_dir / stage / f"{attribute}.json", compute
                        )
                    )
                    if cache_dir is not None
                    else compute()["clues"]
                )
                self._validate_stage_clues(
                    attribute, str(value), stage, clues, budget_context, previous
                )
                descriptions[stage][attribute] = clues
        return descriptions

    def describe_modification(
        self,
        category: str,
        fake_values: dict[str, str],
        true_descriptions: dict[str, dict[str, object]],
        budget_context: dict[str, float] | None = None,
        cache_dir: Path | None = None,
    ) -> dict[str, dict[str, dict[str, list[str]]]]:
        fake_descriptions: dict[str, dict[str, list[str]]] = {
            "buying": {},
            "browsing": {},
            "discovery": {},
        }
        correction_messages: dict[str, dict[str, list[str]]] = {
            "buying": {},
            "browsing": {},
            "discovery": {},
        }
        for stage in ("buying", "browsing", "discovery"):
            previous = {
                previous_stage: fake_descriptions[previous_stage]
                for previous_stage in ("buying", "browsing")
                if fake_descriptions[previous_stage]
            }
            for attribute, fake_value in fake_values.items():
                compute = lambda: {
                    "clues": self.describe_attribute(
                        category,
                        attribute,
                        str(fake_value),
                        stage,
                        previous,
                        budget_context,
                    )
                }
                clues = (
                    self._parse_clue_value(
                        cached_json_call(
                            cache_dir / stage / f"{attribute}.json", compute
                        )
                    )
                    if cache_dir is not None
                    else compute()["clues"]
                )
                self._validate_stage_clues(
                    attribute, str(fake_value), stage, clues, budget_context, previous
                )
                fake_descriptions[stage][attribute] = clues
                true_clues = self._coerce_clues(
                    true_descriptions.get(stage, {}).get(attribute)
                )
                if not true_clues:
                    raise ValueError(
                        f"missing true intent description for {stage}.{attribute}"
                    )
                prior_corrections = [
                    correction_messages[prior_stage][attribute][0]
                    for prior_stage in ("buying", "browsing")
                    if attribute in correction_messages[prior_stage]
                ]
                correction_compute = lambda: {
                    "message": self.correct_attribute(
                        attribute, stage, clues, true_clues, prior_corrections
                    )
                }
                correction = (
                    self._parse_correction_value(
                        cached_json_call(
                            cache_dir / "corrections-v1" / stage / f"{attribute}.json",
                            correction_compute,
                        )
                    )
                    if cache_dir is not None
                    else correction_compute()["message"]
                )
                self._validate_correction(correction, true_clues, prior_corrections)
                correction_messages[stage][attribute] = [correction]
        return {
            "fake_descriptions": fake_descriptions,
            "correction_messages": correction_messages,
        }

    def correct_attribute(
        self,
        attribute: str,
        stage: str,
        fake_clues: list[str],
        true_clues: list[str],
        previous_corrections: list[str],
    ) -> str:
        payload = {
            "attribute": attribute,
            "stage": stage,
            "false_clues": fake_clues,
            "true_clues": true_clues,
            "previous_stage_corrections": previous_corrections,
        }
        messages: list[dict[str, str]] = [
            {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        for attempt in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=300,
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content
            try:
                correction = self._parse_correction(content)
                self._validate_correction(correction, true_clues, previous_corrections)
                return correction
            except ValueError as error:
                if attempt == 1:
                    raise
                messages.append({"role": "assistant", "content": content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f'Invalid output: {error}. Regenerate only {{"message":"..."}}. '
                            "Faithfully express every true clue, preserve exact numbers and named "
                            "values, and do not repeat an earlier correction."
                        ),
                    }
                )
        raise RuntimeError("unreachable")

    @staticmethod
    def _loads(content: str) -> object:
        """Parse a JSON body that may arrive fenced or wrapped in prose."""
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for start in (
            position for position, character in enumerate(text) if character == "{"
        ):
            try:
                candidate, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                return candidate
        preview = " ".join(text.split())[:240]
        raise ValueError(
            f"DeepSeek response must contain a JSON object; received: {preview!r}"
        )

    @staticmethod
    def _parse_clues(content: object) -> list[str]:
        if not isinstance(content, str):
            raise ValueError("DeepSeek response content must be a string")
        parsed = DeepSeekAttributeWriter._loads(content)
        return DeepSeekAttributeWriter._parse_clue_value(parsed)

    @staticmethod
    def _parse_clue_value(parsed: object) -> list[str]:
        clues = parsed.get("clues") if isinstance(parsed, dict) else None
        if not isinstance(clues, list) or not clues:
            raise ValueError("DeepSeek response must contain a non-empty clue list")
        cleaned: list[str] = []
        for clue in clues:
            if not isinstance(clue, str) or not clue.strip():
                raise ValueError("DeepSeek clue fragments must be non-empty strings")
            text = clue.strip()
            if len(text.split()) > 20:
                raise ValueError("DeepSeek clue fragment exceeds 20 words")
            cleaned.append(text)
        return cleaned

    @staticmethod
    def _coerce_clues(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(clue).strip() for clue in value if str(clue).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _parse_correction(content: object) -> str:
        if not isinstance(content, str):
            raise ValueError("DeepSeek correction content must be a string")
        return DeepSeekAttributeWriter._parse_correction_value(
            DeepSeekAttributeWriter._loads(content)
        )

    @staticmethod
    def _parse_correction_value(parsed: object) -> str:
        message = parsed.get("message") if isinstance(parsed, dict) else None
        if not isinstance(message, str) or not message.strip():
            raise ValueError("DeepSeek correction must contain a non-empty message")
        message = message.strip()
        if len(message) > 500 or not message.endswith((".", "!", "?")):
            raise ValueError(
                "DeepSeek correction must be one punctuated sentence <= 500 chars"
            )
        return message

    @staticmethod
    def _validate_correction(
        message: str, true_clues: list[str], previous_corrections: list[str]
    ) -> None:
        if not true_clues:
            raise ValueError("correction requires current-intent true clues")
        folded = message.casefold()
        if folded in {correction.casefold() for correction in previous_corrections}:
            raise ValueError("correction must differ across intent stages")

    @staticmethod
    def _validate_stage_clues(
        attribute: str,
        value: str,
        stage: str,
        clues: list[str],
        budget_context: dict[str, float] | None,
        previous: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        combined = " ".join(clues).casefold()
        if stage == "buying":
            if attribute == "budget" and budget_context is not None:
                exact_price = float(budget_context["exact_price"])
                if not any(
                    abs(float(number) - exact_price) < 0.01
                    for number in re.findall(r"\d+(?:\.\d+)?", combined)
                ):
                    raise ValueError("buying budget must include the exact price")
            return
        raw_value = value.strip().casefold()
        if attribute == "brand" and any(
            token in combined
            for token in re.findall(r"[a-z0-9]+", raw_value)
            if len(token) >= 3
        ):
            raise ValueError(f"{stage} must not reveal the brand")
        if stage == "discovery" and re.search(r"\d", combined):
            raise ValueError("discovery must not contain numeric details")
        if stage == "browsing" and re.search(
            r"\b(?:unspecified|undecided)\b", combined
        ):
            raise ValueError(
                "browsing must generalize known facts, not mark them unknown"
            )
        unsupported = (
            "durable",
            "premium",
            "affordable",
            "lightweight",
            "comfortable",
            "waterproof",
            "weatherproof",
        )
        invented = [
            word for word in unsupported if word in combined and word not in raw_value
        ]
        if invented:
            raise ValueError(
                f"{stage} must not invent or strengthen qualities: {invented!r}"
            )
        if (
            stage == "discovery"
            and attribute in {"category", "feature"}
            and any(
                phrase in combined
                for phrase in (
                    "timekeeping",
                    "tells time",
                    "wrist-worn",
                    "neckwear",
                    "waist accessory",
                )
            )
        ):
            raise ValueError(
                "discovery must not identify the product through its function"
            )
        if attribute == "budget" and budget_context is not None:
            exact_price = float(budget_context["exact_price"])
            for number in re.findall(r"\d+(?:\.\d+)?", combined):
                if abs(float(number) - exact_price) <= max(1.0, exact_price * 0.05):
                    raise ValueError(
                        f"{stage} must not reveal or round the exact price"
                    )
