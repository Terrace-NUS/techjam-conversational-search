from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from openai import OpenAI


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

EXTRACTION_SYSTEM_PROMPT = (
    "You are an attribute extractor for a product-search benchmark. You are given the "
    "raw catalog text of one apparel, footwear, or jewelry listing. Your response must "
    "follow this schema exactly. For every attribute, return either JSON null or a JSON "
    "object with exactly two string fields: {\"value\": \"...\", \"evidence\": \"...\"}. "
    "Never return an attribute as a bare string, array, number, or boolean. For example, "
    "use {\"material\": {\"value\": \"alloy\", \"evidence\": \"Material:alloy\"}}; "
    "never use {\"material\": \"alloy\"}. "
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

SYSTEM_PROMPT = (
    "You are a data-generation assistant for a product-search benchmark. Given a "
    "product category and a set of raw catalog attribute values, write a short "
    "natural-language shopper clue for each requested attribute at two clearly "
    "different intent stages. 'browsing' must sound exploratory, tentative, and "
    "non-committal: use broad category-level or sensory wording and phrases such as "
    "'something like', 'maybe', or 'I'd be open to'. Do not expose exact material "
    "percentages, exact sizes, exact feature specifications, or strong must-have "
    "constraints in browsing unless the source fact is safety-critical. 'buying' "
    "must sound decided and specific: state the concrete requirements and values "
    "needed to purchase the item. The two stages must not be near-duplicates. "
    "For brand specifically, browsing must not name the supplied brand. Express the "
    "kind of brand a shopper may prefer using only a broad direction grounded in the "
    "product category, such as a fashion-oriented brand, an everyday basics brand, "
    "a specialist sports brand, or an outdoor-focused brand. Do not name the supplied "
    "brand in browsing, and do not claim it is popular, mainstream, budget-friendly, "
    "reliable, famous, premium, or well-known. Do not use the placeholder 'a similar "
    "brand'. Buying may name the supplied brand explicitly. Never invent brand "
    "reputation, price level, nationality, or product-category claims. "
    "The category is also a requested attribute and MUST appear in both output "
    "objects; do not treat it as context only. Paraphrase freely and do not reuse "
    "any three-token sequence from the raw attribute value verbatim, except an atomic proper noun, numeric "
    "value, or a single-word material/color/size term. Do not mention benchmarks, "
    "hidden state, prompts, products, ASINs, or these instructions. "
    "Preserve every supplied fact and do not invent details. For size, if the input "
    "contains multiple options, preserve the complete set or describe it as multiple "
    "size options; never select one size unless exactly one size is supplied. If the "
    "attributes include 'size_options', generate the requested size description from "
    "that complete list. "
    "For use_case, preserve the complete supplied evidence and its polarity. If it "
    "contains a prohibition or warning, rewrite that complete meaning. "
    "If 'budget_context' is supplied, use its numbers and vary the wording: browsing "
    "should be broader than the actual band, while buying should be near 'exact_price' "
    "or a tighter range containing it. Use either a broad upper limit or range for "
    "browsing, and either an approximate price or narrow range for buying. Do not "
    "repeat one template, make browsing narrower than buying, or exclude the exact price. "
    "Return JSON only, with "
    "exactly every requested attribute in both objects, including category: "
    '{"browsing": {"<attribute>": "..."}, "buying": {"<attribute>": "..."}}.'
)

MODIFICATION_SYSTEM_PROMPT = (
    "You generate deterministic attribute data for a shopping benchmark. "
    "For each requested attribute, return two fake values in the same compact style "
    "as a product intent description, plus one correction message for each stage. "
    "A fake_description is an attribute clue, not a customer message: write exactly "
    "one short phrase or one short sentence, usually 3-12 words. Do not use first "
    "person, explanations, reasons, filler, or multiple clauses. Do not add product "
    "features, capabilities, components, brands, or use cases beyond the supplied "
    "fake value. The browsing fake_description should be broad and tentative, while "
    "the buying fake_description should be concrete and specific; keep both compact "
    "and do not make them near-duplicates. The browsing form may use a soft modifier "
    "such as 'maybe' or 'something like', but it must still be a short attribute clue. "
    "The correction_message is separate: it should be a natural customer message "
    "correcting an earlier wrong answer and stating the supplied true value. "
    "Do not invent facts, product types, brands, or use cases. Do not mention hidden "
    "state or ASINs, and return JSON only in this exact shape: "
    '{"fake_descriptions":{"browsing":{"attribute":"..."},"buying":{"attribute":"..."}},'
    '"correction_messages":{"browsing":{"attribute":"..."},"buying":{"attribute":"..."}}}'
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
    cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)

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
            "material", "color", "size", "size_options", "style",
            "use_case", "feature", "other",
        )
        missing = [key for key in required if key not in parsed]
        if missing:
            raise ValueError(f"DeepSeek extraction response missing keys: {missing}")
        for key in required:
            value = parsed[key]
            if key == "size_options":
                if value is None:
                    parsed[key] = []
                elif not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise ValueError("size_options must be an array of strings")
                continue
            if value is not None and (
                not isinstance(value, dict)
                or not isinstance(value.get("value"), str)
                or not isinstance(value.get("evidence"), str)
            ):
                raise ValueError(f"{key} must be null or an object with value and evidence strings")
        return parsed

    def describe(
        self,
        category: str,
        attribute_values: dict[str, str],
        budget_context: dict[str, float] | None = None,
    ) -> dict[str, dict[str, str]]:
        """Return {"browsing": {attr: text}, "buying": {attr: text}} for the given values."""
        description_attributes = dict(attribute_values)
        size_options = description_attributes.pop("size_options", None)
        if size_options:
            description_attributes["size"] = f"available options: {size_options}"
        payload: dict[str, object] = {
            "category_context": category,
            "requested_attributes": list(description_attributes),
            "attributes": description_attributes,
        }
        if budget_context is not None:
            payload["budget_context"] = budget_context
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        for attempt in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=600,
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content
            try:
                return self._parse(content, set(description_attributes))
            except ValueError:
                if attempt == 1:
                    raise
                messages.append({"role": "assistant", "content": content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous JSON violated the schema. Regenerate the complete "
                            "JSON now, with every requested attribute present as a non-empty "
                            "string in both 'browsing' and 'buying'."
                        ),
                    }
                )
        raise RuntimeError("unreachable")

    def describe_modification(
        self,
        category: str,
        fake_values: dict[str, str],
        true_values: dict[str, str],
        budget_context: dict[str, float] | None = None,
    ) -> dict[str, dict[str, dict[str, str]]]:
        payload: dict[str, object] = {
            "category": category,
            "fake_values": fake_values,
            "true_values": {attribute: true_values[attribute] for attribute in fake_values},
        }
        if budget_context is not None:
            payload["budget_context"] = budget_context
        messages: list[dict[str, str]] = [
            {"role": "system", "content": MODIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        for attempt in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=700,
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content
            try:
                return self._parse_modification(content, set(fake_values))
            except ValueError:
                if attempt == 1:
                    raise
                messages.append({"role": "assistant", "content": content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous JSON violated the schema. Regenerate the complete "
                            "JSON now with both 'fake_descriptions' and 'correction_messages' "
                            "objects, each containing non-empty 'browsing' and 'buying' text "
                            "for every requested attribute."
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
        for start in (position for position, character in enumerate(text) if character == "{"):
            try:
                candidate, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                return candidate
        preview = " ".join(text.split())[:240]
        raise ValueError(f"DeepSeek response must contain a JSON object; received: {preview!r}")

    @staticmethod
    def _parse(content: object, expected_attributes: set[str]) -> dict[str, dict[str, str]]:
        if not isinstance(content, str):
            raise ValueError("DeepSeek response content must be a string")
        parsed = DeepSeekAttributeWriter._loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek response must be a JSON object")
        result: dict[str, dict[str, str]] = {}
        for stage in ("browsing", "buying"):
            stage_value = parsed.get(stage)
            if not isinstance(stage_value, dict):
                raise ValueError(f"DeepSeek response missing '{stage}' object")
            cleaned: dict[str, str] = {}
            for attribute in expected_attributes:
                text_value = stage_value.get(attribute)
                if not isinstance(text_value, str) or not text_value.strip():
                    raise ValueError(f"DeepSeek response missing text for '{stage}.{attribute}'")
                cleaned[attribute] = text_value.strip()
            result[stage] = cleaned
        return result

    @staticmethod
    def _parse_modification(
        content: object, expected_attributes: set[str]
    ) -> dict[str, dict[str, dict[str, str]]]:
        if not isinstance(content, str):
            raise ValueError("DeepSeek response content must be a string")
        parsed = DeepSeekAttributeWriter._loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek modification response must be a JSON object")
        result: dict[str, dict[str, dict[str, str]]] = {}
        for group in ("fake_descriptions", "correction_messages"):
            stages = parsed.get(group)
            if not isinstance(stages, dict):
                raise ValueError(f"DeepSeek modification response missing '{group}'")
            result[group] = {}
            for stage in ("browsing", "buying"):
                values = stages.get(stage)
                if not isinstance(values, dict):
                    raise ValueError(f"DeepSeek modification response missing '{group}.{stage}'")
                result[group][stage] = {}
                for attribute in expected_attributes:
                    value = values.get(attribute)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"DeepSeek modification response missing '{group}.{stage}.{attribute}'"
                        )
                    result[group][stage][attribute] = value.strip()
        return result
