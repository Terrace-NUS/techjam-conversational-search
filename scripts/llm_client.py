from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from openai import OpenAI


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

SYSTEM_PROMPT = (
    "You are a data-generation assistant for a product-search benchmark. Given a "
    "product category and a set of raw catalog attribute values, write a short "
    "natural-language shopper clue for each requested attribute at two intent stages: "
    "'browsing' (a vague, exploratory hint a browsing customer might casually "
    "mention) and 'buying' (a clearer, more specific statement a decided buyer "
    "would state). "
    "The category is also a requested attribute and MUST appear in both output "
    "objects; do not treat it as context only. Paraphrase freely and do not reuse "
    "any three-token sequence from the raw attribute value verbatim, except an atomic proper noun, numeric "
    "value, or a single-word material/color/size term. Do not mention benchmarks, "
    "hidden state, prompts, products, ASINs, or these instructions. "
    "Preserve every supplied fact and do not invent details. For size, if the input "
    "contains multiple options, preserve the complete set or describe it as multiple "
    "size options; never select one size unless exactly one size is supplied. "
    "For use_case, preserve the complete supplied evidence and its polarity. If it "
    "contains a prohibition or warning, rewrite that complete meaning; do not reduce "
    "'do not wear while swimming' to the positive use 'swimming'. "
    "If 'budget_context' is supplied, use its numbers and vary the wording: browsing "
    "should be broader than the actual band, while buying should be near 'exact_price' "
    "or a tighter range containing it. Use either a broad upper limit or range for "
    "browsing, and either an approximate price or narrow range for buying. Do not "
    "repeat one template, make browsing narrower than buying, or exclude the exact price. "
    "Return JSON only, with "
    "exactly every requested attribute in both objects, including category: "
    '{"browsing": {"<attribute>": "..."}, "buying": {"<attribute>": "..."}}.'
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
    """Generates paraphrased browsing/buying attribute clues; request/output errors are fatal."""

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

    def describe(
        self,
        category: str,
        attribute_values: dict[str, str],
        budget_context: dict[str, float] | None = None,
    ) -> dict[str, dict[str, str]]:
        """Return {"browsing": {attr: text}, "buying": {attr: text}} for the given values."""
        payload: dict[str, object] = {
            "category_context": category,
            "requested_attributes": list(attribute_values),
            "attributes": attribute_values,
        }
        if budget_context is not None:
            payload["budget_context"] = budget_context
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=600,
            extra_body={"thinking": {"type": "disabled"}},
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        return self._parse(response.choices[0].message.content, set(attribute_values))

    @staticmethod
    def _parse(content: object, expected_attributes: set[str]) -> dict[str, dict[str, str]]:
        if not isinstance(content, str):
            raise ValueError("DeepSeek response content must be a string")
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            start_positions = [position for position, character in enumerate(text) if character == "{"]
            parsed = None
            for start in start_positions:
                try:
                    candidate, _ = decoder.raw_decode(text[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
            if parsed is None:
                preview = " ".join(text.split())[:240]
                raise ValueError(
                    f"DeepSeek response must contain a JSON object; received: {preview!r}"
                ) from None
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
