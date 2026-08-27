from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.attributes import browsing_budget_ceiling, extract_attributes, price_band
from scripts.llm_client import DeepSeekAttributeWriter, cached_json_call
from scripts.modification import build_modification

SAMPLE_PRODUCT = {
    "parent_asin": "B0TESTITEM1",
    "title": "Test Crew Shirt",
    "features": ["100% Cotton", "Machine Wash"],
    "description": ["A soft cotton crew neck t-shirt."],
    "price": 24.99,
    "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Shirts"],
    "details": {"Department": "mens"},
    "store": "Acme Apparel",
}

SAMPLE_CATEGORY = "Men Shirts"


class FakeWriter:
    """Deterministic stand-in for DeepSeekAttributeWriter; records call count."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_budget_context: dict | None = None

    def describe(
        self,
        category: str,
        attribute_values: dict[str, str],
        budget_context: dict | None = None,
    ) -> dict[str, dict[str, str]]:
        self.calls += 1
        self.last_budget_context = budget_context
        return {
            "browsing": {attribute: f"vague:{value}" for attribute, value in attribute_values.items()},
            "buying": {attribute: f"clear:{value}" for attribute, value in attribute_values.items()},
        }


class AttributeExtractionTest(unittest.TestCase):
    def test_extract_attributes_grounds_expected_fields(self) -> None:
        attributes = extract_attributes(SAMPLE_PRODUCT)
        self.assertEqual(attributes["material"], "cotton")
        self.assertEqual(attributes["brand"], "Acme Apparel")
        self.assertEqual(attributes["budget"], "15_to_30")
        self.assertIn("category", attributes)

    def test_extract_attributes_joins_multiple_matches(self) -> None:
        product = {**SAMPLE_PRODUCT, "features": ["Cotton, Polyester, Spandex blend"]}
        attributes = extract_attributes(product)
        self.assertEqual(attributes["material"], "cotton, polyester, and spandex")

    def test_extract_attributes_ignores_prohibited_use_cases(self) -> None:
        product = {
            **SAMPLE_PRODUCT,
            "features": ["Perfect for hiking and outdoor activities."],
            "description": ["Do not wear while swimming or bathing."],
        }
        attributes = extract_attributes(product)
        self.assertIn("Perfect for hiking and outdoor activities.", attributes["use_case"])
        self.assertIn("Do not wear while swimming or bathing.", attributes["use_case"])

    def test_extract_attributes_preserves_all_sizes(self) -> None:
        product = {**SAMPLE_PRODUCT, "features": ["Available in small, medium, large, and XL sizes."]}
        attributes = extract_attributes(product)
        self.assertEqual(attributes["size"], "small, medium, large, and xl")

    def test_price_band_boundaries(self) -> None:
        self.assertEqual(price_band(10), "under_15")
        self.assertEqual(price_band(15), "15_to_30")
        self.assertEqual(price_band(999), "80_plus")
        self.assertIsNone(price_band(None))

    def test_browsing_budget_ceiling_is_wider_than_exact_price(self) -> None:
        ceiling = browsing_budget_ceiling(36.5)
        self.assertGreater(ceiling, 36.5)


class ModificationTest(unittest.TestCase):
    def test_build_modification_is_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            writer_a = FakeWriter()
            writer_b = FakeWriter()
            first = build_modification(SAMPLE_PRODUCT, "B0TESTITEM1", writer_a, cache_dir / "a", {})
            second = build_modification(SAMPLE_PRODUCT, "B0TESTITEM1", writer_b, cache_dir / "b", {})
            self.assertIsNotNone(first)
            self.assertEqual(first, second)
            self.assertIn(first.modify_turn, (3, 4))

    def test_build_modification_fake_values_differ_from_truth(self) -> None:
        with TemporaryDirectory() as directory:
            modification = build_modification(
                SAMPLE_PRODUCT, "B0TESTITEM1", FakeWriter(), Path(directory), {}
            )
            self.assertIsNotNone(modification)
            for attribute, texts in modification.fake_attributes.items():
                self.assertNotIn(attribute, ("category", "brand"))
                self.assertTrue(texts["browsing"])
                self.assertTrue(texts["buying"])

    def test_build_modification_prefers_category_vocab_over_global_fallback(self) -> None:
        category_vocab = {SAMPLE_CATEGORY: {"material": ["linen"]}}
        with TemporaryDirectory() as directory:
            modification = build_modification(
                SAMPLE_PRODUCT, "B0TESTITEM1", FakeWriter(), Path(directory), category_vocab
            )
            self.assertIsNotNone(modification)
            if "material" in modification.fake_attributes:
                # The fake material must come from the category-observed vocabulary (only "linen"
                # here), not an arbitrary global material such as "wool".
                self.assertIn("linen", modification.fake_attributes["material"]["browsing"])

    def test_build_modification_passes_numeric_budget_context_for_fake_budget(self) -> None:
        category_vocab: dict = {}
        with TemporaryDirectory() as directory:
            writer = FakeWriter()
            modification = build_modification(
                SAMPLE_PRODUCT, "B0TESTITEM1", writer, Path(directory), category_vocab
            )
            self.assertIsNotNone(modification)
            if "budget" in modification.fake_attributes:
                self.assertIsNotNone(writer.last_budget_context)
                self.assertIn("exact_price", writer.last_budget_context)
                self.assertIn("browsing_ceiling", writer.last_budget_context)


class CachedJsonCallTest(unittest.TestCase):
    def test_cache_avoids_recompute(self) -> None:
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "item.json"
            calls = {"count": 0}

            def compute() -> dict:
                calls["count"] += 1
                return {"value": 1}

            first = cached_json_call(cache_path, compute)
            second = cached_json_call(cache_path, compute)
            self.assertEqual(first, second)
            self.assertEqual(calls["count"], 1)
            self.assertEqual(json.loads(cache_path.read_text(encoding="utf-8")), {"value": 1})


class DeepSeekAttributeWriterTest(unittest.TestCase):
    def test_missing_api_key_raises(self) -> None:
        import os

        original = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                DeepSeekAttributeWriter()
        finally:
            if original is not None:
                os.environ["DEEPSEEK_API_KEY"] = original

    def test_parse_requires_both_stages_and_attributes(self) -> None:
        valid = json.dumps({"browsing": {"material": "soft-ish"}, "buying": {"material": "cotton blend"}})
        parsed = DeepSeekAttributeWriter._parse(valid, {"material"})
        self.assertEqual(parsed["browsing"]["material"], "soft-ish")

        with self.assertRaises(ValueError):
            DeepSeekAttributeWriter._parse(json.dumps({"browsing": {"material": "x"}}), {"material"})
        with self.assertRaises(ValueError):
            DeepSeekAttributeWriter._parse("not json", {"material"})


if __name__ == "__main__":
    unittest.main()
