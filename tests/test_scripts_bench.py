from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.attributes import (
    browsing_budget_ceiling,
    derived_attributes,
    price_band,
    verify_extraction,
)
from scripts.llm_client import DeepSeekAttributeWriter, cached_json_call
from scripts.modification import _conflicts_with_truth, build_modification
from scripts.query_handler import ACTIVE_ATTRIBUTE_COUNT, QueryHandler
from scripts.session import MODIFICATION_SESSION_RATE, create_session
from scripts.schema import Item
from scripts.schema import Modification

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

SAMPLE_ATTRIBUTES = derived_attributes(SAMPLE_PRODUCT)


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

    def describe_modification(
        self,
        category: str,
        fake_values: dict[str, str],
        true_values: dict[str, str],
        budget_context: dict | None = None,
    ) -> dict:
        self.calls += 1
        self.last_budget_context = budget_context
        return {
            "fake_descriptions": {
                stage: {attribute: f"fake:{value}" for attribute, value in fake_values.items()}
                for stage in ("browsing", "buying")
            },
            "correction_messages": {
                stage: {
                    attribute: f"correction:{true_values[attribute]}"
                    for attribute in fake_values
                }
                for stage in ("browsing", "buying")
            },
        }


class PriceBandTest(unittest.TestCase):
    def test_price_band_boundaries(self) -> None:
        self.assertEqual(price_band(10), "under_15")
        self.assertEqual(price_band(15), "15_to_30")
        self.assertEqual(price_band(999), "80_plus")
        self.assertIsNone(price_band(None))

    def test_browsing_budget_ceiling_is_wider_than_exact_price(self) -> None:
        ceiling = browsing_budget_ceiling(36.5)
        self.assertGreater(ceiling, 36.5)


class QueryHandlerTest(unittest.TestCase):
    def _item(self) -> Item:
        descriptions = {
            intent: {attribute: f"{intent}:{attribute}" for attribute in (
                "category", "brand", "budget", "material", "feature", "other"
            )}
            for intent in ("browsing", "buying")
        }
        return Item("ITEM", {}, descriptions)

    def test_selects_exactly_four_attributes_deterministically(self) -> None:
        first = QueryHandler("session-1", self._item())
        second = QueryHandler("session-1", self._item())
        self.assertEqual(first.active_attributes, second.active_attributes)
        self.assertEqual(len(first.active_attributes), ACTIVE_ATTRIBUTE_COUNT)

    def test_other_reveals_only_one_active_attribute_at_a_time(self) -> None:
        handler = QueryHandler("session-2", self._item())
        first = handler.answer("other")
        self.assertIsNotNone(first)
        self.assertEqual(len(handler.disclosed_attributes), 1)
        second = handler.answer("other")
        self.assertIsNotNone(second)
        self.assertEqual(len(handler.disclosed_attributes), 2)

    def test_inactive_attribute_is_not_revealed(self) -> None:
        handler = QueryHandler("session-3", self._item())
        inactive = next(name for name in ("category", "brand", "budget", "material", "feature", "other") if name not in handler.active_attributes)
        self.assertIsNone(handler.answer(inactive))

    def test_fewer_than_four_attributes_is_rejected(self) -> None:
        item = Item(
            "ITEM_WITH_THREE_ATTRIBUTES",
            {},
            {"browsing": {"category": "shirts", "brand": "Acme", "budget": "under $30"}},
        )
        with self.assertRaises(ValueError):
            QueryHandler("session-4", item)

    def test_modification_switches_fake_to_true_and_corrects_prior_disclosure(self) -> None:
        item = Item(
            "ITEM",
            {},
            {
                "browsing": {
                    "category": "browsing:category",
                    "material": "browsing:material",
                    "style": "browsing:style",
                    "feature": "browsing:feature",
                },
                "buying": {
                    "category": "buying:category",
                    "material": "buying:material",
                    "style": "buying:style",
                    "feature": "buying:feature",
                },
            },
        )
        modification = Modification(
            item_id="ITEM",
            fake_attributes={"material": {"browsing": "fake material", "buying": "fake material"}},
            correction_messages={"material": {"browsing": "correction text", "buying": "correction text"}},
            modify_turn=3,
        )
        handler = QueryHandler("session-mod", item, modification=modification)
        self.assertIn("material", handler.active_attributes)
        self.assertEqual(handler.answer("material", turn=1), "fake material")
        result = handler.answer("category", turn=3)
        self.assertIn("category", result)
        self.assertIn("correct", result)
        self.assertIn("correction text", result)

    def test_modification_is_enabled_at_session_level(self) -> None:
        item = self._item()
        modification = Modification(
            "ITEM",
            {"material": {"browsing": "fake", "buying": "fake"}},
            {"material": {"browsing": "correction", "buying": "correction"}},
            3,
        )
        sessions = [create_session(f"session-{index}", item, modification) for index in range(100)]
        enabled = sum(session.modification is not None for session in sessions)
        self.assertGreater(enabled, 0)
        self.assertLess(enabled, 100)
        self.assertEqual(MODIFICATION_SESSION_RATE, 0.30)

    def test_enabled_modification_attributes_are_active(self) -> None:
        item = self._item()
        modification = Modification(
            "ITEM",
            {"other": {"browsing": "fake", "buying": "fake"}},
            {"other": {"browsing": "correction", "buying": "correction"}},
            3,
        )
        for index in range(100):
            session = create_session(f"session-{index}", item, modification)
            if session.modification is not None:
                self.assertIn("other", session.query_handler.active_attributes)


class SpanVerificationTest(unittest.TestCase):
    """The gates that make a model-extracted value trustworthy enough to score against."""

    NECKLACE = {
        "parent_asin": "B0TESTNECK1",
        "title": "Pendant Necklace",
        "features": ["Material:alloy"],
        "description": ["5.it is not real gold and silver products and It will fade away slowly"],
        "price": 9.99,
        "categories": ["Clothing, Shoes & Jewelry", "Jewelry", "Necklaces"],
        "details": {},
        "store": "QIAN0813",
    }

    def test_verified_claim_is_kept(self) -> None:
        verified, rejected = verify_extraction(
            self.NECKLACE, {"material": {"value": "alloy", "evidence": "Material:alloy"}}
        )
        self.assertEqual(verified["material"], "alloy")
        self.assertEqual(rejected, [])

    def test_verified_other_claim_is_kept(self) -> None:
        verified, rejected = verify_extraction(
            self.NECKLACE,
            {"other": {"value": "pendant necklace", "evidence": "Pendant Necklace"}},
        )
        self.assertEqual(verified["other"], "pendant necklace")
        self.assertEqual(rejected, [])

    def test_verified_size_options_are_kept_for_description_generation(self) -> None:
        verified, rejected = verify_extraction(
            self.NECKLACE,
            {"size": None, "size_options": ["S", "M", "XL"]},
        )
        self.assertEqual(verified["size_options"], "S, M, XL")
        self.assertNotIn("size", verified)

    def test_invented_evidence_is_rejected(self) -> None:
        verified, rejected = verify_extraction(
            self.NECKLACE,
            {"use_case": {"value": "not for swimming", "evidence": "Do not wear while swimming."}},
        )
        self.assertNotIn("use_case", verified)
        self.assertIn("evidence not found", rejected[0])

    def test_use_case_negation_is_preserved(self) -> None:
        product = {
            **self.NECKLACE,
            "description": ["Do not wear while swimming."],
        }
        verified, rejected = verify_extraction(
            product,
            {"use_case": {"value": "do not wear while swimming", "evidence": "Do not wear while swimming."}},
        )
        self.assertEqual(verified["use_case"], "do not wear while swimming")
        self.assertEqual(rejected, [])

    def test_negated_evidence_is_rejected(self) -> None:
        # The quote is verbatim, so a substring check alone would accept it.
        verified, rejected = verify_extraction(
            self.NECKLACE,
            {
                "material": {
                    "value": "gold and silver",
                    "evidence": "it is not real gold and silver products",
                }
            },
        )
        self.assertNotIn("material", verified)
        self.assertIn("negated", rejected[0])

    def test_value_unsupported_by_its_evidence_is_rejected(self) -> None:
        verified, rejected = verify_extraction(
            self.NECKLACE, {"color": {"value": "turquoise", "evidence": "Material:alloy"}}
        )
        self.assertNotIn("color", verified)
        self.assertIn("not supported", rejected[0])

    def test_missing_claim_is_silently_absent(self) -> None:
        verified, rejected = verify_extraction(self.NECKLACE, {"color": None})
        self.assertEqual(verified, {})
        self.assertEqual(rejected, [])

    def test_size_is_dropped_when_the_listing_stocks_a_range(self) -> None:
        product = {**self.NECKLACE, "features": ["Available in XL"]}
        verified, rejected = verify_extraction(
            product,
            {
                "size": {"value": "XL", "evidence": "Available in XL"},
                "size_options": ["S", "M", "L", "XL"],
            },
        )
        self.assertNotIn("size", verified)
        self.assertIn("stocks 4 sizes", rejected[0])

    def test_evidence_may_be_reflowed_but_not_reworded(self) -> None:
        verified, _ = verify_extraction(
            self.NECKLACE, {"material": {"value": "alloy", "evidence": "Material: alloy"}}
        )
        self.assertEqual(verified["material"], "alloy")

    def test_combined_evidence_spans_are_rejected(self) -> None:
        verified, rejected = verify_extraction(
            self.NECKLACE,
            {
                "feature": {
                    "value": "alloy and pendant",
                    "evidence": "Material:alloy; Pendant Necklace",
                }
            },
        )
        self.assertNotIn("feature", verified)
        self.assertIn("evidence not found", rejected[0])


class FakeValueTest(unittest.TestCase):
    def test_fake_value_may_not_overlap_the_true_value(self) -> None:
        # "mesh" describes an item whose true material is "polyester and mesh", so it
        # is not a modification at all — the agent's original answer stays correct.
        self.assertTrue(_conflicts_with_truth("mesh", "polyester and mesh"))
        self.assertFalse(_conflicts_with_truth("linen", "polyester and mesh"))

    def test_build_modification_fakes_do_not_restate_the_truth(self) -> None:
        attributes = {**SAMPLE_ATTRIBUTES, "material": "polyester and mesh"}
        with TemporaryDirectory() as directory:
            modification = build_modification(
                SAMPLE_PRODUCT, "B0TESTITEM1", attributes, FakeWriter(), Path(directory)
            )
            self.assertIsNotNone(modification)
            if "material" in modification.fake_attributes:
                self.assertNotIn("polyester and mesh", modification.fake_attributes["material"]["browsing"])
            self.assertEqual(set(modification.correction_messages), set(modification.fake_attributes))


class ModificationTest(unittest.TestCase):
    def test_build_modification_is_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            writer_a = FakeWriter()
            writer_b = FakeWriter()
            first = build_modification(SAMPLE_PRODUCT, "B0TESTITEM1", SAMPLE_ATTRIBUTES, writer_a, cache_dir / "a")
            second = build_modification(SAMPLE_PRODUCT, "B0TESTITEM1", SAMPLE_ATTRIBUTES, writer_b, cache_dir / "b")
            self.assertIsNotNone(first)
            self.assertEqual(first, second)
            self.assertIn(first.modify_turn, (3, 4))
            self.assertEqual(first.correction_messages, second.correction_messages)

    def test_build_modification_fake_values_differ_from_truth(self) -> None:
        with TemporaryDirectory() as directory:
            modification = build_modification(
                SAMPLE_PRODUCT, "B0TESTITEM1", SAMPLE_ATTRIBUTES, FakeWriter(), Path(directory)
            )
            self.assertIsNotNone(modification)
            for attribute, texts in modification.fake_attributes.items():
                self.assertNotIn(attribute, ("category", "brand"))
                self.assertTrue(texts["browsing"])
                self.assertTrue(texts["buying"])

    def test_build_modification_passes_numeric_budget_context_for_fake_budget(self) -> None:
        with TemporaryDirectory() as directory:
            writer = FakeWriter()
            modification = build_modification(
                SAMPLE_PRODUCT, "B0TESTITEM1", SAMPLE_ATTRIBUTES, writer, Path(directory)
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
        valid = json.dumps({"browsing": {"material": "soft-ish", "feature": "stretch"}, "buying": {"material": "cotton blend", "feature": "stretchy"}})
        parsed = DeepSeekAttributeWriter._parse(valid, {"material", "feature"})
        self.assertEqual(parsed["browsing"]["material"], "soft-ish")

        with self.assertRaises(ValueError):
            DeepSeekAttributeWriter._parse(json.dumps({"browsing": {"material": "x"}}), {"material"})
        with self.assertRaises(ValueError):
            DeepSeekAttributeWriter._parse("not json", {"material"})

    def test_validate_extraction_shape_rejects_bare_attribute_strings(self) -> None:
        invalid = {
            "material": "Stainless Steel",
            "color": None,
            "size": None,
            "size_options": [],
            "style": None,
            "use_case": None,
            "feature": None,
            "other": None,
        }
        with self.assertRaises(ValueError):
            DeepSeekAttributeWriter._validate_extraction_shape(invalid)

    def test_validate_extraction_shape_accepts_value_and_evidence_objects(self) -> None:
        valid = {
            "material": {"value": "stainless steel", "evidence": "Stainless Steel Band"},
            "color": None,
            "size": None,
            "size_options": [],
            "style": None,
            "use_case": None,
            "feature": {"value": "water resistant", "evidence": "Water Resistant"},
            "other": None,
        }
        self.assertEqual(DeepSeekAttributeWriter._validate_extraction_shape(valid), valid)

    def test_validate_extraction_shape_normalizes_missing_size_options(self) -> None:
        claims = {
            "material": None,
            "color": None,
            "size": None,
            "size_options": None,
            "style": None,
            "use_case": None,
            "feature": None,
            "other": None,
        }
        normalized = DeepSeekAttributeWriter._validate_extraction_shape(claims)
        self.assertEqual(normalized["size_options"], [])


if __name__ == "__main__":
    unittest.main()
