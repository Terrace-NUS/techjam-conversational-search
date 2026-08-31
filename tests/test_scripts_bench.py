from __future__ import annotations

import json
import unittest
from collections import Counter
from copy import deepcopy
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.attributes import (
    browsing_budget_ceiling,
    derived_attributes,
    price_band,
    verify_extraction,
)
from scripts.build_dataset import _label_v2_samples, _v2_row
from scripts.llm_client import DeepSeekAttributeWriter, cached_json_call
from scripts.modification import _conflicts_with_truth, build_modification
from scripts.query_handler import QueryHandler
from scripts.session import create_session
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


def true_descriptions(attributes: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    values = dict(attributes)
    size_options = values.pop("size_options", None)
    if size_options:
        values["size"] = f"available options: {size_options}"
    return {
        stage: {
            attribute: [f"{stage}-true:{value}"] for attribute, value in values.items()
        }
        for stage in ("discovery", "browsing", "buying")
    }


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
        cache_dir: Path | None = None,
    ) -> dict[str, dict[str, list[str]]]:
        self.calls += 1
        self.last_budget_context = budget_context
        return {
            "discovery": {
                attribute: [f"open:{value}"]
                for attribute, value in attribute_values.items()
            },
            "browsing": {
                attribute: [f"vague:{value}"]
                for attribute, value in attribute_values.items()
            },
            "buying": {
                attribute: [f"clear:{value}"]
                for attribute, value in attribute_values.items()
            },
        }

    def describe_modification(
        self,
        category: str,
        fake_values: dict[str, str],
        true_intents: dict[str, dict[str, list[str]]],
        budget_context: dict | None = None,
        cache_dir: Path | None = None,
    ) -> dict:
        self.calls += 1
        self.last_budget_context = budget_context
        return {
            "fake_descriptions": {
                stage: {
                    attribute: [f"fake:{value}"]
                    for attribute, value in fake_values.items()
                }
                for stage in ("discovery", "browsing", "buying")
            },
            "correction_messages": {
                stage: {
                    attribute: [f"correction:{true_intents[stage][attribute][0]}"]
                    for attribute in fake_values
                }
                for stage in ("discovery", "browsing", "buying")
            },
        }


class DatasetExportTest(unittest.TestCase):
    def test_full_dataset_intent_distribution(self) -> None:
        scenarios = (
            ["buying"] * 80
            + ["browsing"] * 80
            + ["intent_override"] * 30
            + ["boundary"] * 10
        )
        labeled = _label_v2_samples(
            [{"scenario_type": scenario} for scenario in scenarios]
        )
        self.assertEqual(
            Counter(intent for _, intent, _ in labeled),
            {"buying": 50, "discovery": 50, "browsing": 100},
        )

    def test_v2_export_flattens_item_and_modification_fields(self) -> None:
        sample = {
            "sample_id": "sample_1",
            "scenario_type": "intent_override",
            "ground_truth": {"parent_asin": "A"},
        }
        item = Item(
            item_id="A",
            features={"parent_asin": "A"},
            intent_descriptions={"browsing": {}, "buying": {}},
        )
        modification = Modification(
            item_id="A",
            fake_attributes={"style": {"browsing": "fake", "buying": "fake"}},
            correction_messages={"style": {"browsing": "true", "buying": "true"}},
            modify_turn=3,
        )
        [(labeled_sample, intent, override)] = _label_v2_samples([sample])
        row = _v2_row(labeled_sample, intent, override, item, modification)
        self.assertEqual(row["version"], "v2")
        self.assertEqual(row["intent"], "buying")
        self.assertTrue(row["override"])
        self.assertNotIn("scenario_type", row)
        self.assertEqual(row["item_id"], "A")
        self.assertEqual(row["modify_turn"], 3)


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
            intent: {
                attribute: f"{intent}:{attribute}"
                for attribute in (
                    "category",
                    "brand",
                    "budget",
                    "material",
                    "feature",
                    "other",
                )
            }
            for intent in ("browsing", "buying")
        }
        return Item("ITEM", {}, descriptions)

    def test_all_available_attributes_are_active(self) -> None:
        handler = QueryHandler("session-1", self._item())
        self.assertEqual(
            handler.active_attributes,
            ("brand", "budget", "category", "feature", "material", "other"),
        )

    def test_active_other_returns_only_other_description(self) -> None:
        handler = QueryHandler("session-2", self._item())
        result = handler.answer("other")
        self.assertEqual(result, "browsing:other")
        self.assertEqual(len(handler.disclosed_attributes), 1)

    def test_list_fragments_are_joined_only_at_message_boundary(self) -> None:
        item = Item(
            "ITEM", {}, {"browsing": {"feature": ["water resistant", "daily use"]}}
        )
        handler = QueryHandler("session-list", item)
        self.assertEqual(handler.answer("feature"), "water resistant; daily use")

    def test_unknown_attribute_is_not_revealed(self) -> None:
        handler = QueryHandler("session-3", self._item())
        self.assertIsNone(handler.answer("unknown"))

    def test_intent_transition_adds_latest_different_disclosed_attribute(self) -> None:
        handler = QueryHandler("session-transition", self._item())
        handler.answer("brand")
        handler.answer("budget")
        handler.answer("material")

        handler.set_intent("buying")

        self.assertEqual(
            handler.answer("material"),
            "buying:material Also, for budget: buying:budget",
        )

    def test_fewer_than_four_attributes_are_all_active(self) -> None:
        item = Item(
            "ITEM_WITH_THREE_ATTRIBUTES",
            {},
            {
                "browsing": {
                    "category": "shirts",
                    "brand": "Acme",
                    "budget": "under $30",
                }
            },
        )
        handler = QueryHandler("session-4", item)
        self.assertEqual(handler.active_attributes, ("brand", "budget", "category"))

    def test_modification_switches_fake_to_true_and_corrects_prior_disclosure(
        self,
    ) -> None:
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
            fake_attributes={
                "material": {"browsing": "fake material", "buying": "fake material"}
            },
            correction_messages={
                "material": {"browsing": "correction text", "buying": "correction text"}
            },
            modify_turn=3,
        )
        handler = QueryHandler("session-mod", item, modification=modification)
        self.assertIn("material", handler.active_attributes)
        self.assertEqual(handler.answer("material", turn=1), "fake material")
        result = handler.answer("category", turn=3)
        self.assertIn("category", result)
        self.assertIn("correct", result)
        self.assertIn("correction text", result)

    def test_modification_locks_the_first_queried_fakeable_attribute(self) -> None:
        item = self._item()
        modification = Modification(
            "ITEM",
            {
                "budget": {"browsing": "fake budget"},
                "material": {"browsing": "fake material"},
            },
            {
                "budget": {"browsing": "correct budget"},
                "material": {"browsing": "correct material"},
            },
            5,
        )
        handler = QueryHandler("session-first-fake", item, modification=modification)

        self.assertEqual(handler.answer("feature", turn=1), "browsing:feature")
        self.assertEqual(handler.answer("material", turn=2), "fake material")
        self.assertEqual(handler.selected_modification_attribute, "material")
        self.assertEqual(handler.answer("budget", turn=3), "browsing:budget")
        self.assertEqual(handler.answer("material", turn=4), "fake material")
        self.assertEqual(
            handler.answer("category", turn=5),
            "browsing:category correct material",
        )

    def test_unselected_modification_expires_silently_at_modify_turn(self) -> None:
        item = self._item()
        modification = Modification(
            "ITEM",
            {"material": {"browsing": "fake material"}},
            {"material": {"browsing": "correct material"}},
            3,
        )
        handler = QueryHandler("session-expired-fake", item, modification=modification)

        self.assertEqual(handler.answer("feature", turn=1), "browsing:feature")
        self.assertEqual(handler.answer("material", turn=3), "browsing:material")
        self.assertIsNone(handler.selected_modification_attribute)
        self.assertTrue(handler.modification_applied)

    def test_modification_is_enabled_when_supplied(self) -> None:
        item = self._item()
        modification = Modification(
            "ITEM",
            {"material": {"browsing": "fake", "buying": "fake"}},
            {"material": {"browsing": "correction", "buying": "correction"}},
            3,
        )
        sessions = [
            create_session(f"session-{index}", item, modification)
            for index in range(100)
        ]
        enabled = sum(session.modification is not None for session in sessions)
        self.assertEqual(enabled, 100)

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
            self.assertIsNotNone(session.modification)
            self.assertIn("other", session.query_handler.active_attributes)


class SpanVerificationTest(unittest.TestCase):
    """The gates that make a model-extracted value trustworthy enough to score against."""

    NECKLACE = {
        "parent_asin": "B0TESTNECK1",
        "title": "Pendant Necklace",
        "features": ["Material:alloy"],
        "description": [
            "5.it is not real gold and silver products and It will fade away slowly"
        ],
        "price": 9.99,
        "categories": ["Clothing, Shoes & Jewelry", "Jewelry", "Necklaces"],
        "details": {},
        "store": "QIAN0813",
    }

    def test_verified_claim_is_kept(self) -> None:
        verified, rejected = verify_extraction(
            self.NECKLACE,
            {"material": {"value": "alloy", "evidence": "Material:alloy"}},
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
            {
                "use_case": {
                    "value": "not for swimming",
                    "evidence": "Do not wear while swimming.",
                }
            },
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
            {
                "use_case": {
                    "value": "do not wear while swimming",
                    "evidence": "Do not wear while swimming.",
                }
            },
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
            self.NECKLACE,
            {"color": {"value": "turquoise", "evidence": "Material:alloy"}},
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
            self.NECKLACE,
            {"material": {"value": "alloy", "evidence": "Material: alloy"}},
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
                SAMPLE_PRODUCT,
                "B0TESTITEM1",
                attributes,
                true_descriptions(attributes),
                FakeWriter(),
                Path(directory),
            )
            self.assertIsNotNone(modification)
            if "material" in modification.fake_attributes:
                self.assertNotIn(
                    "polyester and mesh",
                    modification.fake_attributes["material"]["browsing"],
                )
            self.assertEqual(
                set(modification.correction_messages), set(modification.fake_attributes)
            )


class ModificationTest(unittest.TestCase):
    def test_build_modification_is_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            writer_a = FakeWriter()
            writer_b = FakeWriter()
            first = build_modification(
                SAMPLE_PRODUCT,
                "B0TESTITEM1",
                SAMPLE_ATTRIBUTES,
                true_descriptions(SAMPLE_ATTRIBUTES),
                writer_a,
                cache_dir / "a",
            )
            second = build_modification(
                SAMPLE_PRODUCT,
                "B0TESTITEM1",
                SAMPLE_ATTRIBUTES,
                true_descriptions(SAMPLE_ATTRIBUTES),
                writer_b,
                cache_dir / "b",
            )
            self.assertIsNotNone(first)
            self.assertEqual(first, second)
            self.assertIn(first.modify_turn, range(3, 8))
            self.assertEqual(first.correction_messages, second.correction_messages)

    def test_build_modification_generates_every_available_fakeable_attribute(
        self,
    ) -> None:
        attributes = {
            **SAMPLE_ATTRIBUTES,
            "material": "cotton",
            "color": "blue",
            "style": "casual",
            "size": "medium",
            "use_case": "work",
        }
        with TemporaryDirectory() as directory:
            modification = build_modification(
                SAMPLE_PRODUCT,
                "B0TESTITEM1",
                attributes,
                true_descriptions(attributes),
                FakeWriter(),
                Path(directory),
            )
        self.assertIsNotNone(modification)
        self.assertEqual(
            set(modification.fake_attributes),
            {"budget", "color", "material", "size", "style", "use_case"},
        )
        self.assertEqual(
            set(modification.correction_messages), set(modification.fake_attributes)
        )

    def test_build_modification_treats_size_options_as_size(self) -> None:
        attributes = {**SAMPLE_ATTRIBUTES, "size_options": "S, M, L, XL"}
        with TemporaryDirectory() as directory:
            modification = build_modification(
                SAMPLE_PRODUCT,
                "B0TESTITEM1",
                attributes,
                true_descriptions(attributes),
                FakeWriter(),
                Path(directory),
            )
        self.assertIn("size", modification.fake_attributes)
        correction = modification.correction_messages["size"]["buying"]
        self.assertIn("buying-true:available options: S, M, L, XL", correction[0])

    def test_build_modification_fake_values_differ_from_truth(self) -> None:
        with TemporaryDirectory() as directory:
            modification = build_modification(
                SAMPLE_PRODUCT,
                "B0TESTITEM1",
                SAMPLE_ATTRIBUTES,
                true_descriptions(SAMPLE_ATTRIBUTES),
                FakeWriter(),
                Path(directory),
            )
            self.assertIsNotNone(modification)
            for attribute, texts in modification.fake_attributes.items():
                self.assertNotIn(attribute, ("category", "brand"))
                self.assertTrue(texts["discovery"])
                self.assertTrue(texts["browsing"])
                self.assertTrue(texts["buying"])

    def test_build_modification_passes_numeric_budget_context_for_fake_budget(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            writer = FakeWriter()
            modification = build_modification(
                SAMPLE_PRODUCT,
                "B0TESTITEM1",
                SAMPLE_ATTRIBUTES,
                true_descriptions(SAMPLE_ATTRIBUTES),
                writer,
                Path(directory),
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
            self.assertEqual(
                json.loads(cache_path.read_text(encoding="utf-8")), {"value": 1}
            )


class DeepSeekAttributeWriterTest(unittest.TestCase):
    def test_missing_api_key_raises(self) -> None:
        import os

        with (
            patch("scripts.llm_client._load_dotenv"),
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}),
        ):
            with self.assertRaises(RuntimeError):
                DeepSeekAttributeWriter()

    def test_parse_requires_compact_clue_list(self) -> None:
        self.assertEqual(
            DeepSeekAttributeWriter._parse_clues(
                '{"clues":["cotton blend", "machine washable"]}'
            ),
            ["cotton blend", "machine washable"],
        )
        self.assertEqual(
            len(
                DeepSeekAttributeWriter._parse_clues(
                    '{"clues":["S","M","L","XL","XXL"]}'
                )
            ),
            5,
        )
        self.assertEqual(
            DeepSeekAttributeWriter._parse_clues(
                '{"clues":["I want cotton", "cotton."]}'
            ),
            ["I want cotton", "cotton."],
        )
        for invalid in (
            '{"clues":"cotton"}',
            '{"clues":[]}',
            "not json",
        ):
            with self.assertRaises(ValueError):
                DeepSeekAttributeWriter._parse_clues(invalid)

    def test_describe_runs_complete_stages_in_order_and_caches_each_attribute(
        self,
    ) -> None:
        class RecordingWriter(DeepSeekAttributeWriter):
            def __init__(self) -> None:
                self.calls: list[tuple[str, str, dict]] = []

            def describe_attribute(
                self,
                category,
                attribute,
                value,
                stage,
                previous=None,
                budget_context=None,
            ) -> list[str]:
                self.calls.append((stage, attribute, deepcopy(previous or {})))
                return [f"{stage}:{attribute}"]

        writer = RecordingWriter()
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            first = writer.describe(
                "shirts",
                {"category": "shirts", "material": "cotton"},
                cache_dir=cache_dir,
            )
            second = writer.describe(
                "shirts",
                {"category": "shirts", "material": "cotton"},
                cache_dir=cache_dir,
            )

        self.assertEqual(first, second)
        self.assertEqual(
            [(stage, attribute) for stage, attribute, _ in writer.calls],
            [
                ("buying", "category"),
                ("buying", "material"),
                ("browsing", "category"),
                ("browsing", "material"),
                ("discovery", "category"),
                ("discovery", "material"),
            ],
        )
        browsing_context = writer.calls[2][2]
        discovery_context = writer.calls[4][2]
        self.assertEqual(set(browsing_context["buying"]), {"category", "material"})
        self.assertEqual(set(discovery_context), {"buying", "browsing"})

    def test_modification_corrections_use_each_intents_true_clues(self) -> None:
        class RecordingWriter(DeepSeekAttributeWriter):
            def __init__(self) -> None:
                self.correction_calls: list[tuple[str, list[str], list[str]]] = []

            def describe_attribute(
                self,
                category,
                attribute,
                value,
                stage,
                previous=None,
                budget_context=None,
            ) -> list[str]:
                return {
                    "buying": ["denim"],
                    "browsing": ["fabric-based"],
                    "discovery": ["material preference open"],
                }[stage]

            def correct_attribute(
                self, attribute, stage, fake_clues, true_clues, previous_corrections
            ) -> str:
                self.correction_calls.append(
                    (stage, list(true_clues), list(previous_corrections))
                )
                return f"Actually, {true_clues[0]}."

        truths = {
            "buying": {"material": ["alloy"]},
            "browsing": {"material": ["metal-based"]},
            "discovery": {"material": ["material undecided"]},
        }
        writer = RecordingWriter()
        with TemporaryDirectory() as directory:
            generated = writer.describe_modification(
                "necklaces",
                {"material": "denim"},
                truths,
                cache_dir=Path(directory),
            )

        self.assertEqual(
            generated["correction_messages"],
            {
                "buying": {"material": ["Actually, alloy."]},
                "browsing": {"material": ["Actually, metal-based."]},
                "discovery": {"material": ["Actually, material undecided."]},
            },
        )
        self.assertEqual(
            [call[:2] for call in writer.correction_calls],
            [
                ("buying", ["alloy"]),
                ("browsing", ["metal-based"]),
                ("discovery", ["material undecided"]),
            ],
        )
        self.assertEqual(len(writer.correction_calls[2][2]), 2)

    def test_correction_validation_allows_paraphrases_and_requires_unique_stages(
        self,
    ) -> None:
        self.assertEqual(
            DeepSeekAttributeWriter._parse_correction(
                '{"message":"Actually, I need metal-based construction."}'
            ),
            "Actually, I need metal-based construction.",
        )
        DeepSeekAttributeWriter._validate_correction(
            "Actually, my material preference is still open.",
            ["material undecided"],
            [],
        )
        with self.assertRaisesRegex(ValueError, "true clues"):
            DeepSeekAttributeWriter._validate_correction("Actually, I changed my mind.", [], [])
        with self.assertRaisesRegex(ValueError, "differ"):
            DeepSeekAttributeWriter._validate_correction(
                "Actually, metal-based.",
                ["metal-based"],
                ["Actually, metal-based."],
            )

    def test_stage_validation_rejects_inexact_price_and_invented_qualities(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "exact price"):
            DeepSeekAttributeWriter._validate_stage_clues(
                "budget",
                "under_15",
                "buying",
                ["under $15"],
                {"exact_price": 9.99, "browsing_ceiling": 20.0},
            )
        with self.assertRaisesRegex(ValueError, "invent or strengthen"):
            DeepSeekAttributeWriter._validate_stage_clues(
                "use_case",
                "water resistance for rainy days",
                "discovery",
                ["weatherproof confidence"],
                None,
            )
        with self.assertRaisesRegex(ValueError, "invent or strengthen"):
            DeepSeekAttributeWriter._validate_stage_clues(
                "other",
                "5 x 8 x 12 inches; 1.41 pounds",
                "browsing",
                ["lightweight build"],
                None,
            )

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
            "material": {
                "value": "stainless steel",
                "evidence": "Stainless Steel Band",
            },
            "color": None,
            "size": None,
            "size_options": [],
            "style": None,
            "use_case": None,
            "feature": {"value": "water resistant", "evidence": "Water Resistant"},
            "other": None,
        }
        self.assertEqual(
            DeepSeekAttributeWriter._validate_extraction_shape(valid), valid
        )

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
