from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


PRODUCTS = [
    {
        "parent_asin": "A", "title": "Blue Cotton Running Shirt",
        "categories": ["Clothing", "Shirts"],
        "features": ["breathable fabric", "running"],
        "details": {"material": "cotton", "color": "blue"},
        "store": "Demo", "description": ["lightweight workout top"],
        "average_rating": 4.7, "rating_number": 100,
    },
    {
        "parent_asin": "B", "title": "Black Leather Winter Boot",
        "categories": ["Clothing", "Boots"],
        "features": ["warm lining", "winter"],
        "details": {"material": "leather", "color": "black"},
        "store": "Demo", "description": ["outdoor boot"],
        "average_rating": 4.5, "rating_number": 80,
    },
    {
        "parent_asin": "C", "title": "Red Polyester Running Shirt",
        "categories": ["Clothing", "Shirts"],
        "features": ["quick dry", "running"],
        "details": {"material": "polyester", "color": "red"},
        "store": "Demo", "description": ["workout top"],
        "average_rating": 4.0, "rating_number": 20,
    },
    {
        "parent_asin": "D", "title": "Quick Dry Running Top",
        "categories": ["Clothing", "Shirts"],
        "features": ["quick dry", "running"],
        "details": {"fit": "regular"},
        "store": "Demo", "description": ["workout top"],
        "average_rating": 4.1, "rating_number": 30,
    },
]


class AgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        catalog = Path(self.temp.name) / "catalog.jsonl"
        catalog.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS),
            encoding="utf-8",
        )
        self.agent = Agent(catalog)
        self.agent.reset("s", {"preference_tags": ["comfort"]})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_searches_catalog_and_ranks_matching_product_first(self) -> None:
        response = self.agent.respond("s", "blue cotton running shirt", 1, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "A")
        self.assertEqual(response["ask_attribute"], "other")

    def test_continuation_does_not_repeat_previous_misses(self) -> None:
        first = self.agent.respond("s", "running shirt", 1, 1)
        second = self.agent.respond("s", "I don't have an additional preference for feature.", 2, 1)
        self.assertNotEqual(
            first["recommendations"][0]["parent_asin"],
            second["recommendations"][0]["parent_asin"],
        )
        self.assertEqual(second["ask_attribute"], "other")

    def test_override_replaces_stale_constraint_but_keeps_category(self) -> None:
        self.agent.respond("s", "I'm looking for shirts. blue cotton", 1, 1)
        response = self.agent.respond(
            "s", "Actually, ignore my earlier preference. What I need is: red polyester.", 2, 2
        )
        self.assertEqual(response["recommendations"][0]["parent_asin"], "C")

    def test_override_clears_pre_override_negative_candidates(self) -> None:
        first = self.agent.respond("s", "I'm looking for shirts. polyester", 1, 3)
        self.assertIn("C", [item["parent_asin"] for item in first["recommendations"]])
        second = self.agent.respond(
            "s", "Actually, ignore my earlier preference. What I need is: red polyester.", 2, 3
        )
        self.assertIn("C", [item["parent_asin"] for item in second["recommendations"]])
        self.assertEqual(second["ask_attribute"], "other")

    def test_exact_evaluator_constraint_outranks_generic_keywords(self) -> None:
        response = self.agent.respond(
            "s",
            "I'm looking for Shirts. A key requirement is: polyester.",
            1,
            3,
        )
        self.assertEqual(response["recommendations"][0]["parent_asin"], "C")

    def test_card_prefix_filters_products_with_regex_inserted_material(self) -> None:
        response = self.agent.respond(
            "s",
            "For that, what matters is: quick dry; running.",
            2,
            3,
        )
        asins = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(asins[0], "D")
        self.assertNotIn("A", asins)


if __name__ == "__main__":
    unittest.main()
