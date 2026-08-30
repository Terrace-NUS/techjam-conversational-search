from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from visualizer.backend.app.main import SimulatorService, create_app


class VisualizerApiTest(unittest.TestCase):
    def test_intent_override_redacts_target_and_only_scores_after_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            dataset = root / "samples.jsonl"
            products = [
                {
                    "parent_asin": "TARGET",
                    "title": "Leather walking shoe",
                    "features": ["leather", "cushioned"],
                    "categories": ["Clothing", "Shoes"],
                    "price": 80,
                },
                {"parent_asin": "OTHER", "title": "Canvas shoe", "categories": ["Shoes"]},
            ]
            sample = {
                "sample_id": "override-1",
                "scenario_type": "intent_override",
                "difficulty_bucket": "hard",
                "category_bucket": "shoes",
                "user_profile": {"summary": "likes comfortable shoes"},
                "ground_truth": {"parent_asin": "TARGET"},
                "intent_card": {
                    "target_category": "shoe",
                    "hard_constraints": ["leather"],
                    "soft_preferences": ["canvas"],
                },
                "behavior": {
                    "scenario_type": "intent_override",
                    "override": {
                        "turn": 3,
                        "old_value": "canvas",
                        "new_value": "leather",
                        "message": "Actually, leather is required now.",
                    },
                },
            }
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products), encoding="utf-8")
            dataset.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            with TestClient(create_app(service=SimulatorService(catalog, dataset))) as client:
                response = client.post("/api/sessions", json={"sample_id": "override-1"})
                self.assertEqual(response.status_code, 201)
                view = response.json()
                self.assertNotIn("TARGET", json.dumps(view))

                response = client.post(
                    f"/api/sessions/{view['id']}/turn",
                    json={"message": "Try this.", "recommendations": ["TARGET"]},
                )
                view = response.json()
                self.assertEqual(view["status"], "waiting_for_agent")
                self.assertIsNone(view["turns"][0]["hit_rank"])

                response = client.post(
                    f"/api/sessions/{view['id']}/turn",
                    json={"message": "Any change?", "recommendations": []},
                )
                view = response.json()
                self.assertEqual(view["current_turn"], 3)
                self.assertEqual(view["current_user_message"], "Actually, leather is required now.")

                response = client.post(
                    f"/api/sessions/{view['id']}/turn",
                    json={"message": "Try again.", "recommendations": ["TARGET", "TARGET"]},
                )
                view = response.json()
                self.assertEqual(view["status"], "hit")
                self.assertEqual(view["turns"][-1]["hit_rank"], 1)
                self.assertEqual(view["outcome"]["target_product"]["parent_asin"], "TARGET")

                response = client.post(
                    f"/api/sessions/{view['id']}/turn", json={"message": "late"}
                )
                self.assertEqual(response.status_code, 409)
                self.assertEqual(
                    client.post("/api/sessions", json={"sample_id": "missing"}).status_code,
                    404,
                )

    def test_boundary_reply_and_turn_limit_match_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            dataset = root / "samples.jsonl"
            products = [
                {"parent_asin": "TARGET", "title": "Blue shirt", "categories": ["Clothing"]},
                {"parent_asin": "OTHER", "title": "Black shirt", "categories": ["Clothing"]},
            ]
            sample = {
                "sample_id": "boundary-1",
                "scenario_type": "boundary",
                "difficulty_bucket": "medium",
                "category_bucket": "clothing",
                "user_profile": {"summary": "A careful shopper", "preference_tags": []},
                "ground_truth": {"parent_asin": "TARGET"},
                "intent_card": {
                    "target_category": "shirt",
                    "hard_constraints": ["color: blue"],
                    "soft_preferences": [],
                },
                "behavior": {"scenario_type": "boundary"},
            }
            catalog.write_text("".join(json.dumps(row) + "\n" for row in products), encoding="utf-8")
            dataset.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            with TestClient(create_app(service=SimulatorService(catalog, dataset))) as client:
                view = client.post("/api/sessions", json={"sample_id": "boundary-1"}).json()
                session_id = view["id"]
                response = client.post(
                    f"/api/sessions/{session_id}/turn",
                    json={
                        "message": "",
                        "ask_attribute": "color",
                        "recommendations": ["OTHER", "OTHER", "MISSING"],
                    },
                )
                self.assertEqual(response.status_code, 200)
                view = response.json()
                self.assertEqual(view["turns"][0]["agent_message"], "")
                self.assertEqual(
                    view["current_user_message"],
                    "I don't have a preference for color; please use your judgment.",
                )
                self.assertEqual(
                    [item["parent_asin"] for item in view["turns"][0]["recommendations"]],
                    ["OTHER"],
                )

                for _ in range(9):
                    view = client.post(
                        f"/api/sessions/{session_id}/turn",
                        json={"message": "No match yet.", "recommendations": []},
                    ).json()
                self.assertEqual(view["status"], "exhausted")
                self.assertFalse(view["outcome"]["hit"])
                self.assertEqual(len(view["turns"]), 10)
                self.assertEqual(view["outcome"]["target_product"]["parent_asin"], "TARGET")

    def test_catalog_subsequence_search_filters_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            dataset = root / "samples.jsonl"
            products = [
                {
                    "parent_asin": "RUNNER",
                    "title": "Trail Running Sneakers",
                    "categories": ["Shoes", "Men"],
                    "store": "Acme",
                    "price": 80.0,
                    "average_rating": 4.8,
                    "rating_number": 500,
                    "features": ["Cushioned sole"],
                    "description": ["Built for trails."],
                    "details": {"Department": "Men"},
                },
                {
                    "parent_asin": "LOAFER",
                    "title": "Leather Office Loafer",
                    "categories": ["Shoes"],
                    "store": "Beta",
                    "price": 120.0,
                    "average_rating": 4.2,
                    "rating_number": 50,
                },
                {
                    "parent_asin": "BLENDER",
                    "title": "Kitchen Blender",
                    "categories": ["Home"],
                    "store": "Acme",
                    "price": None,
                    "average_rating": 4.9,
                    "rating_number": 1000,
                },
                {
                    "parent_asin": "UNKNOWN_PRICE",
                    "title": "Everyday Shoe",
                    "categories": ["Shoes"],
                    "store": "Acme",
                    "price": "from 10.00",
                    "average_rating": 4.7,
                    "rating_number": 700,
                },
            ]
            sample = {
                "sample_id": "catalog-1",
                "scenario_type": "buying",
                "difficulty_bucket": "easy",
                "category_bucket": "shoes",
                "user_profile": {},
                "ground_truth": {"parent_asin": "RUNNER"},
                "intent_card": {"hard_constraints": [], "soft_preferences": []},
                "behavior": {"scenario_type": "buying"},
            }
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in products), encoding="utf-8"
            )
            dataset.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            with TestClient(create_app(service=SimulatorService(catalog, dataset))) as client:
                metadata = client.get("/api/catalog/filters").json()
                self.assertEqual(metadata["categories"], ["Home", "Men", "Shoes"])
                self.assertEqual(metadata["stores"], ["Acme", "Beta"])
                self.assertEqual(metadata["price"], {"min": 80.0, "max": 120.0})
                self.assertEqual(metadata["average_rating"], {"min": 4.2, "max": 4.9})
                self.assertEqual(metadata["rating_number"], {"min": 50, "max": 1000})

                results = client.get(
                    "/api/catalog/search", params={"q": "trl rn snkr", "limit": 2}
                ).json()
                self.assertEqual(results[0]["parent_asin"], "RUNNER")
                self.assertEqual(results[0]["features"], ["Cushioned sole"])
                self.assertEqual(results[0]["description"], ["Built for trails."])
                self.assertEqual(results[0]["details"], {"Department": "Men"})
                self.assertEqual(
                    client.get("/api/catalog/search", params={"q": "rlt"}).json(), []
                )

                first_page = client.get(
                    "/api/catalog/search", params={"limit": 2}
                ).json()
                second_page = client.get(
                    "/api/catalog/search", params={"limit": 2, "offset": 2}
                ).json()
                self.assertEqual(len(first_page), 2)
                self.assertEqual(len(second_page), 2)
                self.assertTrue(
                    {item["parent_asin"] for item in first_page}.isdisjoint(
                        item["parent_asin"] for item in second_page
                    )
                )

                results = client.get(
                    "/api/catalog/search",
                    params={
                        "category": "Shoes",
                        "store": "Acme",
                        "min_price": 70,
                        "max_price": 90,
                        "min_rating": 4.5,
                        "min_rating_count": 400,
                    },
                ).json()
                self.assertEqual([item["parent_asin"] for item in results], ["RUNNER"])

                results = client.get(
                    "/api/catalog/search", params={"category": "Home"}
                ).json()
                self.assertEqual([item["parent_asin"] for item in results], ["BLENDER"])
                results = client.get(
                    "/api/catalog/search", params={"store": "Beta"}
                ).json()
                self.assertEqual([item["parent_asin"] for item in results], ["LOAFER"])
                results = client.get(
                    "/api/catalog/search", params={"min_price": 80, "max_price": 80}
                ).json()
                self.assertEqual([item["parent_asin"] for item in results], ["RUNNER"])
                results = client.get(
                    "/api/catalog/search", params={"min_rating": 4.85}
                ).json()
                self.assertEqual([item["parent_asin"] for item in results], ["BLENDER"])
                results = client.get(
                    "/api/catalog/search", params={"min_rating_count": 800}
                ).json()
                self.assertEqual([item["parent_asin"] for item in results], ["BLENDER"])


if __name__ == "__main__":
    unittest.main()
