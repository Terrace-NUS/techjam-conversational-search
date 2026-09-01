from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts.intent_manager import IntentManager
from visualizer.backend.app.main import SimulatorService, create_app


class VisualizerApiTest(unittest.TestCase):
    def test_agent_events_are_queued_for_sse(self) -> None:
        class EventAgent:
            sink = None

            def set_event_sink(self, session_id, sink):
                self.sink = sink

        service = SimulatorService.__new__(SimulatorService)
        service.sessions = {"session-1": {}}
        service.event_queues = {}
        service.lock = Lock()
        agent = EventAgent()

        service.attach_agent_events("session-1", agent)
        event = {"stage": "query_understanding", "status": "completed", "turn": 1}
        agent.sink(event)

        self.assertEqual(service.event_queue("session-1").get_nowait(), event)

    def test_turn_subscore_is_maximum_recommendation_score(self) -> None:
        class FixedScores:
            def score_turn(self, ranked, target, products):
                return {"LOW": 0.2, "HIGH": 0.9}[ranked[0]]

        service = SimulatorService.__new__(SimulatorService)
        service.products = {"LOW": {}, "HIGH": {}, "TARGET": {}}
        service.reward_calculators = {"gemini": FixedScores()}
        session = {
            "intent_manager": IntentManager("browsing"),
            "embedding_provider": "gemini",
            "target": "TARGET",
            "debug": True,
            "score_error": None,
        }

        metrics = service.turn_metrics(session, ["LOW", "HIGH"], update_intent=False)

        self.assertEqual(metrics["recommendation_scores"], {"LOW": 0.2, "HIGH": 0.9})
        self.assertEqual(metrics["subscore"], 0.9)

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
                self.assertEqual(view["status"], "initializing")
                self.assertNotIn("TARGET", json.dumps(view))
                self.assertEqual(
                    client.post(f"/api/sessions/{view['id']}/initialize").json()["status"],
                    "waiting_for_agent",
                )

                response = client.post(
                    f"/api/sessions/{view['id']}/turn",
                    json={"message": "Try this.", "recommendations": ["TARGET"]},
                )
                view = response.json()
                self.assertEqual(view["status"], "waiting_for_agent")
                self.assertIsNone(view["turns"][0]["hit_rank"])
                self.assertEqual(view["turns"][0]["subscore"], 1.0)
                self.assertTrue(view["turns"][0]["intent_changed"])
                self.assertEqual(view["metrics"]["current_intent"], "buying")
                self.assertEqual(view["metrics"]["threshold"], 0.5)
                self.assertEqual(view["turns"][0]["recommendation_scores"], {})

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

                debug_view = client.post(
                    "/api/sessions",
                    json={
                        "sample_id": "override-1",
                        "dataset": "samples",
                        "embedding_provider": "siliconflow",
                        "debug": True,
                    },
                ).json()
                self.assertEqual(debug_view["debug_target_product"]["parent_asin"], "TARGET")
                self.assertEqual(debug_view["embedding_provider"], "siliconflow")
                client.post(f"/api/sessions/{debug_view['id']}/initialize")
                debug_view = client.post(
                    f"/api/sessions/{debug_view['id']}/turn",
                    json={
                        "message": "Try this.",
                        "ask_attribute": "color",
                        "recommendations": ["TARGET"],
                    },
                ).json()
                self.assertEqual(
                    debug_view["turns"][0]["recommendation_scores"],
                    {"TARGET": 1.0},
                )
                self.assertEqual(debug_view["turns"][0]["queried_attribute"], "color")

                human_view = client.post(
                    "/api/human-sessions",
                    json={"sample_id": "override-1", "dataset": "samples", "agent": "baseline"},
                ).json()
                self.assertEqual(human_view["status"], "initializing")
                human_view = client.post(
                    f"/api/human-sessions/{human_view['id']}/initialize"
                ).json()
                self.assertEqual(human_view["status"], "waiting_for_simulator")
                human_view = client.post(
                    f"/api/human-sessions/{human_view['id']}/reply",
                    json={"message": "I need a leather walking shoe."},
                ).json()
                self.assertEqual(human_view["status"], "hit")

                auto_view = client.post(
                    "/api/auto-sessions",
                    json={
                        "sample_id": "override-1",
                        "dataset": "samples",
                        "agent": "baseline",
                        "reply_model": "template",
                    },
                ).json()
                self.assertEqual(auto_view["mode"], "agent_simulator")
                self.assertEqual(auto_view["agent"], "baseline")
                auto_view = client.post(
                    f"/api/auto-sessions/{auto_view['id']}/initialize"
                ).json()
                self.assertEqual(auto_view["status"], "waiting_for_agent")
                auto_view = client.post(
                    f"/api/auto-sessions/{auto_view['id']}/step"
                ).json()
                self.assertEqual(len(auto_view["turns"]), 1)
                self.assertEqual(auto_view["current_turn"], 2)

                with patch("visualizer.backend.app.main.build_agent") as build_agent:
                    terrace_view = client.post(
                        "/api/human-sessions",
                        json={
                            "sample_id": "override-1",
                            "dataset": "samples",
                            "agent": "terrace",
                        },
                    ).json()
                    terrace_view = client.post(
                        f"/api/human-sessions/{terrace_view['id']}/initialize"
                    ).json()

                    self.assertEqual(terrace_view["agent"], "terrace")
                    self.assertEqual(terrace_view["status"], "waiting_for_simulator")
                    build_agent.assert_called_once_with("terrace", catalog)

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
                client.post(f"/api/sessions/{session_id}/initialize")
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
                self.assertIsNone(view["turns"][0]["queried_attribute"])
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
