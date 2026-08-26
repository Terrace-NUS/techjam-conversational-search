from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile

from evaluator.local_evaluator import (
    DeepSeekReplyModel,
    TemplateReplyModel,
    build_reply_model,
    catalog_index,
    evaluate,
    metric_summary,
    normalize_recommendations,
)
from starter.agent import Agent, build_agent
from starter.baseline import BaselineAgent
from starter.v1 import V1Agent


class EchoTargetAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        asin = "A"
        if "B" in user_message:
            asin = "B"
        return {"message": "ok", "ask_attribute": None, "recommendations": [{"parent_asin": asin}]}


class EvaluatorTest(unittest.TestCase):
    def test_agent_factory_uses_abc_implementations(self) -> None:
        self.assertTrue(issubclass(BaselineAgent, Agent))
        self.assertTrue(issubclass(V1Agent, Agent))
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(json.dumps({"parent_asin": "A"}) + "\n", encoding="utf-8")
            baseline = build_agent("baseline", catalog)
            self.assertIsInstance(baseline, BaselineAgent)
            baseline.connection.close()

    def test_reply_model_selection_and_json_parsing(self) -> None:
        self.assertIsInstance(build_reply_model("template"), TemplateReplyModel)
        self.assertEqual(
            DeepSeekReplyModel._parse_message('{"message":"A natural reply."}'),
            "A natural reply.",
        )
        with self.assertRaises(ValueError):
            DeepSeekReplyModel._parse_message("not json")

    def test_deepseek_request_errors_are_not_replaced_with_template(self) -> None:
        model = DeepSeekReplyModel(api_key="test-key")

        class FailingClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise RuntimeError("network failure")

        model.client = FailingClient()
        with self.assertRaises(RuntimeError):
            model._rewrite("canonical", "initial message")

    def test_deepseek_request_includes_few_shot_messages(self) -> None:
        model = DeepSeekReplyModel(api_key="test-key")
        captured = {}

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        captured.update(kwargs)
                        return type("Response", (), {
                            "choices": [type("Choice", (), {
                                "message": type("Message", (), {"content": '{"message":"paraphrase"}'})()
                            })()]
                        })()

        model.client = FakeClient()
        self.assertEqual(model._rewrite("canonical", "initial message"), "paraphrase")
        messages = captured["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            len(messages),
            1 + len(DeepSeekReplyModel.FEW_SHOT_MESSAGES) + 1,
        )
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(captured["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(captured["temperature"], 0)
        self.assertEqual(captured["max_tokens"], 256)

    def test_normalization_preserves_first_valid_unique_order(self) -> None:
        payload = [
            {"parent_asin": "A"}, {"parent_asin": "bad"}, {"parent_asin": "A"},
            "B", {"parent_asin": "C"},
        ]
        self.assertEqual(normalize_recommendations(payload, {"A", "B", "C"}), ["A", "B", "C"])

    def test_metric_summary_assigns_turn_11_to_miss(self) -> None:
        sessions = [
            {"hit": True, "reciprocal_rank": .5, "first_hit_turn": 2},
            {"hit": False, "reciprocal_rank": 0.0, "first_hit_turn": None},
        ]
        self.assertEqual(metric_summary(sessions), {
            "sample_count": 2,
            "hit_rate_at_10": .5,
            "mrr": .25,
            "mttc": 6.5,
        })

    def test_evaluate_derives_hidden_fields_when_public_set_omits_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            catalog_rows = [
                {
                    "parent_asin": "A",
                    "title": "Blue running shoe",
                    "features": ["cotton"],
                    "details": {"department": "womens"},
                    "description": ["walking shoe"],
                    "categories": ["Clothing", "Shoes"],
                    "store": "Example",
                    "average_rating": 4.2,
                    "rating_number": 10,
                    "price": 49.0,
                },
                {
                    "parent_asin": "B",
                    "title": "Black winter boot",
                    "features": ["leather"],
                    "details": {"department": "womens"},
                    "description": ["winter boot"],
                    "categories": ["Clothing", "Boots"],
                    "store": "Example",
                    "average_rating": 4.4,
                    "rating_number": 12,
                    "price": 89.0,
                },
            ]
            catalog_path.write_text("".join(json.dumps(row) + "\n" for row in catalog_rows), encoding="utf-8")
            catalog_ids, categories, products = catalog_index(catalog_path)
            samples = [{
                "sample_id": "public_v2_0001",
                "scenario_type": "buying",
                "user_profile": {"summary": "x"},
                "ground_truth": {"parent_asin": "A"},
            }]
            result = evaluate(EchoTargetAgent(), samples, catalog_ids, categories, products)
            self.assertEqual(result["hit_rate_at_10"], 1.0)


if __name__ == "__main__":
    unittest.main()
