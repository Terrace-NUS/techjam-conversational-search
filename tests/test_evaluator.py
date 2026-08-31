from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_jsonl,
    metric_summary,
    normalize_recommendations,
)
from evaluator.reply_model import (
    DeepSeekReplyModel,
    TemplateReplyModel,
    build_reply_model,
)
from evaluator.simulators import Simulator, V1Simulator, V2Simulator
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


class ModificationAgent:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.messages.append(user_message)
        recommendations = [{"parent_asin": "A"}] if "Correction" in user_message else []
        return {
            "message": "ok",
            "ask_attribute": "style",
            "recommendations": recommendations,
        }


class IntentEscalationAgent:
    """Miss once, then hit; captures the query-handler reply between turns."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.messages.append(user_message)
        asin = "B" if turn == 1 else "A"
        return {
            "message": "ok",
            "ask_attribute": "style",
            "recommendations": [{"parent_asin": asin}],
        }


class FixedRewardCalculator:
    """Offline reward double: forces escalation without Gemini/network access."""

    def __init__(self, subscore: float) -> None:
        self.subscore = subscore
        self.calls: list[tuple[list[str], str]] = []

    def score_turn(self, ranked: list[str], target_asin: str, products: dict[str, dict]) -> float:
        self.calls.append((ranked, target_asin))
        return self.subscore


class EvaluatorTest(unittest.TestCase):
    def test_versioned_simulators_implement_abc(self) -> None:
        self.assertTrue(issubclass(V1Simulator, Simulator))
        self.assertTrue(issubclass(V2Simulator, Simulator))

    def test_unknown_dataset_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dataset version"):
            evaluate(EchoTargetAgent(), [{"version": "v3"}], set(), {}, {})

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
            self.assertEqual(result["sessions"][0]["scenario_type"], "buying")
            self.assertNotIn("override", result["sessions"][0])

    def test_legacy_public_set_is_not_implicitly_upgraded(self) -> None:
        sample = {
            "sample_id": "legacy_override",
            "scenario_type": "intent_override",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public_set.jsonl"
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            self.assertEqual(load_jsonl(path), [sample])

    def test_subscore_escalates_custom_session_and_query_handler_uses_buying_intent(self) -> None:
        products = {
            "A": {"parent_asin": "A", "title": "target shirt"},
            "B": {"parent_asin": "B", "title": "other shirt"},
        }
        sample = {
            "version": "v2",
            "sample_id": "intent_escalation_1",
            "intent": "browsing",
            "override": False,
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
            "item_id": "A",
            "features": products["A"],
            "intent_descriptions": {
                    "browsing": {
                        "style": "browsing style",
                        "material": "browsing material",
                        "color": "browsing color",
                        "size": "browsing size",
                    },
                    "buying": {
                        "style": "buying style",
                        "material": "buying material",
                        "color": "buying color",
                        "size": "buying size",
                    },
                },
            "fake_attributes": {},
            "correction_messages": {},
            "modify_turn": None,
        }
        agent = IntentEscalationAgent()
        reward_calculator = FixedRewardCalculator(0.8)
        result = evaluate(
            agent,
            [sample],
            {"A", "B"},
            {"A": ["Clothing", "Shirts"]},
            products,
            reward_calculator=reward_calculator,
            intent_threshold=0.5,
        )

        self.assertEqual(reward_calculator.calls, [(["B"], "A")])
        self.assertEqual(result["sessions"][0]["final_intent"], "buying")
        self.assertIn("buying style", agent.messages)
        self.assertEqual(result["sessions"][0]["first_hit_turn"], 2)

    def test_custom_override_uses_fake_then_correction_at_modify_turn(self) -> None:
        catalog_ids = {"A"}
        categories = {"A": ["Clothing", "Shirts"]}
        products = {"A": {"parent_asin": "A"}}
        sample = {
            "version": "v2",
            "sample_id": "override_1",
            "intent": "buying",
            "override": True,
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
            "item_id": "A",
            "features": products["A"],
            "intent_descriptions": {
                "browsing": {"style": "browsing true", "material": "cotton", "size": "medium", "color": "blue"},
                "buying": {"style": "buying true", "material": "cotton", "size": "medium", "color": "blue"},
            },
            "fake_attributes": {"style": {"browsing": "fake style", "buying": "fake style"}},
            "correction_messages": {"style": {"browsing": "Correction: true style", "buying": "Correction: true style"}},
            "modify_turn": 3,
        }
        agent = ModificationAgent()
        result = evaluate(agent, [sample], catalog_ids, categories, products)
        self.assertEqual(result["hit_rate_at_10"], 1.0)
        self.assertEqual(result["sessions"][0]["version"], "v2")
        self.assertIn("fake style", agent.messages)
        self.assertTrue(any("Correction: true style" in message for message in agent.messages))

    def test_v2_without_modification_uses_embedded_item(self) -> None:
        product = {"parent_asin": "A"}
        sample = {
            "version": "v2",
            "sample_id": "browse_1",
            "intent": "browsing",
            "override": False,
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
            "item_id": "A",
            "features": product,
            "intent_descriptions": {
                intent: {"style": "classic", "material": "cotton", "size": "medium", "color": "blue"}
                for intent in ("browsing", "buying")
            },
            "fake_attributes": {},
            "correction_messages": {},
            "modify_turn": None,
        }
        result = evaluate(EchoTargetAgent(), [sample], {"A"}, {"A": []}, {"A": product})
        self.assertEqual(result["hit_rate_at_10"], 1.0)
        self.assertEqual(result["override_metrics"]["sample_count"], 0)

    def test_v2_initial_message_reads_current_intent_category_directly(self) -> None:
        product = {"parent_asin": "A"}
        sample = {
            "version": "v2",
            "sample_id": "buy_1",
            "intent": "buying",
            "override": False,
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
            "item_id": "A",
            "features": product,
            "intent_descriptions": {
                "browsing": {
                    "style": "relaxed",
                    "material": "natural fabric",
                    "size": "several sizes",
                    "category": "shirts",
                },
                "buying": {
                    "style": "classic",
                    "material": "cotton",
                    "size": "medium",
                    "category": "fitted shirts",
                },
            },
            "fake_attributes": {},
            "correction_messages": {},
            "modify_turn": None,
        }
        simulator = V2Simulator(
            sample,
            {"A": ["Clothing", "Shirts"]},
            {"A": product},
            TemplateReplyModel(),
            "session",
        )
        self.assertFalse(hasattr(simulator, "intent_card"))
        self.assertEqual(
            simulator.initial_message(),
            "I'm looking for fitted shirts.",
        )

    def test_mixed_dataset_switches_logic_per_record(self) -> None:
        product = {"parent_asin": "A"}
        legacy = {
            "sample_id": "legacy_1",
            "scenario_type": "boundary",
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
        }
        v2 = {
            "version": "v2",
            "sample_id": "v2_1",
            "intent": "browsing",
            "override": False,
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
            "item_id": "A",
            "features": product,
            "intent_descriptions": {
                intent: {"style": "classic", "material": "cotton", "size": "medium", "color": "blue"}
                for intent in ("browsing", "buying")
            },
            "fake_attributes": {},
            "correction_messages": {},
            "modify_turn": None,
        }
        result = evaluate(
            EchoTargetAgent(), [legacy, v2], {"A"}, {"A": []}, {"A": product}
        )
        self.assertEqual(
            [session["scenario_type"] for session in result["sessions"]],
            ["boundary", "browsing"],
        )
        self.assertNotIn("version", result["sessions"][0])
        self.assertEqual(result["sessions"][1]["version"], "v2")


if __name__ == "__main__":
    unittest.main()
