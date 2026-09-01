from __future__ import annotations

import unittest

from shopping_copilot.application.response_generation import DeterministicResponseComposer
from shopping_copilot.query_understanding import build_reconcile_request, request_payload
from shopping_copilot.session_context import IntentState
from scripts.simulator.evaluate_full_pipeline_other import _intent_diff


class ResponseGenerationTest(unittest.TestCase):
    def test_first_query_understanding_can_include_user_profile(self) -> None:
        intent = IntentState(
            goal=None,
            preferences=(),
            dont_care_facets=frozenset(),
            version=0,
        )
        profile = {"preference_tags": ["black shoes"], "summary": "Prefers black shoes."}

        first = build_reconcile_request(
            turn=1,
            latest_utterance="I'm looking for men's shoes.",
            current_intent=intent,
            category_options=(),
            user_profile=profile,
        )
        later = build_reconcile_request(
            turn=2,
            latest_utterance="Something comfortable.",
            current_intent=intent,
            category_options=(),
        )

        self.assertEqual(request_payload(first)["user_profile"], profile)
        self.assertIsNone(request_payload(later)["user_profile"])

    def test_query_understanding_diff_tracks_structured_changes(self) -> None:
        before = IntentState(
            goal=None,
            preferences=(),
            dont_care_facets=frozenset({"brand"}),
            version=0,
        )
        after = IntentState(
            goal="necklace",
            preferences=(),
            dont_care_facets=frozenset({"color"}),
            version=1,
        )

        diff = _intent_diff(before, after)

        self.assertEqual(
            diff["goal"],
            {"before": None, "after": "necklace", "changed": True},
        )
        self.assertEqual(diff["dont_care"], {"added": ["color"], "removed": ["brand"]})

    def test_response_is_one_explicit_unasked_attribute_question(self) -> None:
        composer = DeterministicResponseComposer()
        intent = IntentState(
            goal="necklace",
            preferences=(),
            dont_care_facets=frozenset({"material"}),
            version=1,
        )

        response = composer.compose(
            recommendations=("A", "B"),
            transparency=0.2,
            previous_transparency=None,
            ranking=None,
            intent=intent,
            product_metadata={"A": {"title": "First"}, "B": {"title": "Second"}},
            asked_attributes=("feature",),
        )

        self.assertEqual(response.question_attribute, "budget")
        self.assertEqual(response.message, "What budget should I stay within?")
        self.assertNotIn("strongest options", response.message.casefold())


if __name__ == "__main__":
    unittest.main()
