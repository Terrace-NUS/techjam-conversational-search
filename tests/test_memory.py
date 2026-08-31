from __future__ import annotations

import tempfile
import unittest

from threadline_memory import InvalidSessionIdError, MemoryService
from threadline_memory.llm import FakeProfileUpdateClient
from threadline_memory.merge import merge_patch
from threadline_memory.schema import empty_profile


class MemoryServiceTest(unittest.TestCase):
    def test_same_subject_preferences_are_scoped_by_category(self) -> None:
        profile = empty_profile("alice", created_at="2026-01-01T00:00:00Z")
        outcome = merge_patch(
            profile,
            {
                "preferences": {
                    "attributes": [
                        {
                            "subject": "size",
                            "value": "L",
                            "category": "shirt",
                            "source": "explicit",
                        },
                        {
                            "subject": "size",
                            "value": "42",
                            "category": "shoes",
                            "source": "explicit",
                        },
                    ]
                }
            },
        )

        by_category = {
            item["category"]: item["value"]
            for item in outcome.profile["shopping_preferences"]["attributes"]
        }
        self.assertEqual(by_category, {"shirt": "L", "shoes": "42"})

    def test_category_scoped_budgets_coexist(self) -> None:
        profile = empty_profile("alice", created_at="2026-01-01T00:00:00Z")
        outcome = merge_patch(
            profile,
            {
                "preferences": {
                    "price": [
                        {"value": 30, "category": "shirt", "source": "explicit"},
                        {"value": 500, "category": "laptop", "source": "explicit"},
                    ]
                }
            },
        )

        prices = outcome.profile["shopping_preferences"]["price"]
        self.assertEqual({item["category"] for item in prices}, {"shirt", "laptop"})

    def test_category_correction_does_not_remove_other_category(self) -> None:
        profile = empty_profile("alice", created_at="2026-01-01T00:00:00Z")
        profile = merge_patch(
            profile,
            {
                "preferences": {
                    "attributes": [
                        {
                            "subject": "size",
                            "value": "L",
                            "category": "shirt",
                            "source": "explicit",
                        },
                        {
                            "subject": "size",
                            "value": "42",
                            "category": "shoes",
                            "source": "explicit",
                        },
                    ]
                }
            },
        ).profile
        outcome = merge_patch(
            profile,
            {
                "corrections": [
                    {
                        "section": "attributes",
                        "subject": "size",
                        "category": "shoes",
                        "superseded_values": ["42"],
                        "final_value": "43",
                        "source": "explicit",
                        "evidence": "Actually my shoe size is 43.",
                    }
                ]
            },
        )

        by_category = {
            item["category"]: item["value"]
            for item in outcome.profile["shopping_preferences"]["attributes"]
        }
        self.assertEqual(by_category, {"shirt": "L", "shoes": "43"})

    def test_shopping_preferences_alias_is_normalized(self) -> None:
        profile = empty_profile("alice", created_at="2026-01-01T00:00:00Z")
        outcome = merge_patch(
            profile,
            {
                "shopping_preferences": {
                    "attributes": [
                        {
                            "subject": "waterproofing",
                            "value": "waterproof",
                            "source": "explicit",
                            "evidence": "I need waterproof shoes",
                        }
                    ]
                }
            },
        )

        self.assertEqual(
            outcome.profile["shopping_preferences"]["attributes"][0]["value"],
            "waterproof",
        )

    def test_explicit_correction_replaces_superseded_price(self) -> None:
        profile = empty_profile("alice", created_at="2026-01-01T00:00:00Z")
        first = merge_patch(
            profile,
            {
                "preferences": {
                    "price": [
                        {
                            "subject": "budget",
                            "value": "$30-$50",
                            "source": "explicit",
                            "evidence": "My budget is $30-$50",
                        }
                    ]
                }
            },
        ).profile
        corrected = merge_patch(
            first,
            {
                "corrections": [
                    {
                        "section": "price",
                        "subject": "budget",
                        "superseded_values": ["$30-$50"],
                        "final_value": "under $15",
                        "source": "explicit",
                        "evidence": "Actually, keep it under $15",
                    }
                ]
            },
        ).profile

        values = [item["value"] for item in corrected["shopping_preferences"]["price"]]
        self.assertEqual(values, ["under $15"])

    def test_explicit_retraction_removes_preference(self) -> None:
        profile = empty_profile("alice", created_at="2026-01-01T00:00:00Z")
        profile["shopping_preferences"]["attributes"].append(
            {
                "subject": "waterproofing",
                "value": "waterproof",
                "polarity": "positive",
                "confidence": 0.9,
                "source": "explicit",
                "evidence": "I need waterproof shoes",
            }
        )
        outcome = merge_patch(
            profile,
            {
                "corrections": [
                    {
                        "section": "attributes",
                        "subject": "waterproofing",
                        "superseded_values": ["waterproof"],
                        "final_value": None,
                        "source": "explicit",
                        "evidence": "Actually, waterproofing does not matter",
                    }
                ]
            },
        )

        self.assertEqual(outcome.profile["shopping_preferences"]["attributes"], [])

    def test_session_retry_is_idempotent_and_wrong_session_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryService.from_json_directory(directory)

            memory.start_session("alice", "s1")
            retried = memory.start_session("alice", "s1")

            self.assertEqual(retried.user_profile["metadata"]["session_count"], 1)
            with self.assertRaises(InvalidSessionIdError):
                memory.update_from_dialogue("alice", "s2", [])

    def test_profile_survives_a_new_service_instance(self) -> None:
        patch = {
            "occupation": {
                "value": "engineer",
                "source": "explicit",
                "evidence": "I am an engineer",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryService.from_json_directory(
                directory,
                llm=FakeProfileUpdateClient([patch]),
            )
            memory.start_session("alice", "s1")
            memory.update_from_dialogue(
                "alice",
                "s1",
                [{"role": "user", "content": "I am an engineer"}],
            )

            reopened = MemoryService.from_json_directory(directory)
            profile = reopened.get_profile("alice")

            self.assertEqual(
                profile["personal_context"]["occupation"]["value"],
                "engineer",
            )


if __name__ == "__main__":
    unittest.main()
