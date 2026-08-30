from __future__ import annotations

import tempfile
import unittest

from threadline_memory import InvalidSessionIdError, MemoryService
from threadline_memory.llm import FakeProfileUpdateClient


class MemoryServiceTest(unittest.TestCase):
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
