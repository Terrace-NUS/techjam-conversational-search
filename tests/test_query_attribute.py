from __future__ import annotations

import json
import os
import unittest

from scripts.query_attribute import (
    DeepSeekAttributeExtractor,
    extract_attribute,
    set_default_extractor,
)


class FakeClient:
    """Records requests and replays a queued response per call."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[dict] = []

    @property
    def chat(self):
        outer = self

        class _Chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    outer.calls.append(kwargs)
                    content = outer.contents.pop(0)
                    return type("Response", (), {
                        "choices": [type("Choice", (), {
                            "message": type("Message", (), {"content": content})()
                        })()]
                    })()

        return _Chat()


class DeepSeekAttributeExtractorTest(unittest.TestCase):
    def test_missing_api_key_raises(self) -> None:
        original = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                DeepSeekAttributeExtractor()
        finally:
            if original is not None:
                os.environ["DEEPSEEK_API_KEY"] = original

    def test_parse_requires_attribute_within_candidates(self) -> None:
        self.assertEqual(
            DeepSeekAttributeExtractor._parse('{"attribute":"material"}', {"material", "color"}),
            "material",
        )
        self.assertIsNone(DeepSeekAttributeExtractor._parse('{"attribute":null}', {"material"}))
        with self.assertRaises(ValueError):
            DeepSeekAttributeExtractor._parse('{"attribute":"brand"}', {"material"})
        with self.assertRaises(ValueError):
            DeepSeekAttributeExtractor._parse("not json", {"material"})
        with self.assertRaises(ValueError):
            DeepSeekAttributeExtractor._parse("{}", {"material"})

    def test_extract_sends_candidates_and_caches_result(self) -> None:
        extractor = DeepSeekAttributeExtractor(api_key="test-key")
        client = FakeClient(['{"attribute":"material"}'])
        extractor.client = client

        result = extractor.extract("Could you tell me what fabric you prefer?", ("material", "color"))
        self.assertEqual(result, "material")
        self.assertEqual(len(client.calls), 1)
        payload = json.loads(client.calls[0]["messages"][-1]["content"])
        self.assertEqual(set(payload["candidates"]), {"material", "color"})
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})

        # A repeated call with the same query/candidates is served from cache.
        result_again = extractor.extract("Could you tell me what fabric you prefer?", ("material", "color"))
        self.assertEqual(result_again, "material")
        self.assertEqual(len(client.calls), 1)

    def test_extract_retries_once_on_malformed_json_then_raises(self) -> None:
        extractor = DeepSeekAttributeExtractor(api_key="test-key")
        extractor.client = FakeClient(["not json", "still not json"])
        with self.assertRaises(ValueError):
            extractor.extract("what color is it?", ("color",))

    def test_extract_recovers_after_one_malformed_response(self) -> None:
        extractor = DeepSeekAttributeExtractor(api_key="test-key")
        client = FakeClient(["not json", '{"attribute":"color"}'])
        extractor.client = client
        self.assertEqual(extractor.extract("what color is it?", ("color",)), "color")
        self.assertEqual(len(client.calls), 2)

    def test_network_errors_are_not_swallowed(self) -> None:
        extractor = DeepSeekAttributeExtractor(api_key="test-key")

        class FailingClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise RuntimeError("network failure")

        extractor.client = FailingClient()
        with self.assertRaises(RuntimeError):
            extractor.extract("what color is it?", ("color",))

    def test_extract_attribute_delegates_to_default_extractor(self) -> None:
        class FakeExtractor:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[str, ...]]] = []

            def extract(self, query: str, candidates: tuple[str, ...]) -> str | None:
                self.calls.append((query, candidates))
                return "size" if "fit" in query else None

        fake = FakeExtractor()
        set_default_extractor(fake)
        try:
            self.assertEqual(extract_attribute("how does the fit run?", {"size", "color"}), "size")
            self.assertEqual(fake.calls[-1][0], "how does the fit run?")
            self.assertEqual(set(fake.calls[-1][1]), {"size", "color"})
            self.assertIsNone(extract_attribute("", {"size"}))
            self.assertIsNone(extract_attribute("hello", []))
        finally:
            set_default_extractor(None)


if __name__ == "__main__":
    unittest.main()
