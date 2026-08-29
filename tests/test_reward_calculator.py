from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.reward_calculator import GeminiEmbeddingClient, RewardCalculator, cosine_similarity


class FakeEmbeddingClient:
    """Deterministic asin -> vector map; no network calls."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []

    def embed(self, asin: str, text: str) -> list[float]:
        self.calls.append(asin)
        return self.vectors[asin]


class CosineSimilarityTest(unittest.TestCase):
    def test_identical_vectors_score_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector_is_zero(self) -> None:
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)


class RewardCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.products = {
            "A": {"title": "target product"},
            "B": {"title": "rank1 product"},
        }

    def test_hit_scores_one_without_calling_embedding_client(self) -> None:
        client = FakeEmbeddingClient({})
        calculator = RewardCalculator(client, text_fn=lambda product: product["title"])
        self.assertEqual(calculator.score_turn(["A", "C"], "A", self.products), 1.0)
        self.assertEqual(client.calls, [])

    def test_empty_ranked_scores_zero(self) -> None:
        client = FakeEmbeddingClient({})
        calculator = RewardCalculator(client, text_fn=lambda product: product["title"])
        self.assertEqual(calculator.score_turn([], "A", self.products), 0.0)

    def test_miss_scores_clipped_cosine_similarity_of_rank1_and_target(self) -> None:
        client = FakeEmbeddingClient({"A": [1.0, 0.0], "B": [-1.0, 0.0]})
        calculator = RewardCalculator(client, text_fn=lambda product: product["title"])
        # rank1 is "B", target is "A"; opposite vectors would cosine to -1, clipped to 0.
        self.assertEqual(calculator.score_turn(["B"], "A", self.products), 0.0)
        self.assertEqual(sorted(client.calls), ["A", "B"])

    def test_miss_scores_relative_to_catalog_baseline(self) -> None:
        products = {asin: {"title": asin} for asin in ("A", "B", "C", "D")}
        client = FakeEmbeddingClient({
            "A": [1.0, 0.0],
            "B": [0.9, (1.0 - 0.9 ** 2) ** 0.5],
            "C": [1.0, 0.0],
            "D": [0.6, 0.8],
        })
        calculator = RewardCalculator(client, text_fn=lambda product: product["title"], baseline_sample_size=2, margin_scale=1.0)
        # The result is the positive margin rather than the raw high cosine.
        self.assertAlmostEqual(calculator.score_turn(["B"], "A", products), 0.1, places=6)


class GeminiEmbeddingClientTest(unittest.TestCase):
    def test_missing_api_key_raises(self) -> None:
        with mock.patch("scripts.reward_calculator._load_dotenv"):
            with mock.patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(RuntimeError):
                    GeminiEmbeddingClient(api_key=None)

    def test_embed_caches_to_disk_and_skips_repeat_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            client = GeminiEmbeddingClient(api_key="test-key", cache_dir=cache_dir)
            with mock.patch.object(client, "_request", return_value=[0.1, 0.2]) as request:
                first = client.embed("A", "target product")
                second = client.embed("A", "target product")
            self.assertEqual(first, [0.1, 0.2])
            self.assertEqual(second, [0.1, 0.2])
            request.assert_called_once()
            self.assertEqual(
                json.loads((cache_dir / "A.json").read_text(encoding="utf-8")),
                {
                    "model": client.model,
                    "text_hash": hashlib.sha256("target product".encode("utf-8")).hexdigest(),
                    "values": [0.1, 0.2],
                },
            )


if __name__ == "__main__":
    unittest.main()
