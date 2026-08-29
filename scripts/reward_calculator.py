from __future__ import annotations

import json
import hashlib
import math
import os
import random
import statistics
import ssl
import urllib.error
import urllib.request
import threading
from pathlib import Path
from typing import Callable



def _load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without adding a dotenv dependency.

    Keep configuration loading local to this module: reward calculation belongs to
    ``scripts`` now and must not import the evaluator just to obtain an API key.
    That also avoids a scripts -> evaluator import cycle when the evaluator imports
    this module at runtime.
    """
    dotenv_path = Path(path)
    if not dotenv_path.is_file():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value

try:
    import certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    # certifi ships transitively via openai/httpx; fall back to the interpreter's
    # own trust store if it's ever absent.
    _SSL_CONTEXT = None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 for a zero vector."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class GeminiEmbeddingClient:
    """Gemini (Google AI Studio) text embeddings with a per-`parent_asin` disk cache.

    Request and output errors are fatal: a reward signal is worse unbuilt than built
    on values nobody checked.
    """

    DEFAULT_MODEL = "gemini-embedding-001"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
    TASK_TYPE = "SEMANTIC_SIMILARITY"
    MAX_INPUT_CHARS = 20000

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "data/cache/embeddings",
        timeout: float = 30.0,
    ) -> None:
        _load_dotenv()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for embedding-based reward scoring; "
                "set it in the environment or a .env file."
            )
        self.model = model or os.environ.get("GEMINI_EMBEDDING_MODEL", self.DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("GEMINI_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout

    def embed(self, asin: str, text: str) -> list[float]:
        """Return the embedding for `text`, invalidating stale ASIN caches."""
        cache_path = self.cache_dir / f"{asin}.json"
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("model") == self.model
                and cached.get("text_hash") == text_hash
                and isinstance(cached.get("values"), list)
            ):
                return cached["values"]
        values = self._request(text)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"model": self.model, "text_hash": text_hash, "values": values}),
            encoding="utf-8",
        )
        return values

    def _request(self, text: str) -> list[float]:
        url = f"{self.base_url}/v1beta/models/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text[: self.MAX_INPUT_CHARS]}]},
            "taskType": self.TASK_TYPE,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=_SSL_CONTEXT) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini embedContent request failed ({error.code}): {detail}") from error
        values = body.get("embedding", {}).get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Gemini embedContent response missing values: {body}")
        return values


class SiliconFlowEmbeddingClient(GeminiEmbeddingClient):
    """SiliconFlow embeddings through its OpenAI-compatible API."""

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "data/cache/embeddings",
        timeout: float = 30.0,
    ) -> None:
        _load_dotenv()
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "SILICONFLOW_API_KEY is required for SiliconFlow embedding scoring; "
                "set it in the environment or a .env file."
            )
        self.model = model or os.environ.get("SILICONFLOW_EMBEDDING_MODEL", self.DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("SILICONFLOW_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)

    def _request(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text[: self.MAX_INPUT_CHARS])
        values = response.data[0].embedding if response.data else None
        if not isinstance(values, list) or not values:
            raise ValueError("SiliconFlow embeddings response missing data")
        return values


class RewardCalculator:
    """Per-turn subscore: rank-1 recommendation relevance to the session's ground truth."""

    def __init__(
        self,
        embedding_client: GeminiEmbeddingClient,
        text_fn: Callable[[dict], str],
        baseline_sample_size: int = 32,
        margin_scale: float = 10.0,
        baseline_cache_path: str | Path = "data/cache/baselines.json",
    ) -> None:
        self.embedding_client = embedding_client
        self.text_fn = text_fn
        self.baseline_sample_size = max(0, baseline_sample_size)
        self.margin_scale = max(0.0, margin_scale)
        self.baseline_cache_path = Path(baseline_cache_path)
        self._baseline_cache: dict[str, float] = {}
        self._baseline_lock = threading.Lock()
        self._baseline_metadata: dict[str, object] | None = None

    def score_turn(self, ranked: list[str], target_asin: str, products: dict[str, dict]) -> float:
        """Return a subscore in [0, 1] for the current turn's ranked recommendations.

        A hit (`target_asin` present in `ranked`) scores 1.0; the caller's session loop
        already ends the session on a hit, so this branch is defensive rather than load
        bearing. Otherwise the score is the cosine similarity, clipped to [0, 1], between
        the rank-1 recommendation and the target product.
        """
        if target_asin in ranked:
            return 1.0
        if not ranked:
            return 0.0
        rank1_asin = ranked[0]
        rank1_product = products.get(rank1_asin)
        target_product = products.get(target_asin)
        if rank1_product is None or target_product is None:
            return 0.0
        target_vector = self.embedding_client.embed(target_asin, self.text_fn(target_product))
        rank1_vector = self.embedding_client.embed(rank1_asin, self.text_fn(rank1_product))
        raw_similarity = max(0.0, cosine_similarity(rank1_vector, target_vector))
        baseline = self._target_baseline(target_asin, rank1_asin, target_vector, products)
        if baseline is None:
            return raw_similarity
        margin = max(0.0, raw_similarity - baseline)
        return min(1.0, margin * self.margin_scale)

    def _target_baseline(
        self,
        target_asin: str,
        rank1_asin: str,
        target_vector: list[float],
        products: dict[str, dict],
    ) -> float | None:
        """Return a deterministic in-memory median similarity baseline for target."""
        cached = self._baseline_cache.get(target_asin)
        if cached is not None:
            return cached
        # Keep the rank-1 item out of the negative reference set so it cannot
        # inflate its own baseline.
        candidates = [asin for asin in products if asin not in {target_asin, rank1_asin}]
        if not candidates or self.baseline_sample_size <= 0:
            return None
        with self._baseline_lock:
            cached = self._baseline_cache.get(target_asin)
            if cached is not None:
                return cached
            self._load_baselines(products)
            cached = self._baseline_cache.get(target_asin)
            if cached is not None:
                return cached
            rng = random.Random(target_asin)
            sample = rng.sample(candidates, min(self.baseline_sample_size, len(candidates)))
            similarities = []
            for asin in sample:
                product = products.get(asin)
                if product is not None:
                    vector = self.embedding_client.embed(asin, self.text_fn(product))
                    similarities.append(max(0.0, cosine_similarity(vector, target_vector)))
            if not similarities:
                return None
            baseline = statistics.median(similarities)
            self._baseline_cache[target_asin] = baseline
            self._save_baselines(products)
            return baseline

    def _load_baselines(self, products: dict[str, dict]) -> None:
        if self._baseline_metadata is not None:
            return
        fingerprint = hashlib.sha256("\0".join(sorted(products)).encode()).hexdigest()
        metadata = {"model": getattr(self.embedding_client, "model", None), "sample_size": self.baseline_sample_size, "catalog": fingerprint}
        self._baseline_metadata = metadata
        if not self.baseline_cache_path.is_file():
            return
        try:
            payload = json.loads(self.baseline_cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if payload.get("metadata") == metadata and isinstance(payload.get("baselines"), dict):
            self._baseline_cache.update({str(k): float(v) for k, v in payload["baselines"].items()})

    def _save_baselines(self, products: dict[str, dict]) -> None:
        self.baseline_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": self._baseline_metadata, "baselines": self._baseline_cache}
        temporary_path = self.baseline_cache_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(payload), encoding="utf-8")
        temporary_path.replace(self.baseline_cache_path)
