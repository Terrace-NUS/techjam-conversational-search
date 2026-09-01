from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import hashlib
import math
import os
import random
import statistics
import ssl
import time
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
    DEFAULT_OUTPUT_DIMENSIONALITY = 768
    MAX_INPUT_CHARS = 1000

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "data/cache/embeddings",
        timeout: float = 30.0,
        output_dimensionality: int | None = None,
        retries: int = 4,
    ) -> None:
        _load_dotenv()
        self.api_key = (
            api_key
            or os.environ.get("AI_STUIDIO_API_KEY", "").strip()
            or os.environ.get("AI_STUDIO_API_KEY", "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()
        )
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or AI_STUIDIO_API_KEY is required for embedding-based "
                "reward scoring; set it in the environment or a .env file."
            )
        self.model = model or os.environ.get("GEMINI_EMBEDDING_MODEL", self.DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("GEMINI_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.output_dimensionality = output_dimensionality or int(
            os.environ.get("GEMINI_EMBEDDING_DIMENSIONS", self.DEFAULT_OUTPUT_DIMENSIONALITY)
        )
        if self.output_dimensionality <= 0:
            raise ValueError("output_dimensionality must be positive")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.retries = retries

    def embed(self, asin: str, text: str) -> list[float]:
        """Return the embedding for `text`, invalidating stale ASIN caches."""
        cache_path = self.cache_dir / f"{asin}.json"
        input_text = text[: self.MAX_INPUT_CHARS]
        text_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        cached = self._cached_values(cache_path, text_hash)
        if cached is not None:
            return cached
        values = self._request(input_text)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "dimensions": self.output_dimensionality,
                    "task_type": self.TASK_TYPE,
                    "text_hash": text_hash,
                    "values": values,
                }
            ),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)
        return values

    def _cached_values(self, cache_path: Path, text_hash: str) -> list[float] | None:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            cached.get("model") == self.model
            and cached.get("dimensions") == self.output_dimensionality
            and cached.get("task_type") == self.TASK_TYPE
            and cached.get("text_hash") == text_hash
            and isinstance(cached.get("values"), list)
            and len(cached["values"]) == self.output_dimensionality
        ):
            return cached["values"]
        return None

    def _request(self, text: str) -> list[float]:
        url = f"{self.base_url}/v1beta/models/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": self.TASK_TYPE,
            "outputDimensionality": self.output_dimensionality,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=_SSL_CONTEXT
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise RuntimeError(
                        f"Gemini embedContent request failed ({error.code}): {detail}"
                    ) from error
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries:
                    raise
            else:
                values = body.get("embedding", {}).get("values")
                if not isinstance(values, list) or not values:
                    raise ValueError(f"Gemini embedContent response missing values: {body}")
                return values
            time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise AssertionError("embedding retry loop must return or raise")


class SiliconFlowEmbeddingClient(GeminiEmbeddingClient):
    """SiliconFlow embeddings through its OpenAI-compatible API."""

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
    MAX_INPUT_CHARS = 20000

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "data/cache/embeddings",
        timeout: float = 30.0,
        output_dimensionality: int | None = None,
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
        self.output_dimensionality = output_dimensionality or int(
            os.environ.get("SILICONFLOW_EMBEDDING_DIMENSIONS", 1024)
        )
        if self.output_dimensionality <= 0:
            raise ValueError("output_dimensionality must be positive")
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
    """Per-turn subscore: maximum recommendation relevance to the ground truth."""

    def __init__(
        self,
        embedding_client: GeminiEmbeddingClient,
        text_fn: Callable[[dict], str],
        baseline_sample_size: int = 32,
        baseline_cache_path: str | Path = "data/cache/baselines.json",
        embedding_workers: int = 8,
    ) -> None:
        if embedding_workers < 1:
            raise ValueError("embedding_workers must be positive")
        self.embedding_client = embedding_client
        self.text_fn = text_fn
        self.baseline_sample_size = max(0, baseline_sample_size)
        self.baseline_cache_path = Path(baseline_cache_path)
        self._embedding_executor = ThreadPoolExecutor(max_workers=embedding_workers)
        self._embedding_lock = threading.Lock()
        self._embedding_cache: dict[tuple[str, str], list[float]] = {}
        self._embedding_futures: dict[tuple[str, str], Future[list[float]]] = {}
        self._baseline_cache: dict[str, float] = {}
        self._baseline_lock = threading.Lock()
        self._baseline_target_locks: dict[str, threading.Lock] = {}
        self._baseline_metadata: dict[str, object] | None = None

    def score_turn(self, ranked: list[str], target_asin: str, products: dict[str, dict]) -> float:
        """Return a subscore in [0, 1] for the current turn's ranked recommendations.

        A hit (`target_asin` present in `ranked`) scores 1.0; the caller's session loop
        already ends the session on a hit, so this branch is defensive rather than load
        bearing. Otherwise each recommendation is scored in [0, 1] and the turn score is
        their maximum.
        """
        if target_asin in ranked:
            return 1.0
        if not ranked:
            return 0.0
        target_product = products.get(target_asin)
        if target_product is None:
            return 0.0
        ranked_asins = [asin for asin in dict.fromkeys(ranked) if asin in products]
        if not ranked_asins:
            return 0.0
        vectors = self._embed_many([target_asin, *ranked_asins], products)
        target_vector = vectors[target_asin]
        baseline = self._target_baseline(
            target_asin, ranked_asins[0], target_vector, products
        )
        scores = []
        for asin in ranked_asins:
            vector = vectors[asin]
            raw_similarity = min(1.0, max(0.0, cosine_similarity(vector, target_vector)))
            score = (
                raw_similarity
                if baseline is None
                else min(
                    1.0,
                    max(0.0, raw_similarity - baseline) / max(1e-12, 1.0 - baseline),
                )
            )
            scores.append(score)
        return max(scores, default=0.0)

    def _embed_many(self, asins: list[str], products: dict[str, dict]) -> dict[str, list[float]]:
        results: dict[str, list[float]] = {}
        pending: list[tuple[str, tuple[str, str], Future[list[float]]]] = []
        for asin in dict.fromkeys(asins):
            product = products.get(asin)
            if product is None:
                continue
            text = self.text_fn(product)
            key = (asin, hashlib.sha256(text.encode("utf-8")).hexdigest())
            with self._embedding_lock:
                cached = self._embedding_cache.get(key)
                if cached is not None:
                    results[asin] = cached
                    continue
                future = self._embedding_futures.get(key)
                if future is None:
                    future = self._embedding_executor.submit(
                        self.embedding_client.embed, asin, text
                    )
                    self._embedding_futures[key] = future
            pending.append((asin, key, future))

        for asin, key, future in pending:
            try:
                vector = future.result()
            except Exception:
                with self._embedding_lock:
                    if self._embedding_futures.get(key) is future:
                        self._embedding_futures.pop(key, None)
                raise
            with self._embedding_lock:
                self._embedding_cache[key] = vector
                if self._embedding_futures.get(key) is future:
                    self._embedding_futures.pop(key, None)
            results[asin] = vector
        return results

    def _target_baseline(
        self,
        target_asin: str,
        rank1_asin: str,
        target_vector: list[float],
        products: dict[str, dict],
    ) -> float | None:
        """Return a deterministic in-memory median similarity baseline for target."""
        # Keep the rank-1 item out of the negative reference set so it cannot
        # inflate its own baseline.
        candidates = [asin for asin in products if asin not in {target_asin, rank1_asin}]
        if not candidates or self.baseline_sample_size <= 0:
            return None
        with self._baseline_lock:
            self._load_baselines(products)
            cached = self._baseline_cache.get(target_asin)
            if cached is not None:
                return cached
            target_lock = self._baseline_target_locks.setdefault(target_asin, threading.Lock())

        with target_lock:
            with self._baseline_lock:
                cached = self._baseline_cache.get(target_asin)
                if cached is not None:
                    return cached
            rng = random.Random(target_asin)
            sample = rng.sample(candidates, min(self.baseline_sample_size, len(candidates)))
            vectors = self._embed_many(sample, products)
            similarities = [
                max(0.0, cosine_similarity(vectors[asin], target_vector))
                for asin in sample
                if asin in vectors
            ]
            if not similarities:
                return None
            baseline = statistics.median(similarities)
            with self._baseline_lock:
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
