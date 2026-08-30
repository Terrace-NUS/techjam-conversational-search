"""Replaceable LLM boundary for profile-patch extraction.

The service depends only on the :class:`ProfileUpdateLLM` protocol, so business
logic is never bound to DeepSeek. Three implementations ship here:

* :class:`FakeProfileUpdateClient`     — returns a canned/queued patch (tests);
* :class:`RuleBasedProfileUpdateClient`— tiny offline keyword extractor so the
  whole module runs and is demoable with no network and no API key;
* :class:`DeepSeekProfileUpdateClient` — calls a DeepSeek chat-completions
  endpoint, configured purely from environment variables.

No implementation performs network I/O at import time.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol, runtime_checkable

from .prompt import SYSTEM_PROMPT, build_user_payload


@runtime_checkable
class ProfileUpdateLLM(Protocol):
    def extract_profile_patch(
        self,
        current_profile: dict[str, object],
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        """Return a partial profile patch (never the whole profile)."""
        ...


class FakeProfileUpdateClient:
    """Deterministic client for tests: returns queued patches, then ``{}``."""

    def __init__(self, patches: list[dict[str, Any]] | None = None) -> None:
        self._patches = list(patches or [])
        self.calls: list[tuple[dict[str, object], list[dict[str, str]]]] = []

    def queue(self, patch: dict[str, Any]) -> None:
        self._patches.append(patch)

    def extract_profile_patch(
        self,
        current_profile: dict[str, object],
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        self.calls.append((current_profile, messages))
        if self._patches:
            return self._patches.pop(0)
        return {}


class RuleBasedProfileUpdateClient:
    """Keyword-based extractor so the module works with no LLM and no network.

    It is intentionally conservative and covers the demo phrases: an explicit
    job statement, explicit interests, explicit likes/dislikes, and "for mom"
    gifting. It never guesses occupation from a purchase.
    """

    _OCCUPATION = re.compile(
        r"(?:我是(?:一名|一个|个)?|我的职业是|职业是|i(?:'m| am) an? )\s*([^\s，。,\.!?、]+)"
    )
    _INTEREST = re.compile(r"(?:我(?:很)?喜欢|我热爱|i love |i really like )\s*([^\s，。,\.!?、]+)")
    _GIFT_MOTHER = re.compile(r"(给|帮)?\s*(妈妈|母亲|我妈)\s*(买|挑|选)")

    _JOB_HINTS = ("摄影师", "工程师", "老师", "医生", "设计师", "程序员", "photographer", "engineer")

    def extract_profile_patch(
        self,
        current_profile: dict[str, object],
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        text = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        patch: dict[str, Any] = {}

        occ = self._OCCUPATION.search(text)
        if occ and any(hint in occ.group(1) for hint in self._JOB_HINTS):
            patch["occupation"] = {
                "value": occ.group(1),
                "confidence": 0.95,
                "source": "explicit",
                "evidence": occ.group(0),
            }

        interests = []
        for match in self._INTEREST.finditer(text):
            interests.append(
                {
                    "name": match.group(1),
                    "strength": 0.7,
                    "confidence": 0.85,
                    "source": "explicit",
                    "evidence": match.group(0),
                }
            )
        if interests:
            patch["interests"] = interests

        if self._GIFT_MOTHER.search(text):
            patch["recipient_cards"] = [
                {
                    "recipient_id": "mother",
                    "relationship": "mother",
                    "display_name": "妈妈",
                }
            ]
        return patch


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


class DeepSeekProfileUpdateClient:
    """DeepSeek chat-completions client configured from the environment.

    Environment variables (all optional except the key at call time):
    ``DEEPSEEK_API_KEY``, ``DEEPSEEK_BASE_URL`` (default
    ``https://api.deepseek.com``), ``DEEPSEEK_MODEL`` (default
    ``deepseek-chat``). Import never touches the network.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self._base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL")
                          or "https://api.deepseek.com").rstrip("/")
        self._model = model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
        self._timeout = timeout

    def extract_profile_patch(
        self,
        current_profile: dict[str, object],
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        if not self._api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        raw = self._call(build_user_payload(dict(current_profile), list(messages)))
        try:
            parsed = json.loads(_strip_code_fence(raw))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _call(self, user_content: str) -> str:  # pragma: no cover - network path
        import urllib.request

        body = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"])
