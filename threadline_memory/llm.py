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

from .prompt import (
    CORRECTION_AUDIT_PROMPT,
    SYSTEM_PROMPT,
    build_correction_audit_payload,
    build_user_payload,
)

_CORRECTION_SIGNAL = re.compile(
    r"\b(?:actually|correction|correct(?:ing)?\s+(?:that|my)|i meant|instead|sorry[, ]+i meant|not)\b"
    r"|更正|纠正|改成|不是|而是",
    re.IGNORECASE,
)


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


def _has_correction_signal(messages: list[dict[str, str]]) -> bool:
    return any(
        message.get("role") == "user"
        and bool(_CORRECTION_SIGNAL.search(message.get("content", "")))
        for message in messages
    )


class DeepSeekProfileUpdateClient:
    """DeepSeek chat-completions client configured from the environment.

    Environment variables (all optional except the key at call time):
    ``DEEPSEEK_API_KEY``, ``DEEPSEEK_BASE_URL`` (default
    ``https://api.deepseek.com``), ``DEEPSEEK_MODEL`` (default
    ``deepseek-v4-flash``). Import never touches the network.
    """

    DEFAULT_MODEL = "deepseek-v4-flash"
    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self._base_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._model = model or os.environ.get("DEEPSEEK_MODEL") or self.DEFAULT_MODEL
        self._timeout = timeout

    def extract_profile_patch(
        self,
        current_profile: dict[str, object],
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        if not self._api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        dialogue = list(messages)
        raw = self._call(build_user_payload(dict(current_profile), dialogue))
        try:
            parsed = json.loads(_strip_code_fence(raw))
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        if _has_correction_signal(dialogue):
            try:
                audit_raw = self._chat(
                    CORRECTION_AUDIT_PROMPT,
                    build_correction_audit_payload(dialogue, parsed),
                )
                audit = json.loads(_strip_code_fence(audit_raw))
                corrections = audit.get("corrections") if isinstance(audit, dict) else None
                if isinstance(corrections, list):
                    parsed["corrections"] = corrections
            except (KeyError, TypeError, json.JSONDecodeError, OSError):
                # The first extraction remains useful if the optional audit fails.
                pass
        return parsed

    def _call(self, user_content: str) -> str:  # pragma: no cover - network path
        return self._chat(SYSTEM_PROMPT, user_content)

    def _chat(self, system_prompt: str, user_content: str) -> str:  # pragma: no cover
        import urllib.request

        body = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
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
