"""Adapters that project a stored profile onto the host Shopping Copilot APIs.

This module owns the whole coupling to the main project. It never imports host
code and never mutates it; it only shapes plain dicts that match the host
contracts:

* :func:`to_profile_prior` — the exact five ``ProfilePrior`` fields;
* :func:`to_ranking_user_profile` — kwargs for ``RankingUserProfile`` as
  ``{schema, version, payload}``.
"""

from __future__ import annotations

from typing import Any

from ..schema import (
    RANKING_PROFILE_SCHEMA,
    RANKING_PROFILE_VERSION,
    base_profile_defaults,
    normalize_base_profile,
)

_PROFILE_PRIOR_FIELDS = (
    "purchase_frequency",
    "average_prior_rating",
    "rating_style",
    "preference_tags",
    "summary",
)


def to_profile_prior(profile: dict[str, Any]) -> dict[str, Any]:
    """Return ONLY the five host ``ProfilePrior`` fields, in contract order.

    ``preference_tags`` is a tuple to match the host's declared type. No
    long-term extras (occupation, interests, ...) are leaked into this view.
    """

    base = profile.get("base_profile") if isinstance(profile, dict) else None
    normalized = normalize_base_profile(base if isinstance(base, dict) else base_profile_defaults())
    return {
        "purchase_frequency": normalized["purchase_frequency"],
        "average_prior_rating": normalized["average_prior_rating"],
        "rating_style": normalized["rating_style"],
        "preference_tags": tuple(normalized["preference_tags"]),
        "summary": normalized["summary"],
    }


def to_reset_request(session_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Build the host ``Agent.reset`` request: ``{session_id, user_profile}``."""

    if type(session_id) is not str or not session_id.strip() or session_id != session_id.strip():
        raise ValueError("session_id must be a non-blank trimmed string")
    prior = to_profile_prior(profile)
    prior["preference_tags"] = list(prior["preference_tags"])
    return {"session_id": session_id, "user_profile": prior}


def to_ranking_user_profile(
    profile: dict[str, Any],
    *,
    selected_recipient: str | None = None,
) -> dict[str, Any]:
    """Return ``{schema, version, payload}`` kwargs for ``RankingUserProfile``.

    The payload is a compact weak prior only. Full chat, raw evidence text and
    unrelated recipient data are deliberately excluded. ``selected_recipient``
    is set only when the CURRENT query explicitly targets a named recipient.
    """

    occupation = _occupation_value(profile)
    interests = _interest_names(profile)
    prefs = profile.get("shopping_preferences", {}) if isinstance(profile, dict) else {}

    payload: dict[str, Any] = {
        "occupation": occupation,
        "interests": interests,
        "category_priors": _positive_values(prefs.get("categories")),
        "attribute_priors": _attribute_priors(prefs.get("attributes")),
        "negative_priors": _negative_priors(prefs),
        "selected_recipient": selected_recipient,
    }
    return {
        "schema": RANKING_PROFILE_SCHEMA,
        "version": RANKING_PROFILE_VERSION,
        "payload": payload,
    }


def _occupation_value(profile: dict[str, Any]) -> str | None:
    ctx = profile.get("personal_context") if isinstance(profile, dict) else None
    if isinstance(ctx, dict) and isinstance(ctx.get("occupation"), dict):
        value = ctx["occupation"].get("value")
        return value if isinstance(value, str) else None
    return None


def _interest_names(profile: dict[str, Any]) -> list[dict[str, Any]]:
    ctx = profile.get("personal_context") if isinstance(profile, dict) else None
    items = ctx.get("interests") if isinstance(ctx, dict) else None
    if not isinstance(items, list):
        return []
    return [
        {"name": i["name"], "strength": i.get("strength", 0.5)}
        for i in items
        if isinstance(i, dict) and isinstance(i.get("name"), str)
    ]


def _positive_values(bucket: Any) -> list[dict[str, Any]]:
    if not isinstance(bucket, list):
        return []
    return [
        {"value": e["value"], "confidence": e.get("confidence", 0.5)}
        for e in bucket
        if isinstance(e, dict) and e.get("polarity", "positive") == "positive" and e.get("value")
    ]


def _attribute_priors(bucket: Any) -> list[dict[str, Any]]:
    if not isinstance(bucket, list):
        return []
    return [
        {
            "subject": e.get("subject", e.get("value")),
            "value": e.get("value"),
            "category": e.get("category"),
            "polarity": e.get("polarity", "positive"),
            "confidence": e.get("confidence", 0.5),
        }
        for e in bucket
        if isinstance(e, dict) and e.get("value")
    ]


def _negative_priors(prefs: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for section in ("categories", "attributes", "brands", "constraints", "negative_preferences"):
        bucket = prefs.get(section)
        if not isinstance(bucket, list):
            continue
        for e in bucket:
            if isinstance(e, dict) and (
                e.get("polarity") == "negative" or section == "negative_preferences"
            ):
                result.append(
                    {
                        "section": section,
                        "value": e.get("value"),
                        "confidence": e.get("confidence", 0.5),
                    }
                )
    return result
