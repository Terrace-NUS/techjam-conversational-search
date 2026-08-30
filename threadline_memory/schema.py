"""The on-disk user-profile document: schema, defaults, and normalization.

Everything the module persists is plain JSON. This module is the single source
of truth for the document shape so that the store, the patch merger and the
adapters all agree on field names.

The five ``base_profile`` fields are intentionally identical to the official
dataset ``user_profile`` and to the host ``ProfilePrior`` contract. They are
never renamed or repurposed. All long-term additions live in sibling sections.
"""

from __future__ import annotations

from typing import Any

PROFILE_SCHEMA = "threadline/user-profile"
PROFILE_VERSION = 2

RANKING_PROFILE_SCHEMA = "threadline/ranking-profile"
RANKING_PROFILE_VERSION = 1

# The five official dataset / ProfilePrior fields, with safe defaults.
_BASE_PROFILE_DEFAULTS: dict[str, Any] = {
    "purchase_frequency": "unknown",
    "average_prior_rating": None,
    "rating_style": "unknown",
    "preference_tags": [],
    "summary": "",
}

# Shopping-preference dimensions kept as lists of evidence-carrying entries.
_PREFERENCE_SECTIONS = (
    "categories",
    "brands",
    "attributes",
    "price",
    "constraints",
    "negative_preferences",
)


def base_profile_defaults() -> dict[str, Any]:
    return {
        "purchase_frequency": "unknown",
        "average_prior_rating": None,
        "rating_style": "unknown",
        "preference_tags": [],
        "summary": "",
    }


def empty_profile(user_id: str, *, created_at: str) -> dict[str, Any]:
    """Return a fresh, valid profile document for a brand-new user."""

    return {
        "schema": PROFILE_SCHEMA,
        "version": PROFILE_VERSION,
        "user_id": user_id,
        "base_profile": base_profile_defaults(),
        "personal_context": {
            "occupation": None,
            "interests": [],
        },
        "shopping_preferences": {section: [] for section in _PREFERENCE_SECTIONS},
        "purchase_anchors": [],
        "recipient_cards": [],
        "episode_seeds": [],
        "metadata": {
            "created_at": created_at,
            "updated_at": created_at,
            "session_count": 0,
            "session_ids": [],
        },
    }


def preference_sections() -> tuple[str, ...]:
    return _PREFERENCE_SECTIONS


def normalize_base_profile(raw: Any) -> dict[str, Any]:
    """Coerce an official-format ``user_profile`` into the five base fields.

    Unknown keys are dropped; missing keys fall back to defaults. This keeps the
    initial-profile merge safe against slightly different dataset rows.
    """

    result = base_profile_defaults()
    if not isinstance(raw, dict):
        return result
    if isinstance(raw.get("purchase_frequency"), str):
        result["purchase_frequency"] = raw["purchase_frequency"]
    rating = raw.get("average_prior_rating")
    if isinstance(rating, (int, float)) and not isinstance(rating, bool):
        result["average_prior_rating"] = float(rating)
    if isinstance(raw.get("rating_style"), str):
        result["rating_style"] = raw["rating_style"]
    tags = raw.get("preference_tags")
    if isinstance(tags, (list, tuple)):
        result["preference_tags"] = [t for t in tags if isinstance(t, str) and t.strip()]
    if isinstance(raw.get("summary"), str):
        result["summary"] = raw["summary"]
    return result
