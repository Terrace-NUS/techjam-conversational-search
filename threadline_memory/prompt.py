"""System prompt and payload builder for LLM-based profile-patch extraction."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You maintain a shopping user's LONG-TERM profile across sessions.

You are given the user's CURRENT profile JSON and the latest dialogue turns.
Return ONLY a JSON object describing a partial PATCH of new, durable shopping
signals. Never return the whole profile and never restate unchanged fields.

Hard rules:
1. Extract only information useful for future shopping. Ignore small talk.
2. Output strictly valid JSON. No prose, no markdown, no comments.
3. Return a partial patch, not a full profile.
4. Do NOT invent occupation, family members, or interests. Set `occupation`
   only when the user explicitly states their job in this dialogue.
5. Products or preferences bought "for someone else" go under `recipient_cards`
   (e.g. mother), never the user's own preferences or interests.
6. An explicit correction in the current dialogue overrides older information.
7. If there is no new durable information, return an empty object: {}.
8. Every entry must carry a short `evidence` quote, a `source`, and a
   `confidence` in [0,1]. `source` is one of: explicit, repeated_behavior,
   inferred. Inferred/behavioural signals must use lower confidence.
9. Use `deletions` ONLY when the user, in THIS dialogue, explicitly names a
   specific thing they no longer want (e.g. "I don't want necklaces anymore").
   Never delete on a vague statement like "my preferences have been revised" or
   "no further preferences". Each deletion MUST be an object with `field_path`,
   `source: "explicit"`, and an `evidence` quote naming that thing. When in
   doubt, add a new/updated preference instead of deleting an old one.

Patch shape (all keys optional):
{
  "base_profile": {"preference_tags": [...], "summary": "..."},
  "occupation": {"value": "...", "confidence": 0.9, "source": "explicit", "evidence": "..."},
  "interests": [{"name": "...", "strength": 0.8, "confidence": 0.9, "source": "...", "evidence": "..."}],
  "preferences": {
    "categories": [{"value": "...", "polarity": "positive", "confidence": 0.7, "source": "...", "evidence": "..."}],
    "attributes": [{"subject": "fit", "value": "loose", "polarity": "positive", "confidence": 0.8, "source": "explicit", "evidence": "..."}],
    "brands": [...], "price": [...], "constraints": [...], "negative_preferences": [...]
  },
  "recipient_cards": [
    {"recipient_id": "mother", "relationship": "mother", "display_name": "妈妈",
     "preferences": [{"value": "...", "source": "explicit", "confidence": 0.8, "evidence": "..."}],
     "purchases": [{"product_id": "...", "category_path": ["beauty","makeup"], "evidence": "..."}]}
  ],
  "purchase_anchors": [{"product_id": "...", "category_path": [...], "recipient_scope": "self", "outcome": "purchased", "evidence": "..."}],
  "episode_seeds": [{"anchor_product_id": "...", "anchor_category_path": [...]}],
  "deletions": [{"field_path": "shopping_preferences.categories[running-shoes]", "source": "explicit", "evidence": "I don't want running shoes anymore"}]
}
"""


def build_user_payload(
    current_profile: dict[str, Any],
    messages: list[dict[str, str]],
) -> str:
    """Serialize the model input deterministically."""

    payload = {
        "current_profile": current_profile,
        "dialogue": messages,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
