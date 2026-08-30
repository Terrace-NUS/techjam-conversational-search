"""Validate a structured profile patch and merge it into a profile document.

This module is the trust boundary. The LLM only ever proposes a *patch* — it
never returns a whole profile. Every entry is validated here; anything invalid
is dropped and reported as a warning rather than raising, so a bad model
response can never corrupt the stored profile. Merging always starts from the
current document and returns a new one plus the list of changed field paths.

Merge principles (mirrors the product rules):
* occupation is sensitive identity — updated ONLY from an explicit statement;
* inferred / repeated-behavior signals get their confidence capped down;
* a purchase or preference "for someone else" goes to that recipient card and
  never touches the user's own profile;
* an explicit preference on the same subject overwrites the older one
  (correction), while keeping a merged evidence trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import normalize_base_profile, preference_sections

VALID_SOURCES = ("explicit", "repeated_behavior", "inferred")
VALID_POLARITIES = ("positive", "negative")
_INFERRED_CONFIDENCE_CAP = 0.6

# Key that identifies "the same preference" within a section, for correction.
_SECTION_KEYS: dict[str, str] = {
    "categories": "value",
    "brands": "value",
    "attributes": "subject",
    "price": "subject",
    "constraints": "subject",
    "negative_preferences": "value",
}


@dataclass(slots=True)
class MergeOutcome:
    profile: dict[str, Any]
    changed_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _clamp(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(0.0, min(1.0, float(value)))


def _cap_confidence(source: str, confidence: float) -> float:
    if source != "explicit":
        return min(confidence, _INFERRED_CONFIDENCE_CAP)
    return confidence


def _norm_source(raw: Any, warnings: list[str], where: str) -> str | None:
    if isinstance(raw, str) and raw in VALID_SOURCES:
        return raw
    warnings.append(f"{where}: unknown source {raw!r}")
    return None


def _str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def merge_patch(profile: dict[str, Any], patch: Any) -> MergeOutcome:
    """Return a new profile with the valid parts of ``patch`` applied."""

    import copy

    result = copy.deepcopy(profile)
    outcome = MergeOutcome(profile=result)
    if not isinstance(patch, dict):
        outcome.warnings.append("patch: not an object; ignored")
        return outcome

    _merge_base_profile(result, patch.get("base_profile"), outcome)
    _merge_occupation(result, patch.get("occupation"), outcome)
    _merge_interests(result, patch.get("interests"), outcome)
    _merge_preferences(result, patch.get("preferences"), outcome)
    _merge_recipient_cards(result, patch.get("recipient_cards"), outcome)
    _merge_purchase_anchors(result, patch.get("purchase_anchors"), outcome)
    _merge_episode_seeds(result, patch.get("episode_seeds"), outcome)
    _apply_deletions(result, patch.get("deletions"), outcome)
    return outcome


def _merge_base_profile(profile: dict[str, Any], raw: Any, outcome: MergeOutcome) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        outcome.warnings.append("base_profile: not an object; ignored")
        return
    base = profile["base_profile"]
    incoming = normalize_base_profile({**base, **raw})
    # preference_tags are unioned rather than replaced, preserving history.
    tags = list(dict.fromkeys([*base.get("preference_tags", []), *incoming["preference_tags"]]))
    incoming["preference_tags"] = tags
    for key, value in incoming.items():
        if base.get(key) != value:
            base[key] = value
            outcome.changed_fields.append(f"base_profile.{key}")


def _merge_occupation(profile: dict[str, Any], raw: Any, outcome: MergeOutcome) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        outcome.warnings.append("occupation: not an object; ignored")
        return
    value = _str(raw.get("value"))
    source = _norm_source(raw.get("source"), outcome.warnings, "occupation")
    if value is None or source is None:
        outcome.warnings.append("occupation: missing value/source; ignored")
        return
    if source != "explicit":
        # Sensitive identity: never inferred from behaviour or a single buy.
        outcome.warnings.append("occupation: only explicit statements update occupation; ignored")
        return
    entry = {
        "value": value,
        "confidence": _clamp(raw.get("confidence"), 0.9),
        "source": "explicit",
        "evidence": _str(raw.get("evidence")) or "",
    }
    if profile["personal_context"].get("occupation") != entry:
        profile["personal_context"]["occupation"] = entry
        outcome.changed_fields.append("personal_context.occupation")


def _merge_interests(profile: dict[str, Any], raw: Any, outcome: MergeOutcome) -> None:
    if raw is None:
        return
    if not isinstance(raw, list):
        outcome.warnings.append("interests: not a list; ignored")
        return
    interests: list[dict[str, Any]] = profile["personal_context"]["interests"]
    index = {i["name"]: i for i in interests if isinstance(i, dict) and "name" in i}
    for item in raw:
        if not isinstance(item, dict):
            outcome.warnings.append("interests[]: not an object; skipped")
            continue
        name = _str(item.get("name"))
        source = _norm_source(item.get("source"), outcome.warnings, "interests[]")
        if name is None or source is None:
            outcome.warnings.append("interests[]: missing name/source; skipped")
            continue
        strength = _clamp(item.get("strength"), 0.5)
        confidence = _cap_confidence(source, _clamp(item.get("confidence"), 0.5))
        evidence = _str(item.get("evidence")) or ""
        if name in index:
            existing = index[name]
            existing["observations"] = int(existing.get("observations", 1)) + 1
            existing["strength"] = max(_clamp(existing.get("strength"), 0.0), strength)
            existing["confidence"] = max(_clamp(existing.get("confidence"), 0.0), confidence)
            if evidence:
                existing["evidence"] = evidence
            existing["source"] = source if source == "explicit" else existing.get("source", source)
            outcome.changed_fields.append(f"personal_context.interests[{name}]")
        else:
            index[name] = {
                "name": name,
                "strength": strength,
                "confidence": confidence,
                "source": source,
                "evidence": evidence,
                "observations": 1,
            }
            interests.append(index[name])
            outcome.changed_fields.append(f"personal_context.interests[{name}]")


def _preference_entry(
    section: str, item: dict[str, Any], outcome: MergeOutcome
) -> dict[str, Any] | None:
    source = _norm_source(item.get("source"), outcome.warnings, f"preferences.{section}[]")
    if source is None:
        return None
    value = _str(item.get("value"))
    subject = _str(item.get("subject")) or value
    if value is None and subject is None:
        outcome.warnings.append(f"preferences.{section}[]: missing value/subject; skipped")
        return None
    polarity = item.get("polarity", "positive")
    if polarity not in VALID_POLARITIES:
        polarity = "negative" if section == "negative_preferences" else "positive"
    return {
        "subject": subject or value,
        "value": value or subject,
        "polarity": polarity,
        "confidence": _cap_confidence(source, _clamp(item.get("confidence"), 0.5)),
        "source": source,
        "evidence": _str(item.get("evidence")) or "",
    }


def _upsert_preference(bucket: list[dict[str, Any]], section: str, entry: dict[str, Any]) -> bool:
    key_field = _SECTION_KEYS[section]
    key = entry.get(key_field)
    for i, existing in enumerate(bucket):
        if existing.get(key_field) == key:
            if existing == entry:
                return False
            bucket[i] = entry
            return True
    bucket.append(entry)
    return True


def _merge_preferences(profile: dict[str, Any], raw: Any, outcome: MergeOutcome) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        outcome.warnings.append("preferences: not an object; ignored")
        return
    prefs = profile["shopping_preferences"]
    for section in preference_sections():
        items = raw.get(section)
        if items is None:
            continue
        if not isinstance(items, list):
            outcome.warnings.append(f"preferences.{section}: not a list; ignored")
            continue
        for item in items:
            if not isinstance(item, dict):
                outcome.warnings.append(f"preferences.{section}[]: not an object; skipped")
                continue
            entry = _preference_entry(section, item, outcome)
            if entry is None:
                continue
            if _upsert_preference(prefs[section], section, entry):
                outcome.changed_fields.append(f"shopping_preferences.{section}")


def _merge_recipient_cards(profile: dict[str, Any], raw: Any, outcome: MergeOutcome) -> None:
    if raw is None:
        return
    if not isinstance(raw, list):
        outcome.warnings.append("recipient_cards: not a list; ignored")
        return
    cards: list[dict[str, Any]] = profile["recipient_cards"]
    index = {c["recipient_id"]: c for c in cards if isinstance(c, dict) and "recipient_id" in c}
    for item in raw:
        if not isinstance(item, dict):
            outcome.warnings.append("recipient_cards[]: not an object; skipped")
            continue
        recipient_id = _str(item.get("recipient_id"))
        if recipient_id is None:
            outcome.warnings.append("recipient_cards[]: missing recipient_id; skipped")
            continue
        card = index.get(recipient_id)
        if card is None:
            card = {
                "recipient_id": recipient_id,
                "relationship": _str(item.get("relationship")) or recipient_id,
                "display_name": _str(item.get("display_name")) or recipient_id,
                "preferences": [],
                "purchases": [],
            }
            index[recipient_id] = card
            cards.append(card)
            outcome.changed_fields.append(f"recipient_cards[{recipient_id}]")
        else:
            # relationship/display_name may be corrected; recipient_id is immutable.
            for attr in ("relationship", "display_name"):
                new_val = _str(item.get(attr))
                if new_val and card.get(attr) != new_val:
                    card[attr] = new_val
                    outcome.changed_fields.append(f"recipient_cards[{recipient_id}].{attr}")
        _merge_card_preferences(card, recipient_id, item.get("preferences"), outcome)
        _merge_card_purchases(card, recipient_id, item.get("purchases"), outcome)


def _merge_card_preferences(
    card: dict[str, Any], recipient_id: str, raw: Any, outcome: MergeOutcome
) -> None:
    if raw is None:
        return
    if not isinstance(raw, list):
        outcome.warnings.append(f"recipient_cards[{recipient_id}].preferences: not a list; ignored")
        return
    for item in raw:
        if not isinstance(item, dict):
            continue
        source = _norm_source(item.get("source"), outcome.warnings, "recipient preference")
        value = _str(item.get("value"))
        if source is None or value is None:
            continue
        entry = {
            "value": value,
            "category": _str(item.get("category")),
            "confidence": _cap_confidence(source, _clamp(item.get("confidence"), 0.5)),
            "source": source,
            "evidence": _str(item.get("evidence")) or "",
        }
        bucket = card.setdefault("preferences", [])
        if not any(e.get("value") == value for e in bucket):
            bucket.append(entry)
            outcome.changed_fields.append(f"recipient_cards[{recipient_id}].preferences")


def _merge_card_purchases(
    card: dict[str, Any], recipient_id: str, raw: Any, outcome: MergeOutcome
) -> None:
    if raw is None:
        return
    if not isinstance(raw, list):
        return
    for item in raw:
        if not isinstance(item, dict):
            continue
        product_id = _str(item.get("product_id"))
        if product_id is None:
            continue
        bucket = card.setdefault("purchases", [])
        if not any(p.get("product_id") == product_id for p in bucket):
            bucket.append(
                {
                    "product_id": product_id,
                    "category_path": _category_path(item.get("category_path")),
                    "evidence": _str(item.get("evidence")) or "",
                }
            )
            outcome.changed_fields.append(f"recipient_cards[{recipient_id}].purchases")


def _category_path(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [seg for seg in raw if isinstance(seg, str) and seg.strip()]
    if isinstance(raw, str) and raw.strip():
        return [seg for seg in raw.split(".") if seg]
    return []


def _merge_purchase_anchors(profile: dict[str, Any], raw: Any, outcome: MergeOutcome) -> None:
    if raw is None:
        return
    if not isinstance(raw, list):
        outcome.warnings.append("purchase_anchors: not a list; ignored")
        return
    anchors: list[dict[str, Any]] = profile["purchase_anchors"]
    seen = {a.get("product_id") for a in anchors if isinstance(a, dict)}
    for item in raw:
        if not isinstance(item, dict):
            continue
        product_id = _str(item.get("product_id"))
        if product_id is None:
            outcome.warnings.append("purchase_anchors[]: missing product_id; skipped")
            continue
        scope = item.get("recipient_scope", "self")
        if scope not in ("self", "other", "shared", "unknown"):
            scope = "unknown"
        if scope != "self":
            # Purchases for others belong to a recipient card, not the self anchors.
            outcome.warnings.append(
                f"purchase_anchors[{product_id}]: non-self scope routed away from self profile"
            )
            continue
        if product_id in seen:
            continue
        seen.add(product_id)
        anchors.append(
            {
                "product_id": product_id,
                "category_path": _category_path(item.get("category_path")),
                "recipient_scope": "self",
                "outcome": item.get("outcome") if item.get("outcome") in
                ("purchased", "returned", "cancelled") else "purchased",
                "evidence": _str(item.get("evidence")) or "",
            }
        )
        outcome.changed_fields.append(f"purchase_anchors[{product_id}]")


def _merge_episode_seeds(profile: dict[str, Any], raw: Any, outcome: MergeOutcome) -> None:
    if raw is None:
        return
    if not isinstance(raw, list):
        outcome.warnings.append("episode_seeds: not a list; ignored")
        return
    seeds: list[dict[str, Any]] = profile["episode_seeds"]
    seen = {s.get("anchor_product_id") for s in seeds if isinstance(s, dict)}
    for item in raw:
        if not isinstance(item, dict):
            continue
        anchor = _str(item.get("anchor_product_id"))
        if anchor is None or anchor in seen:
            continue
        seen.add(anchor)
        seeds.append(
            {
                "anchor_product_id": anchor,
                "anchor_category_path": _category_path(item.get("anchor_category_path")),
                "route_hints": [
                    r for r in item.get("route_hints", []) if r in ("complete", "extend", "discover")
                ]
                or ["complete", "extend", "discover"],
                "excluded_product_ids": [anchor],
            }
        )
        outcome.changed_fields.append(f"episode_seeds[{anchor}]")


def _apply_deletions(profile: dict[str, Any], raw: Any, outcome: MergeOutcome) -> None:
    """Honor LLM-proposed deletions ONLY for explicit, evidenced corrections.

    A vague filler turn ("my preferences have been revised") must not be able to
    wipe a durable signal — especially one carried over from an earlier session.
    So an LLM deletion is accepted only when it is an object that names the exact
    ``field_path`` AND carries ``source == "explicit"`` with a non-empty
    ``evidence`` quote. Bare-string paths and non-explicit deletions are dropped
    with a warning; they never mutate the stored profile. The caller-driven
    ``MemoryService.delete_profile_field`` bypasses this guard on purpose — that
    is a human/host action, not the model's.
    """

    if raw is None:
        return
    if not isinstance(raw, list):
        outcome.warnings.append("deletions: not a list; ignored")
        return
    for item in raw:
        if isinstance(item, str):
            outcome.warnings.append(
                f"deletions: bare path {item!r} ignored; LLM deletions must be an "
                "explicit, evidenced correction object"
            )
            continue
        if not isinstance(item, dict):
            outcome.warnings.append("deletions[]: not an object; skipped")
            continue
        path = _str(item.get("field_path"))
        source = item.get("source")
        evidence = _str(item.get("evidence"))
        if path is None:
            outcome.warnings.append("deletions[]: missing field_path; skipped")
            continue
        if source != "explicit" or evidence is None:
            outcome.warnings.append(
                f"deletions[{path!r}]: only explicit, evidenced corrections may "
                "delete durable signals; skipped"
            )
            continue
        if delete_field(profile, path):
            outcome.changed_fields.append(f"-{path}")
        else:
            outcome.warnings.append(f"deletions: path not found {path!r}")


def delete_field(profile: dict[str, Any], field_path: str) -> bool:
    """Delete a value addressed by a dotted path or ``section[key]`` selector.

    Supported forms:
    * ``personal_context.occupation`` — set a scalar section back to null;
    * ``shopping_preferences.categories[running-shoes]`` — remove one entry;
    * ``personal_context.interests[photography]`` — remove one interest;
    * ``recipient_cards[mother]`` — remove a whole card.
    Returns ``True`` if something was removed.
    """

    tokens = _tokenize_path(field_path)
    if not tokens:
        return False
    return _delete_tokens(profile, tokens)


def _tokenize_path(field_path: str) -> list[tuple[str, str | None]]:
    tokens: list[tuple[str, str | None]] = []
    for part in field_path.split("."):
        part = part.strip()
        if not part:
            return []
        if part.endswith("]") and "[" in part:
            name, _, key = part[:-1].partition("[")
            tokens.append((name.strip(), key.strip()))
        else:
            tokens.append((part, None))
    return tokens


def _delete_tokens(node: Any, tokens: list[tuple[str, str | None]]) -> bool:
    name, key = tokens[0]
    if not isinstance(node, dict) or name not in node:
        return False
    if len(tokens) == 1:
        return _delete_leaf(node, name, key)
    return _delete_tokens(node[name], tokens[1:])


def _delete_leaf(node: dict[str, Any], name: str, key: str | None) -> bool:
    target = node[name]
    if key is None:
        if isinstance(target, list):
            if not target:
                return False
            node[name] = []
        elif isinstance(target, dict):
            node[name] = {}
        else:
            node[name] = None
        return True
    if not isinstance(target, list):
        return False
    for i, entry in enumerate(target):
        if isinstance(entry, dict) and key in (
            entry.get("value"),
            entry.get("subject"),
            entry.get("name"),
            entry.get("recipient_id"),
            entry.get("product_id"),
        ):
            target.pop(i)
            return True
    return False
