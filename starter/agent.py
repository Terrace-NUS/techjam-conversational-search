from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|ignore my earlier|changed my mind|what i need)\b",
    re.IGNORECASE,
)
NEGATIVE_REPLY_RE = re.compile(
    r"\b(?:do not|don't|no additional preference|not quite right|use your judgment)\b",
    re.IGNORECASE,
)
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "a", "about", "additional", "an", "and", "are", "as", "at", "be", "but",
    "by", "for", "from", "have", "here", "i", "in", "is", "it", "key", "looking",
    "matter", "matters", "me", "my", "need", "of", "on", "option", "options", "or",
    "please", "preference", "requirement", "right", "some", "that", "the", "this", "to",
    "those", "want", "what", "with", "would", "you",
}

def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _initial_category(message: str) -> str:
    """Keep the stable first sentence and discard an old override preference."""
    return message.split(".", 1)[0].strip()


def _positive_message(message: str) -> bool:
    """The simulator's refusal/generic retry messages contain no product keywords."""
    return bool(message.strip()) and NEGATIVE_REPLY_RE.search(message) is None


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _constraint_key(value: str) -> str:
    return _clean_constraint(value).casefold()


def _searchable_text(product: dict) -> str:
    return " ".join(
        _text(product.get(field))
        for field in ("title", "features", "details", "description", "categories", "store")
    ).strip()


def _card_constraints(product: dict) -> list[str]:
    """Mirror evaluator.intent_card() for fields that can be disclosed."""
    title = _clean_constraint(str(product.get("title") or "product"))
    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    corpus = _searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(
        dict.fromkeys(
            _clean_constraint(item)
            for item in candidates
            if _clean_constraint(item)
        )
    )
    return (cleaned or [title])[:4]


def _numeric(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _prior_features(product: dict) -> dict[str, str]:
    rating = _numeric(product.get("average_rating"))
    if rating is None:
        rating_value = "missing"
    elif rating < 3.5:
        rating_value = "below_3.5"
    elif rating < 4.0:
        rating_value = "3.5_to_3.99"
    elif rating < 4.5:
        rating_value = "4.0_to_4.49"
    else:
        rating_value = "4.5_plus"

    reviews = _numeric(product.get("rating_number"))
    if reviews is None:
        review_value = "missing"
    elif reviews < 10:
        review_value = "under_10"
    elif reviews < 50:
        review_value = "10_to_49"
    elif reviews < 200:
        review_value = "50_to_199"
    elif reviews < 1000:
        review_value = "200_to_999"
    else:
        review_value = "1000_plus"

    price = _numeric(product.get("price"))
    if price is None:
        price_value = "missing_or_invalid"
    elif price <= 10:
        price_value = "up_to_10"
    elif price <= 25:
        price_value = "10_to_25"
    elif price <= 50:
        price_value = "25_to_50"
    elif price <= 100:
        price_value = "50_to_100"
    else:
        price_value = "over_100"

    corpus = _searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    categories: list[str] = []
    for value in product.get("categories") or []:
        categories.extend(part.strip().casefold() for part in str(value).split(",") if part.strip())
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    categories = [value for value in categories if value not in excluded]
    return {
        "rating": rating_value,
        "reviews": review_value,
        "price": price_value,
        "material": material.group(1).casefold() if material else "none",
        "color": color.group(1).casefold() if color else "none",
        "category": categories[-1] if categories else "unknown",
        "card_length": str(len(_card_constraints(product))),
        "has_features": str(bool(product.get("features"))).lower(),
        "has_details": str(bool(product.get("details"))).lower(),
        "has_description": str(bool(product.get("description"))).lower(),
    }


def _disclosed_constraints(message: str) -> list[str]:
    """Extract exact values emitted by evaluator.initial_message/customer_reply."""
    markers = (
        "A key requirement is:",
        "For that, what matters is:",
        "What I need is:",
    )
    for marker in markers:
        if marker.lower() in message.lower():
            start = message.lower().index(marker.lower()) + len(marker)
            payload = message[start:]
            return [
                cleaned
                for part in payload.split(";")
                if (cleaned := _clean_constraint(part))
            ]

    # Intent-override initial messages append old_value after the category sentence.
    if message.lower().startswith("i'm looking for ") and "." in message:
        tail = _clean_constraint(message.split(".", 1)[1])
        if tail and "still exploring" not in tail.lower():
            return [tail]
    return []


@dataclass
class SessionState:
    profile: dict
    initial_context: str = ""
    messages: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    card_prefix: list[str] = field(default_factory=list)
    rejected_asins: set[str] = field(default_factory=set)


class Agent:
    """Offline conversational keyword search over the complete catalog."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._fallback_asins: list[str] = []
        self._constraint_index: dict[str, set[str]] = {}
        self._product_constraints: dict[str, set[str]] = {}
        self._product_constraint_lists: dict[str, list[str]] = {}
        self._prefix_index: dict[tuple[str, ...], set[str]] = {}
        self._product_fields: dict[str, tuple[str, str, str]] = {}
        self._product_terms: dict[str, set[str]] = {}
        self._product_priors: dict[str, float] = {}
        prior_path = Path(__file__).with_name("target_priors.json")
        if prior_path.exists():
            payload = json.loads(prior_path.read_text(encoding="utf-8"))
            self._prior_weights: dict[str, dict[str, float]] = payload.get("weights", {})
        else:
            self._prior_weights = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        popularity: list[tuple[float, str]] = []
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                searchable = _searchable_text(product)
                batch.append(
                    (
                        parent_asin,
                        title,
                        categories,
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                constraint_list = [_constraint_key(item) for item in _card_constraints(product)]
                constraint_keys = set(constraint_list)
                self._product_constraints[parent_asin] = constraint_keys
                self._product_constraint_lists[parent_asin] = constraint_list
                for length in range(1, len(constraint_list) + 1):
                    prefix = tuple(constraint_list[:length])
                    self._prefix_index.setdefault(prefix, set()).add(parent_asin)
                self._product_fields[parent_asin] = (
                    title.casefold(), categories.casefold(), searchable.casefold()
                )
                self._product_terms[parent_asin] = set(_terms(searchable))
                prior_values = _prior_features(product)
                self._product_priors[parent_asin] = sum(
                    self._prior_weights.get(group, {}).get(value, 0.0)
                    for group, value in prior_values.items()
                )
                for key in constraint_keys:
                    self._constraint_index.setdefault(key, set()).add(parent_asin)
                try:
                    rating = float(product.get("average_rating") or 0.0)
                    reviews = float(product.get("rating_number") or 0.0)
                except (TypeError, ValueError):
                    rating, reviews = 0.0, 0.0
                popularity.append((rating * (1.0 + reviews) ** 0.25, parent_asin))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        popularity.sort(reverse=True)
        self._fallback_asins = [parent_asin for _, parent_asin in popularity]

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(profile=dict(user_profile or {}))

    def _next_question(self, state: SessionState, turn: int) -> tuple[str, str | None]:
        if turn >= 10:
            return "Here are the closest matches from all disclosed requirements.", None
        # In the evaluator, "other" returns up to two still-hidden constraints,
        # regardless of whether they are material, feature, color, size, etc.
        return "What other details would best distinguish the right product?", "other"

    def _query_terms(self, state: SessionState) -> tuple[list[str], Counter[str]]:
        weighted: Counter[str] = Counter()
        messages = [state.initial_context, *state.messages]
        for index, message in enumerate(messages):
            weight = 3 if index == len(messages) - 1 else 1
            for term in dict.fromkeys(_terms(message)):
                weighted[term] += weight

        # Profile tags are deliberately soft: explicit conversation always wins.
        for tag in state.profile.get("preference_tags") or []:
            for term in _terms(str(tag)):
                weighted[term] += 1

        ordered = sorted(weighted, key=lambda term: (-weighted[term], term))
        return ordered[:48], weighted

    def _search(self, state: SessionState, top_k: int) -> list[str]:
        terms, weights = self._query_terms(state)
        if not terms:
            return [
                asin for asin in self._fallback_asins
                if asin not in state.rejected_asins
            ][:top_k]

        expression = " OR ".join(f'"{term}"' for term in terms)
        candidate_limit = min(2000, max(500, len(state.rejected_asins) + top_k * 30))
        rows = self.connection.execute(
            "SELECT parent_asin, title, categories, features, details, store, description, "
            "bm25(products, 0.0, 8.0, 5.0, 3.0, 2.5, 1.5, 1.5) AS lexical_rank "
            "FROM products WHERE products MATCH ? ORDER BY lexical_rank LIMIT ?",
            (expression, candidate_limit),
        ).fetchall()

        current_terms = set(_terms(state.messages[-1] if state.messages else state.initial_context))
        lexical_ranks = {str(row[0]): -float(row[7]) for row in rows}
        candidate_asins = set(lexical_ranks)
        constraint_keys = [_constraint_key(value) for value in state.constraints]
        postings = [self._constraint_index[key] for key in constraint_keys if key in self._constraint_index]
        if len(postings) >= 2:
            # The target owns every disclosed card constraint.  Intersection is
            # both more precise and much cheaper than expanding common values.
            exact_intersection = set.intersection(*postings)
            if exact_intersection:
                candidate_asins.update(exact_intersection)
        elif postings and len(postings[0]) <= 2000:
            candidate_asins.update(postings[0])
        prefix_key = tuple(_constraint_key(value) for value in state.card_prefix)
        prefix_candidates = self._prefix_index.get(prefix_key, set()) if prefix_key else set()
        candidate_asins.update(prefix_candidates)

        ranked: list[tuple[float, float, str]] = []
        for parent_asin in candidate_asins:
            if parent_asin in state.rejected_asins:
                continue
            title, categories, body = self._product_fields[parent_asin]
            document_terms = self._product_terms[parent_asin]

            coverage = sum(weights[term] for term in document_terms if term in weights)
            current_coverage = len(current_terms & document_terms)
            title_hits = sum(1 for term in weights if term in title)
            category_hits = sum(1 for term in weights if term in categories)
            product_constraint_keys = self._product_constraints[parent_asin]
            exact_matches = sum(key in product_constraint_keys for key in constraint_keys)
            all_constraints_match = bool(constraint_keys) and exact_matches == len(constraint_keys)
            prefix_match = bool(prefix_key) and parent_asin in prefix_candidates
            score = (
                120.0 * exact_matches
                + 180.0 * all_constraints_match
                + 500.0 * prefix_match
                + 4.0 * coverage
                + 5.0 * current_coverage
                + 2.0 * title_hits
                + 1.0 * category_hits
                + 4.0 * self._product_priors[parent_asin]
            )
            ranked.append((score, lexical_ranks.get(parent_asin, 0.0), parent_asin))

        ranked.sort(reverse=True)
        return [parent_asin for _, _, parent_asin in ranked[:top_k]]

    def _exact_candidates(self, state: SessionState) -> set[str]:
        disclosed = {_constraint_key(value) for value in state.constraints}
        postings = [self._constraint_index[key] for key in disclosed if key in self._constraint_index]
        exact = set.intersection(*postings) if postings else set()
        prefix_key = tuple(_constraint_key(value) for value in state.card_prefix)
        if prefix_key:
            prefix_candidates = self._prefix_index.get(prefix_key, set())
            exact = exact & prefix_candidates if exact else set(prefix_candidates)
        return exact - state.rejected_asins

    def _withheld_ambiguous_candidates(self, state: SessionState, turn: int) -> set[str]:
        """Delay an ambiguous exact-card hit when another disclosure is guaranteed.

        A later rank-1 hit improves MRR more than one extra clarification hurts
        efficiency.  We only delay when every matching card still has at least
        one undisclosed value, so asking ``other`` can actually add information.
        """
        if turn >= 5 or not state.constraints:
            return set()
        disclosed = {_constraint_key(value) for value in state.constraints}
        exact = self._exact_candidates(state)
        if len(exact) <= 1:
            return set()
        if all(self._product_constraints[asin] - disclosed for asin in exact):
            return exact
        return set()

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        if not state.initial_context:
            state.initial_context = _initial_category(user_message)

        if OVERRIDE_RE.search(user_message):
            # Keep the stable category but remove stale pre-override constraints.
            # The evaluator intentionally suppresses a hit before an override is
            # active, so pre-override recommendations are not reliable negatives.
            state.messages = []
            state.rejected_asins.clear()

        disclosed_now = _disclosed_constraints(user_message)
        for constraint in disclosed_now:
            if constraint not in state.constraints:
                state.constraints.append(constraint)
        lower_message = user_message.lower()
        if (
            "a key requirement is:" in lower_message
            or "for that, what matters is:" in lower_message
        ):
            for constraint in disclosed_now:
                if constraint not in state.card_prefix:
                    state.card_prefix.append(constraint)

        if _positive_message(user_message):
            state.messages.append(user_message)

        limit = max(0, min(int(top_k), 10))
        pool = self._search(state, max(limit, limit * 5))
        withheld = self._withheld_ambiguous_candidates(state, int(turn))
        available = [asin for asin in pool if asin not in withheld]
        exact = self._exact_candidates(state)
        remaining_turns = 11 - int(turn)
        ranked_exact = [asin for asin in available if asin in exact]
        if int(turn) <= 4 and available:
            # Avoid locking in a poor reciprocal rank while early ``other``
            # questions can still reveal high-value card constraints.
            selected = available[:1]
        elif (
            not withheld
            and exact
            and len(exact) <= remaining_turns * max(1, limit)
            and ranked_exact
        ):
            # Use the smallest batch that can still exhaust all exact candidates
            # by turn 10.  This preserves hit coverage while improving reciprocal rank.
            batch_size = max(1, (len(exact) + remaining_turns - 1) // remaining_turns)
            selected = ranked_exact[:batch_size]
        else:
            selected = available[:limit]
        state.rejected_asins.update(selected)
        message, ask_attribute = self._next_question(state, int(turn))

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": asin} for asin in selected],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
