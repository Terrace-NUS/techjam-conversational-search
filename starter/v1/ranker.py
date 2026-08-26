from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .catalog import CatalogIndex
from .state import SessionState


@dataclass(slots=True)
class RankedResult:
    pids: list[int]
    strict_match: bool
    matched_constraint_count: int
    candidate_count: int


class ProductRanker:
    def __init__(self, catalog: CatalogIndex) -> None:
        self.catalog = catalog

    def rank(self, state: SessionState) -> RankedResult:
        scores: dict[int, float] = defaultdict(float)
        filters: list[set[int]] = []
        union_candidates: set[int] = set()

        category_postings: tuple[int, ...] = ()
        if state.category:
            category_postings = self.catalog.category_candidates(state.category)
            if category_postings:
                category_set = set(category_postings)
                filters.append(category_set)
                union_candidates.update(category_set)
                for pid in category_postings:
                    scores[pid] += 1.5

        matched_constraints = 0
        for constraint in state.active_constraints:
            postings = self.catalog.postings_for_constraint(constraint.text)
            if not postings:
                continue
            matched_constraints += 1
            posting_set = set(postings)
            filters.append(posting_set)
            union_candidates.update(posting_set)
            weight = 4.0 * self.catalog.idf(len(postings))
            # Long, exact catalog phrases are much more diagnostic than isolated
            # attribute tokens, while IDF still controls generic marketing text.
            length_bonus = min(2.0, len(constraint.normalized) / 60.0)
            for pid in postings:
                scores[pid] += weight + length_bonus
            for position, positional_postings in enumerate(
                self.catalog.positional_phrase_candidates(constraint.text)
            ):
                position_weight = 2.5 * (4 - position)
                for pid in positional_postings:
                    scores[pid] += position_weight

        fts_query = state.query_text or " ".join(state.messages[-2:])
        fts_pids = self.catalog.fts_search(fts_query, limit=750)
        for rank, pid in enumerate(fts_pids, start=1):
            union_candidates.add(pid)
            scores[pid] += 3.0 / math.sqrt(rank)

        strict_match = False
        if filters:
            strict_candidates = set.intersection(*filters)
            if strict_candidates:
                pool = strict_candidates
                strict_match = True
            else:
                # A single parser or normalization mismatch must not permanently
                # remove the target. Fall back to a scored union.
                pool = union_candidates
        elif union_candidates:
            pool = union_candidates
        else:
            pool = set(self.catalog.all_pids)

        if not pool:
            pool = set(fts_pids) or set(self.catalog.all_pids)

        ranked = sorted(
            pool,
            key=lambda pid: (
                -scores.get(pid, 0.0),
                -math.log1p(self.catalog.rating_count_by_pid[pid]),
                self.catalog.asins[pid],
            ),
        )
        return RankedResult(
            pids=ranked,
            strict_match=strict_match,
            matched_constraint_count=matched_constraints,
            candidate_count=len(ranked),
        )
