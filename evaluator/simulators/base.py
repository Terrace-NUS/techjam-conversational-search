from __future__ import annotations

from abc import ABC, abstractmethod

from evaluator.reply_model import ReplyModel


def coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


class Simulator(ABC):
    """Version-specific customer simulation behind the shared evaluation loop."""

    def __init__(
        self,
        sample: dict,
        categories: dict[str, list[str]],
        products: dict[str, dict],
        reply_model: ReplyModel,
        session_id: str,
    ) -> None:
        self.sample = sample
        self.categories = categories
        self.products = products
        self.reply_model = reply_model
        self.session_id = session_id
        self.target = str(sample["ground_truth"]["parent_asin"])

    @abstractmethod
    def initial_message(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def ready_for_hit(self) -> bool:
        raise NotImplementedError

    def query_attribute(self, response: dict) -> str | None:
        attribute = response.get("ask_attribute")
        return attribute if isinstance(attribute, str) else None

    @abstractmethod
    def next_message(self, response: dict, next_turn: int) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def scenario_type(self) -> str:
        raise NotImplementedError

    def result_metadata(self) -> dict:
        return {}

    def result(self, hit_turn: int | None, best_rank: int | None) -> dict:
        return {
            "sample_id": self.sample["sample_id"],
            "scenario_type": self.scenario_type,
            **self.result_metadata(),
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        }
