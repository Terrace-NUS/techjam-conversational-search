from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence
import torch

from evaluator.local_evaluator import (
    behavior_for,
    classify_constraint,
    coarse_category,
    catalog_index,
    intent_candidates,
    load_jsonl,
    materialize_hidden_fields,
)


ATTRIBUTE_NAMES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)
ATTRIBUTE_TO_ID = {name: index for index, name in enumerate(ATTRIBUTE_NAMES)}
NO_ATTRIBUTE = -1
OTHER_ATTRIBUTE = ATTRIBUTE_TO_ID["other"]

SCENARIO_NAMES = ("buying", "browsing", "intent_override", "boundary")
SCENARIO_TO_ID = {name: index for index, name in enumerate(SCENARIO_NAMES)}
BUYING = SCENARIO_TO_ID["buying"]
INTENT_OVERRIDE = SCENARIO_TO_ID["intent_override"]
BOUNDARY = SCENARIO_TO_ID["boundary"]

MESSAGE_KINDS = (
    "initial",
    "preference",
    "no_preference",
    "ask_specific",
    "override",
    "terminal",
)
INITIAL, PREFERENCE, NO_PREFERENCE, ASK_SPECIFIC, OVERRIDE, TERMINAL = range(len(MESSAGE_KINDS))


def rollout_device() -> torch.device:
    _require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "GPU rollouts require PyTorch; install project dependencies with `uv sync`"
        )


@dataclass(frozen=True)
class RolloutBatch:
    target_ids: torch.Tensor
    scenario_ids: torch.Tensor
    override_turns: torch.Tensor
    constraint_attribute_ids: torch.Tensor
    constraint_mask: torch.Tensor
    initial_revealed: torch.Tensor
    overridden_constraint_indices: torch.Tensor
    override_constraint_indices: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.target_ids.shape[0])

    def to(self, device: torch.device | str) -> RolloutBatch:
        return RolloutBatch(**{
            name: value.to(device)
            for name, value in self.__dict__.items()
        })


@dataclass(frozen=True)
class RolloutObservation:
    turn: torch.Tensor
    done: torch.Tensor
    scenario_ids: torch.Tensor
    revealed_constraints: torch.Tensor
    newly_revealed_constraints: torch.Tensor
    message_kind: torch.Tensor
    asked_attribute: torch.Tensor


@dataclass(frozen=True)
class RolloutStep:
    observation: RolloutObservation
    reward: torch.Tensor
    hit_rank: torch.Tensor


@dataclass(frozen=True)
class RolloutDataset:
    batch: RolloutBatch
    catalog_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    initial_messages: tuple[str, ...]
    constraint_texts: tuple[tuple[str, ...], ...]
    override_messages: tuple[str, ...]

    def messages(self, observation: RolloutObservation) -> list[str]:
        """Materialize text only when a text policy needs it; this synchronizes the device."""
        kinds = observation.message_kind.detach().cpu().tolist()
        asks = observation.asked_attribute.detach().cpu().tolist()
        newly_revealed = observation.newly_revealed_constraints.detach().cpu().tolist()
        messages: list[str] = []
        for index, kind in enumerate(kinds):
            if kind == INITIAL:
                message = self.initial_messages[index]
            elif kind == PREFERENCE:
                values = [
                    text for text, revealed in zip(self.constraint_texts[index], newly_revealed[index])
                    if revealed
                ]
                message = "For that, what matters is: " + "; ".join(values) + "."
            elif kind == NO_PREFERENCE:
                attribute = ATTRIBUTE_NAMES[asks[index]] if 0 <= asks[index] < len(ATTRIBUTE_NAMES) else "that"
                message = f"I don't have an additional preference for {attribute}."
            elif kind == ASK_SPECIFIC:
                message = "Those options are not quite right yet. Ask me about one specific attribute."
            elif kind == OVERRIDE:
                message = self.override_messages[index]
            else:
                message = ""
            messages.append(message)
        return messages


class TensorRolloutSimulator:
    """Vectorized first-hit simulator for policy rollout on CUDA, MPS, or CPU."""

    def __init__(
        self,
        batch: RolloutBatch,
        *,
        device: torch.device | str | None = None,
        max_turns: int = 10,
    ) -> None:
        _require_torch()
        requested_device = torch.device(device) if device is not None else rollout_device()
        self.batch = batch.to(requested_device)
        # MPS tensors materialize as ``mps:0`` even when requested as ``mps``.
        self.device = self.batch.target_ids.device
        self.max_turns = max_turns
        self._validate_batch()
        self.reset()

    def _validate_batch(self) -> None:
        batch_size = self.batch.size
        constraint_shape = self.batch.constraint_attribute_ids.shape
        if len(constraint_shape) != 2 or constraint_shape[0] != batch_size:
            raise ValueError("constraint_attribute_ids must have shape [batch, constraints]")
        for name in (
            "constraint_mask",
            "initial_revealed",
        ):
            if getattr(self.batch, name).shape != constraint_shape:
                raise ValueError(f"{name} must match constraint_attribute_ids")
        for name in (
            "scenario_ids",
            "override_turns",
            "overridden_constraint_indices",
            "override_constraint_indices",
        ):
            if getattr(self.batch, name).shape != (batch_size,):
                raise ValueError(f"{name} must have shape [batch]")

    def reset(self) -> RolloutObservation:
        size = self.batch.size
        self.turn = torch.ones(size, dtype=torch.long, device=self.device)
        self.done = torch.zeros(size, dtype=torch.bool, device=self.device)
        self.revealed = self.batch.initial_revealed.clone()
        self.override_applied = self.batch.scenario_ids != INTENT_OVERRIDE
        self.boundary_used = torch.zeros(size, dtype=torch.bool, device=self.device)
        self.first_hit_turn = torch.zeros(size, dtype=torch.long, device=self.device)
        self.best_rank = torch.zeros(size, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(size, dtype=torch.float32, device=self.device)
        return self._observation(
            newly_revealed=self.revealed,
            message_kind=torch.full((size,), INITIAL, dtype=torch.long, device=self.device),
            asked_attribute=torch.full((size,), NO_ATTRIBUTE, dtype=torch.long, device=self.device),
        )

    def step(self, recommendations: torch.Tensor, ask_attribute: torch.Tensor) -> RolloutStep:
        if recommendations.ndim != 2 or recommendations.shape[0] != self.batch.size:
            raise ValueError("recommendations must have shape [batch, top_k]")
        if ask_attribute.shape != (self.batch.size,):
            raise ValueError("ask_attribute must have shape [batch]")
        if recommendations.device != self.device or ask_attribute.device != self.device:
            raise ValueError(f"actions must be on {self.device}")
        no_attribute = ask_attribute.lt(0)
        ask_attribute = torch.where(
            no_attribute,
            torch.full_like(ask_attribute, NO_ATTRIBUTE),
            torch.where(
                ask_attribute.ge(len(ATTRIBUTE_NAMES)),
                torch.full_like(ask_attribute, OTHER_ATTRIBUTE),
                ask_attribute,
            ),
        )

        active = ~self.done
        matches = recommendations.eq(self.batch.target_ids[:, None])
        eligible_hit = active & self.override_applied & matches.any(dim=1)
        rank = matches.to(torch.int32).argmax(dim=1).to(torch.long) + 1
        hit_rank = torch.where(eligible_hit, rank, torch.zeros_like(rank))
        efficiency = (self.max_turns + 1 - self.turn).to(torch.float32) / self.max_turns
        reward = torch.where(
            eligible_hit,
            0.5 + 0.3 / rank.to(torch.float32) + 0.2 * efficiency,
            torch.zeros_like(efficiency),
        )
        self.first_hit_turn = torch.where(eligible_hit, self.turn, self.first_hit_turn)
        self.best_rank = torch.where(eligible_hit, rank, self.best_rank)
        self.episode_return = self.episode_return + reward

        timed_out = active & (self.turn >= self.max_turns)
        self.done = self.done | eligible_hit | timed_out
        continuing = ~self.done
        next_turn = self.turn + continuing.to(torch.long)

        override_now = (
            continuing
            & ~self.override_applied
            & next_turn.eq(self.batch.override_turns)
        )
        self.override_applied = self.override_applied | override_now

        newly_revealed = torch.zeros_like(self.revealed)
        revealed = self.revealed.clone()
        constraint_count = revealed.shape[1]
        if constraint_count:
            row_ids = torch.arange(self.batch.size, device=self.device)
            new_indices = self.batch.override_constraint_indices.clamp(min=0, max=constraint_count - 1)
            add_override = override_now & self.batch.override_constraint_indices.ge(0)
            newly_revealed[row_ids, new_indices] |= add_override

        boundary_now = (
            continuing
            & ~override_now
            & self.batch.scenario_ids.eq(BOUNDARY)
            & ~self.boundary_used
            & ask_attribute.ge(0)
        )
        self.boundary_used = self.boundary_used | boundary_now

        valid_question = continuing & ~override_now & ~boundary_now & ask_attribute.ge(0)
        requested = ask_attribute[:, None]
        attribute_matches = self.batch.constraint_attribute_ids.eq(requested)
        attribute_matches |= requested.eq(OTHER_ATTRIBUTE)
        candidates = self.batch.constraint_mask & ~revealed & attribute_matches & valid_question[:, None]
        first_two = candidates & candidates.to(torch.int32).cumsum(dim=1).le(2)
        newly_revealed |= first_two
        revealed |= newly_revealed
        self.revealed = revealed
        self.turn = next_turn

        disclosed_any = first_two.any(dim=1)
        message_kind = torch.full_like(self.turn, ASK_SPECIFIC)
        message_kind = torch.where(valid_question, torch.full_like(message_kind, NO_PREFERENCE), message_kind)
        message_kind = torch.where(disclosed_any, torch.full_like(message_kind, PREFERENCE), message_kind)
        message_kind = torch.where(boundary_now, torch.full_like(message_kind, NO_PREFERENCE), message_kind)
        message_kind = torch.where(override_now, torch.full_like(message_kind, OVERRIDE), message_kind)
        message_kind = torch.where(self.done, torch.full_like(message_kind, TERMINAL), message_kind)

        observation = self._observation(newly_revealed, message_kind, ask_attribute)
        return RolloutStep(observation=observation, reward=reward, hit_rank=hit_rank)

    def _observation(
        self,
        newly_revealed: torch.Tensor,
        message_kind: torch.Tensor,
        asked_attribute: torch.Tensor,
    ) -> RolloutObservation:
        return RolloutObservation(
            turn=self.turn,
            done=self.done,
            scenario_ids=self.batch.scenario_ids,
            revealed_constraints=self.revealed,
            newly_revealed_constraints=newly_revealed,
            message_kind=message_kind,
            asked_attribute=asked_attribute,
        )


def _constraint_index(values: Sequence[str], value: object) -> int:
    text = str(value) if value is not None else ""
    try:
        return values.index(text)
    except ValueError:
        return -1


def random_intent_card(product: dict, rng: random.Random, limit: int = 180) -> dict:
    constraints = intent_candidates(product, limit)
    selected = rng.sample(constraints, k=min(4, len(constraints)))
    return {
        "target_category": str(product.get("title") or "product")[:limit],
        "hard_constraints": selected[:2],
        "soft_preferences": selected[2:4] or selected[:1],
    }


def load_rollout_dataset(
    catalog_path: str | Path = "data/catalog.jsonl",
    dataset_path: str | Path = "data/public_set.jsonl",
    *,
    randomize_cards: bool = False,
    seed: int | None = None,
) -> RolloutDataset:
    """Build tensor episodes, optionally resampling each target's intent card."""
    _require_torch()
    rng = random.Random(seed)
    samples = load_jsonl(dataset_path)
    _, categories, products = catalog_index(catalog_path)
    catalog_ids = tuple(products)
    catalog_id_to_index = {parent_asin: index for index, parent_asin in enumerate(catalog_ids)}

    targets: list[int] = []
    scenarios: list[int] = []
    override_turns: list[int] = []
    constraints_by_sample: list[list[str]] = []
    attributes_by_sample: list[list[int]] = []
    initial_revealed_indices: list[int] = []
    overridden_indices: list[int] = []
    override_indices: list[int] = []
    initial_messages: list[str] = []
    override_messages: list[str] = []

    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        if randomize_cards:
            card = random_intent_card(products[target], rng)
            behavior = behavior_for(str(sample["scenario_type"]), card, rng)
        else:
            card, behavior = materialize_hidden_fields(sample, products)
        hard = [str(value) for value in card.get("hard_constraints", [])]
        soft = [str(value) for value in card.get("soft_preferences", [])]
        constraints = list(dict.fromkeys([*hard, *soft]))
        scenario = str(sample["scenario_type"])
        override = behavior.get("override") or {}

        targets.append(catalog_id_to_index[target])
        scenarios.append(SCENARIO_TO_ID[scenario])
        constraints_by_sample.append(constraints)
        attributes_by_sample.append([ATTRIBUTE_TO_ID[classify_constraint(value)] for value in constraints])
        override_turns.append(int(override.get("turn", 0)))
        overridden_indices.append(_constraint_index(constraints, override.get("old_value")))
        override_indices.append(_constraint_index(constraints, override.get("new_value")))
        if scenario == "buying" and hard:
            initial_revealed_indices.append(_constraint_index(constraints, hard[0]))
        else:
            initial_revealed_indices.append(-1)

        category = coarse_category(categories.get(target, []))
        if scenario == "buying" and hard:
            initial_messages.append(f"I'm looking for {category}. A key requirement is: {hard[0]}.")
        elif scenario == "intent_override":
            initial_messages.append(f"I'm looking for {category}. {override.get('old_value', '')}")
        else:
            initial_messages.append(f"I'm looking for {category}, but I'm still exploring.")
        override_messages.append(str(override.get("message", "")))

    batch_size = len(samples)
    max_constraints = max((len(values) for values in constraints_by_sample), default=0)
    attribute_ids = torch.full((batch_size, max_constraints), NO_ATTRIBUTE, dtype=torch.long)
    constraint_mask = torch.zeros((batch_size, max_constraints), dtype=torch.bool)
    initial_revealed = torch.zeros_like(constraint_mask)
    for row, attributes in enumerate(attributes_by_sample):
        count = len(attributes)
        if count:
            attribute_ids[row, :count] = torch.tensor(attributes, dtype=torch.long)
            constraint_mask[row, :count] = True
        if initial_revealed_indices[row] >= 0:
            initial_revealed[row, initial_revealed_indices[row]] = True

    return RolloutDataset(
        batch=RolloutBatch(
            target_ids=torch.tensor(targets, dtype=torch.long),
            scenario_ids=torch.tensor(scenarios, dtype=torch.long),
            override_turns=torch.tensor(override_turns, dtype=torch.long),
            constraint_attribute_ids=attribute_ids,
            constraint_mask=constraint_mask,
            initial_revealed=initial_revealed,
            overridden_constraint_indices=torch.tensor(overridden_indices, dtype=torch.long),
            override_constraint_indices=torch.tensor(override_indices, dtype=torch.long),
        ),
        catalog_ids=catalog_ids,
        sample_ids=tuple(str(sample["sample_id"]) for sample in samples),
        initial_messages=tuple(initial_messages),
        constraint_texts=tuple(tuple(values) for values in constraints_by_sample),
        override_messages=tuple(override_messages),
    )
