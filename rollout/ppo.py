from __future__ import annotations

import math
from contextlib import nullcontext
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .gpu_simulator import (
    ASK_SPECIFIC,
    NO_PREFERENCE,
    RolloutDataset,
    RolloutObservation,
    TensorRolloutSimulator,
)


class BPETokenizer:
    PAD = 0
    UNK = 1
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, tokenizer: Tokenizer) -> None:
        if tokenizer.token_to_id(self.PAD_TOKEN) != self.PAD:
            raise ValueError("BPE tokenizer must assign [PAD] token ID 0")
        if tokenizer.token_to_id(self.UNK_TOKEN) != self.UNK:
            raise ValueError("BPE tokenizer must assign [UNK] token ID 1")
        self.tokenizer = tokenizer

    @classmethod
    def fit(cls, texts: Iterable[str], max_vocab: int = 20_000) -> BPETokenizer:
        if max_vocab < 258:
            raise ValueError("byte-level BPE vocabulary must contain at least 258 tokens")
        tokenizer = Tokenizer(BPE(unk_token=cls.UNK_TOKEN))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.train_from_iterator(
            texts,
            trainer=BpeTrainer(
                vocab_size=max_vocab,
                special_tokens=[cls.PAD_TOKEN, cls.UNK_TOKEN],
                initial_alphabet=ByteLevel.alphabet(),
                show_progress=False,
            ),
        )
        return cls(tokenizer)

    @classmethod
    def from_str(cls, serialized: str) -> BPETokenizer:
        return cls(Tokenizer.from_str(serialized))

    def to_str(self) -> str:
        return self.tokenizer.to_str()

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode_batch(
        self,
        texts: Iterable[str],
        max_length: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [
            encoding.ids[-max_length:] or [self.UNK]
            for encoding in self.tokenizer.encode_batch(list(texts))
        ]
        token_ids = torch.full((len(rows), max_length), self.PAD, dtype=torch.long)
        mask = torch.zeros((len(rows), max_length), dtype=torch.bool)
        for row_id, row in enumerate(rows):
            length = min(len(row), max_length)
            token_ids[row_id, :length] = torch.tensor(row[-length:], dtype=torch.long)
            mask[row_id, :length] = True
        return token_ids.to(device), mask.to(device)


class DualEncoderActorCritic(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128, attribute_actions: int = 11) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=BPETokenizer.PAD)
        self.query_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.product_projection = nn.Linear(hidden_size, hidden_size)
        self.attribute_head = nn.Linear(hidden_size, attribute_actions)
        self.value_head = nn.Linear(hidden_size, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_queries(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.query_encoder(self.embedding(token_ids), hidden)
        last = mask.sum(dim=1).clamp(min=1) - 1
        features = output[torch.arange(output.shape[0], device=output.device), last]
        return F.normalize(features, dim=-1), features.unsqueeze(0)

    def encode_products(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        pooled = (embedded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return F.normalize(self.product_projection(pooled), dim=-1)

    def policy_value(self, query_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attribute_head(query_features), self.value_head(query_features).squeeze(-1)

    def scores(self, queries: torch.Tensor, products: torch.Tensor) -> torch.Tensor:
        return queries @ products.T * self.logit_scale.exp().clamp(max=100.0)


@torch.no_grad()
def encode_catalog(
    model: DualEncoderActorCritic,
    token_ids: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    return torch.cat([
        model.encode_products(token_ids[start:start + chunk_size], mask[start:start + chunk_size])
        for start in range(0, token_ids.shape[0], chunk_size)
    ])


def ordered_log_prob_and_entropy(
    logits: torch.Tensor,
    selected_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plackett-Luce log probability for an ordered sample without replacement."""
    available = torch.ones_like(logits, dtype=torch.bool)
    log_probability = torch.zeros(logits.shape[0], device=logits.device)
    entropy = torch.zeros_like(log_probability)
    for column in range(selected_positions.shape[1]):
        masked_logits = logits.masked_fill(~available, -torch.inf)
        distribution = torch.distributions.Categorical(logits=masked_logits)
        selected = selected_positions[:, column]
        log_probability += distribution.log_prob(selected)
        entropy += distribution.entropy()
        available.scatter_(1, selected[:, None], False)
    return log_probability, entropy


def sample_without_replacement(logits: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = torch.multinomial(F.softmax(logits, dim=-1), count, replacement=False)
    log_probability, entropy = ordered_log_prob_and_entropy(logits, selected)
    return selected, log_probability, entropy


def target_rank_potential(scores: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    target_scores = scores.gather(1, target_ids[:, None])
    ranks = scores.gt(target_scores).sum(dim=1).to(torch.float32) + 1.0
    denominator = math.log(max(scores.shape[1], 2))
    return 1.0 - ranks.log() / denominator


def fixed_minibatches(
    count: int,
    minibatch_size: int,
    device: torch.device | str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if count <= 0:
        raise ValueError("at least one active transition is required")
    order = torch.randperm(count, device=device)
    padding = (-count) % minibatch_size
    if padding:
        repeated = order.repeat(math.ceil(padding / count))[:padding]
        order = torch.cat((order, repeated))
    valid = torch.arange(order.shape[0], device=device).lt(count)
    return list(zip(order.split(minibatch_size), valid.split(minibatch_size)))


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(values.dtype)
    return (values * weights).sum() / weights.sum()


@dataclass(frozen=True)
class DenseRewardConfig:
    potential_scale: float = 0.4
    revealed_constraint_reward: float = 0.03
    no_information_penalty: float = 0.02
    turn_cost: float = 0.01


def dense_reward(
    official_reward: torch.Tensor,
    current_potential: torch.Tensor,
    next_potential: torch.Tensor,
    next_observation: RolloutObservation,
    active: torch.Tensor,
    config: DenseRewardConfig,
) -> torch.Tensor:
    revealed = next_observation.newly_revealed_constraints.sum(dim=1).to(torch.float32)
    no_information = (
        next_observation.message_kind.eq(NO_PREFERENCE)
        | next_observation.message_kind.eq(ASK_SPECIFIC)
    ).to(torch.float32)
    reward = (
        official_reward
        + config.potential_scale * (next_potential - current_potential)
        + config.revealed_constraint_reward * revealed
        - config.no_information_penalty * no_information
        - config.turn_cost
    )
    return reward * active.to(reward.dtype)


@dataclass
class Trajectory:
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    candidate_ids: torch.Tensor
    selected_positions: torch.Tensor
    attribute_actions: torch.Tensor
    old_log_prob: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    active: torch.Tensor
    target_ids: torch.Tensor


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    contrastive_coefficient: float = 0.1
    learning_rate: float = 3e-4
    epochs: int = 2
    minibatch_size: int = 64
    max_grad_norm: float = 1.0


class PPOTrainer:
    def __init__(
        self,
        model: DualEncoderActorCritic,
        tokenizer: BPETokenizer,
        catalog_token_ids: torch.Tensor,
        catalog_mask: torch.Tensor,
        *,
        device: torch.device | str,
        candidate_count: int = 128,
        top_k: int = 10,
        max_query_length: int = 128,
        max_turns: int = 10,
        precision: str = "bf16",
        reward_config: DenseRewardConfig = DenseRewardConfig(),
        ppo_config: PPOConfig = PPOConfig(),
    ) -> None:
        if candidate_count < top_k:
            raise ValueError("candidate_count must be at least top_k")
        if catalog_token_ids.shape[0] < top_k:
            raise ValueError("catalog must contain at least top_k products")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.catalog_token_ids = catalog_token_ids.to(self.device)
        self.catalog_mask = catalog_mask.to(self.device)
        self.candidate_count = min(candidate_count, catalog_token_ids.shape[0])
        self.top_k = top_k
        self.max_query_length = max_query_length
        self.max_turns = max_turns
        if precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be 'bf16' or 'fp32'")
        self.precision = precision
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else None
        self.reward_config = reward_config
        self.config = ppo_config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ppo_config.learning_rate)

    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    @torch.no_grad()
    def collect(self, dataset: RolloutDataset) -> tuple[Trajectory, dict[str, float]]:
        simulator = TensorRolloutSimulator(dataset.batch, device=self.device, max_turns=self.max_turns)
        observation = simulator.reset()
        messages = list(dataset.initial_messages)
        hidden: torch.Tensor | None = None
        with self._autocast():
            product_embeddings = encode_catalog(self.model, self.catalog_token_ids, self.catalog_mask)
        storage: dict[str, list[torch.Tensor]] = {
            name: [] for name in (
                "input_ids", "input_mask", "candidate_ids", "selected_positions",
                "attribute_actions", "old_log_prob", "old_values", "rewards", "dones", "active",
            )
        }

        for _ in range(simulator.max_turns):
            active = ~observation.done
            input_ids, input_mask = self.tokenizer.encode_batch(
                messages, self.max_query_length, self.device
            )
            with self._autocast():
                queries, next_hidden = self.model.encode_queries(input_ids, input_mask, hidden)
                all_scores = self.model.scores(queries, product_embeddings)
                current_potential = target_rank_potential(all_scores, simulator.batch.target_ids)
                candidate_ids = all_scores.topk(self.candidate_count, dim=1).indices
                candidate_scores = all_scores.gather(1, candidate_ids)
                selected_positions, recommendation_log_prob, _ = sample_without_replacement(
                    candidate_scores, self.top_k
                )
                recommendations = candidate_ids.gather(1, selected_positions)
                attribute_logits, values = self.model.policy_value(queries)
                attribute_distribution = torch.distributions.Categorical(logits=attribute_logits)
                attribute_actions = attribute_distribution.sample()
                old_log_prob = recommendation_log_prob + attribute_distribution.log_prob(attribute_actions)

            step = simulator.step(recommendations, attribute_actions - 1)
            replies = dataset.messages(step.observation)
            next_ids, next_mask = self.tokenizer.encode_batch(
                replies, self.max_query_length, self.device
            )
            with self._autocast():
                encoded_next_queries, _ = self.model.encode_queries(next_ids, next_mask, next_hidden)
                next_queries = torch.where(
                    step.observation.done[:, None], queries, encoded_next_queries
                )
                next_scores = self.model.scores(next_queries, product_embeddings)
                next_potential = target_rank_potential(next_scores, simulator.batch.target_ids)
            rewards = dense_reward(
                step.reward,
                current_potential,
                next_potential,
                step.observation,
                active,
                self.reward_config,
            )

            for name, value in (
                ("input_ids", input_ids),
                ("input_mask", input_mask),
                ("candidate_ids", candidate_ids),
                ("selected_positions", selected_positions),
                ("attribute_actions", attribute_actions),
                ("old_log_prob", old_log_prob),
                ("old_values", values),
                ("rewards", rewards),
                ("dones", step.observation.done),
                ("active", active),
            ):
                storage[name].append(value.detach())
            hidden = next_hidden * (~step.observation.done)[None, :, None].to(next_hidden.dtype)
            messages = replies
            observation = step.observation

        trajectory = Trajectory(
            **{name: torch.stack(values) for name, values in storage.items()},
            target_ids=simulator.batch.target_ids[None].expand(simulator.max_turns, -1),
        )
        metrics = {
            "mean_return": float(simulator.episode_return.mean().cpu()),
            "hit_rate": float(simulator.first_hit_turn.gt(0).to(torch.float32).mean().cpu()),
            "dense_reward": float(trajectory.rewards.sum(dim=0).mean().cpu()),
        }
        return trajectory, metrics

    def _advantages(self, trajectory: Trajectory) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(trajectory.rewards)
        gae = torch.zeros(trajectory.rewards.shape[1], device=self.device)
        next_value = torch.zeros_like(gae)
        for turn in reversed(range(trajectory.rewards.shape[0])):
            nonterminal = ~trajectory.dones[turn]
            delta = (
                trajectory.rewards[turn]
                + self.config.gamma * next_value * nonterminal
                - trajectory.old_values[turn]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
            advantages[turn] = gae
            next_value = trajectory.old_values[turn]
        returns = advantages + trajectory.old_values
        return advantages, returns

    def update(self, trajectory: Trajectory) -> dict[str, float]:
        advantages, returns = self._advantages(trajectory)
        active = trajectory.active
        active_advantages = advantages[active]
        normalized_advantages = torch.zeros_like(advantages)
        normalized_advantages[active] = (
            active_advantages - active_advantages.mean()
        ) / (active_advantages.std(unbiased=False) + 1e-8)
        totals = Counter()
        updates = 0
        turn_count, session_count = active.shape
        sessions_per_minibatch = max(1, self.config.minibatch_size // turn_count)
        padded_sessions = (-session_count) % sessions_per_minibatch
        count = int(active.sum())

        for _ in range(self.config.epochs):
            for indices, valid_sessions in fixed_minibatches(
                session_count, sessions_per_minibatch, self.device
            ):
                with self._autocast():
                    hidden: torch.Tensor | None = None
                    query_steps: list[torch.Tensor] = []
                    for turn in range(turn_count):
                        queries, hidden = self.model.encode_queries(
                            trajectory.input_ids[turn, indices],
                            trajectory.input_mask[turn, indices],
                            hidden,
                        )
                        query_steps.append(queries)
                        continuing = ~trajectory.dones[turn, indices] & valid_sessions
                        hidden = hidden * continuing[None, :, None].to(hidden.dtype)

                    queries = torch.stack(query_steps).flatten(0, 1)
                    valid = (active[:, indices] & valid_sessions[None]).flatten()
                    attribute_logits, values = self.model.policy_value(queries)
                    attribute_distribution = torch.distributions.Categorical(logits=attribute_logits)

                    candidate_ids = trajectory.candidate_ids[:, indices].flatten(0, 1)
                    candidate_tokens = self.catalog_token_ids[candidate_ids]
                    candidate_masks = self.catalog_mask[candidate_ids]
                    shape = candidate_tokens.shape
                    candidate_embeddings = self.model.encode_products(
                        candidate_tokens.flatten(0, 1), candidate_masks.flatten(0, 1)
                    ).view(shape[0], shape[1], -1)
                    candidate_logits = (
                        queries[:, None] * candidate_embeddings
                    ).sum(dim=-1) * self.model.logit_scale.exp().clamp(max=100.0)
                    recommendation_log_prob, recommendation_entropy = ordered_log_prob_and_entropy(
                        candidate_logits,
                        trajectory.selected_positions[:, indices].flatten(0, 1),
                    )
                    new_log_prob = (
                        recommendation_log_prob
                        + attribute_distribution.log_prob(
                            trajectory.attribute_actions[:, indices].flatten()
                        )
                    )
                    entropy = recommendation_entropy + attribute_distribution.entropy()
                    old_log_prob = trajectory.old_log_prob[:, indices].flatten()
                    batch_advantages = normalized_advantages[:, indices].flatten()
                    ratio = (new_log_prob - old_log_prob).exp()
                    unclipped = ratio * batch_advantages
                    clipped = ratio.clamp(
                        1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                    ) * batch_advantages
                    policy_loss = -_masked_mean(torch.minimum(unclipped, clipped), valid)
                    value_loss = _masked_mean(
                        F.mse_loss(values, returns[:, indices].flatten(), reduction="none"), valid
                    )

                    target_ids = trajectory.target_ids[:, indices].flatten()
                    target_embeddings = self.model.encode_products(
                        self.catalog_token_ids[target_ids],
                        self.catalog_mask[target_ids],
                    )
                    contrastive_logits = self.model.scores(queries, target_embeddings)
                    contrastive_logits = contrastive_logits.masked_fill(~valid[None], -torch.inf)
                    contrastive_logits = torch.where(
                        valid[:, None], contrastive_logits, torch.zeros_like(contrastive_logits)
                    )
                    contrastive_loss = _masked_mean(
                        F.cross_entropy(
                            contrastive_logits,
                            torch.arange(queries.shape[0], device=self.device),
                            reduction="none",
                        ),
                        valid,
                    )
                    mean_entropy = _masked_mean(entropy, valid)
                    loss = (
                        policy_loss
                        + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * mean_entropy
                        + self.config.contrastive_coefficient * contrastive_loss
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                for name, value in (
                    ("loss", loss),
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("contrastive_loss", contrastive_loss),
                    ("entropy", mean_entropy),
                    ("gradient_norm", gradient_norm),
                ):
                    totals[name] += float(value.detach().cpu())
                updates += 1

        return {
            **{name: value / updates for name, value in totals.items()},
            "active_transitions": float(count),
            "padding_fraction": padded_sessions / (session_count + padded_sessions),
        }
