from __future__ import annotations

import unittest

import torch

from rollout.gpu_simulator import ASK_SPECIFIC, RolloutObservation
from rollout.ppo import (
    DenseRewardConfig,
    WordTokenizer,
    dense_reward,
    fixed_minibatches,
    ordered_log_prob_and_entropy,
    target_rank_potential,
)


class PPOTest(unittest.TestCase):
    def test_minibatches_are_fixed_size_and_padding_is_masked(self) -> None:
        minibatches = fixed_minibatches(70, 64, "cpu")
        self.assertEqual([len(indices) for indices, _ in minibatches], [64, 64])
        self.assertEqual(sum(int(valid.sum()) for _, valid in minibatches), 70)
        self.assertTrue(all(int(indices.max()) < 70 for indices, _ in minibatches))

    def test_tokenizer_is_bounded_and_handles_unknown_words(self) -> None:
        tokenizer = WordTokenizer.fit(["red shoe", "blue shoe"], max_vocab=4)
        token_ids, mask = tokenizer.encode_batch(["unknown shoe"], 4, "cpu")
        self.assertEqual(token_ids.shape, (1, 4))
        self.assertEqual(int(mask.sum()), 2)
        self.assertIn(WordTokenizer.UNK, token_ids[0].tolist())

    def test_ordered_action_probability_is_finite(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        selected = torch.tensor([[0, 2]])
        log_probability, entropy = ordered_log_prob_and_entropy(logits, selected)
        self.assertTrue(torch.isfinite(log_probability).all())
        self.assertGreater(float(entropy[0]), 0.0)

    def test_rank_potential_and_dense_reward_improve_with_rank(self) -> None:
        current = target_rank_potential(torch.tensor([[0.0, 1.0, 2.0]]), torch.tensor([0]))
        improved = target_rank_potential(torch.tensor([[3.0, 1.0, 2.0]]), torch.tensor([0]))
        observation = RolloutObservation(
            turn=torch.tensor([2]),
            done=torch.tensor([False]),
            scenario_ids=torch.tensor([0]),
            revealed_constraints=torch.tensor([[False]]),
            newly_revealed_constraints=torch.tensor([[False]]),
            message_kind=torch.tensor([ASK_SPECIFIC]),
            asked_attribute=torch.tensor([-1]),
        )
        reward = dense_reward(
            torch.zeros(1), current, improved, observation, torch.tensor([True]), DenseRewardConfig()
        )
        self.assertGreater(float(improved[0]), float(current[0]))
        self.assertGreater(float(reward[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
