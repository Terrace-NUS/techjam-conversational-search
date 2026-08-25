from __future__ import annotations

import unittest

import torch

from rollout.gpu_simulator import ASK_SPECIFIC, RolloutObservation
from rollout.ppo import (
    BPETokenizer,
    DenseRewardConfig,
    DualEncoderActorCritic,
    dense_reward,
    fixed_minibatches,
    multi_positive_contrastive_loss,
    ordered_log_prob_and_entropy,
    target_rank_potential,
)


class PPOTest(unittest.TestCase):
    def test_minibatches_are_fixed_size_and_padding_is_masked(self) -> None:
        minibatches = fixed_minibatches(70, 64, "cpu")
        self.assertEqual([len(indices) for indices, _ in minibatches], [64, 64])
        self.assertEqual(sum(int(valid.sum()) for _, valid in minibatches), 70)
        self.assertTrue(all(int(indices.max()) < 70 for indices, _ in minibatches))

    def test_byte_bpe_is_bounded_and_serializable(self) -> None:
        tokenizer = BPETokenizer.fit(["red shoe", "blue shoe"], max_vocab=300)
        token_ids, mask = tokenizer.encode_batch(["新款 shoe"], 16, "cpu")
        restored = BPETokenizer.from_str(tokenizer.to_str())
        restored_ids, restored_mask = restored.encode_batch(["新款 shoe"], 16, "cpu")
        self.assertLessEqual(len(tokenizer), 300)
        self.assertNotIn(BPETokenizer.UNK, token_ids[0, mask[0]].tolist())
        self.assertTrue(torch.equal(token_ids, restored_ids))
        self.assertTrue(torch.equal(mask, restored_mask))

    def test_ordered_action_probability_is_finite(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        selected = torch.tensor([[0, 2]])
        log_probability, entropy = ordered_log_prob_and_entropy(logits, selected)
        self.assertTrue(torch.isfinite(log_probability).all())
        self.assertGreater(float(entropy[0]), 0.0)

    def test_gru_hidden_state_changes_the_next_turn_encoding(self) -> None:
        tokenizer = BPETokenizer.fit(
            ["first preference", "second preference"], max_vocab=300
        )
        model = DualEncoderActorCritic(len(tokenizer), hidden_size=8)
        first_ids, first_mask = tokenizer.encode_batch(["first preference"], 4, "cpu")
        second_ids, second_mask = tokenizer.encode_batch(["second preference"], 4, "cpu")
        _, hidden = model.encode_queries(first_ids, first_mask)
        stateful, _ = model.encode_queries(second_ids, second_mask, hidden)
        stateless, _ = model.encode_queries(second_ids, second_mask)
        self.assertFalse(torch.allclose(stateful, stateless))

    def test_contrastive_loss_uses_one_target_column_per_session(self) -> None:
        logits = torch.tensor([
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        target_ids = torch.tensor([10, 20])
        loss = multi_positive_contrastive_loss(
            logits,
            target_ids.repeat(2),
            target_ids,
            torch.ones(4, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
        )
        expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
        self.assertTrue(torch.allclose(loss, expected))

    def test_contrastive_loss_treats_duplicate_targets_as_positives(self) -> None:
        loss = multi_positive_contrastive_loss(
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([10]),
            torch.tensor([10, 10]),
            torch.tensor([True]),
            torch.tensor([True, True]),
        )
        self.assertAlmostEqual(float(loss), 0.0)

    def test_contrastive_loss_ignores_padded_sessions(self) -> None:
        loss = multi_positive_contrastive_loss(
            torch.tensor([[1.0, 100.0], [0.0, 0.0]]),
            torch.tensor([10, 20]),
            torch.tensor([10, 20]),
            torch.tensor([True, False]),
            torch.tensor([True, False]),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), 0.0)

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
