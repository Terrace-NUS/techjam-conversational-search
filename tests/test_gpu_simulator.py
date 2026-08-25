from __future__ import annotations

import random
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - dependency is optional locally
    torch = None

from rollout.gpu_simulator import (
    BOUNDARY,
    INTENT_OVERRIDE,
    RolloutBatch,
    TensorRolloutSimulator,
    random_intent_card,
)


@unittest.skipUnless(torch is not None, "run `uv sync` before running PyTorch tests")
class TensorRolloutSimulatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = RolloutBatch(
            target_ids=torch.tensor([2, 3, 4]),
            scenario_ids=torch.tensor([0, INTENT_OVERRIDE, BOUNDARY]),
            override_turns=torch.tensor([0, 3, 0]),
            constraint_attribute_ids=torch.tensor([[1, 2], [2, 1], [1, 0]]),
            constraint_mask=torch.ones((3, 2), dtype=torch.bool),
            initial_revealed=torch.tensor([[True, False], [False, False], [False, False]]),
            overridden_constraint_indices=torch.tensor([-1, 1, -1]),
            override_constraint_indices=torch.tensor([-1, 0, -1]),
        )
        self.simulator = TensorRolloutSimulator(self.batch, device="cpu")

    def test_batch_hit_reward_and_rank(self) -> None:
        self.simulator.reset()
        step = self.simulator.step(
            torch.tensor([[9, 2], [9, 9], [9, 9]]),
            torch.tensor([-1, -1, -1]),
        )
        self.assertEqual(step.hit_rank.tolist(), [2, 0, 0])
        self.assertAlmostEqual(float(step.reward[0]), 0.5 + 0.3 / 2 + 0.2)
        self.assertEqual(self.simulator.done.tolist(), [True, False, False])

    def test_override_blocks_old_turn_and_reveals_new_constraint(self) -> None:
        self.simulator.reset()
        self.simulator.step(torch.full((3, 2), 9), torch.full((3,), -1))
        self.assertFalse(bool(self.simulator.override_applied[1]))
        step = self.simulator.step(torch.full((3, 2), 9), torch.full((3,), -1))
        self.assertTrue(bool(self.simulator.override_applied[1]))
        self.assertTrue(bool(step.observation.revealed_constraints[1, 0]))

    def test_boundary_uses_one_no_preference_turn(self) -> None:
        self.simulator.reset()
        step = self.simulator.step(
            torch.full((3, 2), 9),
            torch.tensor([-1, -1, 1]),
        )
        self.assertEqual(int(step.observation.message_kind[2]), 2)
        self.assertTrue(bool(self.simulator.boundary_used[2]))

    def test_invalid_attribute_is_mapped_to_other(self) -> None:
        self.simulator.reset()
        step = self.simulator.step(
            torch.full((3, 2), 9),
            torch.tensor([99, -1, -1]),
        )
        self.assertEqual(int(step.observation.asked_attribute[0]), 9)
        self.assertEqual(int(step.observation.message_kind[0]), 1)

    def test_random_card_samples_four_constraints_reproducibly(self) -> None:
        product = {
            "title": "Running shoe",
            "features": [f"feature {index}" for index in range(8)],
        }
        first = random_intent_card(product, random.Random(7))
        repeated = random_intent_card(product, random.Random(7))
        different = random_intent_card(product, random.Random(8))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        self.assertEqual(len(first["hard_constraints"]), 2)
        self.assertEqual(len(first["soft_preferences"]), 2)
        self.assertEqual(len(set(first["hard_constraints"] + first["soft_preferences"])), 4)


if __name__ == "__main__":
    unittest.main()
