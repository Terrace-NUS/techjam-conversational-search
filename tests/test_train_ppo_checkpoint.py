from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from rollout.ppo import DualEncoderActorCritic, PPOTrainer, WordTokenizer
from scripts.train_ppo import save_checkpoint


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_contains_training_state(self) -> None:
        tokenizer = WordTokenizer.fit(["a product"])
        model = DualEncoderActorCritic(len(tokenizer), hidden_size=8)
        trainer = PPOTrainer(
            model,
            tokenizer,
            torch.ones((2, 3), dtype=torch.long),
            torch.ones((2, 3), dtype=torch.bool),
            device="cpu",
            candidate_count=2,
            top_k=2,
        )
        args = Namespace(experiment_name="test", checkpoint_interval=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, model, trainer, tokenizer, ("A", "B"), args, 3)
            checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(checkpoint["iteration"], 3)
        self.assertIn("model", checkpoint)
        self.assertIn("optimizer", checkpoint)


if __name__ == "__main__":
    unittest.main()
