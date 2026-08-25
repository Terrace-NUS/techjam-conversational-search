from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from rollout.gpu_simulator import BUYING, INTENT_OVERRIDE, SCENARIO_NAMES
from scripts.train_ppo import RolloutShardSampler, load_config


class TrainConfigTest(unittest.TestCase):
    def test_loads_toml_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text("[run]\nseed = 7\n[ppo]\nclip_ratio = 0.1\n", encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config["run"]["seed"], 7)
        self.assertEqual(config["ppo"]["clip_ratio"], 0.1)

    def test_rollout_shards_produce_a_new_randomized_batch_each_iteration(self) -> None:
        catalog_ids = ("A", "B", "C")
        records = [
            (0, BUYING, "Shoes", ("a", "b", "c", "d"), 0),
            (1, INTENT_OVERRIDE, "Shirts", ("e", "f", "g", "h"), 3),
            (2, 1, "Hats", ("i", "j", "k", "l"), 0),
            (0, 3, "Shoes", ("m", "n", "o", "p"), 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = {
                "format": "techjam-rollout-pickle-v1",
                "scenario_names": SCENARIO_NAMES,
                "catalog_ids": catalog_ids,
                "num_samples": len(records),
                "part_count": 2,
            }
            with (path / "metadata.pkl").open("wb") as handle:
                pickle.dump(metadata, handle)
            for part_id, part in enumerate((records[:2], records[2:])):
                with (path / f"part-{part_id:06d}.pkl").open("wb") as handle:
                    pickle.dump(part, handle)

            sampler = RolloutShardSampler(path, seed=7)
            first = sampler.next_batch(2)
            second = sampler.next_batch(2)

        self.assertEqual(first.batch.size, 2)
        self.assertEqual(second.batch.size, 2)
        self.assertEqual(sampler.samples_seen, 4)
        self.assertTrue(set(first.sample_ids).isdisjoint(second.sample_ids))
        self.assertNotEqual(first.constraint_texts, second.constraint_texts)
        for dataset in (first, second):
            self.assertTrue(all(len(set(card)) == 4 for card in dataset.constraint_texts))
            for row, scenario in enumerate(dataset.batch.scenario_ids.tolist()):
                self.assertEqual(bool(dataset.batch.initial_revealed[row, 0]), scenario == BUYING)
                self.assertEqual(
                    int(dataset.batch.override_constraint_indices[row]),
                    0 if scenario == INTENT_OVERRIDE else -1,
                )


if __name__ == "__main__":
    unittest.main()
