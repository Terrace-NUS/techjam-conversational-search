from __future__ import annotations

import unittest
from collections import Counter

from scripts.generate_rollout_data import generate_records


class GenerateRolloutDataTest(unittest.TestCase):
    def test_generation_is_reproducible_and_randomizes_cards(self) -> None:
        sampler = {
            "entries": ((7, "Shoes", tuple(f"constraint {index}" for index in range(8))),),
            "stratum_entries": ((0,),),
            "stratum_weights": (1,),
        }
        first = generate_records(sampler, 10_000, seed=5)
        repeated = generate_records(sampler, 10_000, seed=5)
        self.assertEqual(first, repeated)
        self.assertGreater(len({record[3] for record in first}), 100)
        scenario_counts = Counter(record[1] for record in first)
        self.assertTrue(3_500 < scenario_counts[0] < 4_500)
        self.assertTrue(3_500 < scenario_counts[1] < 4_500)
        self.assertTrue(1_200 < scenario_counts[2] < 1_800)
        self.assertTrue(300 < scenario_counts[3] < 700)
        self.assertTrue(all(len(record[3]) == 4 for record in first))
        self.assertTrue(all((record[4] in (3, 4)) == (record[1] == 2) for record in first))

if __name__ == "__main__":
    unittest.main()
