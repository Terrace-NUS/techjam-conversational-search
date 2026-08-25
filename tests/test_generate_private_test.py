from __future__ import annotations

import unittest

from scripts.generate_private_test import _capacity_constrained_counts, _sample_stratum_counts


class GeneratePrivateTestTest(unittest.TestCase):
    def test_stratum_quotas_sum_exactly(self) -> None:
        quotas = _sample_stratum_counts([1, 9, 60], 800)
        self.assertEqual(quotas, [11, 103, 686])
        self.assertEqual(sum(quotas), 800)

    def test_capacity_constraints_redistribute_unavailable_stratum(self) -> None:
        quotas = _capacity_constrained_counts([1, 9, 60], [20, 200, 85], 200)
        self.assertEqual(sum(quotas), 200)
        self.assertLessEqual(quotas[2], 85)


if __name__ == "__main__":
    unittest.main()
