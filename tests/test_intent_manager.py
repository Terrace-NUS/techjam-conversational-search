from __future__ import annotations

import unittest

from scripts.intent_manager import IntentManager


class IntentManagerTest(unittest.TestCase):
    def test_rejects_unknown_intent(self) -> None:
        with self.assertRaises(ValueError):
            IntentManager("unknown")

    def test_browsing_escalates_when_subscore_clears_threshold(self) -> None:
        manager = IntentManager("browsing", threshold=0.5)
        self.assertFalse(manager.update(0.49))
        self.assertEqual(manager.intent, "browsing")
        self.assertTrue(manager.update(0.5))
        self.assertEqual(manager.intent, "buying")

    def test_discovery_advances_one_stage_per_update(self) -> None:
        manager = IntentManager("discovery")
        self.assertFalse(manager.update(0.29))
        self.assertTrue(manager.update(0.3))
        self.assertEqual(manager.intent, "browsing")
        self.assertFalse(manager.update(0.49))
        self.assertTrue(manager.update(0.5))
        self.assertEqual(manager.intent, "buying")

    def test_buying_never_escalates_or_reverts(self) -> None:
        manager = IntentManager("browsing", threshold=0.5)
        manager.update(0.9)
        self.assertEqual(manager.intent, "buying")
        # Already buying: further updates, even low ones, are no-ops.
        self.assertFalse(manager.update(0.0))
        self.assertEqual(manager.intent, "buying")

    def test_initial_buying_stays_buying(self) -> None:
        manager = IntentManager("buying", threshold=0.5)
        self.assertFalse(manager.update(0.0))
        self.assertEqual(manager.intent, "buying")


if __name__ == "__main__":
    unittest.main()
