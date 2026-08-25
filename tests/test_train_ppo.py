from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.train_ppo import load_config


class TrainConfigTest(unittest.TestCase):
    def test_loads_toml_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text("[run]\nseed = 7\n[ppo]\nclip_ratio = 0.1\n", encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config["run"]["seed"], 7)
        self.assertEqual(config["ppo"]["clip_ratio"], 0.1)


if __name__ == "__main__":
    unittest.main()
