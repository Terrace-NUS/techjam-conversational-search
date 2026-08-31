from __future__ import annotations

from evaluator.reply_model import ReplyModel

from .base import Simulator
from .v1 import V1Simulator
from .v2 import V2Simulator

SIMULATORS: dict[str, type[Simulator]] = {
    "v1": V1Simulator,
    "v2": V2Simulator,
}


def build_simulator(
    sample: dict,
    categories: dict[str, list[str]],
    products: dict[str, dict],
    reply_model: ReplyModel,
    session_id: str,
) -> Simulator:
    version = sample.get("version", "v1")
    simulator_class = SIMULATORS.get(version)
    if simulator_class is None:
        raise ValueError(f"unsupported dataset version: {version!r}")
    return simulator_class(sample, categories, products, reply_model, session_id)


__all__ = ["Simulator", "V1Simulator", "V2Simulator", "build_simulator"]
