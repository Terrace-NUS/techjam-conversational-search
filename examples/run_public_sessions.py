"""Run independent public-set sessions through DeepSeek simulator + memory."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

_memory = importlib.import_module("threadline_memory")
DeepSeekProfileUpdateClient = _memory.DeepSeekProfileUpdateClient
MemoryService = _memory.MemoryService

QUESTION_TEXT = {
    "brand": "Do you have a preferred brand?",
    "budget": "What budget should I stay within?",
    "category": "What product category do you have in mind?",
    "color": "Do you have a color preference?",
    "feature": "Which features matter most to you?",
    "material": "What material do you prefer?",
    "other": "Are there any other requirements I should know?",
    "size": "What size or fit do you need?",
    "style": "What style do you prefer?",
    "use_case": "What will you mainly use it for?",
}


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def select_samples(
    samples: list[dict[str, Any]],
    limit: int,
    *,
    balanced: bool,
) -> list[dict[str, Any]]:
    if not balanced:
        return samples[:limit]
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[(str(sample.get("intent")), bool(sample.get("override")))].append(sample)
    order = [
        ("buying", False),
        ("buying", True),
        ("browsing", False),
        ("browsing", True),
    ]
    selected: list[dict[str, Any]] = []
    offsets = defaultdict(int)
    while len(selected) < limit:
        added = False
        for key in order:
            offset = offsets[key]
            if offset < len(groups[key]):
                selected.append(groups[key][offset])
                offsets[key] += 1
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return selected


def run_sample(
    sample: dict[str, Any],
    *,
    build_simulator: Any,
    reply_model_type: Any,
    memory_dir: Path,
    simulator_model: str,
    memory_model: str,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    reply_model = reply_model_type(model=simulator_model)
    memory_llm = DeepSeekProfileUpdateClient(model=memory_model)
    simulator = build_simulator(sample, {}, {}, reply_model, sample_id)
    memory = MemoryService.from_json_directory(memory_dir / sample_id, llm=memory_llm)
    started = memory.start_session("user", sample_id, sample["user_profile"])

    dialogue: list[dict[str, str]] = []
    user_message = simulator.initial_message()
    dialogue.append({"role": "user", "content": user_message})

    active_attributes = list(simulator.query_handler.active_attributes)
    for offset, attribute in enumerate(active_attributes, start=2):
        question = QUESTION_TEXT.get(attribute, "Could you tell me more?")
        dialogue.append({"role": "assistant", "content": question})
        user_message = simulator.next_message(
            {"ask_attribute": attribute},
            next_turn=offset,
        )
        dialogue.append({"role": "user", "content": user_message})

    updated = memory.update_from_dialogue("user", sample_id, dialogue)
    return {
        "sample_id": sample_id,
        "intent": sample.get("intent"),
        "override": bool(sample.get("override")),
        "active_attributes": active_attributes,
        "initial_profile": started.profile_prior,
        "dialogue": dialogue,
        "changed_fields": updated.changed_fields,
        "warnings": updated.warnings,
        "final_profile": updated.user_profile,
        "ranking_user_profile": updated.ranking_user_profile,
    }


def write_results(
    path: Path,
    sessions: list[dict[str, Any]],
    *,
    simulator_model: str,
    memory_model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "simulator_model": simulator_model,
                "memory_model": memory_model,
                "sessions": sessions,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--simulator-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--simulator-model", default="deepseek-v4-flash")
    parser.add_argument("--memory-model", default="deepseek-v4-flash")
    args = parser.parse_args()

    sys.path.insert(0, str(args.simulator_root.resolve()))
    from evaluator.reply_model import DeepSeekReplyModel
    from evaluator.simulators import build_simulator

    if args.workers < 1:
        raise ValueError("workers must be positive")
    samples = select_samples(load_samples(args.dataset), args.limit, balanced=args.balanced)
    results: list[dict[str, Any] | None] = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_sample,
                sample,
                build_simulator=build_simulator,
                reply_model_type=DeepSeekReplyModel,
                memory_dir=args.memory_dir,
                simulator_model=args.simulator_model,
                memory_model=args.memory_model,
            ): index
            for index, sample in enumerate(samples)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            finished = [result for result in results if result is not None]
            write_results(
                args.output.with_suffix(args.output.suffix + ".partial"),
                finished,
                simulator_model=args.simulator_model,
                memory_model=args.memory_model,
            )
            print(f"completed {completed}/{len(samples)}", flush=True)

    completed_results = [result for result in results if result is not None]
    write_results(
        args.output,
        completed_results,
        simulator_model=args.simulator_model,
        memory_model=args.memory_model,
    )
    print(f"wrote {len(completed_results)} sessions to {args.output}")


if __name__ == "__main__":
    main()
