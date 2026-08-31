from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import uuid
from typing import TYPE_CHECKING

from evaluator.reply_model import (
    ReplyModel,
    TemplateReplyModel,
    build_reply_model,
)
from evaluator.simulators import build_simulator
from starter.agent import Agent, build_agent
from scripts.intent_manager import IntentManager
from scripts.structured_text import structured_product_text

if TYPE_CHECKING:
    from scripts.reward_calculator import RewardCalculator


MAX_TURNS = 10
TOP_K = 10
DEFAULT_INTENT_THRESHOLD = 0.5


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_recommendations(payload: object, catalog_ids: set[str]) -> list[str]:
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        parent_asin = str(value).strip()
        if not parent_asin or parent_asin in seen or parent_asin not in catalog_ids:
            continue
        seen.add(parent_asin)
        result.append(parent_asin)
        if len(result) >= TOP_K:
            break
    return result


def catalog_index(catalog_path: str | Path) -> tuple[set[str], dict[str, list[str]], dict[str, dict]]:
    identifiers: set[str] = set()
    categories: dict[str, list[str]] = {}
    products: dict[str, dict] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            identifiers.add(parent_asin)
            categories[parent_asin] = [str(value) for value in product.get("categories") or []]
            products[parent_asin] = product
    return identifiers, categories, products


def metric_summary(sessions: list[dict]) -> dict:
    if not sessions:
        return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None}
    hit_rate = sum(int(item["hit"]) for item in sessions) / len(sessions)
    mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)
    mttc = statistics.fmean(
        item["first_hit_turn"] if item["first_hit_turn"] is not None else MAX_TURNS + 1
        for item in sessions
    )
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
    }


def _evaluation_result(sessions: list[dict], prompt_tokens: int, completion_tokens: int) -> dict:
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0)) if sessions else 0.0
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    result = {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "sessions": sessions,
    }
    if any("override" in session for session in sessions):
        result["override_metrics"] = metric_summary(
            [session for session in sessions if session.get("override")]
        )
    return result


def _agent_response(
    agent: Agent,
    session_id: str,
    user_message: str,
    turn: int,
    catalog_ids: set[str],
    agent_lock: threading.Lock,
) -> tuple[dict, int, int, list[str]]:
    try:
        with agent_lock:
            response = agent.respond(session_id, user_message, turn, TOP_K)
    except Exception:
        response = {"message": "", "ask_attribute": None, "recommendations": []}
    if not isinstance(response, dict) or not isinstance(response.get("message"), str):
        response = {"message": "", "ask_attribute": None, "recommendations": []}
    usage = response.get("usage")
    prompt_tokens = 0
    completion_tokens = 0
    if isinstance(usage, dict):
        if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] >= 0:
            prompt_tokens = usage["prompt_tokens"]
        if isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] >= 0:
            completion_tokens = usage["completion_tokens"]
    return (
        response,
        prompt_tokens,
        completion_tokens,
        normalize_recommendations(response.get("recommendations"), catalog_ids),
    )


def _evaluate_sample(
    agent: Agent,
    sample: dict,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    reply_model: ReplyModel,
    agent_lock: threading.Lock,
    reward_calculator: "RewardCalculator | None" = None,
    intent_threshold: float = DEFAULT_INTENT_THRESHOLD,
) -> tuple[dict, int, int]:
    session_id = f"public_{uuid.uuid4().hex}"
    simulator = build_simulator(sample, categories, products, reply_model, session_id)
    initial_intent = sample.get("intent")
    if initial_intent not in {"buying", "browsing"}:
        initial_intent = "buying" if sample.get("scenario_type") == "buying" else "browsing"
    intent_manager = IntentManager(initial_intent, threshold=intent_threshold)
    with agent_lock:
        agent.reset(session_id, sample["user_profile"])
    user_message = simulator.initial_message()
    prompt_tokens = 0
    completion_tokens = 0
    hit_turn: int | None = None
    best_rank: int | None = None
    for turn in range(1, MAX_TURNS + 1):
        response, prompt_used, completion_used, ranked = _agent_response(
            agent, session_id, user_message, turn, catalog_ids, agent_lock
        )
        prompt_tokens += prompt_used
        completion_tokens += completion_used
        if simulator.ready_for_hit and simulator.target in ranked:
            best_rank = ranked.index(simulator.target) + 1
            hit_turn = turn
            break
        if turn == MAX_TURNS:
            break
        if reward_calculator is not None:
            subscore = reward_calculator.score_turn(ranked, simulator.target, products)
            if intent_manager.update(subscore):
                query_handler = getattr(simulator, "query_handler", None)
                if query_handler is not None:
                    query_handler.set_intent(intent_manager.intent)
        user_message = simulator.next_message(response, turn + 1)
    result = simulator.result(hit_turn, best_rank)
    result["final_intent"] = intent_manager.intent
    return result, prompt_tokens, completion_tokens


def evaluate(
    agent: Agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    reply_model: ReplyModel | None = None,
    checkpoint_path: str | Path | None = None,
    progress: bool = False,
    max_workers: int = 1,
    reward_calculator: "RewardCalculator | None" = None,
    intent_threshold: float = DEFAULT_INTENT_THRESHOLD,
) -> dict:
    reply_model = reply_model or TemplateReplyModel()
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    sessions: list[dict | None] = [None] * len(samples)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_samples = len(samples)
    agent_lock = threading.Lock()
    worker = lambda index_sample: _evaluate_sample(
        agent,
        index_sample[1],
        catalog_ids,
        categories,
        products,
        reply_model,
        agent_lock,
        reward_calculator,
        intent_threshold,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker, (index, sample)): index
            for index, sample in enumerate(samples)
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            sample_index = futures[future]
            session, prompt_used, completion_used = future.result()
            sessions[sample_index] = session
            total_prompt_tokens += prompt_used
            total_completion_tokens += completion_used
            completed_sessions = [item for item in sessions if item is not None]
            partial = _evaluation_result(completed_sessions, total_prompt_tokens, total_completion_tokens)
            partial["completed_sessions"] = completed_count
            partial["total_sessions"] = total_samples
            if checkpoint_path:
                Path(checkpoint_path).write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
            if progress:
                hit_rate = partial["hit_rate_at_10"]
                print(
                    f"\rEvaluated {completed_count}/{total_samples} sessions "
                    f"({completed_count / total_samples:.1%}), HR@10={hit_rate:.3f}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    if progress and total_samples:
        print(file=sys.stderr)
    return _evaluation_result(
        [item for item in sessions if item is not None],
        total_prompt_tokens,
        total_completion_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TechJam public-set local evaluator")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument(
        "--agent",
        choices=("baseline", "v1"),
        default=None,
        help="Agent implementation; defaults to TECHJAM_AGENT or baseline.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Write a resumable partial result after every completed session.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print session progress to stderr.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent sessions (DeepSeek mode benefits from values such as 8).",
    )
    parser.add_argument(
        "--reply-model",
        choices=("template", "deepseek"),
        default=None,
        help="Customer wording model; defaults to TECHJAM_REPLY_MODEL or template.",
    )
    parser.add_argument(
        "--intent-threshold",
        type=float,
        default=DEFAULT_INTENT_THRESHOLD,
        help="Subscore threshold for the Intent Manager's browsing->buying escalation.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("gemini", "siliconflow"),
        default=os.environ.get("EMBEDDING_PROVIDER", "gemini"),
        help="Embedding API provider (defaults to EMBEDDING_PROVIDER or gemini).",
    )
    args = parser.parse_args()
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    from scripts.reward_calculator import (
        GeminiEmbeddingClient,
        RewardCalculator,
        SiliconFlowEmbeddingClient,
    )

    embedding_client = (
        SiliconFlowEmbeddingClient()
        if args.embedding_provider == "siliconflow"
        else GeminiEmbeddingClient()
    )
    reward_calculator = RewardCalculator(embedding_client, text_fn=structured_product_text)
    result = evaluate(
        build_agent(args.agent, args.catalog),
        samples,
        catalog_ids,
        categories,
        products,
        reply_model=build_reply_model(args.reply_model),
        checkpoint_path=args.checkpoint or f"{args.output}.partial",
        progress=args.progress,
        max_workers=args.workers,
        reward_calculator=reward_calculator,
        intent_threshold=args.intent_threshold,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
