from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from evaluator.local_evaluator import (
    DEFAULT_INTENT_THRESHOLD,
    MAX_TURNS,
    TOP_K,
    catalog_index,
    load_jsonl,
    normalize_recommendations,
)
from evaluator.reply_model import ReplyModel, build_reply_model
from evaluator.simulators import build_simulator
from evaluator.simulators.v1 import searchable_text
from starter.agent import Agent, build_agent
from scripts.intent_manager import IntentManager
from scripts.reward_calculator import GeminiEmbeddingClient, RewardCalculator


AskAttribute = Literal[
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
]


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    dataset: str | None = Field(default=None, min_length=1)
    reply_model: Literal["template", "deepseek"] = "template"
    debug: bool = False


class AgentTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    ask_attribute: AskAttribute | None = None
    recommendations: list[str] = Field(default_factory=list, max_length=TOP_K)


class CreateHumanSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    dataset: str | None = Field(default=None, min_length=1)
    agent: Literal["baseline", "v1"] = "v1"


class CreateAutoSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    dataset: str | None = Field(default=None, min_length=1)
    agent: Literal["baseline", "v1"] = "v1"
    reply_model: Literal["template", "deepseek"] = "deepseek"
    debug: bool = False


class HumanReplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)


class RewriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)


SUMMARY_FIELDS = (
    "parent_asin",
    "title",
    "thumb",
    "price",
    "categories",
    "store",
    "average_rating",
    "rating_number",
)
DETAIL_FIELDS = ("features", "description", "details")
SAMPLE_FIELDS = ("sample_id", "scenario_type", "difficulty_bucket", "category_bucket")
ASK_ATTRIBUTE_VALUES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}


class RecordingReplyModel(ReplyModel):
    def __init__(self, reply_model: ReplyModel) -> None:
        self.reply_model = reply_model
        self.original: str | None = None

    def rewrite_initial_message(self, canonical_message: str) -> str:
        self.original = canonical_message
        return self.reply_model.rewrite_initial_message(canonical_message)

    def override_message(self, override: dict) -> str:
        self.original = str(override.get("message", "Actually, please ignore my earlier preference."))
        return self.reply_model.override_message(override)

    def rewrite_query_answer(self, canonical_message: str) -> str:
        self.original = canonical_message
        return self.reply_model.rewrite_query_answer(canonical_message)


class SimulatorService:
    def __init__(self, catalog_path: str | Path, dataset_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog_ids, self.categories, self.products = catalog_index(catalog_path)
        self.thumbs_dir = Path(catalog_path).parent / "thumbs"
        local_thumbs = {
            path.stem for path in self.thumbs_dir.glob("*.jpg") if path.is_file()
        }
        thumbnail_urls = {
            str(row["parent_asin"]): str(row["thumb"])
            for path in sorted((self.thumbs_dir / ".metadata").glob("*.jsonl"))
            for row in load_jsonl(path)
            if row.get("parent_asin") and row.get("thumb")
        }
        for parent_asin, product in self.products.items():
            product["thumb"] = (
                f"/api/thumbs/{parent_asin}.jpg"
                if parent_asin in local_thumbs
                else thumbnail_urls.get(parent_asin)
            )
        default_dataset = Path(dataset_path)
        dataset_paths = {default_dataset, *default_dataset.parent.glob("public_set*.jsonl")}
        self.datasets = {
            path.stem: {
                sample["sample_id"]: sample
                for sample in load_jsonl(path)
            }
            for path in sorted(dataset_paths)
            if path.is_file()
        }
        self.default_dataset = default_dataset.stem
        self.catalog_order = sorted(
            self.products,
            key=lambda parent_asin: (
                -(self._number(self.products[parent_asin].get("rating_number")) or 0),
                -(self._number(self.products[parent_asin].get("average_rating")) or 0),
                str(self.products[parent_asin].get("title") or "").casefold(),
                parent_asin,
            ),
        )
        self.catalog_rank = {
            parent_asin: rank for rank, parent_asin in enumerate(self.catalog_order)
        }
        self.search_documents = {
            parent_asin: [
                (parent_asin.casefold(), 0),
                (str(self.products[parent_asin].get("title") or "").casefold(), 0),
                (
                    str(self.products[parent_asin].get("store") or "").casefold(),
                    20,
                ),
                (
                    " ".join(
                        str(value)
                        for value in self.products[parent_asin].get("features") or []
                        if value not in (None, "")
                    ).casefold(),
                    40,
                ),
            ]
            for parent_asin in self.catalog_order
        }
        self.filter_options = {
            "categories": sorted(
                {category for values in self.categories.values() for category in values},
                key=lambda value: (value.casefold(), value),
            ),
            "stores": sorted(
                {
                    str(product["store"])
                    for product in self.products.values()
                    if product.get("store") not in (None, "")
                },
                key=lambda value: (value.casefold(), value),
            ),
            **{
                field: self._numeric_range(field)
                for field in ("price", "average_rating", "rating_number")
            },
        }
        self.sessions: dict[str, dict] = {}
        self.agents: dict[str, Agent] = {}
        self.reward_calculator: RewardCalculator | None = None
        # ponytail: one lock is enough for a local visualizer; split per session if contention appears.
        self.lock = Lock()

    @staticmethod
    def initial_intent(sample: dict) -> str:
        intent = sample.get("intent")
        if intent in {"browsing", "buying"}:
            return intent
        return "buying" if sample.get("scenario_type") == "buying" else "browsing"

    def turn_metrics(self, session: dict, ranked: list[str], update_intent: bool) -> dict:
        before = session["intent_manager"].intent
        score_error = None
        scores: dict[str, float | None] = {}
        for parent_asin in ranked if session["debug"] else ranked[:1]:
            if parent_asin == session["target"]:
                scores[parent_asin] = 1.0
                continue
            if score_error is not None:
                scores[parent_asin] = None
                continue
            try:
                if self.reward_calculator is None:
                    self.reward_calculator = RewardCalculator(
                        GeminiEmbeddingClient(),
                        text_fn=searchable_text,
                    )
                scores[parent_asin] = self.reward_calculator.score_turn(
                    [parent_asin],
                    session["target"],
                    self.products,
                )
            except Exception as error:
                scores[parent_asin] = None
                score_error = str(error)
        score = 0.0 if not ranked else scores.get(ranked[0])
        changed = score is not None and update_intent and session["intent_manager"].update(score)
        if changed:
            query_handler = getattr(session.get("simulator"), "query_handler", None)
            if query_handler is not None:
                query_handler.set_intent(session["intent_manager"].intent)
        session["score_error"] = score_error
        return {
            "subscore": score,
            "intent_before": before,
            "intent_after": session["intent_manager"].intent,
            "intent_changed": changed,
            "recommendation_scores": scores if session["debug"] else {},
        }

    @staticmethod
    def metrics_view(session: dict) -> dict:
        turns = session["turns"]
        return {
            "current_intent": session["intent_manager"].intent,
            "threshold": session["intent_manager"].threshold,
            "last_subscore": turns[-1]["subscore"] if turns else None,
            "score_error": session["score_error"],
        }

    @staticmethod
    def sample_summary(sample: dict) -> dict:
        summary = {field: sample.get(field) for field in SAMPLE_FIELDS}
        summary["scenario_type"] = sample.get("scenario_type") or sample.get("intent")
        return summary

    @staticmethod
    def product_summary(product: dict) -> dict:
        return {field: product.get(field) for field in SUMMARY_FIELDS}

    @staticmethod
    def product_detail(product: dict) -> dict:
        return {field: product.get(field) for field in (*SUMMARY_FIELDS, *DETAIL_FIELDS)}

    @staticmethod
    def _number(value: object) -> int | float | None:
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def _numeric_range(self, field: str) -> dict[str, int | float | None]:
        values = [
            value
            for product in self.products.values()
            if (value := self._number(product.get(field))) is not None
        ]
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    @staticmethod
    def _subsequence_score(needle: str, haystack: str) -> int | None:
        exact_at = haystack.find(needle)
        if exact_at >= 0:
            return exact_at
        positions: list[int] = []
        position = -1
        for character in needle:
            position = haystack.find(character, position + 1)
            if position < 0:
                return None
            positions.append(position)
        gaps = positions[-1] - positions[0] + 1 - len(needle)
        boundary_bonus = (
            10
            if positions[0] == 0 or not haystack[positions[0] - 1].isalnum()
            else 0
        )
        return 100 + gaps * 10 + positions[0] - boundary_bonus

    @classmethod
    def _search_score(cls, query: str, documents: list[tuple[str, int]]) -> int | None:
        total = 0
        for term in query.split():
            scores = [
                score + field_penalty
                for document, field_penalty in documents
                if (score := cls._subsequence_score(term, document)) is not None
            ]
            if not scores:
                return None
            total += min(scores)
        return total

    def list_datasets(self) -> list[dict]:
        return [
            {
                "id": dataset_id,
                "label": dataset_id.replace("_", " ").title(),
                "sample_count": len(samples),
                "default": dataset_id == self.default_dataset,
            }
            for dataset_id, samples in self.datasets.items()
        ]

    def list_samples(
        self,
        dataset_id: str | None = None,
        scenario: str | None = None,
    ) -> list[dict]:
        dataset_id = dataset_id or self.default_dataset
        samples = self.datasets.get(dataset_id)
        if samples is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        return [
            self.sample_summary(sample)
            for sample in samples.values()
            if scenario is None
            or (sample.get("scenario_type") or sample.get("intent")) == scenario
        ]

    def search_catalog(
        self,
        query: str,
        limit: int,
        offset: int = 0,
        category: str | None = None,
        store: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        min_rating: float | None = None,
        min_rating_count: int | None = None,
    ) -> list[dict]:
        def matches_filters(parent_asin: str) -> bool:
            product = self.products[parent_asin]
            if category is not None and category not in self.categories.get(parent_asin, []):
                return False
            if store is not None and product.get("store") != store:
                return False
            constraints = (
                ("price", min_price, max_price),
                ("average_rating", min_rating, None),
                ("rating_number", min_rating_count, None),
            )
            for field, minimum, maximum in constraints:
                if minimum is None and maximum is None:
                    continue
                value = self._number(product.get(field))
                if (
                    value is None
                    or minimum is not None
                    and value < minimum
                    or maximum is not None
                    and value > maximum
                ):
                    return False
            return True

        has_filters = any(
            value is not None
            for value in (
                category,
                store,
                min_price,
                max_price,
                min_rating,
                min_rating_count,
            )
        )
        choices = (
            [
                parent_asin
                for parent_asin in self.catalog_order
                if matches_filters(parent_asin)
            ]
            if has_filters
            else self.catalog_order
        )
        query = query.strip().casefold()
        if not query:
            return [
                self.product_detail(self.products[parent_asin])
                for parent_asin in choices[offset : offset + limit]
            ]
        matches = []
        for parent_asin in choices:
            score = self._search_score(query, self.search_documents[parent_asin])
            if score is not None:
                matches.append((score, self.catalog_rank[parent_asin], parent_asin))
        return [
            self.product_detail(self.products[parent_asin])
            for _, _, parent_asin in sorted(matches)[offset : offset + limit]
        ]

    def get_product(self, parent_asin: str) -> dict:
        product = self.products.get(parent_asin)
        if product is None:
            raise HTTPException(status_code=404, detail="product not found")
        return self.product_detail(product)

    def create_session(
        self,
        sample_id: str,
        dataset_id: str,
        reply_model_name: str,
        debug: bool = False,
    ) -> dict:
        samples = self.datasets.get(dataset_id)
        if samples is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        sample = samples.get(sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail="sample not found")

        target = str(sample["ground_truth"]["parent_asin"])
        if target not in self.products:
            raise RuntimeError(f"target product {target!r} is missing from the catalog")
        session_id = uuid4().hex
        reply_model = RecordingReplyModel(build_reply_model(reply_model_name))
        simulator = build_simulator(
            sample,
            self.categories,
            self.products,
            reply_model,
            session_id,
        )
        session = {
            "id": session_id,
            "mode": "human_as_agent",
            "status": "initializing",
            "sample": sample,
            "dataset": dataset_id,
            "reply_model": reply_model_name,
            "debug": debug,
            "reply_model_recorder": reply_model,
            "simulator": simulator,
            "target": target,
            "intent_manager": IntentManager(
                self.initial_intent(sample),
                threshold=DEFAULT_INTENT_THRESHOLD,
            ),
            "score_error": None,
            "current_turn": 1,
            "current_user_message": None,
            "current_user_message_original": None,
            "initialization_error": None,
            "turns": [],
            "outcome": None,
        }
        with self.lock:
            self.sessions[session_id] = session
        return self.session_view(session)

    def initialize_session(self, session_id: str) -> dict:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")
            if session["status"] != "initializing":
                return self.session_view(session)
        try:
            message = session["simulator"].initial_message()
        except Exception as error:
            with self.lock:
                session["status"] = "error"
                session["initialization_error"] = str(error)
                return self.session_view(session)
        with self.lock:
            session["current_user_message"] = message
            session["current_user_message_original"] = session["reply_model_recorder"].original
            session["status"] = "waiting_for_agent"
            return self.session_view(session)

    def create_auto_session(
        self,
        sample_id: str,
        dataset_id: str,
        agent_name: str,
        reply_model_name: str,
        debug: bool = False,
    ) -> dict:
        view = self.create_session(sample_id, dataset_id, reply_model_name, debug)
        with self.lock:
            session = self.sessions[view["id"]]
            session["mode"] = "agent_simulator"
            session["agent_name"] = agent_name
            session["agent"] = None
            return self.session_view(session)

    def initialize_auto_session(self, session_id: str) -> dict:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None or session.get("mode") != "agent_simulator":
                raise HTTPException(status_code=404, detail="automatic session not found")
            if session["status"] != "initializing":
                return self.session_view(session)
            agent_name = session["agent_name"]
        try:
            agent = self.agents.get(agent_name) or build_agent(agent_name, self.catalog_path)
            self.agents[agent_name] = agent
            agent.reset(session_id, session["sample"]["user_profile"])
        except Exception as error:
            with self.lock:
                session["status"] = "error"
                session["initialization_error"] = str(error)
                return self.session_view(session)
        with self.lock:
            session["agent"] = agent
        return self.initialize_session(session_id)

    def step_auto_session(self, session_id: str) -> dict:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None or session.get("mode") != "agent_simulator":
                raise HTTPException(status_code=404, detail="automatic session not found")
            if session["status"] != "waiting_for_agent":
                raise HTTPException(status_code=409, detail="session is not ready for a turn")
            agent = session["agent"]
            user_message = session["current_user_message"]
            turn = session["current_turn"]
        response = agent.respond(session_id, user_message, turn, TOP_K)
        response = response if isinstance(response, dict) else {}
        ask_attribute = response.get("ask_attribute")
        recommendations = normalize_recommendations(
            response.get("recommendations"),
            self.catalog_ids,
        )
        return self.submit_turn(
            session_id,
            AgentTurnRequest(
                message=str(response.get("message") or ""),
                ask_attribute=ask_attribute if ask_attribute in ASK_ATTRIBUTE_VALUES else None,
                recommendations=recommendations,
            ),
        )

    def create_human_session(
        self,
        sample_id: str,
        dataset_id: str,
        agent_name: str,
    ) -> dict:
        samples = self.datasets.get(dataset_id)
        if samples is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        sample = samples.get(sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail="sample not found")
        target = str(sample["ground_truth"]["parent_asin"])
        if target not in self.products:
            raise RuntimeError(f"target product {target!r} is missing from the catalog")
        session_id = uuid4().hex
        session = {
            "id": session_id,
            "mode": "human_as_simulator",
            "status": "initializing",
            "sample": sample,
            "dataset": dataset_id,
            "agent_name": agent_name,
            "agent": None,
            "debug": True,
            "target": target,
            "intent_manager": IntentManager(
                self.initial_intent(sample),
                threshold=DEFAULT_INTENT_THRESHOLD,
            ),
            "score_error": None,
            "current_turn": 1,
            "turns": [],
            "outcome": None,
            "initialization_error": None,
        }
        with self.lock:
            self.sessions[session_id] = session
        return self.session_view(session)

    def initialize_human_session(self, session_id: str) -> dict:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None or session.get("mode") != "human_as_simulator":
                raise HTTPException(status_code=404, detail="human simulator session not found")
            if session["status"] != "initializing":
                return self.session_view(session)
            agent_name = session["agent_name"]
        try:
            agent = self.agents.get(agent_name) or build_agent(agent_name, self.catalog_path)
            self.agents[agent_name] = agent
            agent.reset(session_id, session["sample"]["user_profile"])
        except Exception as error:
            with self.lock:
                session["status"] = "error"
                session["initialization_error"] = str(error)
                return self.session_view(session)
        with self.lock:
            session["agent"] = agent
            session["status"] = "waiting_for_simulator"
            return self.session_view(session)

    def submit_human_reply(self, session_id: str, message: str) -> dict:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None or session.get("mode") != "human_as_simulator":
                raise HTTPException(status_code=404, detail="human simulator session not found")
            if session["status"] != "waiting_for_simulator":
                raise HTTPException(status_code=409, detail="session is not waiting for a reply")
            response = session["agent"].respond(session_id, message, session["current_turn"], TOP_K)
            recommendations = normalize_recommendations(
                response.get("recommendations") if isinstance(response, dict) else None,
                self.catalog_ids,
            )
            ask_attribute = response.get("ask_attribute") if isinstance(response, dict) else None
            session["turns"].append(
                {
                    "user_message": message,
                    "user_message_original": None,
                    "agent_message": str(response.get("message") or "") if isinstance(response, dict) else "",
                    "ask_attribute": ask_attribute if ask_attribute in ASK_ATTRIBUTE_VALUES else None,
                    "recommendations": recommendations,
                    "hit_rank": None,
                }
            )
            turn = session["current_turn"]
            target = session["target"]
            rank = recommendations.index(target) + 1 if target in recommendations else None
            metrics = self.turn_metrics(
                session,
                recommendations,
                update_intent=rank is None and turn < MAX_TURNS,
            )
            session["turns"][-1].update(metrics)
            if rank is not None or turn == MAX_TURNS:
                session["turns"][-1]["hit_rank"] = rank
                session["status"] = "hit" if rank is not None else "exhausted"
                session["outcome"] = {
                    "hit": rank is not None,
                    "first_hit_turn": turn if rank is not None else None,
                    "best_rank": rank,
                    "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                }
            else:
                session["current_turn"] = turn + 1
                session["status"] = "waiting_for_simulator"
            return self.session_view(session)

    def submit_turn(self, session_id: str, request: AgentTurnRequest) -> dict:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")
            if session["status"] != "waiting_for_agent":
                raise HTTPException(status_code=409, detail="session has already ended")

            ranked = normalize_recommendations(request.recommendations, self.catalog_ids)
            simulator = session["simulator"]
            target = simulator.target
            hit_rank = (
                ranked.index(target) + 1
                if simulator.ready_for_hit and target in ranked
                else None
            )
            turn = session["current_turn"]
            metrics = self.turn_metrics(
                session,
                ranked,
                update_intent=hit_rank is None and turn < MAX_TURNS,
            )
            session["turns"].append(
                {
                    "user_message": session["current_user_message"],
                    "user_message_original": session["current_user_message_original"],
                    "agent_message": request.message,
                    "ask_attribute": request.ask_attribute,
                    "recommendations": ranked,
                    "hit_rank": hit_rank,
                    **metrics,
                }
            )

            if hit_rank is not None or turn == MAX_TURNS:
                session["status"] = "hit" if hit_rank is not None else "exhausted"
                session["current_user_message"] = None
                session["current_user_message_original"] = None
                session["outcome"] = {
                    "hit": hit_rank is not None,
                    "first_hit_turn": turn if hit_rank is not None else None,
                    "best_rank": hit_rank,
                    "reciprocal_rank": 0.0 if hit_rank is None else 1.0 / hit_rank,
                }
                return self.session_view(session)

            next_message = simulator.next_message(
                {
                    "message": request.message,
                    "ask_attribute": request.ask_attribute,
                    "recommendations": request.recommendations,
                },
                turn + 1,
            )
            session["current_turn"] = turn + 1
            session["current_user_message"] = next_message
            session["current_user_message_original"] = session["reply_model_recorder"].original
            return self.session_view(session)

    def session_view(self, session: dict) -> dict:
        if session.get("mode") == "human_as_simulator":
            return self.human_session_view(session)
        outcome = None
        if session["outcome"] is not None:
            outcome = {
                **session["outcome"],
                "target_product": self.product_summary(self.products[session["target"]]),
            }
        return {
            "id": session["id"],
            "mode": session["mode"],
            "status": session["status"],
            "sample": self.sample_summary(session["sample"]),
            "dataset": session["dataset"],
            "reply_model": session["reply_model"],
            "agent": session.get("agent_name"),
            "debug": session["debug"],
            "initialization_error": session["initialization_error"],
            "debug_target_product": (
                self.product_summary(self.products[session["target"]])
                if session["debug"]
                else None
            ),
            "user_profile": session["sample"]["user_profile"],
            "current_turn": session["current_turn"],
            "current_user_message": session["current_user_message"],
            "current_user_message_original": (
                session["current_user_message_original"]
                if session["debug"] and session["reply_model"] == "deepseek"
                else None
            ),
            "metrics": self.metrics_view(session),
            "turns": [
                {
                    **turn,
                    "user_message_original": (
                        turn["user_message_original"]
                        if session["debug"] and session["reply_model"] == "deepseek"
                        else None
                    ),
                    "recommendations": [
                        self.product_summary(self.products[parent_asin])
                        for parent_asin in turn["recommendations"]
                    ],
                }
                for turn in session["turns"]
            ],
            "outcome": outcome,
        }

    def human_session_view(self, session: dict) -> dict:
        sample = session["sample"]
        intent = sample.get("intent") or sample.get("scenario_type")
        descriptions = sample.get("intent_descriptions") or {}
        intent_description = descriptions.get(intent) if isinstance(descriptions, dict) else None
        outcome = None
        if session["outcome"] is not None:
            outcome = {
                **session["outcome"],
                "target_product": self.product_summary(self.products[session["target"]]),
            }
        return {
            "id": session["id"],
            "mode": session["mode"],
            "status": session["status"],
            "sample": self.sample_summary(sample),
            "dataset": session["dataset"],
            "reply_model": None,
            "agent": session["agent_name"],
            "debug": session["debug"],
            "initialization_error": session["initialization_error"],
            "debug_target_product": self.product_detail(self.products[session["target"]]),
            "human_context": {
                "intent": intent,
                "override": bool(sample.get("override")),
                "intent_description": intent_description,
                "fake_attributes": sample.get("fake_attributes") or {},
                "correction_messages": sample.get("correction_messages") or {},
                "modify_turn": sample.get("modify_turn"),
            },
            "user_profile": sample["user_profile"],
            "current_turn": session["current_turn"],
            "current_user_message": None,
            "current_user_message_original": None,
            "metrics": self.metrics_view(session),
            "turns": [
                {
                    **turn,
                    "recommendations": [
                        self.product_summary(self.products[parent_asin])
                        for parent_asin in turn["recommendations"]
                    ],
                }
                for turn in session["turns"]
            ],
            "outcome": outcome,
        }


def create_app(
    catalog_path: str | Path = "data/catalog.jsonl",
    dataset_path: str | Path = "data/public_set_v2.jsonl",
    service: SimulatorService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.simulator = service or SimulatorService(catalog_path, dataset_path)
        yield

    app = FastAPI(title="TechJam Simulator Visualizer", lifespan=lifespan)
    thumbs_dir = service.thumbs_dir if service is not None else Path(catalog_path).parent / "thumbs"
    if thumbs_dir.is_dir():
        app.mount("/api/thumbs", StaticFiles(directory=thumbs_dir), name="thumbs")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def simulator() -> SimulatorService:
        return app.state.simulator

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/samples")
    def samples(dataset: str | None = None, scenario: str | None = None) -> list[dict]:
        return simulator().list_samples(dataset, scenario)

    @app.get("/api/datasets")
    def datasets() -> list[dict]:
        return simulator().list_datasets()

    @app.get("/api/catalog/search")
    def catalog_search(
        q: str = Query(default="", max_length=200),
        category: str | None = Query(default=None, max_length=200),
        store: str | None = Query(default=None, max_length=500),
        min_price: float | None = Query(default=None, ge=0),
        max_price: float | None = Query(default=None, ge=0),
        min_rating: float | None = Query(default=None, ge=0, le=5),
        min_rating_count: int | None = Query(default=None, ge=0),
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict]:
        return simulator().search_catalog(
            query=q,
            limit=limit,
            offset=offset,
            category=category,
            store=store,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            min_rating_count=min_rating_count,
        )

    @app.get("/api/catalog/filters")
    def catalog_filters() -> dict:
        return simulator().filter_options

    @app.get("/api/catalog/{parent_asin}")
    def catalog_product(parent_asin: str) -> dict:
        return simulator().get_product(parent_asin)

    @app.post("/api/sessions", status_code=201)
    def create_session(request: CreateSessionRequest) -> dict:
        try:
            return simulator().create_session(
                request.sample_id,
                request.dataset or simulator().default_dataset,
                request.reply_model,
                request.debug,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/sessions/{session_id}/initialize")
    def initialize_session(session_id: str) -> dict:
        return simulator().initialize_session(session_id)

    @app.post("/api/auto-sessions", status_code=201)
    def create_auto_session(request: CreateAutoSessionRequest) -> dict:
        try:
            return simulator().create_auto_session(
                request.sample_id,
                request.dataset or simulator().default_dataset,
                request.agent,
                request.reply_model,
                request.debug,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/auto-sessions/{session_id}/initialize")
    def initialize_auto_session(session_id: str) -> dict:
        return simulator().initialize_auto_session(session_id)

    @app.post("/api/auto-sessions/{session_id}/step")
    def step_auto_session(session_id: str) -> dict:
        return simulator().step_auto_session(session_id)

    @app.post("/api/human-sessions", status_code=201)
    def create_human_session(request: CreateHumanSessionRequest) -> dict:
        return simulator().create_human_session(
            request.sample_id,
            request.dataset or simulator().default_dataset,
            request.agent,
        )

    @app.post("/api/human-sessions/{session_id}/initialize")
    def initialize_human_session(session_id: str) -> dict:
        return simulator().initialize_human_session(session_id)

    @app.post("/api/human-sessions/{session_id}/reply")
    def submit_human_reply(session_id: str, request: HumanReplyRequest) -> dict:
        return simulator().submit_human_reply(session_id, request.message)

    @app.post("/api/rewrite")
    def rewrite_message(request: RewriteRequest) -> dict:
        try:
            return {"message": build_reply_model("deepseek").rewrite_query_answer(request.message)}
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/sessions/{session_id}/turn")
    def submit_turn(session_id: str, request: AgentTurnRequest) -> dict:
        return simulator().submit_turn(session_id, request)

    return app


app = create_app()
