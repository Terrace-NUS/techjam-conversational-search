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
    MAX_TURNS,
    TOP_K,
    catalog_index,
    load_jsonl,
    normalize_recommendations,
)
from evaluator.reply_model import build_reply_model
from evaluator.simulators import build_simulator


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


class AgentTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    ask_attribute: AskAttribute | None = None
    recommendations: list[str] = Field(default_factory=list, max_length=TOP_K)


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


class SimulatorService:
    def __init__(self, catalog_path: str | Path, dataset_path: str | Path) -> None:
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
        # ponytail: one lock is enough for a local visualizer; split per session if contention appears.
        self.lock = Lock()

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

    def create_session(
        self,
        sample_id: str,
        dataset_id: str,
        reply_model_name: str,
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
        simulator = build_simulator(
            sample,
            self.categories,
            self.products,
            build_reply_model(reply_model_name),
            session_id,
        )
        session = {
            "id": session_id,
            "status": "waiting_for_agent",
            "sample": sample,
            "dataset": dataset_id,
            "reply_model": reply_model_name,
            "simulator": simulator,
            "target": target,
            "current_turn": 1,
            "current_user_message": simulator.initial_message(),
            "turns": [],
            "outcome": None,
        }
        with self.lock:
            self.sessions[session_id] = session
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
            session["turns"].append(
                {
                    "user_message": session["current_user_message"],
                    "agent_message": request.message,
                    "ask_attribute": request.ask_attribute,
                    "recommendations": ranked,
                    "hit_rank": hit_rank,
                }
            )

            if hit_rank is not None or turn == MAX_TURNS:
                session["status"] = "hit" if hit_rank is not None else "exhausted"
                session["current_user_message"] = None
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
            return self.session_view(session)

    def session_view(self, session: dict) -> dict:
        outcome = None
        if session["outcome"] is not None:
            outcome = {
                **session["outcome"],
                "target_product": self.product_summary(self.products[session["target"]]),
            }
        return {
            "id": session["id"],
            "status": session["status"],
            "sample": self.sample_summary(session["sample"]),
            "dataset": session["dataset"],
            "reply_model": session["reply_model"],
            "user_profile": session["sample"]["user_profile"],
            "current_turn": session["current_turn"],
            "current_user_message": session["current_user_message"],
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
    dataset_path: str | Path = "data/public_set.jsonl",
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

    @app.post("/api/sessions", status_code=201)
    def create_session(request: CreateSessionRequest) -> dict:
        try:
            return simulator().create_session(
                request.sample_id,
                request.dataset or simulator().default_dataset,
                request.reply_model,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/sessions/{session_id}/turn")
    def submit_turn(session_id: str, request: AgentTurnRequest) -> dict:
        return simulator().submit_turn(session_id, request)

    return app


app = create_app()
