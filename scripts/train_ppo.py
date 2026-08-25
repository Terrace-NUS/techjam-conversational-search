from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback.
    import tomli as tomllib

from evaluator.local_evaluator import catalog_index, searchable_text
from rollout.gpu_simulator import (
    ATTRIBUTE_TO_ID,
    BUYING,
    INTENT_OVERRIDE,
    SCENARIO_NAMES,
    RolloutBatch,
    RolloutDataset,
    classify_constraint,
    rollout_device,
)
from rollout.ppo import (
    BPETokenizer,
    DenseRewardConfig,
    DualEncoderActorCritic,
    PPOConfig,
    PPOTrainer,
)


class RolloutShardSampler:
    """Streams shuffled fixed-size batches from generated pickle parts."""

    def __init__(self, directory: str | Path, seed: int) -> None:
        self.directory = Path(directory)
        with (self.directory / "metadata.pkl").open("rb") as handle:
            metadata = pickle.load(handle)
        if metadata.get("format") != "techjam-rollout-pickle-v1":
            raise ValueError(f"unsupported rollout data format in {self.directory}")
        if tuple(metadata.get("scenario_names", ())) != SCENARIO_NAMES:
            raise ValueError("rollout scenario names do not match the simulator")
        self.catalog_ids = tuple(metadata["catalog_ids"])
        self.sample_count = int(metadata["num_samples"])
        self.part_paths = tuple(sorted(self.directory.glob("part-*.pkl")))
        if len(self.part_paths) != int(metadata["part_count"]):
            raise ValueError(
                f"expected {metadata['part_count']} rollout parts, found {len(self.part_paths)}"
            )
        self.rng = random.Random(seed)
        self.samples_seen = 0
        self.epochs_completed = 0
        self._part_order: list[int] = []
        self._records: list[tuple] = []
        self._record_offset = 0

    def _load_next_part(self) -> None:
        if not self._part_order:
            self._part_order = list(range(len(self.part_paths)))
            self.rng.shuffle(self._part_order)
            if self.samples_seen:
                self.epochs_completed += 1
        path = self.part_paths[self._part_order.pop()]
        with path.open("rb") as handle:
            records = pickle.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"{path} does not contain a record list")
        self.rng.shuffle(records)
        self._records = records
        self._record_offset = 0

    def next_batch(self, batch_size: int) -> RolloutDataset:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        records: list[tuple] = []
        while len(records) < batch_size:
            if self._record_offset >= len(self._records):
                self._load_next_part()
            count = min(batch_size - len(records), len(self._records) - self._record_offset)
            records.extend(self._records[self._record_offset:self._record_offset + count])
            self._record_offset += count

        start = self.samples_seen
        self.samples_seen += batch_size
        randomized = [
            (*record[:3], tuple(self.rng.sample(tuple(record[3]), len(record[3]))), record[4])
            for record in records
        ]
        return rollout_dataset_from_records(randomized, self.catalog_ids, start)


def rollout_dataset_from_records(
    records: list[tuple],
    catalog_ids: tuple[str, ...],
    sample_offset: int = 0,
) -> RolloutDataset:
    target_ids: list[int] = []
    scenario_ids: list[int] = []
    override_turns: list[int] = []
    attributes: list[list[int]] = []
    initial_messages: list[str] = []
    constraint_texts: list[tuple[str, ...]] = []
    override_messages: list[str] = []

    for record in records:
        if len(record) != 5:
            raise ValueError("rollout records must contain five fields")
        target_id, scenario_id, category, values, override_turn = record
        constraints = tuple(str(value) for value in values)
        if len(constraints) != 4:
            raise ValueError("rollout cards must contain four constraints")
        if not 0 <= int(target_id) < len(catalog_ids):
            raise ValueError(f"target_id {target_id} is outside the catalog")
        if not 0 <= int(scenario_id) < len(SCENARIO_NAMES):
            raise ValueError(f"scenario_id {scenario_id} is invalid")

        target_ids.append(int(target_id))
        scenario_ids.append(int(scenario_id))
        override_turns.append(int(override_turn))
        constraint_texts.append(constraints)
        attributes.append([ATTRIBUTE_TO_ID[classify_constraint(value)] for value in constraints])
        if scenario_id == BUYING:
            initial_messages.append(
                f"I'm looking for {category}. A key requirement is: {constraints[0]}."
            )
        elif scenario_id == INTENT_OVERRIDE:
            initial_messages.append(f"I'm looking for {category}. {constraints[-1]}")
        else:
            initial_messages.append(f"I'm looking for {category}, but I'm still exploring.")
        override_messages.append(
            f"Actually, ignore my earlier preference. What I need is: {constraints[0]}."
            if scenario_id == INTENT_OVERRIDE else ""
        )

    size = len(records)
    constraint_attribute_ids = torch.tensor(attributes, dtype=torch.long).reshape(size, 4)
    constraint_mask = torch.ones((size, 4), dtype=torch.bool)
    initial_revealed = torch.zeros((size, 4), dtype=torch.bool)
    if size:
        buying_rows = torch.tensor(scenario_ids).eq(BUYING)
        initial_revealed[buying_rows, 0] = True
    return RolloutDataset(
        batch=RolloutBatch(
            target_ids=torch.tensor(target_ids, dtype=torch.long),
            scenario_ids=torch.tensor(scenario_ids, dtype=torch.long),
            override_turns=torch.tensor(override_turns, dtype=torch.long),
            constraint_attribute_ids=constraint_attribute_ids,
            constraint_mask=constraint_mask,
            initial_revealed=initial_revealed,
            overridden_constraint_indices=torch.tensor(
                [3 if scenario == INTENT_OVERRIDE else -1 for scenario in scenario_ids], dtype=torch.long
            ),
            override_constraint_indices=torch.tensor(
                [0 if scenario == INTENT_OVERRIDE else -1 for scenario in scenario_ids], dtype=torch.long
            ),
        ),
        catalog_ids=catalog_ids,
        sample_ids=tuple(f"synthetic_{sample_offset + index}" for index in range(size)),
        initial_messages=tuple(initial_messages),
        constraint_texts=tuple(constraint_texts),
        override_messages=tuple(override_messages),
    )


def subset_dataset(
    dataset: RolloutDataset,
    session_count: int,
    catalog_count: int,
) -> tuple[RolloutDataset, list[int]]:
    session_count = min(session_count, dataset.batch.size)
    targets = dataset.batch.target_ids[:session_count].tolist()
    catalog_indices = list(dict.fromkeys([*targets, *range(min(catalog_count, len(dataset.catalog_ids)))]))
    old_to_new = {old: new for new, old in enumerate(catalog_indices)}
    fields = {}
    for name, tensor in dataset.batch.__dict__.items():
        sliced = tensor[:session_count].clone()
        if name == "target_ids":
            sliced = torch.tensor([old_to_new[int(value)] for value in sliced], dtype=torch.long)
        fields[name] = sliced
    return RolloutDataset(
        batch=RolloutBatch(**fields),
        catalog_ids=tuple(dataset.catalog_ids[index] for index in catalog_indices),
        sample_ids=dataset.sample_ids[:session_count],
        initial_messages=dataset.initial_messages[:session_count],
        constraint_texts=dataset.constraint_texts[:session_count],
        override_messages=dataset.override_messages[:session_count],
    ), catalog_indices


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _value(cli_value: object, section: dict, name: str, default: object) -> object:
    return cli_value if cli_value is not None else section.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="From-scratch dense-reward PPO trainer")
    parser.add_argument("--config", default="configs/ppo.toml")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--rollout-data", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--max-vocab", type=int, default=None)
    parser.add_argument("--max-product-length", type=int, default=None)
    parser.add_argument("--max-query-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be at least 1")
    config = load_config(args.config)
    paths = config.get("paths", {})
    run = config.get("run", {})
    model = config.get("model", {})
    rollout = config.get("rollout", {})
    args.catalog = _value(args.catalog, paths, "catalog", "data/catalog.jsonl")
    args.rollout_data = _value(
        args.rollout_data, paths, "rollout_data", "data/generated/rollout-10m/pkl"
    )
    args.device = _value(args.device, run, "device", "auto")
    args.precision = _value(args.precision, run, "precision", "bf16")
    args.iterations = _value(args.iterations, run, "iterations", 10)
    args.seed = _value(args.seed, run, "seed", 0)
    for name, section, default in (
        ("hidden_size", model, 128),
        ("max_vocab", model, 20_000),
        ("max_product_length", model, 64),
        ("max_query_length", model, 128),
        ("batch_size", rollout, 200),
        ("candidate_count", rollout, 128),
        ("top_k", rollout, 10),
        ("max_turns", rollout, 10),
    ):
        setattr(args, name, _value(getattr(args, name), section, name, default))
    args.ppo_config = config.get("ppo", {})
    args.reward_config = config.get("reward", {})
    args.experiment_dir = Path(args.experiment_dir)
    args.tensorboard_dir = str(args.experiment_dir / "tensorboard")
    args.experiment_checkpoint_dir = args.experiment_dir / "checkpoints"
    return args


def save_checkpoint(
    path: Path,
    model: DualEncoderActorCritic,
    trainer: PPOTrainer,
    tokenizer: BPETokenizer,
    catalog_ids: tuple[str, ...],
    args: argparse.Namespace,
    iteration: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "tokenizer": tokenizer.to_str(),
            "catalog_ids": catalog_ids,
            "iteration": iteration,
            "config": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = rollout_device() if args.device == "auto" else torch.device(args.device)

    if args.smoke:
        args.iterations = 1
        args.batch_size = min(args.batch_size, 8)
        args.candidate_count = min(args.candidate_count, 32)
        args.hidden_size = min(args.hidden_size, 64)
        args.max_vocab = min(args.max_vocab, 2_000)
        args.max_product_length = min(args.max_product_length, 48)
        args.max_query_length = min(args.max_query_length, 64)
    sampler = RolloutShardSampler(args.rollout_data, args.seed)
    dataset = sampler.next_batch(args.batch_size)
    _, _, products = catalog_index(args.catalog)
    if tuple(products) != sampler.catalog_ids:
        raise ValueError("rollout metadata catalog IDs do not match the configured catalog")
    if args.smoke:
        dataset, _ = subset_dataset(dataset, args.batch_size, 128)
    catalog_ids = dataset.catalog_ids

    product_texts = [searchable_text(products[parent_asin]) for parent_asin in catalog_ids]
    tokenizer = BPETokenizer.fit(
        [
            *product_texts,
            *dataset.initial_messages,
            *(text for values in dataset.constraint_texts for text in values),
            *dataset.override_messages,
        ],
        max_vocab=args.max_vocab,
    )
    product_token_ids, product_mask = tokenizer.encode_batch(
        product_texts, args.max_product_length, "cpu"
    )
    model = DualEncoderActorCritic(len(tokenizer), args.hidden_size)
    ppo_values = {**args.ppo_config}
    if args.smoke:
        ppo_values.update(epochs=1, minibatch_size=32)
    ppo_config = PPOConfig(**ppo_values)
    reward_config = DenseRewardConfig(**args.reward_config)
    trainer = PPOTrainer(
        model,
        tokenizer,
        product_token_ids,
        product_mask,
        device=device,
        candidate_count=args.candidate_count,
        top_k=args.top_k,
        max_query_length=args.max_query_length,
        max_turns=args.max_turns,
        precision=args.precision,
        reward_config=reward_config,
        ppo_config=ppo_config,
    )

    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    print(json.dumps({"tensorboard_dir": writer.log_dir}), flush=True)
    print(json.dumps({"checkpoint_dir": str(args.experiment_checkpoint_dir)}), flush=True)
    writer.add_text("config", json.dumps(vars(args), indent=2, default=str))
    update_tags = {
        "loss": "loss/total",
        "policy_loss": "loss/policy",
        "value_loss": "loss/value",
        "contrastive_loss": "loss/contrastive",
        "entropy": "train/entropy",
        "gradient_norm": "train/gradient_norm",
    }
    try:
        for iteration in range(args.iterations):
            if iteration:
                dataset = sampler.next_batch(args.batch_size)
            trajectory, rollout_metrics = trainer.collect(dataset)
            update_metrics = trainer.update(trajectory)
            step = iteration + 1
            metrics = {
                "iteration": step,
                "samples_seen": sampler.samples_seen,
                "data_epochs": sampler.epochs_completed,
                **rollout_metrics,
                **update_metrics,
            }
            print(json.dumps({name: round(value, 6) for name, value in metrics.items()}), flush=True)
            for name, value in rollout_metrics.items():
                writer.add_scalar(f"rollout/{name}", value, step)
            for name, value in update_metrics.items():
                writer.add_scalar(update_tags.get(name, f"train/{name}"), value, step)
            writer.add_scalar("data/samples_seen", sampler.samples_seen, step)
            writer.add_scalar("data/epochs", sampler.epochs_completed, step)
            writer.add_scalar("train/learning_rate", trainer.optimizer.param_groups[0]["lr"], step)
            writer.flush()
            save_checkpoint(
                args.experiment_checkpoint_dir / "latest.pt",
                model,
                trainer,
                tokenizer,
                catalog_ids,
                args,
                step,
            )
            if step % args.checkpoint_interval == 0:
                save_checkpoint(
                    args.experiment_checkpoint_dir / f"step-{step:06d}.pt",
                    model,
                    trainer,
                    tokenizer,
                    catalog_ids,
                    args,
                    step,
                )
    finally:
        writer.close()

    if not args.smoke:
        save_checkpoint(
            args.experiment_checkpoint_dir / "final.pt",
            model,
            trainer,
            tokenizer,
            catalog_ids,
            args,
            args.iterations,
        )


if __name__ == "__main__":
    main()
