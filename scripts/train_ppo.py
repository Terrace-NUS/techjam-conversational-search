from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback.
    import tomli as tomllib

from evaluator.local_evaluator import catalog_index, searchable_text
from rollout.gpu_simulator import RolloutBatch, RolloutDataset, load_rollout_dataset, rollout_device
from rollout.ppo import (
    DenseRewardConfig,
    DualEncoderActorCritic,
    PPOConfig,
    PPOTrainer,
    WordTokenizer,
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
    parser.add_argument("--dataset", default=None)
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
    parser.add_argument("--output", default=None)
    parser.add_argument("--tensorboard-dir", default=None)
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
    args.dataset = _value(args.dataset, paths, "dataset", "data/public_set.jsonl")
    args.output = _value(args.output, paths, "output", "ppo-checkpoint.pt")
    args.tensorboard_dir = _value(args.tensorboard_dir, paths, "tensorboard_dir", None)
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
    args.randomize_cards = bool(rollout.get("randomize_cards", True))
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
    tokenizer: WordTokenizer,
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
            "vocabulary": tokenizer.vocabulary,
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
    dataset = load_rollout_dataset(
        args.catalog,
        args.dataset,
        randomize_cards=args.randomize_cards,
        seed=args.seed,
    )
    _, _, products = catalog_index(args.catalog)

    if args.smoke:
        args.iterations = 1
        args.batch_size = min(args.batch_size, 8)
        args.candidate_count = min(args.candidate_count, 32)
        args.hidden_size = min(args.hidden_size, 64)
        args.max_vocab = min(args.max_vocab, 2_000)
        args.max_product_length = min(args.max_product_length, 48)
        args.max_query_length = min(args.max_query_length, 64)
        dataset, _ = subset_dataset(dataset, args.batch_size, 128)
    else:
        dataset, _ = subset_dataset(dataset, args.batch_size, len(dataset.catalog_ids))
    catalog_ids = dataset.catalog_ids

    product_texts = [searchable_text(products[parent_asin]) for parent_asin in catalog_ids]
    tokenizer = WordTokenizer.fit(
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
            trajectory, rollout_metrics = trainer.collect(dataset)
            update_metrics = trainer.update(trajectory)
            step = iteration + 1
            metrics = {"iteration": step, **rollout_metrics, **update_metrics}
            print(json.dumps({name: round(value, 6) for name, value in metrics.items()}), flush=True)
            for name, value in rollout_metrics.items():
                writer.add_scalar(f"rollout/{name}", value, step)
            for name, value in update_metrics.items():
                writer.add_scalar(update_tags.get(name, f"train/{name}"), value, step)
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
