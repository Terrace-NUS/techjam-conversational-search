# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is required. Install all project dependencies with uv:

```bash
uv sync
```

The starter itself uses only the Python standard library.

```bash
uv run python -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## GPU Rollouts (Optional)

For batched policy training, `rollout.gpu_simulator.TensorRolloutSimulator`
keeps episode state and reward calculation in PyTorch tensors. It selects
CUDA, then Apple MPS, then CPU automatically. Dependencies are installed by
the root `uv sync` command.

The rollout action format is integer catalog indices shaped `[batch, top_k]`
and attribute IDs shaped `[batch]`; use `RolloutDataset.catalog_ids` to map a
selected index back to `parent_asin`. The official evaluator remains the
reproducibility check and is intentionally unchanged.

Run the from-scratch byte-level BPE tokenizer, dual-encoder, dense-reward PPO
smoke test with:

```bash
uv run python -m scripts.train_ppo \
  --config configs/ppo.toml \
  --experiment-dir experiments/ppo-exp-001 \
  --checkpoint-interval 10
uv run tensorboard --logdir experiments/ppo-exp-001/tensorboard
```

Training defaults are in `configs/ppo.toml`. Use any CLI option to override
its TOML value, for example `--batch-size 256` or `--precision fp32`. BF16
autocast is enabled by default while model parameters and optimizer state stay
FP32. Each experiment directory
contains `tensorboard/` and `checkpoints/`. `latest.pt` is updated every iteration,
periodic `step-*.pt` files are written at `--checkpoint-interval`, and
`final.pt` is written after normal training. TensorBoard logs
rollout return, hit rate, dense reward, PPO/value/contrastive losses, entropy,
gradient norm, and learning rate once per training iteration.

Training streams shuffled batches from the generated 10-million-session
pickle shards configured by `paths.rollout_data`. A new batch is loaded every
iteration, and each record's four constraints are reshuffled into a fresh
two-hard/two-soft intent card. The default `19,532` iterations cover one pass
over 10 million samples at batch size 512; the final fixed-size batch wraps by
384 samples into the next shuffled pass. TensorBoard also logs samples seen
and completed data epochs.

The policy sees one customer message per turn and carries its GRU hidden state
across the session. Recurrent PPO minibatches shuffle sessions while preserving
and backpropagating through their 10-turn order; completed turns are masked.
Hidden targets are used for rank-based training rewards and in-batch
contrastive learning, never as model inputs.
The default dense reward is `official + 0.4 * delta_rank_potential + 0.03 *
new_constraints - 0.02 * no_information - 0.01 * turn`, where rank potential
is the target's normalized log rank over the current catalog.

Generate 10 million synthetic sessions as Ray-produced pickle parts:

```bash
uv run python -m scripts.generate_rollout_data --workers 8
```

Add `--resume` to verify existing pickle parts and continue an interrupted run.
Generated data is written under `data/generated/rollout-10m/pkl/` and is
ignored by Git.

Generate an 800-session private-style test set with targets disjoint from the
public 200 sessions and the specification's exact `320/320/120/40` scenario
mix:

```bash
uv run python -m scripts.generate_private_test
```

This writes `test.pkl`, `test.jsonl`, and `metadata.json` under
`data/generated/private-test-800/`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
