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

Python 3.10 or later is recommended. The baseline starter uses the Python standard library;
`uv sync` also installs the OpenAI SDK for the optional DeepSeek reply model.

```bash
uv sync
uv run python -m evaluator.local_evaluator --progress
```

## Run the Simulator Visualizer

The first visualizer mode lets you play the Agent against the same deterministic
Simulator rules used by the local evaluator. The FastAPI backend shares this
project's `uv` environment.

```bash
uv sync
cd visualizer/frontend
pnpm install
cd ../..
```

Then start both servers with one command:

```bash
uv run python scripts/visualizer.py
# Enable backend hot reload; Vite already hot-reloads the frontend.
uv run python scripts/visualizer.py --reload
```

Open `http://localhost:5173`. Choose a public case, write the Agent message and
`ask_attribute`, then search and rank up to ten catalog products. The hidden
target is only revealed after a hit or turn 10.

`--dataset` accepts both formats. Records with `"version":"v2"` use the
embedded intent descriptions and modification fields; records without it keep
the legacy simulator behavior. For the generated v2 development set, run:

```bash
uv run python -m evaluator.local_evaluator --dataset data/custom/public_set_v2.jsonl --progress
```

Agent implementations are selected with `--agent baseline|v1` (or
`TECHJAM_AGENT`). `baseline` is the original BM25 starter; `v1` is the
offline structured-retrieval implementation in `starter/v1/`.

The evaluator uses deterministic template customer wording by default. To
surface-realize each customer message with DeepSeek, set `DEEPSEEK_API_KEY`
in the ignored `.env` file and run:

```bash
uv run python -m evaluator.local_evaluator --reply-model deepseek
```

Different sessions can run concurrently while each session remains turn-ordered:

```bash
uv run python -m evaluator.local_evaluator --reply-model deepseek --workers 8 --progress
```

Evaluation progress is printed to stderr. A partial result is written after
each completed session to `results.json.partial` (or the path passed to
`--checkpoint`), so an interrupted run still leaves usable metrics.

`DEEPSEEK_MODEL` defaults to `deepseek-v4-flash`; `DEEPSEEK_BASE_URL` defaults
to `https://api.deepseek.com`. DeepSeek request or response errors are fatal in
this mode; they are not silently replaced with template text.

Implement your agent against the ABC in `starter/agent.py`; the included
baseline and V1 implementations are in `starter/baseline.py` and `starter/v1/`.
Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

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

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
data/custom/public_set_v2.jsonl   v2 sessions with embedded intent data
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set evaluator and scorer
evaluator/simulators/             versioned simulator implementations
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
