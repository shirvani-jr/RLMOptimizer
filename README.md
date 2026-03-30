# RLMOptimizer

A prompt optimizer built on the [Recursive Language Model](https://arxiv.org/abs/2512.24601) paradigm. The optimizer is an LLM agent with a persistent code environment, optimization tools, and sub-LLM access. It doesn't follow a fixed optimization algorithm — it *is* the algorithm.

## Why RLM for Optimization

Most prompt optimizers — even good ones like [GEPA](https://arxiv.org/abs/2507.03549) and MIPROv2 — are fixed algorithms that use LLMs as components. GEPA reflects on traces to propose mutations; MIPROv2 generates candidates and selects via Bayesian optimization. The LLM contributes, but the algorithm orchestrates.

RLMOptimizer inverts this. The [RLM paper](https://arxiv.org/abs/2512.24601) showed that LLMs can handle problems far beyond their context limits by treating data as part of an external environment they interact with through code — loading it into a REPL, slicing and filtering it programmatically, and recursively calling sub-LLMs on subsets. The RLM paradigm outperformed base LLMs by 2× on long-context tasks while using comparable or lower cost.

RLMOptimizer applies this to prompt optimization: evaluation data (potentially hundreds of examples, each with multi-step execution traces) lives in the agent's code environment as structured data it can process with Python and sub-LLM calls. The agent doesn't try to see all the data at once. It writes code to explore it, builds up understanding across iterations in persistent REPL variables, and makes targeted prompt edits based on what it finds.

This gives the optimizer three capabilities that fixed algorithms don't have:

1. **Computational analysis at scale**: The agent writes code to filter, group, count, and compare evaluation results — processing volumes of diagnostic data that would overwhelm any single LLM reflection prompt.

2. **Hybrid code + LLM reasoning**: Code handles structure (which examples failed? which step? what pattern?). Sub-LLM calls handle semantics (why is this output wrong? classify these 50 failures). The agent composes both freely.

3. **Adaptive strategy**: The agent decides what to evaluate, what to analyze, and what to change — adapting its approach to what it finds rather than following predetermined steps.

## How It Works

When you call `compile()`:

1. **Baseline**: Your program is cloned and evaluated on the training set (and validation set if provided) to establish starting scores. These baseline runs are saved — the agent can inspect them without spending budget.

2. **Agent loop**: A [DSPy RLM](https://dspy.ai) agent takes over. Each iteration, it produces reasoning and Python code. The code executes in a persistent REPL where optimization tools and sub-LLM functions are registered:

   | Function | Budget Cost | Purpose |
   |----------|-------------|---------|
   | `evaluate_program(...)` | 1 per example | Run the program on examples, return scores + traces |
   | `run_data(run_id)` | Free | Re-read any previous evaluation run |
   | `update_prompt(step, text)` | Free | Replace one step's prompt text |
   | `optimization_status()` | Free | Check budget, current prompts, step names |
   | `llm_query(prompt)` | 1 | Query sub-LLM for analysis |
   | `llm_query_batched([...])` | 1 per prompt | Batch sub-LLM queries |

   The REPL is persistent: variables, functions, and analysis results from earlier iterations remain available in later ones.

3. **Submission**: When the agent is satisfied (or budget runs out), it submits `optimized_dspy_program` — a `dict[str, str]` mapping each step name to its final prompt text. This map is applied to your program and returned.

### Evaluation Data

Train evaluations return structured per-example results. Each example includes:
- The inputs passed to the program
- The expected outputs (ground truth)
- The predicted outputs (what the program produced)
- A score (0.0–1.0) and pass/fail status
- **Per-step execution traces**: what each step received as input and produced as output

The agent can also target evaluations: evaluate a subset (`limit=15`), sample randomly (`sample='random'`), pick specific IDs (`ids='3,7,12'`), or re-evaluate only examples that failed in a previous run (`failed_from_run=run_id`).

Validation evaluations return only aggregate metrics (score, pass count, example count). No per-example data — the agent can't overfit to the holdout.

### Budget

```
budget = max_iterations × len(trainset) + len(valset)
```

Each evaluated example costs 1 unit. Each sub-LLM call costs 1 unit. **Root LM reasoning is free** — the agent can continue analyzing existing data, planning, and submitting prompts even after evaluation budget hits zero. Budget constrains how much new information the agent gathers, not how much it thinks.

### What Gets Optimized

Only `signature.instructions` (the prompt text) for each step. Field definitions, program control flow, few-shot demos, and model weights are not modified. This is structurally enforced: a fingerprint check after every update rejects any change that alters program structure.

## Quick Start

```python
import dspy
from rlmoptimizer import RLMDocstringOptimizer

optimizer = RLMDocstringOptimizer(
    max_iterations=5,
    root_lm=dspy.LM("openai/gpt-4o", model_type="responses"),  # the optimizer agent
    sub_lm=dspy.LM("openai/gpt-4o-mini"),     # for batch analysis
    eval_lm=dspy.LM("openai/gpt-4o-mini"),    # runs your program
)

optimized = optimizer.compile(
    student=program,
    trainset=trainset,
    metric=metric,
    valset=valset,   # optional but recommended
)

result = optimized(question="What is the capital of France?")
```

Set `OPENAI_API_KEY` (or the appropriate provider key) in your environment.

### The Three LMs

| Parameter | Role | Recommendation |
|-----------|------|----------------|
| `root_lm` | The optimizer agent — writes analysis code, reasons about failures, writes new prompts | Your strongest available model |
| `sub_lm` | Helper the agent calls via `llm_query` / `llm_query_batched` for semantic analysis tasks | A capable but cheaper model (optional) |
| `eval_lm` | Runs your DSPy program during evaluations | Whatever model your program targets in production |

## What the Agent Actually Does

The agent works in a persistent Python REPL with the tools above registered as callable functions. Here's what its code looks like in practice:

**Evaluate and filter**:
```python
result = evaluate_program(split='train', limit=25)
failed = [ex for ex in result['examples'] if not ex['passed']]
print(f"{len(failed)} failures out of {result['evaluated_count']}")
```

**Trace through steps to find root causes**:
```python
for ex in failed[:5]:
    print(f"Example {ex['example_id']}:")
    for step in ex['steps']:
        print(f"  {step['step_name']}: in={step['inputs'].keys()} → out={step['outputs']}")
    print(f"  Expected: {ex['expected']}")
    print()
```

**Use sub-LLM for semantic analysis**:
```python
analyses = llm_query_batched([
    f"Why does this output not match? Input: {ex['inputs']}, "
    f"Expected: {ex['expected']}, Got: {ex['predicted']}. "
    f"Categorize: (a) wrong format (b) wrong content (c) hallucinated (d) incomplete"
    for ex in failed
])
# Agent now has failure categories to reason about
```

**Update prompts and re-evaluate targeted subsets**:
```python
update_prompt('generate', """You are answering questions based on retrieved context.
Extract only information present in the provided passages...""")

# Re-evaluate just the failures to see if the fix helped
recheck = evaluate_program(split='train', failed_from_run=result['run_id'])
print(f"Previously failed: {len(failed)}, now passing: {recheck['passed_count']}")
```

**Checkpoint with validation**:
```python
val = evaluate_program(split='val')  # aggregate-only, no per-example data
print(f"Validation score: {val['score']}%")
```

Each of these code blocks is one agent iteration. Variables persist across iterations — `failed` from iteration 1 is still available in iteration 5. The agent builds up understanding incrementally, and the system prompt provides strategic guidance on how to use these tools effectively.

## How It Differs from Other Optimizers

| | Algorithm | LLM Role | Analysis Medium | Strategy |
|---|---|---|---|---|
| **MIPROv2** | Bayesian optimization | Generate candidate prompts | — | Fixed: propose, evaluate, select |
| **OPRO** | Score-guided generation | Propose prompts from (prompt, score) history | Natural language | Fixed: generate from history |
| **GEPA** | Evolutionary + Pareto selection | Reflect on traces to propose mutations | Natural language | Fixed: select, mutate, evaluate, prune |
| **RLMOptimizer** | The LLM agent *is* the algorithm | Orchestrate the entire optimization | Code + sub-LLM calls | Adaptive: agent decides based on findings |

GEPA and RLMOptimizer both use execution traces for diagnostics. The difference is in how they process them:

- GEPA passes traces to an LLM in a single reflection prompt that proposes a new prompt. The evolutionary algorithm then decides whether to keep it.
- RLMOptimizer loads traces into a code environment where the agent can filter, aggregate, compare across runs, call sub-LLMs on subsets, and iterate — all before deciding what to change.

This distinction matters most for complex programs with large evaluation sets, where the diagnostic data exceeds what fits in a single reflection prompt.

## Configuration

```python
RLMDocstringOptimizer(
    max_iterations=5,              # Budget = this × trainset size (+ valset size)
    root_lm=...,                   # Required
    sub_lm=None,                   # Optional helper LM
    eval_lm=None,                  # Falls back to dspy.settings.lm
    num_threads=8,                 # Parallel evaluation threads
    rlm_max_iterations=200,        # Max agent REPL iterations
    rlm_max_llm_calls=200,         # Max sub-LM calls the agent can make
    rlm_max_output_chars=10000,    # Max output per REPL iteration
    root_stateful_session=True,    # Thread state via Responses API
    rlm_multiturn_history=False,   # Format history as user/assistant turns
    verbose=False,                 # Print agent trajectory with rich panels
    run_storage_dir=None,          # Persist run data (default: temp dir)
)
```

### Stateful Root Sessions

When `root_stateful_session=True` and `root_lm` uses the Responses API (`model_type="responses"`), the optimizer threads `previous_response_id` across agent turns. The root model maintains conversational context without re-processing the full history each turn. If the root model doesn't support Responses mode, optimization still runs — without state threading.

If you enable both `root_stateful_session=True` and `rlm_multiturn_history=True`, REPL history is rendered as structured user/assistant turns (preserving the conversation shape) while still chaining `previous_response_id` between turns.

### Inspecting Results

```python
optimized.optimized_dspy_program   # dict[str, str] — final prompt for each step
optimized.latest_run_id           # Most recent evaluation run ID
optimized.trial_logs              # All evaluation runs with scores and configs
optimized.agent_trajectory        # Full agent trajectory (reasoning + code + output)
optimized.agent_final_reasoning   # Agent's final summary of what it learned
```

## Demo

```bash
python example.py
```

Runs a self-contained demo with a scripted optimizer session — no API keys needed.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

## License

MIT
```
