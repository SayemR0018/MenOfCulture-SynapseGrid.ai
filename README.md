<div align="center">

# SynapseGrid.ai — Smart Campus Energy Optimization Challenge

<img src="docs/SynapseGrid.ai.jpeg" alt="SynapseGrid.ai Architecture" width="100%">

</div>

LLM-assisted operator directive interpretation + deterministic energy
optimization, built for the BUP CSE FEST 2026 preliminary round
("GridWise" is the organizers' name for this challenge track;
SynapseGrid.ai is our submission's name for the service implementing it).

Pipeline (matches the problem statement's architecture exactly):

```
Operator Notes + 24h Scenario
        |
        v
DirectiveInterpreter (ml/interpreter.py)
    -> LLM provider (ml/providers.py)         [natural language -> raw JSON]
    -> deterministic validator (ml/validator.py) [untrusted JSON -> safe directives]
    -> retry-with-correction on failure (ml/prompts.py: build_correction_prompt)
        |
        v
Validated DirectiveInterpretation list
        |
        v
optimizer/constraints.py   [directives -> per-hour arrays]
optimizer/optimizer.py     [linear program -> hourly_plan]  <- NO LLM involvement
        |
        v
optimizer/final_validator.py  [replay hourly_plan against every rule/directive]
        |
        v
HTTP JSON response (app/routes.py)
```

The LLM never computes a schedule, cost, or any numeric optimization
result — it only turns natural language into one of six structured
directives, which deterministic code then validates, applies, and
optimizes.

## Project layout

```
app/            FastAPI service: routes, request/response schemas, config
ml/             LLM interpretation layer (the ML module)
  schemas.py        Pydantic contract for directives (strict validation)
  prompts.py        System prompt, few-shot examples, prompt builders
  providers.py      LLMProvider abstraction: OpenAI / Anthropic / offline mock
  interpreter.py    Orchestrates LLM call -> parse -> validate -> retry
  validator.py      Deterministic guardrail (never trusts raw LLM output)
  normalizer.py     Regex-based time/percentage parsing (backs the mock
                     provider + pins down semantics in tests)
  exceptions.py     Controlled error types
optimizer/      Deterministic math layer (LLM-free)
  constraints.py    Directives + scenario -> per-hour arrays
  optimizer.py      Linear program (scipy/HiGHS) -> hourly_plan
  cost.py           total_grid_kwh / total_cost_bdt / peak_grid_kwh
  final_validator.py  Replays hourly_plan against every SynapseGrid.ai rule
tests/          pytest suite (65+ tests)
scripts/        scripts/evaluate_interpreter.py — accuracy metrics
docs/           Canonical problem statement, rubric, and public sample cases
```

## How the ML pipeline works

1. **`DirectiveInterpreter.interpret()`** builds one prompt for *all* of a
   scenario's 1-3 operator notes (Section 26: one request per scenario, not
   per note) and calls the configured `LLMProvider.generate_structured()`.
2. The raw text response is parsed defensively (tolerates markdown code
   fences, `{"directive_interpretation": [...]}` or a bare `[...]`).
3. **`validate_interpretation()`** deterministically checks every rule from
   the problem statement's guardrail table: allowed directive types, full
   note coverage with no duplicates, ascending/unique/in-range hours,
   `factor` in `[0,1]`, non-negative finite reserve/grid-cap values, reserve
   not exceeding battery capacity, `no_op` <-> `applies=false` <-> `null`
   adjustment, and rejection of any extra/invented field in
   `structured_adjustment` (via Pydantic's `extra="forbid"`).
4. If validation fails, a **correction prompt** containing only the
   validation errors is sent back to the model (`MAX_RETRIES`, default 1).
5. If validation still fails after all retries, the service **fails safe**:
   every note is returned as `no_op` (never a guess, never a crash) unless
   `on_unsafe_fallback="raise"` is configured, in which case a controlled
   `InterpretationValidationError` propagates instead.
6. Only now does `optimizer/optimizer.py` run — a linear program (via
   `scipy.optimize.linprog`, HiGHS) with variables `grid`, `solar_used`,
   `charge`, `discharge`, `battery_energy_after` per hour, minimizing
   `sum(grid[h] * tariff[h])` subject to the energy-balance equation,
   battery bounds/rate limits, and every applied directive.
7. `optimizer/final_validator.py` independently replays the resulting
   `hourly_plan` hour-by-hour (its own copy of the rules, not shared code
   paths with the LP) as a safety net before the response is returned.

### The six supported directives

`solar_reduction`, `minimum_battery_reserve`, `no_charge_window`,
`no_discharge_window`, `max_grid_window`, `no_op` — see `ml/prompts.py`
for the exact schema and semantics embedded in the system prompt, and
`ml/schemas.py` for their Pydantic models.

## How to configure the LLM

Copy `.env.example` to `.env` and set:

```
MODEL_PROVIDER=openai        # or anthropic, or mock
MODEL_NAME=gpt-4o-mini        # or e.g. claude-sonnet-5 for anthropic
API_KEY=sk-...                 # never hard-coded, never committed
MAX_RETRIES=1
```

- `openai` / `anthropic` use the respective official SDKs with JSON-mode /
  plain text completion and `temperature=0`.
- `mock` is a **deterministic, offline, rule-based** provider
  (`ml/providers.py: MockProvider`, backed by `ml/normalizer.py`) used as
  the default so the project runs and tests pass with zero credentials.
  **It does not satisfy the competition's "LLM must be part of the
  interpretation path" requirement** — set `MODEL_PROVIDER` to `openai` or
  `anthropic` before deploying/submitting.
- Swapping providers never touches application code — `app/routes.py` only
  calls `provider.generate_structured(...)` through the `LLMProvider`
  abstract base class.

## How to run the server

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit MODEL_PROVIDER / MODEL_NAME / API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`GET /health` → `{"status": "ok"}`
`POST /optimize-energy` → see `app/schemas.py` / Section 07 of the problem
statement for the exact request shape.

## Running with Docker

The service is fully containerized (`Dockerfile` + `docker-compose.yml`).
The image installs `requirements.txt`, copies only `app/`, `ml/`,
`optimizer/`, `scripts/`, and the public sample-case JSON (tests, docs
PDFs, and the venv are excluded via `.dockerignore`), and runs as a
non-root user with a built-in `HEALTHCHECK` against `/health`.

```bash
cp .env.example .env        # edit MODEL_PROVIDER / MODEL_NAME / API_KEY first

# Option A: docker compose (recommended — reads .env automatically)
docker compose up --build

# Option B: plain docker
docker build -t synapsegrid-energy-optimizer .
docker run --rm -p 8000:8000 --env-file .env synapsegrid-energy-optimizer
```

Then the same `curl` calls from the section above work against
`http://localhost:8000`. `docker compose up -d` runs it detached;
`docker compose logs -f` follows the structured request logs described
below.

Notes:

- `MODEL_PROVIDER=mock` (the `.env.example` default) needs no `API_KEY` and
  is fine for confirming the container runs end-to-end; switch to `openai`
  or `anthropic` with a real key before deploying/submitting, same as the
  non-Docker path.
- The image was written and sanity-checked by inspection in this
  environment (no container runtime was available here to actually build
  it) — build it once locally / in CI before relying on it for submission.
- Rebuild after dependency changes (`docker compose up --build`); code-only
  changes under `app/`, `ml/`, `optimizer/` also need a rebuild since the
  image copies source rather than mounting it. For live-reload local dev,
  run `uvicorn app.main:app --reload` outside Docker instead.

## How to run tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

All 65+ tests run against the offline `mock` provider (no network/API key
needed) and currently pass, including:

- `test_time_parsing.py` / `test_numeric_parsing.py` — start-inclusive/
  end-exclusive hour semantics, "remaining fraction" vs. "reduction"
  percentage semantics.
- `test_validator.py` — every deterministic guardrail rule, including
  malformed/hallucinated input that must not crash the validator.
- `test_interpreter.py` — one test per directive type, three independent
  paraphrases of the same solar-reduction note (Section 17), five
  adversarial notes that must never hallucinate scenario data or invent
  directives (Section 18), and the retry/safe-fallback path (Section 22-23).
- `test_public_cases.py` — runs all 10 public sample cases through the
  *entire* pipeline (interpret -> optimize -> final-validate) and checks
  schedule validity (24 hours, energy balance, battery bounds, end-of-day
  neutrality, directive constraints) without hard-coding any case's
  expected numbers.

To evaluate interpretation *accuracy* specifically (relevance, directive
classification, hour/numeric extraction, no_op, complete-interpretation
accuracy, plus an error-kind breakdown):

```bash
python scripts/evaluate_interpreter.py
# or against another labeled pack:
python scripts/evaluate_interpreter.py --cases path/to/cases.json
```

This respects the same `MODEL_PROVIDER` env var, so pointing it at a real
LLM gives a true accuracy readout before submission.

## How to test /optimize-energy manually

```bash
uvicorn app.main:app --port 8000 &
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "DEMO-1",
    "operator_notes": [
      "Solar output will drop to about 20% from 1 PM to 3 PM.",
      "Do not charge the battery between 2 AM and 4 AM."
    ],
    "hours": [ ... 24 entries with hour/demand_kwh/solar_kwh/tariff_bdt_per_kwh ... ],
    "battery": {
      "capacity_kwh": 500, "initial_energy_kwh": 200, "minimum_energy_kwh": 50,
      "max_charge_kwh_per_hour": 100, "max_discharge_kwh_per_hour": 100
    }
  }' | python3 -m json.tool
```

Or replay the whole public sample pack:

```bash
python3 -c "
import json, requests
data = json.load(open('docs/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json'))
for case in data['cases']:
    r = requests.post('http://127.0.0.1:8000/optimize-energy', json=case['input'])
    print(case['id'], r.status_code, r.json()['total_cost_bdt'])
"
```

## Known limitations

- The `mock` provider is a hand-written regex engine, not a real language
  model. It reaches 100% accuracy on the 10 public sample cases (see
  `scripts/evaluate_interpreter.py` output) but will not generalize to
  arbitrary hidden paraphrasing the way a real LLM will — it exists purely
  so the project is runnable/testable without credentials. **A real
  provider must be configured for actual grading.**
- The safe-fallback path (`on_unsafe_fallback="fallback"`, the default)
  degrades an unrecoverable interpretation failure to "apply nothing"
  (`no_op` for every note) rather than failing the whole request with a
  5xx. This maximizes uptime/availability but means a genuinely
  misbehaving LLM silently loses directives instead of erroring loudly;
  every occurrence is logged at `WARNING` (`interpretation_fallback`) for
  observability. Set `on_unsafe_fallback="raise"` in `app/routes.py` if a
  hard failure (HTTP 500) is preferred instead.
- `max_grid_window` directives that overlap in the same hour are combined
  with `min()` (the tightest cap wins); the problem statement guarantees
  organizer scenarios won't require contradictory hard directives, so this
  case shouldn't arise in valid scoring scenarios.
- The optimizer solves an exact LP to global optimality (not a heuristic),
  so cost-minimization is provably optimal given the validated directives —
  the only source of suboptimality is misinterpretation upstream.
- `plan_summary` is generated by a simple deterministic template, not the
  LLM, since Section 02 explicitly says using an LLM only for
  `plan_summary`/cosmetic text does not satisfy the LLM requirement — the
  LLM's real job (directive interpretation) already satisfies it.

## Next improvements

- Add a small in-process cache keyed on `(scenario_id, notes)` to avoid
  redundant LLM calls if the judge harness ever retries a request.
- Expand `MockProvider`'s regex coverage (or replace it with a tiny local
  transformer) if a fully offline fallback for production is ever desired.
- Add OpenTelemetry-style structured JSON logging (currently plain
  `logging` text) if downstream log aggregation is needed.
- Add a request-level rate limiter / timeout budget around the LLM call so
  a slow provider can't stall the 4-hour round's response budget.

<br>

---

## Team & Contributors

* **Sayem Rahman** ([@SayemR0018](https://github.com/SayemR0018)) — **Team Lead**
  * Overall system architecture design and project roadmap orchestration.
  * API contract enforcement, end-to-end integration across ML and optimization modules.
  * Benchmark evaluation against official competition rubrics, documentation, and presentation walkthrough.
* **Rabbi Islam Emon** ([@iamrabbiislamemon](https://github.com/iamrabbiislamemon)) — **Initial Codebase & Backend Engineering**
  * Core repository scaffolding, application layout, and environment configuration management.
  * FastAPI service initialization (`app/main.py`, `app/routes.py`, `app/schemas.py`).
  * Initial mathematical formulation setup and baseline endpoint routing.
* **MD. Redwan Hossain Khan** ([@redwan212](https://github.com/redwan212)) — **ML Model & LLM Directive Interpretation**
  * LLM provider abstraction layer (`ml/providers.py`) supporting OpenAI, Anthropic, and offline mock engines.
  * Prompt engineering with Pydantic structured outputs (`ml/prompts.py`) and zero-shot distractor rejection (`no_op`).
  * Time normalization (start-inclusive, end-exclusive hours) and factor inversion parsing (`ml/normalizer.py`).
* **Shamiul Riyad** ([@shamiulriyad](https://github.com/shamiulriyad)) — **Docker Deployment & Guardrail Engineering**
  * Pre-optimization deterministic guardrails (`ml/validator.py`) to eliminate hallucinations and invalid inputs.
  * Post-optimization schedule verification replayer (`optimizer/final_validator.py`) auditing energy balance and neutrality.
  * Multi-stage Docker containerization (`Dockerfile`, `docker-compose.yml`), non-root security, and registry publishing.

---