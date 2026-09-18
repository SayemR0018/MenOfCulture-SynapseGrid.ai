"""HTTP routes: GET /health and POST /optimize-energy.

Wires together, in order, exactly the pipeline mandated by the problem
statement:

    request -> DirectiveInterpreter (LLM + deterministic validation)
            -> optimizer.optimize_schedule (math, no LLM)
            -> final_validator.replay_and_validate (safety net)
            -> OptimizeEnergyResponse
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    OptimizeEnergyRequest,
    OptimizeEnergyResponse,
)
from ml.exceptions import LLMProviderError
from ml.interpreter import DirectiveInterpreter
from ml.providers import get_provider
from ml.schemas import BatteryContext, ScenarioContext
from optimizer.cost import peak_grid_kwh, total_cost_bdt, total_grid_kwh
from optimizer.final_validator import replay_and_validate
from optimizer.optimizer import OptimizerInfeasibleError, optimize_schedule

logger = logging.getLogger("synapsegrid.app.routes")

router = APIRouter()

_settings = get_settings()
try:
    _provider = get_provider(
        _settings.model_provider, _settings.model_name, _settings.api_key,
        _settings.api_base_url,
    )
except LLMProviderError as exc:  # fail fast at startup with a clear message
    logger.error("Failed to initialize LLM provider: %s", exc)
    raise

_interpreter = DirectiveInterpreter(
    provider=_provider,
    max_retries=_settings.max_retries,
    on_unsafe_fallback="fallback",
)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _load_example_request() -> dict | None:
    """Best-effort example for the Swagger "Try it out" box.

    The endpoint takes a raw ``Request`` (not a Pydantic parameter) so it can
    return a custom 400 on malformed JSON before validation runs -- which
    means FastAPI has nothing to auto-generate a request-body schema from.
    ``openapi_extra`` below fills that gap for documentation only; it does
    not affect how the request is actually parsed or validated.
    """
    path = Path(__file__).resolve().parent.parent / "docs" / (
        "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["cases"][0]["input"]
    except Exception:
        return None


def _inline_json_schema_refs(schema: dict) -> dict:
    """Resolves ``$ref: "#/$defs/..."`` in place and drops ``$defs``.

    Swagger UI resolves "#/..." refs relative to the whole OpenAPI document
    root, not the local schema subtree embedded under ``requestBody`` here --
    so raw ``model_json_schema()`` output (which keeps its own local
    ``$defs``) breaks the docs page with "Could not resolve reference".
    Inlining trades a larger schema for one that works wherever it's placed.
    """
    defs = schema.get("$defs", {})

    def resolve(node, seen=frozenset()):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name in seen:
                    return {}
                rest = {k: v for k, v in node.items() if k != "$ref"}
                return {**resolve(defs.get(name, {}), seen | {name}), **rest}
            return {k: resolve(v, seen) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v, seen) for v in node]
        return node

    return resolve(schema)


_REQUEST_BODY_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": _inline_json_schema_refs(
                    OptimizeEnergyRequest.model_json_schema()
                ),
                **(
                    {"example": example}
                    if (example := _load_example_request()) is not None
                    else {}
                ),
            }
        },
    }
}


@router.post(
    "/optimize-energy",
    response_model=OptimizeEnergyResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    openapi_extra=_REQUEST_BODY_OPENAPI,
)
async def optimize_energy(request: Request):
    request_id = str(uuid.uuid4())

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error="malformed_json", detail="Body is not valid JSON").model_dump(),
        )

    try:
        req = OptimizeEnergyRequest.model_validate(body)
    except ValidationError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="invalid_request", detail=str(exc)
            ).model_dump(),
        )

    scenario_id = req.scenario_id
    logger.info(
        "request_received request_id=%s scenario_id=%s num_notes=%d",
        request_id, scenario_id, len(req.operator_notes),
    )
    for i, note in enumerate(req.operator_notes):
        logger.info(
            "operator_note request_id=%s scenario_id=%s note_index=%d note=%r",
            request_id, scenario_id, i, note,
        )

    scenario_context = ScenarioContext(
        battery=BatteryContext(capacity_kwh=req.battery.capacity_kwh)
    )

    try:
        directives = await _interpreter.interpret_async(
            operator_notes=req.operator_notes,
            scenario_context=scenario_context,
            battery_capacity_kwh=req.battery.capacity_kwh,
            request_id=request_id,
            scenario_id=scenario_id,
            timeout_seconds=_settings.llm_timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - controlled 500, no stack trace leak
        logger.exception(
            "interpretation_failed request_id=%s scenario_id=%s", request_id, scenario_id
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="interpretation_failed", detail="Unable to interpret operator notes"
            ).model_dump(),
        )

    try:
        plan = optimize_schedule(req.hours, req.battery, directives)
    except OptimizerInfeasibleError as exc:
        logger.warning(
            "optimizer_infeasible request_id=%s scenario_id=%s error=%s",
            request_id, scenario_id, exc,
        )
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="infeasible_scenario", detail=str(exc)
            ).model_dump(),
        )
    except Exception:
        logger.exception(
            "optimizer_failed request_id=%s scenario_id=%s", request_id, scenario_id
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="optimizer_failed", detail="Unable to compute a schedule"
            ).model_dump(),
        )

    replay_errors = replay_and_validate(req.hours, req.battery, directives, plan)
    if replay_errors:
        logger.error(
            "final_validation_failed request_id=%s scenario_id=%s errors=%s",
            request_id, scenario_id, replay_errors,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="final_validation_failed",
                detail="Internal schedule failed post-optimization validation",
            ).model_dump(),
        )

    tariff_by_hour = [h.tariff_bdt_per_kwh for h in sorted(req.hours, key=lambda h: h.hour)]
    response = OptimizeEnergyResponse(
        scenario_id=scenario_id,
        directive_interpretation=directives,
        hourly_plan=plan,
        total_grid_kwh=round(total_grid_kwh(plan), 6),
        total_cost_bdt=round(total_cost_bdt(plan, tariff_by_hour), 6),
        peak_grid_kwh=round(peak_grid_kwh(plan), 6),
        plan_summary=_build_plan_summary(directives, plan),
    )

    logger.info(
        "request_completed request_id=%s scenario_id=%s total_cost_bdt=%.2f",
        request_id, scenario_id, response.total_cost_bdt,
    )

    return response


def _build_plan_summary(directives, plan) -> str:
    applied = [d for d in directives if d.applies]
    if not applied:
        return "Cost-minimizing 24-hour schedule with no active operator directives."
    kinds = ", ".join(sorted({d.directive_type.value for d in applied}))
    return f"Cost-minimizing 24-hour schedule honoring: {kinds}."
