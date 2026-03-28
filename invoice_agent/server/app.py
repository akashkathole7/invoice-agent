"""FastAPI application for InvoiceAgent OpenEnv environment."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from invoice_agent.models import InvoiceAction, InvoiceObservation
from invoice_agent.server.invoice_environment import InvoiceEnvironment
from invoice_agent.graders import GRADERS
from invoice_agent.data.invoice_templates import get_required_fields

app = FastAPI(
    title="InvoiceAgent",
    description="OpenEnv environment for automated invoice processing — extract fields, validate, cross-reference, and flag discrepancies.",
    version="1.0.0",
)

# Store active environments per session
_sessions: Dict[str, InvoiceEnvironment] = {}


# --- WebSocket Endpoint (primary OpenEnv interface) ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    env = InvoiceEnvironment()
    _sessions[session_id] = env

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            method = msg.get("method", "")
            params = msg.get("params", {})

            if method == "reset":
                task_id = params.get("task_id", "easy")
                seed = params.get("seed", 42)
                obs, reward, done, info = env.reset(task_id=task_id, seed=seed)
                await ws.send_text(json.dumps({
                    "observation": obs.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info,
                }))

            elif method == "step":
                action = InvoiceAction(**params)
                obs, reward, done, info = env.step(action)
                await ws.send_text(json.dumps({
                    "observation": obs.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info,
                }))

            elif method == "state":
                state = env.state()
                await ws.send_text(json.dumps({"state": state}))

            else:
                await ws.send_text(json.dumps({
                    "error": f"Unknown method: {method}",
                }))

    except WebSocketDisconnect:
        pass
    finally:
        _sessions.pop(session_id, None)


# --- HTTP Endpoints ---

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    action: InvoiceAction


# HTTP fallback for reset
@app.post("/reset")
async def http_reset(req: ResetRequest) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())[:8]
    env = InvoiceEnvironment()
    _sessions[session_id] = env
    obs, reward, done, info = env.reset(task_id=req.task_id, seed=req.seed)
    info["session_id"] = session_id
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.post("/step/{session_id}")
async def http_step(session_id: str, req: StepRequest) -> Dict[str, Any]:
    env = _sessions.get(session_id)
    if env is None:
        return {"error": "Session not found. Call /reset first."}
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state/{session_id}")
async def http_state(session_id: str) -> Dict[str, Any]:
    env = _sessions.get(session_id)
    if env is None:
        return {"error": "Session not found."}
    return {"state": env.state()}


# --- Required Competition Endpoints ---

@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    """Return list of tasks and the action schema."""
    return {
        "tasks": [
            {
                "task_id": "easy",
                "name": "Clean Invoice Extraction",
                "description": "Extract all fields from a simple, well-formatted invoice. No errors to detect.",
                "difficulty": "easy",
                "required_fields": get_required_fields("easy"),
                "max_steps": 25,
            },
            {
                "task_id": "medium",
                "name": "Invoice with Validation Errors",
                "description": "Extract fields AND detect 2-3 deliberate errors (math, tax, vendor mismatch).",
                "difficulty": "medium",
                "required_fields": get_required_fields("medium"),
                "max_steps": 25,
            },
            {
                "task_id": "hard",
                "name": "Multi-Document Reconciliation",
                "description": "Extract, cross-reference with PO, reconcile line items, detect unauthorized charges and duplicates.",
                "difficulty": "hard",
                "required_fields": get_required_fields("hard"),
                "max_steps": 25,
            },
        ],
        "action_schema": InvoiceAction.model_json_schema(),
        "observation_schema": InvoiceObservation.model_json_schema(),
    }


@app.post("/baseline")
async def run_baseline() -> Dict[str, Any]:
    """Run baseline inference on all 3 tasks and return scores."""
    results: Dict[str, Any] = {}

    for task_id in ["easy", "medium", "hard"]:
        env = InvoiceEnvironment()
        obs, _, _, _ = env.reset(task_id=task_id, seed=42)

        # Simple heuristic baseline: extract fields by pattern matching
        score = _run_heuristic_baseline(env, obs, task_id)
        results[task_id] = {
            "score": score,
            "seed": 42,
        }

    return {"baseline_scores": results}


@app.post("/grader")
async def run_grader(req: ResetRequest) -> Dict[str, Any]:
    """Run grader on a completed episode. Uses default seed for reproducibility."""
    env = InvoiceEnvironment()
    env.reset(task_id=req.task_id, seed=req.seed)
    grader_fn = GRADERS.get(req.task_id, GRADERS["easy"])
    score = grader_fn(
        env._state.extracted_fields,
        env._state.flagged_discrepancies,
        env._state.ground_truth_fields,
        env._state.ground_truth_discrepancies,
    )
    return {"task_id": req.task_id, "seed": req.seed, "grader_score": score}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "environment": "InvoiceAgent", "version": "1.0.0"}


# --- Heuristic Baseline ---

def _run_heuristic_baseline(
    env: InvoiceEnvironment, obs: InvoiceObservation, task_id: str
) -> float:
    """Simple heuristic baseline that extracts fields via pattern matching."""
    import re

    text = obs.invoice_text

    # Extract common fields with regex
    patterns = {
        "invoice_number": r"INVOICE\s*#?\s*(INV-[\w-]+)",
        "invoice_date": r"Date:\s*(.+?)(?:\n|$)",
        "vendor_name": r"From:\s*(.+?)(?:\n|$)",
        "vendor_address": r"From:.*?\n\s+(.+?)(?:\n|$)",
        "bill_to": r"Bill To:\s*(.+?)(?:\n|$)",
        "po_number": r"PO Reference:\s*(PO-[\w-]+)",
        "subtotal": r"Subtotal:\s*\$?([\d,]+\.?\d*)",
        "tax_amount": r"Tax\s*\([\d.]+%\):\s*\$?([\d,]+\.?\d*)",
        "total_amount": r"TOTAL DUE:\s*\$?([\d,]+\.?\d*)",
        "payment_terms": r"Payment Terms:\s*(.+?)(?:\n|$)",
        "due_date": r"Due Date:\s*(.+?)(?:\n|$)",
    }

    for field_name, pattern in patterns.items():
        if field_name in obs.required_fields:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if field_name in ("subtotal", "tax_amount", "total_amount"):
                    value = f"${value}"
                action = InvoiceAction(
                    action_type="extract_field",
                    field_name=field_name,
                    field_value=value,
                )
                obs, _, done, info = env.step(action)
                if done:
                    return info.get("grader_score", 0.0)

    # Count line items for hard task
    if task_id == "hard" and "line_item_count" in obs.required_fields:
        lines = [l for l in text.split("\n") if l.strip().startswith("$") or re.match(r"^\w.+\d+\s+\$", l.strip())]
        action = InvoiceAction(
            action_type="extract_field",
            field_name="line_item_count",
            field_value=str(len(lines)),
        )
        obs, _, done, info = env.step(action)
        if done:
            return info.get("grader_score", 0.0)

    # Submit
    action = InvoiceAction(action_type="submit")
    obs, _, done, info = env.step(action)
    return info.get("grader_score", 0.0)
