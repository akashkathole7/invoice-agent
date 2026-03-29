"""FastAPI application for InvoiceAgent OpenEnv environment."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Request
from starlette.routing import Route

from openenv.core.env_server import create_app

from invoice_agent.models import InvoiceAction, InvoiceObservation
from invoice_agent.server.invoice_environment import InvoiceEnvironment, _SESSIONS
from invoice_agent.graders import GRADERS
from invoice_agent.data.invoice_templates import get_required_fields

# Create the standard OpenEnv FastAPI app (provides /reset, /state, /health, /schema, /ws, /mcp)
app = create_app(InvoiceEnvironment, InvoiceAction, InvoiceObservation)

# Remove the standard /step route so we can replace it with one that tolerates empty body
app.router.routes = [
    r for r in app.router.routes
    if not (isinstance(r, Route) and r.path == "/step")
]


# --- /step — tolerates empty body, handles both stateless and session-based calls ---

@app.post("/step")
@app.post("/step/{session_id}")
async def step(request: Request, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Execute an action. Accepts empty body (defaults to submit action)."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Parse action from body
    action_data = body.get("action", body) or {}
    action_type = action_data.get("action_type", "submit")
    # Parse optional confidence (float or None)
    raw_conf = action_data.get("confidence")
    conf_val = None
    if raw_conf is not None:
        try:
            conf_val = float(raw_conf)
        except (TypeError, ValueError):
            conf_val = None

    try:
        action = InvoiceAction(
            action_type=action_type,
            field_name=action_data.get("field_name"),
            field_value=action_data.get("field_value"),
            confidence=conf_val,
            vendor_query=action_data.get("vendor_query"),
            po_number=action_data.get("po_number"),
            gr_po_number=action_data.get("gr_po_number"),
            flag_field=action_data.get("flag_field"),
            flag_reason=action_data.get("flag_reason"),
        )
    except Exception:
        action = InvoiceAction(action_type="submit")

    # Look up or create env for this session
    env = _SESSIONS.get(session_id) if session_id else None
    if env is None:
        # Stateless: create fresh env (auto-resets on first step)
        env = InvoiceEnvironment()
        env.reset()

    obs = env.step(action)
    info: Dict[str, Any] = {}
    if obs.done:
        info["grader_score"] = obs.grader_score
        info["termination"] = "submitted"

    return {
        "observation": obs.model_dump(
            exclude={"reward", "done", "metadata", "grader_score", "session_id"}
        ),
        "reward": obs.reward,
        "done": obs.done,
        "info": info,
    }


# --- /tasks ---

@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": "easy",
                "name": "Clean Invoice Extraction",
                "description": "Extract all fields from a simple, well-formatted invoice.",
                "difficulty": "easy",
                "required_fields": get_required_fields("easy"),
                "max_steps": 25,
            },
            {
                "task_id": "medium",
                "name": "Invoice with Validation Errors",
                "description": "Extract fields AND detect 2-3 deliberate errors.",
                "difficulty": "medium",
                "required_fields": get_required_fields("medium"),
                "max_steps": 25,
            },
            {
                "task_id": "hard",
                "name": "Multi-Document Reconciliation",
                "description": "Extract, cross-reference with PO and goods receipts (3-way match), reconcile line items.",
                "difficulty": "hard",
                "required_fields": get_required_fields("hard"),
                "max_steps": 30,
            },
        ],
        "action_schema": InvoiceAction.model_json_schema(),
        "observation_schema": InvoiceObservation.model_json_schema(),
    }


# --- /baseline — no body required ---

@app.post("/baseline")
async def run_baseline(request: Request) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for task_id in ["easy", "medium", "hard"]:
        env = InvoiceEnvironment()
        obs = env.reset(task_id=task_id, seed=42)
        score = _run_heuristic_baseline(env, obs, task_id)
        results[task_id] = {"score": score, "seed": 42}
    return {"baseline_scores": results}


# --- /grader — tolerates empty body ---

@app.post("/grader")
async def run_grader(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id = body.get("task_id", "easy") if body else "easy"
    seed = body.get("seed", 42) if body else 42

    env = InvoiceEnvironment()
    env.reset(task_id=task_id, seed=seed)
    grader_fn = GRADERS.get(task_id, GRADERS["easy"])
    score = grader_fn(
        env._state.extracted_fields,
        env._state.flagged_discrepancies,
        env._state.ground_truth_fields,
        env._state.ground_truth_discrepancies,
    )
    return {"task_id": task_id, "seed": seed, "grader_score": score}


# --- Heuristic baseline helper ---

def _run_heuristic_baseline(
    env: InvoiceEnvironment, obs: InvoiceObservation, task_id: str
) -> float:
    import re

    text = obs.invoice_text
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
        "discount_total": r"Discount Total:\s*\$?([\d,]+\.?\d*)",
        "shipping_cost": r"Shipping & Handling:\s*\$?([\d,]+\.?\d*)",
        "net_amount": r"Net Amount:\s*\$?([\d,]+\.?\d*)",
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
                obs = env.step(action)
                if obs.done:
                    return obs.grader_score

    if task_id == "hard" and "line_item_count" in obs.required_fields:
        lines = [
            l for l in text.split("\n")
            if re.match(r"^\w.+\d+\s+\$", l.strip())
        ]
        action = InvoiceAction(
            action_type="extract_field",
            field_name="line_item_count",
            field_value=str(len(lines)),
        )
        obs = env.step(action)
        if obs.done:
            return obs.grader_score

    action = InvoiceAction(action_type="submit")
    obs = env.step(action)
    return obs.grader_score
