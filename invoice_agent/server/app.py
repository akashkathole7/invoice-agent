"""FastAPI application for InvoiceAgent OpenEnv environment."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel

from openenv.core.env_server import create_app

from invoice_agent.models import InvoiceAction, InvoiceObservation
from invoice_agent.server.invoice_environment import InvoiceEnvironment, _SESSIONS
from invoice_agent.graders import GRADERS
from invoice_agent.data.invoice_templates import get_required_fields

# Create the standard OpenEnv FastAPI app
app = create_app(InvoiceEnvironment, InvoiceAction, InvoiceObservation)


# --- Custom session-based step (for stateful inference.py) ---

class SessionStepRequest(BaseModel):
    action: InvoiceAction


@app.post("/step/{session_id}")
async def session_step(session_id: str, req: SessionStepRequest) -> Dict[str, Any]:
    """Stateful step using session ID returned by /reset."""
    env = _SESSIONS.get(session_id)
    if env is None:
        return {"error": "Session not found. Call /reset first."}

    obs = env.step(req.action)
    info: Dict[str, Any] = {}
    if obs.done:
        info["grader_score"] = obs.grader_score
        info["termination"] = "submitted"

    return {
        "observation": obs.model_dump(exclude={"reward", "done", "metadata", "grader_score", "session_id"}),
        "reward": obs.reward,
        "done": obs.done,
        "info": info,
    }


# --- Tasks endpoint ---

@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    """Return list of tasks and the action/observation schemas."""
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


# --- Baseline endpoint ---

@app.post("/baseline")
async def run_baseline() -> Dict[str, Any]:
    """Run heuristic baseline on all 3 tasks and return scores."""
    results: Dict[str, Any] = {}
    for task_id in ["easy", "medium", "hard"]:
        env = InvoiceEnvironment()
        obs = env.reset(task_id=task_id, seed=42)
        score = _run_heuristic_baseline(env, obs, task_id)
        results[task_id] = {"score": score, "seed": 42}
    return {"baseline_scores": results}


# --- Grader endpoint ---

class GraderRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


@app.post("/grader")
async def run_grader(req: GraderRequest) -> Dict[str, Any]:
    """Run grader on a fresh episode (default seed for reproducibility)."""
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
    }

    for field_name, pattern in patterns.items():
        if field_name in obs.required_fields:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if field_name in ("subtotal", "tax_amount", "total_amount"):
                    value = f"${value}"
                action = InvoiceAction(action_type="extract_field", field_name=field_name, field_value=value)
                obs = env.step(action)
                if obs.done:
                    return obs.grader_score

    if task_id == "hard" and "line_item_count" in obs.required_fields:
        lines = [l for l in text.split("\n") if re.match(r"^\w.+\d+\s+\$", l.strip())]
        action = InvoiceAction(action_type="extract_field", field_name="line_item_count", field_value=str(len(lines)))
        obs = env.step(action)
        if obs.done:
            return obs.grader_score

    action = InvoiceAction(action_type="submit")
    obs = env.step(action)
    return obs.grader_score
