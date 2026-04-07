"""InvoiceAgent inference script — matches official OpenEnv format."""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# --- Environment variables ---
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL")
SPACE_URL = os.getenv("SPACE_URL")
# CRITICAL: Evaluator injects API_BASE_URL and API_KEY.
# Read API_KEY first (evaluator's variable), fall back to HF_TOKEN only for local testing.
# Use os.environ.get with None check — NOT the `or` operator (which treats "" as falsy).
_api_key = os.environ.get("API_KEY")
if _api_key is None:
    _api_key = os.environ.get("HF_TOKEN", "")
API_KEY = _api_key

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "invoice_agent"
MAX_STEPS = {"easy": 25, "medium": 25, "hard": 30}
SUCCESS_THRESHOLD = 0.1
_container_id = None

client = None  # Created inside main() after env vars are confirmed

# --- Structured logging (exact match to official sample) ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- Environment setup (handles Docker, URL, or Space) ---

def setup_environment():
    """Determine how to connect to the environment. Priority:
    1. ENV_URL already set (evaluator started the container)
    2. IMAGE_NAME set (we start the container ourselves)
    3. SPACE_URL set (connect to live HF Space)
    4. Default to localhost:8000
    """
    global ENV_URL, _container_id

    if ENV_URL:
        print(f"[DEBUG] Using existing server at {ENV_URL}", flush=True)
        return

    if IMAGE_NAME:
        print(f"[DEBUG] Starting container from image: {IMAGE_NAME}", flush=True)
        try:
            _container_id = subprocess.check_output([
                "docker", "run", "-d", "--rm",
                "-p", "8000:8000",
                IMAGE_NAME
            ]).decode().strip()
            ENV_URL = "http://localhost:8000"

            # Wait for container to be ready (max 30 seconds)
            for i in range(30):
                try:
                    r = requests.get(f"{ENV_URL}/health", timeout=2)
                    if r.status_code == 200:
                        print(f"[DEBUG] Container ready after {i+1}s", flush=True)
                        return
                except Exception:
                    pass
                time.sleep(1)
            print(f"[DEBUG] Container started but health check timed out", flush=True)
            return
        except Exception as e:
            print(f"[DEBUG] Docker start failed: {e}, falling back", flush=True)

    if SPACE_URL:
        ENV_URL = SPACE_URL.rstrip("/")
        print(f"[DEBUG] Using HF Space at {ENV_URL}", flush=True)
        return

    ENV_URL = "http://localhost:8000"
    print(f"[DEBUG] Defaulting to {ENV_URL}", flush=True)

def cleanup_container():
    """Stop the container if we started one."""
    global _container_id
    if _container_id:
        try:
            subprocess.run(["docker", "stop", _container_id],
                           capture_output=True, timeout=10)
            print(f"[DEBUG] Container stopped", flush=True)
        except Exception:
            pass

atexit.register(cleanup_container)

# --- LLM interaction ---

SYSTEM_PROMPT = """You are an expert accounts payable agent. You process invoices by:
1. Reading the invoice text carefully
2. Extracting required fields one at a time using extract_field actions
3. Looking up vendors and purchase orders for cross-referencing
4. For hard tasks: looking up goods receipts for 3-way matching (invoice vs PO vs GR)
5. Validating the extracted data
6. Flagging any discrepancies found (math errors, vendor mismatches, quantity shortfalls, damaged goods)
7. Submitting when complete

You respond with a single JSON action object. Valid action types:
- {"action_type": "extract_field", "field_name": "<name>", "field_value": "<value>", "confidence": 0.9}
- {"action_type": "lookup_vendor", "vendor_query": "<search string>"}
- {"action_type": "lookup_purchase_order", "po_number": "<PO number>"}
- {"action_type": "lookup_goods_receipt", "gr_po_number": "<PO number>"}
- {"action_type": "flag_discrepancy", "flag_field": "<field>", "flag_reason": "<reason>"}
- {"action_type": "validate"}
- {"action_type": "submit"}

The "confidence" field (0.0-1.0) is optional for extract_field. High confidence + correct = bigger reward. High confidence + wrong = bigger penalty. If unsure, omit it or use a low value.

IMPORTANT: Respond ONLY with a single valid JSON action. No explanation text."""


def build_user_prompt(obs: Dict[str, Any]) -> str:
    """Build the user prompt from the observation."""
    parts = [
        f"INVOICE TEXT:\n{obs['invoice_text']}\n",
        f"REQUIRED FIELDS: {', '.join(obs['required_fields'])}",
        f"EXTRACTED SO FAR: {json.dumps(obs['extracted_fields'])}",
        f"FIELDS REMAINING: {obs['fields_remaining']}",
        f"STEP: {obs['current_step']}/{obs['max_steps']}",
        f"LAST RESULT: {obs['last_action_result']}",
    ]

    if obs.get("vendor_lookup_result"):
        parts.append(f"VENDOR LOOKUP: {json.dumps(obs['vendor_lookup_result'])}")
    if obs.get("po_lookup_result"):
        parts.append(f"PO LOOKUP: {json.dumps(obs['po_lookup_result'])}")
    if obs.get("gr_lookup_result"):
        parts.append(f"GOODS RECEIPT: {json.dumps(obs['gr_lookup_result'])}")
    if obs.get("validation_errors"):
        parts.append(f"VALIDATION ERRORS: {obs['validation_errors']}")
    if obs.get("validation_warnings"):
        parts.append(f"VALIDATION WARNINGS: {obs['validation_warnings']}")
    if obs.get("flagged_discrepancies"):
        parts.append(f"FLAGGED: {json.dumps(obs['flagged_discrepancies'])}")

    parts.append("\nRespond with your next action as a JSON object:")
    return "\n".join(parts)


def parse_action(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response into an action dict."""
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]

    try:
        action = json.loads(text)
        return action
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]+\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"action_type": "submit"}

# --- Episode runner ---

def run_episode(task_id: str, seed: int = 42) -> float:
    global client  # Use the client created in main()
    rewards_list: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    max_steps = MAX_STEPS.get(task_id, 25)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed})
        data = resp.json()
        obs = data["observation"]
        session_id = obs.get("session_id") or data.get("info", {}).get("session_id", "")
        done = data["done"]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, max_steps + 1):
            if done:
                break

            # LLM generates action
            user_prompt = build_user_prompt(obs)
            messages.append({"role": "user", "content": user_prompt})

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME, messages=messages,
                    max_tokens=300, temperature=0.1,
                )
                assistant_msg = response.choices[0].message.content or '{"action_type": "submit"}'
            except Exception as e:
                print(f"[DEBUG] LLM CALL FAILED: {type(e).__name__}: {e}", flush=True)
                assistant_msg = '{"action_type": "submit"}'

            messages.append({"role": "assistant", "content": assistant_msg})
            action = parse_action(assistant_msg)

            # Step environment
            try:
                step_resp = requests.post(
                    f"{ENV_URL}/step/{session_id}", json={"action": action}
                )
                step_data = step_resp.json()
            except Exception as e:
                print(f"[DEBUG] Step error: {e}", flush=True)
                break

            obs = step_data.get("observation", {})
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            info = step_data.get("info", {})

            # Build action string for log
            action_str = action.get("action_type", "unknown")
            param = (action.get("field_name") or action.get("vendor_query")
                     or action.get("po_number") or action.get("gr_po_number") or "")
            if param:
                action_str += f"({param})"

            # Detect error from observation
            last_result = obs.get("last_action_result", "")
            error = last_result if isinstance(last_result, str) and last_result.startswith("✗") else None

            rewards_list.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                score = info.get("grader_score", 0.0)
                success = score >= SUCCESS_THRESHOLD
                break

            # Keep history manageable
            if len(messages) > 20:
                messages = [messages[0]] + messages[-10:]

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards_list)

    return score

# --- Main ---

def main():
    global client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Debug: confirm which credentials are being used
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] API_KEY={'set (' + API_KEY[:8] + '...)' if API_KEY else 'MISSING'}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    setup_environment()

    results = {}
    for task_id in ["easy", "medium", "hard"]:
        score = run_episode(task_id, seed=42)
        results[task_id] = score

    print(f"[DEBUG] All scores: {results}", flush=True)
    cleanup_container()

if __name__ == "__main__":
    main()
