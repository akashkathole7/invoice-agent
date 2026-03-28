"""Baseline inference script for InvoiceAgent OpenEnv environment.

Uses OpenAI Client with environment variables:
  - API_BASE_URL: The API endpoint for the LLM
  - MODEL_NAME: The model identifier to use for inference
  - HF_TOKEN: Your Hugging Face / API key

This script runs the LLM agent on all 3 tasks and reports scores.
Must complete in under 20 minutes on vcpu=2, memory=8gb.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
from openai import OpenAI

# --- Configuration from environment variables ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are an expert accounts payable agent. You process invoices by:
1. Reading the invoice text carefully
2. Extracting required fields one at a time using extract_field actions
3. Looking up vendors and purchase orders for cross-referencing
4. Validating the extracted data
5. Flagging any discrepancies found
6. Submitting when complete

You respond with a single JSON action object. Valid action types:
- {"action_type": "extract_field", "field_name": "<name>", "field_value": "<value>"}
- {"action_type": "lookup_vendor", "vendor_query": "<search string>"}
- {"action_type": "lookup_purchase_order", "po_number": "<PO number>"}
- {"action_type": "flag_discrepancy", "flag_field": "<field>", "flag_reason": "<reason>"}
- {"action_type": "validate"}
- {"action_type": "submit"}

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
        # Try to find JSON in the response
        import re
        match = re.search(r"\{[^{}]+\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    # Fallback: submit
    return {"action_type": "submit"}


def run_episode(task_id: str, seed: int = 42) -> float:
    """Run a single episode using the LLM agent. Returns grader score."""
    # Reset environment via HTTP
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": seed},
    )
    data = resp.json()
    obs = data["observation"]
    session_id = obs.get("session_id") or data.get("info", {}).get("session_id", "")
    done = data["done"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    step_count = 0

    while not done and step_count < 25:
        user_prompt = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=300,
                temperature=0.1,
            )
            assistant_msg = response.choices[0].message.content or '{"action_type": "submit"}'
        except Exception as e:
            print(f"  LLM call failed: {e}")
            assistant_msg = '{"action_type": "submit"}'

        messages.append({"role": "assistant", "content": assistant_msg})
        action = parse_action(assistant_msg)

        # Step the environment
        try:
            step_resp = requests.post(
                f"{ENV_URL}/step/{session_id}",
                json={"action": action},
            )
            step_data = step_resp.json()
        except Exception as e:
            print(f"  Step failed: {e}")
            break

        if "error" in step_data:
            print(f"  Env error: {step_data['error']}")
            break

        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        info = step_data.get("info", {})
        step_count += 1

        print(f"  Step {step_count}: {action.get('action_type', '?')} -> reward={reward:.3f}")

        if done:
            score = info.get("grader_score", 0.0)
            print(f"  Episode done. Grader score: {score:.4f}")
            return score

        # Keep message history manageable
        if len(messages) > 20:
            messages = [messages[0]] + messages[-10:]

    return 0.0


def main() -> None:
    """Run baseline inference on all 3 tasks."""
    print("=" * 60)
    print("InvoiceAgent Baseline Inference")
    print(f"API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_URL}")
    print("=" * 60)

    start_time = time.time()
    results: Dict[str, float] = {}

    for task_id in ["easy", "medium", "hard"]:
        print(f"\n--- Task: {task_id} (seed=42) ---")
        score = run_episode(task_id, seed=42)
        results[task_id] = score
        print(f"  Final score: {score:.4f}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    for task_id, score in results.items():
        print(f"  {task_id:>8}: {score:.4f}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 60)

    # Output as JSON for automated parsing
    print(json.dumps({"baseline_scores": results, "elapsed_seconds": round(elapsed, 1)}))


if __name__ == "__main__":
    main()
