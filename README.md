---
title: InvoiceAgent
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - invoice
  - data-extraction
  - accounts-payable
---

# InvoiceAgent — OpenEnv Environment for Automated Invoice Processing

## Overview

**InvoiceAgent** is an OpenEnv-compliant environment that simulates the real-world task of accounts payable (AP) invoice processing. An AI agent receives raw invoice text and must extract structured fields, validate them against business rules, cross-reference with vendor databases and purchase orders, and flag discrepancies before submitting — exactly what millions of AP clerks do daily.

This environment goes beyond simple field extraction. It tests multi-step reasoning, tool use (vendor/PO lookups), error detection, and document reconciliation — making it a meaningful benchmark for evaluating agentic AI capabilities in a production-relevant domain.

## Motivation

- **Real-world impact**: Invoice processing is a $4B+ automation market. 74% of CFOs plan AI-driven invoice extraction.
- **Agent evaluation gap**: OpenEnv has environments for games, coding, and calendars — but nothing for document processing workflows. InvoiceAgent fills this gap.
- **Multi-step complexity**: Unlike single-shot extraction, this environment requires the agent to reason across multiple steps: extract → lookup → validate → flag → submit.
- **Practical RL training**: The continuous reward signal and graduated difficulty make this environment directly usable with GRPO, TRL, and other RL training pipelines.

## Action Space

The agent can take one action per step:

| Action | Parameters | Description |
|--------|-----------|-------------|
| `extract_field` | `field_name`, `field_value` | Extract a named field from the invoice |
| `lookup_vendor` | `vendor_query` | Search the vendor database by name |
| `lookup_purchase_order` | `po_number` | Look up a purchase order by number |
| `flag_discrepancy` | `flag_field`, `flag_reason` | Flag an error or inconsistency |
| `validate` | — | Run validation checks on current extractions |
| `submit` | — | Finalize and submit all extractions |

**Action schema (Pydantic):**
```python
class InvoiceAction(BaseModel):
    action_type: Literal["extract_field", "lookup_vendor", "lookup_purchase_order",
                          "flag_discrepancy", "validate", "submit"]
    field_name: Optional[str] = None
    field_value: Optional[str] = None
    vendor_query: Optional[str] = None
    po_number: Optional[str] = None
    flag_field: Optional[str] = None
    flag_reason: Optional[str] = None
```

## Observation Space

After each step, the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `invoice_text` | str | Raw invoice text to process |
| `extracted_fields` | Dict | Fields extracted so far |
| `required_fields` | List[str] | Checklist of required fields |
| `last_action_result` | str | Success/failure message from last action |
| `vendor_lookup_result` | Dict or None | Results from vendor database query |
| `po_lookup_result` | Dict or None | Results from PO lookup |
| `validation_errors` | List[str] or None | Errors found during validation |
| `flagged_discrepancies` | List[Dict] | Discrepancies flagged so far |
| `fields_extracted` / `fields_remaining` | int | Progress tracking |
| `current_step` / `max_steps` | int | Step counter (max 25) |

## Tasks

### Task 1: Clean Invoice Extraction (Easy)
- **Objective**: Extract 7 fields from a simple, well-formatted invoice
- **Difficulty**: Solvable by pattern matching or a basic LLM
- **No errors to detect** — just extract and submit
- **Expected baseline score**: 0.70 – 0.90

### Task 2: Invoice with Validation Errors (Medium)
- **Objective**: Extract 11 fields AND detect 2-3 deliberate errors
- **Errors include**: math miscalculations, tax rate mismatches, vendor name inconsistencies
- **Requires**: lookup_vendor() for cross-referencing, validate() for math checks, flag_discrepancy() for each error
- **Expected baseline score**: 0.40 – 0.60

### Task 3: Multi-Document Reconciliation (Hard)
- **Objective**: Extract 12+ fields, reconcile invoice against purchase order, detect subtle discrepancies
- **Challenges**: similar vendor names requiring disambiguation, line items differing from PO, unauthorized charges, duplicate invoice detection, mixed tax rules
- **Why frontier models struggle**: requires multi-source correlation, numerical reconciliation with rounding ambiguity, distinguishing legitimate changes from errors
- **Expected baseline score**: 0.20 – 0.35

## Reward Function

The environment provides step-level rewards (not sparse binary):

| Action Result | Reward |
|--------------|--------|
| Correct field extraction | +0.10 |
| Partial match | +0.03 |
| Wrong extraction | -0.05 |
| Vendor/PO found | +0.05 |
| Correct discrepancy flag | +0.15 |
| False discrepancy flag | -0.10 |
| Submit with high accuracy | +0.20 |
| Timeout (max steps) | -0.15 |

## Setup

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/invoice-agent.git
cd invoice-agent

# Install dependencies
pip install -e .

# Run the server
uvicorn invoice_agent.server.app:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build
docker build -t invoice-agent .

# Run
docker run -p 8000:8000 invoice-agent

# Verify
curl http://localhost:8000/health
curl http://localhost:8000/tasks
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": "easy", "seed": 42}'
```

### Running Inference
```bash
# Set environment variables
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_URL="http://localhost:8000"

# Run baseline inference
python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WebSocket | Persistent session (primary OpenEnv interface) |
| `/reset` | POST | Reset environment (HTTP fallback) |
| `/step/{session_id}` | POST | Take a step |
| `/state/{session_id}` | GET | Get current state |
| `/tasks` | GET | List tasks and action schema |
| `/baseline` | POST | Run heuristic baseline on all tasks |
| `/grader` | POST | Run grader for a task |
| `/health` | GET | Health check |

## Baseline Scores

| Task | Heuristic Baseline | LLM Baseline (estimated) |
|------|-------------------|-------------------------|
| Easy | ~0.75 | ~0.85 |
| Medium | ~0.30 | ~0.55 |
| Hard | ~0.10 | ~0.30 |

## Architecture

```
invoice_agent/
├── __init__.py              # Exports
├── models.py                # Pydantic models (Action, Observation, State)
├── client.py                # WebSocket client
├── inference.py             # LLM baseline inference script
├── openenv.yaml             # OpenEnv manifest
├── data/
│   ├── invoice_templates.py # Procedural invoice generation
│   ├── vendor_database.py   # Vendor DB generation
│   └── purchase_orders.py   # PO generation
├── graders/
│   └── __init__.py          # Deterministic graders for all 3 tasks
└── server/
    ├── app.py               # FastAPI + WebSocket + HTTP endpoints
    ├── invoice_environment.py  # Core environment logic
    ├── Dockerfile
    └── requirements.txt
```

## License

MIT
