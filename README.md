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

**InvoiceAgent** is an OpenEnv-compliant environment that simulates the real-world task of accounts payable (AP) invoice processing. An AI agent receives raw invoice text and must extract structured fields, validate them against business rules, cross-reference with vendor databases, purchase orders, and goods receipts, and flag discrepancies before submitting — exactly what millions of AP clerks do daily.

This environment goes beyond simple field extraction. It tests multi-step reasoning, tool use (vendor/PO/GR lookups), error detection, confidence calibration, and multi-document reconciliation — making it a meaningful benchmark for evaluating agentic AI capabilities in a production-relevant domain.

## Motivation

- **Real-world impact**: Invoice processing is a $4B+ automation market. 74% of CFOs plan AI-driven invoice extraction.
- **Agent evaluation gap**: OpenEnv has environments for games, coding, and calendars — but nothing for document processing workflows. InvoiceAgent fills this gap.
- **Multi-step complexity**: Unlike single-shot extraction, this environment requires the agent to reason across multiple steps: extract → lookup → validate → flag → submit.
- **Practical RL training**: The continuous reward signal and graduated difficulty make this environment directly usable with GRPO, TRL, and other RL training pipelines.

## Action Space

The agent can take one action per step:

| Action | Parameters | Description |
|--------|-----------|-------------|
| `extract_field` | `field_name`, `field_value`, `confidence` (optional, 0.0–1.0) | Extract a named field from the invoice |
| `lookup_vendor` | `vendor_query` | Search the vendor database by name |
| `lookup_purchase_order` | `po_number` | Look up a purchase order by number |
| `lookup_goods_receipt` | `gr_po_number` (or `po_number`) | Look up goods receipt for a PO (hard tasks) |
| `flag_discrepancy` | `flag_field`, `flag_reason` | Flag an error or inconsistency |
| `validate` | — | Run validation checks on current extractions |
| `submit` | — | Finalize and submit all extractions |

**Action schema (Pydantic):**
```python
class InvoiceAction(BaseModel):
    action_type: Literal["extract_field", "lookup_vendor", "lookup_purchase_order",
                          "lookup_goods_receipt", "flag_discrepancy", "validate", "submit"]
    field_name: Optional[str] = None
    field_value: Optional[str] = None
    confidence: Optional[float] = None   # 0.0–1.0, shapes rewards
    vendor_query: Optional[str] = None
    po_number: Optional[str] = None
    gr_po_number: Optional[str] = None   # for lookup_goods_receipt
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
| `gr_lookup_result` | Dict or None | Results from goods receipt lookup |
| `validation_errors` | List[str] or None | Errors found during validation |
| `validation_warnings` | List[str] or None | Warnings from validation |
| `flagged_discrepancies` | List[Dict] | Discrepancies flagged so far |
| `fields_extracted` / `fields_remaining` | int | Progress tracking |
| `current_step` / `max_steps` | int | Step counter |

## Invoice Templates

Invoices are generated from **4 distinct templates**, selected deterministically by seed:

| Template | Key Characteristics |
|----------|-------------------|
| **Standard** | Clean format, PO reference, straightforward fields |
| **Consulting** | Service-based line items, no PO reference, hourly rates |
| **Noisy** | Abbreviated vendor names, numeric dates, rush order notes, remit-to addresses |
| **Detailed** | Discount lines, shipping costs, net amount calculations |

Template selection: `seed % 4` maps to standard/consulting/noisy/detailed. Hard tasks always use standard or noisy (never consulting, since PO matching is required).

## Tasks

### Task 1: Clean Invoice Extraction (Easy) — 25 steps
- **Objective**: Extract 7 fields from a well-formatted invoice
- **Difficulty**: Solvable by pattern matching or a basic LLM
- **No errors to detect** — just extract and submit
- **Expected baseline score**: 0.70 – 0.90

### Task 2: Invoice with Validation Errors (Medium) — 25 steps
- **Objective**: Extract 11 fields AND detect 2–3 deliberate errors
- **Errors include**: math miscalculations, tax rate mismatches, vendor name inconsistencies
- **Requires**: `lookup_vendor` for cross-referencing, `validate` for math checks, `flag_discrepancy` for each error
- **Grading**: 50% field accuracy + 40% discrepancy F1 + 10% efficiency + calibration bonus (up to 5%)
- **Expected baseline score**: 0.30 – 0.55

### Task 3: Multi-Document Reconciliation (Hard) — 30 steps
- **Objective**: Extract 12+ fields, reconcile invoice against PO **and** goods receipts (3-way match), detect subtle discrepancies
- **3-way matching**: Invoice line items vs PO quantities vs goods receipt (received qty, condition, dates)
- **Challenges**: similar vendor names, line item differences from PO, unauthorized charges, duplicate invoices, quantity shortfalls, damaged goods
- **Grading**: 25% field accuracy + 30% discrepancy F1 + 25% three-way match + 20% critical flags − false positive penalty + calibration bonus (up to 5%)
- **Expected baseline score**: 0.10 – 0.30

## Confidence Scoring

The agent can optionally provide a `confidence` value (0.0–1.0) with `extract_field` actions:

| Scenario | Reward |
|----------|--------|
| Confidence + correct (exact) | `0.05 + 0.15 × confidence` |
| Confidence + correct (partial) | `0.02 + 0.06 × confidence` |
| Confidence + wrong | `-0.03 - 0.12 × confidence` |
| No confidence + correct | +0.10 (unchanged) |
| No confidence + wrong | -0.05 (unchanged) |

Well-calibrated confidence earns a **calibration bonus** (up to 0.05) in medium and hard graders. Overconfident-but-wrong agents are penalized. Agents that don't use confidence are unaffected — full backward compatibility.

## Reward Function

| Action Result | Reward |
|--------------|--------|
| Correct field extraction | +0.10 (or confidence-shaped) |
| Partial match | +0.03 (or confidence-shaped) |
| Wrong extraction | -0.05 (or confidence-shaped) |
| Vendor/PO/GR found | +0.05 |
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
pip install -e ".[dev]"

# Run the server
uvicorn invoice_agent.server.app:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v
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
| `/step/{session_id}` | POST | Take a step (accepts empty body) |
| `/state/{session_id}` | GET | Get current state |
| `/tasks` | GET | List tasks and action schema |
| `/baseline` | POST | Run heuristic baseline on all tasks |
| `/grader` | POST | Run grader for a task |
| `/health` | GET | Health check |

All POST endpoints accept empty bodies and will not error.

## Baseline Scores

| Task | Heuristic Baseline | LLM Baseline (estimated) |
|------|-------------------|-------------------------|
| Easy | ~0.75 | ~0.85 |
| Medium | ~0.30 | ~0.55 |
| Hard | ~0.10 | ~0.30 |

## Architecture

```
invoice_agent/
├── __init__.py
├── models.py                    # Pydantic models (Action, Observation, State)
├── client.py                    # WebSocket client
├── data/
│   ├── invoice_templates.py     # 4 invoice template formats, procedural generation
│   ├── vendor_database.py       # Vendor DB generation & fuzzy search
│   ├── purchase_orders.py       # PO generation & duplicate detection
│   └── goods_receipts.py        # Goods receipt generation (3-way matching)
├── graders/
│   └── __init__.py              # Deterministic graders + calibration scoring
└── server/
    ├── app.py                   # FastAPI + WebSocket + HTTP endpoints
    └── invoice_environment.py   # Core environment logic

tests/
├── test_confidence.py           # Confidence scoring & calibration
├── test_endpoints.py            # HTTP endpoint integration tests
├── test_environment.py          # Core environment lifecycle
├── test_graders.py              # Grader determinism, range, variation
├── test_templates.py            # 4 template formats & cross-template grading
└── test_three_way_match.py      # GR generation, lookup, 3-way grading
```

## Test Suite

113 tests covering:
- **Environment lifecycle**: reset, step, submit, max steps, seed reproducibility
- **All 4 templates**: standard, consulting, noisy, detailed — selection, fields, grading
- **3-way matching**: goods receipt generation, lookup, GR actions, hard grader
- **Confidence scoring**: reward shaping, calibration, backward compatibility
- **Graders**: determinism, variation, range [0,1], empty/perfect submissions
- **HTTP endpoints**: /health, /tasks, /reset, /step, /baseline, /grader (all with empty bodies)

```bash
pytest tests/ -v   # runs all 113 tests
```

## License

MIT
