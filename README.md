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

> **OpenEnv hackathon submission** · Phase 1 ✅ · Phase 2 ✅ · Submission #17

## Baseline Scores (seed=42)

| Task | Difficulty | Steps | Heuristic Baseline | LLM Baseline |
|------|-----------|-------|-------------------|--------------|
| Easy | Clean extraction | 25 | 0.79 | **0.999** |
| Medium | Validation errors | 25 | 0.47 | **0.758** |
| Hard | 3-way matching | 30 | 0.19 | **0.226** |

The gap between heuristic and LLM on medium/hard confirms the environment rewards genuine reasoning, not pattern matching.

---

## Motivation

Accounts payable (AP) invoice processing is a **$4B+ automation market** — 74% of CFOs plan AI-driven invoice workflows. AP clerks spend hours per invoice: extracting fields, cross-referencing vendor databases, matching purchase orders, reconciling goods receipts, and flagging discrepancies.

**The gap InvoiceAgent fills:** OpenEnv has environments for games, code, and calendars — nothing for document-processing workflows that require multi-step reasoning across multiple evidence sources.

InvoiceAgent models this end-to-end: extract → lookup → validate → flag → submit, with a graded reward at every step and a hard task (3-way matching) that challenges frontier models.

---

## Architecture

```
Invoice Text
     │
     ▼
┌─────────────────────────────────────────┐
│              Agent Loop                  │
│                                         │
│  extract_field ──► +0.10 (exact)        │
│  lookup_vendor ──► +0.05 (found)        │
│  lookup_po     ──► +0.05 (found)        │
│  lookup_gr     ──► +0.05 (found)        │ ◄── Hard only
│  flag_discrepancy ► +0.15 (correct)     │
│                     -0.10 (false pos)   │
│  validate      ──► +0.05 (found errors) │
│  submit        ──► +0.20 (score ≥ 0.8)  │
└─────────────────────────────────────────┘
     │
     ▼
 Grader Score ∈ (0, 1)
```

---

## Action Space

| Action | Required Parameters | Reward | Notes |
|--------|-------------------|--------|-------|
| `extract_field` | `field_name`, `field_value` | +0.10 exact / +0.03 partial / -0.05 wrong | `confidence` optional (shapes reward) |
| `lookup_vendor` | `vendor_query` | +0.05 found / +0.02 not found | Fuzzy search across vendor DB |
| `lookup_purchase_order` | `po_number` | +0.05 found / +0.01 not found | Validates PO reference |
| `lookup_goods_receipt` | `gr_po_number` | +0.05 found / +0.01 not found | Hard task only |
| `flag_discrepancy` | `flag_field`, `flag_reason` | +0.15 correct / -0.10 false positive | High stakes — penalizes hallucination |
| `validate` | — | +0.05 / +0.02 / -0.01 | Checks math and completeness |
| `submit` | — | +0.20 / +0.05 / -0.10 | Ends episode, triggers grader |

**Pydantic schema:**
```python
class InvoiceAction(Action):
    action_type: Literal["extract_field", "lookup_vendor", "lookup_purchase_order",
                          "lookup_goods_receipt", "flag_discrepancy", "validate", "submit"]
    field_name:   Optional[str]   = None
    field_value:  Optional[str]   = None
    confidence:   Optional[float] = None  # 0.0–1.0
    vendor_query: Optional[str]   = None
    po_number:    Optional[str]   = None
    gr_po_number: Optional[str]   = None
    flag_field:   Optional[str]   = None
    flag_reason:  Optional[str]   = None
```

---

## Observation Space

| Field | Type | Always Present | Description |
|-------|------|---------------|-------------|
| `invoice_text` | `str` | ✅ | Raw invoice text the agent must process |
| `required_fields` | `List[str]` | ✅ | Fields the agent must extract for this task |
| `extracted_fields` | `Dict[str,str]` | ✅ | Fields extracted so far |
| `last_action_result` | `str` | ✅ | Feedback from previous action |
| `fields_extracted` | `int` | ✅ | Progress counter |
| `fields_remaining` | `int` | ✅ | How many required fields are left |
| `current_step` / `max_steps` | `int` | ✅ | Step counter |
| `flagged_discrepancies` | `List[Dict]` | ✅ | Discrepancies flagged so far |
| `vendor_lookup_result` | `Dict` | conditional | Set after `lookup_vendor`, else `null` |
| `po_lookup_result` | `Dict` | conditional | Set after `lookup_purchase_order`, else `null` |
| `gr_lookup_result` | `Dict` | conditional | Set after `lookup_goods_receipt`, else `null` |
| `validation_errors` | `List[str]` | conditional | Set after `validate`, else `null` |
| `validation_warnings` | `List[str]` | conditional | Set after `validate`, else `null` |

---

## Tasks

### Task 1 — Easy: Clean Invoice Extraction (25 steps)

**Objective:** Extract 7 fields from a well-formatted invoice.

**Required fields:** `invoice_number`, `invoice_date`, `vendor_name`, `subtotal`, `tax_amount`, `total_amount`, `po_number`

**Grading:** `score = avg field accuracy` (exact=1.0, partial=0.5, wrong=0.0)

**What makes it solvable:** No errors injected. One document. Fields follow clear patterns.

| Heuristic | LLM |
|-----------|-----|
| 0.79 | 0.999 |

---

### Task 2 — Medium: Invoice with Validation Errors (25 steps)

**Objective:** Extract 11 fields AND detect 2–3 deliberate errors.

**Required fields:** all easy fields + `vendor_address`, `payment_terms`, `due_date`, `bill_to`

**Errors injected:** math miscalculations, tax rate mismatches, vendor name inconsistencies.

**Grading:**
```
score = 0.50 × field_accuracy
      + 0.40 × discrepancy_F1
      + 0.10 × efficiency
      + calibration_bonus (up to 0.05)
```

**What makes it hard:** Agent must look up the vendor database to confirm the vendor name, run `validate` to catch math errors, and flag exact discrepancies (false positives cost -0.10).

| Heuristic | LLM |
|-----------|-----|
| 0.47 | 0.758 |

---

### Task 3 — Hard: Multi-Document Reconciliation (30 steps)

**Objective:** Extract 12 fields, reconcile invoice against PO AND goods receipts (3-way match), detect subtle discrepancies.

**Required fields:** all medium fields + `line_item_count`

**Grading:**
```
score = 0.25 × field_accuracy
      + 0.30 × discrepancy_F1
      + 0.25 × three_way_match_score
      + 0.20 × critical_flag_score
      − false_positive_penalty (up to 0.20)
      + calibration_bonus (up to 0.05)
```

**Three-way match** checks: invoice qty vs PO qty vs goods received — shortfalls, damaged goods, unreceived items, unauthorized charges, duplicate invoices.

**Why it challenges frontier models:** Requires sequential lookups (vendor → PO → GR), reasoning across 3 documents simultaneously, and conservative flagging (every false positive is penalized).

| Heuristic | LLM |
|-----------|-----|
| 0.19 | 0.226 |

---

## Confidence Scoring (Optional Mechanic)

The agent may include `confidence: float` (0.0–1.0) on any `extract_field` action:

| Scenario | Reward |
|----------|--------|
| Confidence + exact match | `0.05 + 0.15 × confidence` |
| Confidence + partial match | `0.02 + 0.06 × confidence` |
| Confidence + wrong | `−0.03 − 0.12 × confidence` |
| No confidence + exact | +0.10 (unchanged) |
| No confidence + wrong | −0.05 (unchanged) |

Well-calibrated agents earn a **calibration bonus** (up to +0.05) in medium and hard tasks. Agents that never use confidence are unaffected — full backward compatibility.

---

## Invoice Templates

Template selection: `seed % 4` → `[standard, consulting, noisy, detailed]`. Hard tasks never use consulting (requires PO matching).

| Template | Key Characteristics |
|----------|-------------------|
| **Standard** | Clean format, PO reference, numeric fields clearly labeled |
| **Consulting** | Service line items, hourly rates, no PO reference |
| **Noisy** | Abbreviated vendor names, numeric dates, rush-order notes |
| **Detailed** | Discount lines, shipping costs, net amount calculations |

---

## Reward Design Rationale

| Action Result | Reward | Why |
|--------------|--------|-----|
| Exact field extraction | +0.10 | Primary task signal |
| Partial field match | +0.03 | Encourages partial progress |
| Wrong extraction | −0.05 | Discourages hallucination |
| Vendor/PO/GR found | +0.05 | Rewards multi-document grounding |
| Correct discrepancy flag | +0.15 | Flags harder than extraction — higher reward |
| False discrepancy flag | −0.10 | Keeps precision high, discourages over-flagging |
| Submit (score ≥ 0.8) | +0.20 | Completion bonus for high quality |
| Submit (score < 0.5) | −0.10 | Discourages early submission |
| Timeout | −0.15 | Penalizes getting stuck |

**Design tension:** flag reward (+0.15) > extraction reward (+0.10) incentivizes error detection in medium/hard without making extraction worthless. False positive penalty (−0.10) keeps precision honest.

---

## Setup

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development
```bash
git clone https://github.com/akashkathole7/invoice-agent.git
cd invoice-agent

pip install -e ".[dev]"

# Start server
uvicorn invoice_agent.server.app:app --host 0.0.0.0 --port 7860

# Run tests
pytest tests/ -v   # 113 tests
```

### Docker
```bash
docker build -t invoice-agent .
docker run -p 7860:7860 invoice-agent

# Verify
curl http://localhost:7860/health
curl http://localhost:7860/tasks
```

### Running Inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export API_KEY="your-api-key"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_URL="http://localhost:7860"

python inference.py
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns `{"status":"healthy"}` |
| `/reset` | POST | Start new episode — body: `{"task_id":"easy","seed":42}` |
| `/step/{session_id}` | POST | Take action — body: `{"action":{...}}` |
| `/state/{session_id}` | GET | Get current state |
| `/schema` | GET | Action/observation/state JSON schemas |
| `/tasks` | GET | List tasks and action schema |
| `/baseline` | POST | Run heuristic baseline on all tasks |
| `/mcp` | POST | MCP-compatible endpoint |
| `/openapi.json` | GET | OpenAPI spec |

---

## Test Suite

113 tests across 6 modules:

| Module | Coverage |
|--------|---------|
| `test_environment.py` | Reset/step lifecycle, session management, max steps |
| `test_graders.py` | Determinism, score range, perfect/empty submissions |
| `test_templates.py` | All 4 templates, field generation, error injection |
| `test_endpoints.py` | All HTTP endpoints with empty and valid bodies |
| `test_confidence.py` | Confidence reward shaping, calibration scoring |
| `test_three_way_match.py` | GR generation, lookup, 3-way grader, backward compat |

```bash
pytest tests/ -v   # All 113 pass
```

---

## Project Structure

```
invoice_agent/
├── models.py                    # Pydantic Action, Observation, State
├── data/
│   ├── invoice_templates.py     # 4 procedural invoice templates
│   ├── vendor_database.py       # Vendor DB with fuzzy search
│   ├── purchase_orders.py       # PO generation, duplicate detection
│   └── goods_receipts.py        # Goods receipt generation (3-way match)
├── graders/__init__.py          # Deterministic graders + calibration
└── server/
    ├── app.py                   # FastAPI + OpenEnv endpoints
    └── invoice_environment.py   # Core environment logic

inference.py                     # Baseline LLM agent
openenv.yaml                     # OpenEnv metadata
Dockerfile                       # Container (public.ecr.aws/docker/library/python:3.11-slim)
```

---

## License

MIT — Author: Akash Kathole
