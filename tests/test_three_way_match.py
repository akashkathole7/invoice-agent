"""Tests for Phase 3: 3-way matching (Invoice vs PO vs Goods Receipt)."""
from __future__ import annotations

import pytest

from invoice_agent.data.goods_receipts import (
    generate_goods_receipts,
    get_gr_discrepancies,
    lookup_goods_receipt,
)
from invoice_agent.data.invoice_templates import generate_invoice
from invoice_agent.graders import grade_hard
from invoice_agent.models import InvoiceAction
from invoice_agent.server.invoice_environment import InvoiceEnvironment


# ---------------------------------------------------------------------------
# Goods receipt generation
# ---------------------------------------------------------------------------

class TestGoodsReceiptGeneration:
    def test_gr_only_for_hard_task(self):
        """Goods receipts are only generated for hard tasks."""
        data, _ = generate_invoice(0, "easy")
        gr = generate_goods_receipts(0, data, "easy")
        assert gr == {}

        data, _ = generate_invoice(0, "medium")
        gr = generate_goods_receipts(0, data, "medium")
        assert gr == {}

    def test_gr_generated_for_hard(self):
        data, _ = generate_invoice(0, "hard")
        gr = generate_goods_receipts(0, data, "hard")
        assert len(gr) > 0
        assert data.po_number in gr

    def test_gr_has_items(self):
        data, _ = generate_invoice(0, "hard")
        gr = generate_goods_receipts(0, data, "hard")
        receipt = gr[data.po_number]
        assert "items" in receipt
        assert len(receipt["items"]) > 0

    def test_gr_deterministic(self):
        data, _ = generate_invoice(42, "hard")
        gr1 = generate_goods_receipts(42, data, "hard")
        gr2 = generate_goods_receipts(42, data, "hard")
        assert gr1 == gr2

    def test_gr_has_quantity_fields(self):
        data, _ = generate_invoice(0, "hard")
        gr = generate_goods_receipts(0, data, "hard")
        for item in gr[data.po_number]["items"]:
            assert "ordered_qty" in item
            assert "received_qty" in item
            assert "receive_date" in item

    def test_gr_has_discrepancies_across_seeds(self):
        """At least some seeds produce quantity discrepancies."""
        has_discrepancy = False
        for seed in range(20):
            data, _ = generate_invoice(seed, "hard")
            gr = generate_goods_receipts(seed, data, "hard")
            discs = get_gr_discrepancies(gr, data.po_number)
            if discs:
                has_discrepancy = True
                break
        assert has_discrepancy, "Expected at least one seed to produce GR discrepancies"


# ---------------------------------------------------------------------------
# Goods receipt lookup
# ---------------------------------------------------------------------------

class TestGoodsReceiptLookup:
    def test_lookup_existing(self):
        data, _ = generate_invoice(0, "hard")
        gr = generate_goods_receipts(0, data, "hard")
        result = lookup_goods_receipt(gr, data.po_number)
        assert result is not None
        assert result["po_number"] == data.po_number

    def test_lookup_nonexistent(self):
        data, _ = generate_invoice(0, "hard")
        gr = generate_goods_receipts(0, data, "hard")
        result = lookup_goods_receipt(gr, "PO-FAKE-9999")
        assert result is None


# ---------------------------------------------------------------------------
# lookup_goods_receipt action in environment
# ---------------------------------------------------------------------------

class TestGRAction:
    def test_lookup_gr_action_works(self):
        env = InvoiceEnvironment()
        obs = env.reset(seed=0, task_id="hard")
        # Get the PO number from the invoice text
        import re
        match = re.search(r"PO Reference:\s*(PO-[\w-]+)", obs.invoice_text)
        assert match, "Hard invoice should have PO reference"
        po_num = match.group(1)

        obs = env.step(InvoiceAction(
            action_type="lookup_goods_receipt",
            gr_po_number=po_num,
        ))
        assert isinstance(obs.reward, float)
        assert obs.reward >= 0
        assert obs.gr_lookup_result is not None

    def test_lookup_gr_with_po_number_field(self):
        """Can also use po_number field as fallback."""
        env = InvoiceEnvironment()
        env.reset(seed=0, task_id="hard")
        data, _ = generate_invoice(0, "hard")

        obs = env.step(InvoiceAction(
            action_type="lookup_goods_receipt",
            po_number=data.po_number,
        ))
        assert obs.reward >= 0

    def test_lookup_gr_missing_po(self):
        env = InvoiceEnvironment()
        env.reset(seed=0, task_id="hard")
        obs = env.step(InvoiceAction(
            action_type="lookup_goods_receipt",
        ))
        assert obs.reward < 0

    def test_gr_result_cleared_after_step(self):
        """GR result is cleared after next step (same as vendor/PO)."""
        env = InvoiceEnvironment()
        env.reset(seed=0, task_id="hard")
        data, _ = generate_invoice(0, "hard")

        env.step(InvoiceAction(
            action_type="lookup_goods_receipt",
            gr_po_number=data.po_number,
        ))
        obs = env.step(InvoiceAction(action_type="validate"))
        assert obs.gr_lookup_result is None


# ---------------------------------------------------------------------------
# Hard task max_steps
# ---------------------------------------------------------------------------

class TestHardMaxSteps:
    def test_hard_max_steps_is_30(self):
        env = InvoiceEnvironment()
        obs = env.reset(seed=0, task_id="hard")
        assert obs.max_steps == 30

    def test_easy_max_steps_unchanged(self):
        env = InvoiceEnvironment()
        obs = env.reset(seed=0, task_id="easy")
        assert obs.max_steps == 25

    def test_medium_max_steps_unchanged(self):
        env = InvoiceEnvironment()
        obs = env.reset(seed=0, task_id="medium")
        assert obs.max_steps == 25


# ---------------------------------------------------------------------------
# Hard grader with 3-way matching
# ---------------------------------------------------------------------------

class TestHardGraderThreeWay:
    def test_hard_grader_range(self):
        """Hard grader scores in [0, 1] across seeds."""
        for seed in range(10):
            data, _ = generate_invoice(seed, "hard")
            gt = data.to_ground_truth("hard")
            # Get discrepancies from environment
            env = InvoiceEnvironment()
            env.reset(seed=seed, task_id="hard")
            gt_discs = env._state.ground_truth_discrepancies
            score = grade_hard({}, [], gt, gt_discs)
            assert 0.0 <= score <= 1.0, f"seed={seed}: score {score}"

    def test_hard_grader_deterministic(self):
        env = InvoiceEnvironment()
        env.reset(seed=42, task_id="hard")
        gt = env._state.ground_truth_fields
        gt_discs = env._state.ground_truth_discrepancies
        scores = [grade_hard({}, [], gt, gt_discs) for _ in range(3)]
        assert scores[0] == scores[1] == scores[2]

    def test_hard_raw_submit_low_score(self):
        """Submitting immediately on hard should give a low score."""
        env = InvoiceEnvironment()
        env.reset(seed=42, task_id="hard")
        obs = env.step(InvoiceAction(action_type="submit"))
        assert obs.grader_score < 0.5, f"Raw submit should be < 0.5, got {obs.grader_score}"

    def test_hard_baseline_not_perfect(self):
        """Heuristic baseline should not get a perfect score on hard."""
        from invoice_agent.server.app import _run_heuristic_baseline
        env = InvoiceEnvironment()
        obs = env.reset(seed=42, task_id="hard")
        score = _run_heuristic_baseline(env, obs, "hard")
        assert score < 0.8, f"Baseline on hard should be < 0.8, got {score}"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_old_action_types_still_work(self):
        """All original action types still function."""
        env = InvoiceEnvironment()
        env.reset(seed=42, task_id="easy")

        obs = env.step(InvoiceAction(
            action_type="extract_field",
            field_name="invoice_number",
            field_value="TEST",
        ))
        assert isinstance(obs.reward, float)

        obs = env.step(InvoiceAction(
            action_type="lookup_vendor",
            vendor_query="Acme",
        ))
        assert isinstance(obs.reward, float)

        obs = env.step(InvoiceAction(action_type="validate"))
        assert isinstance(obs.reward, float)

        obs = env.step(InvoiceAction(action_type="submit"))
        assert obs.done is True
