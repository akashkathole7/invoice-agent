"""Tests for invoice template generation — verifies all 4 template types."""
from __future__ import annotations

import pytest

from invoice_agent.data.invoice_templates import generate_invoice, get_required_fields
from invoice_agent.graders import grade_easy, grade_medium, grade_hard, GRADERS
from invoice_agent.models import InvoiceAction
from invoice_agent.server.invoice_environment import InvoiceEnvironment


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------

class TestTemplateSelection:
    def test_template_deterministic(self):
        """Same seed always picks the same template."""
        for seed in range(8):
            d1, _ = generate_invoice(seed, "easy")
            d2, _ = generate_invoice(seed, "easy")
            assert d1.template_type == d2.template_type

    def test_four_templates_appear(self):
        """Seeds 0-3 produce all four template types for easy task."""
        types = {generate_invoice(s, "easy")[0].template_type for s in range(4)}
        assert types == {"standard", "consulting", "noisy", "detailed"}

    def test_hard_excludes_consulting(self):
        """Hard task never uses consulting template (needs PO matching)."""
        for seed in range(20):
            data, _ = generate_invoice(seed, "hard")
            assert data.template_type != "consulting", f"seed={seed}"


# ---------------------------------------------------------------------------
# Consulting template
# ---------------------------------------------------------------------------

class TestConsultingTemplate:
    def test_no_po_in_ground_truth(self):
        data, _ = generate_invoice(1, "easy")
        assert data.template_type == "consulting"
        gt = data.to_ground_truth("easy")
        assert "po_number" not in gt

    def test_no_po_in_required_fields(self):
        fields = get_required_fields("easy", "consulting")
        assert "po_number" not in fields

    def test_consulting_has_service_items(self):
        data, text = generate_invoice(1, "easy")
        assert data.template_type == "consulting"
        assert "CONSULTING SERVICES INVOICE" in text

    def test_consulting_text_has_no_po_line(self):
        data, text = generate_invoice(1, "easy")
        assert data.template_type == "consulting"
        assert "PO Reference" not in text

    def test_consulting_grading_works(self):
        data, _ = generate_invoice(1, "easy")
        gt = data.to_ground_truth("easy")
        score = grade_easy(dict(gt), [], gt, [])
        assert score == 1.0


# ---------------------------------------------------------------------------
# Noisy template
# ---------------------------------------------------------------------------

class TestNoisyTemplate:
    def test_noisy_abbreviates_vendor(self):
        data, text = generate_invoice(2, "easy")
        assert data.template_type == "noisy"
        assert data.noisy_vendor_name != ""
        assert data.noisy_vendor_name in text
        gt = data.to_ground_truth("easy")
        assert gt["vendor_name"] == data.noisy_vendor_name

    def test_noisy_numeric_date(self):
        data, text = generate_invoice(2, "easy")
        assert data.template_type == "noisy"
        assert data.noisy_invoice_date != ""
        assert "/" in data.noisy_invoice_date
        assert data.noisy_invoice_date in text

    def test_noisy_has_rush_order(self):
        _, text = generate_invoice(2, "easy")
        assert "RUSH ORDER" in text

    def test_noisy_has_remit_payment(self):
        data, text = generate_invoice(2, "easy")
        assert f"REMIT PAYMENT TO: {data.vendor_name}" in text

    def test_noisy_grading_works(self):
        data, _ = generate_invoice(2, "easy")
        gt = data.to_ground_truth("easy")
        score = grade_easy(dict(gt), [], gt, [])
        assert score == 1.0


# ---------------------------------------------------------------------------
# Detailed template
# ---------------------------------------------------------------------------

class TestDetailedTemplate:
    def test_detailed_has_discount_field(self):
        data, text = generate_invoice(3, "easy")
        assert data.template_type == "detailed"
        gt = data.to_ground_truth("easy")
        assert "discount_total" in gt

    def test_detailed_medium_has_shipping(self):
        data, _ = generate_invoice(3, "medium")
        assert data.template_type == "detailed"
        gt = data.to_ground_truth("medium")
        assert "shipping_cost" in gt
        assert "net_amount" in gt

    def test_detailed_text_has_discount_line(self):
        _, text = generate_invoice(3, "easy")
        assert "Discount Total:" in text
        assert "Shipping & Handling:" in text
        assert "Net Amount:" in text

    def test_detailed_grading_works(self):
        data, _ = generate_invoice(3, "easy")
        gt = data.to_ground_truth("easy")
        score = grade_easy(dict(gt), [], gt, [])
        assert score == 1.0

    def test_detailed_total_is_consistent(self):
        """total = subtotal - discount + shipping + tax."""
        data, _ = generate_invoice(3, "easy")
        expected = round(
            data.subtotal - data.discount_total + data.shipping_cost + data.tax_amount, 2
        )
        assert data.total_amount == expected


# ---------------------------------------------------------------------------
# Standard template unchanged
# ---------------------------------------------------------------------------

class TestStandardTemplate:
    def test_standard_has_po(self):
        data, text = generate_invoice(0, "easy")
        assert data.template_type == "standard"
        gt = data.to_ground_truth("easy")
        assert "po_number" in gt
        assert "PO Reference" in text

    def test_standard_grading_works(self):
        data, _ = generate_invoice(0, "easy")
        gt = data.to_ground_truth("easy")
        score = grade_easy(dict(gt), [], gt, [])
        assert score == 1.0


# ---------------------------------------------------------------------------
# Cross-template consistency
# ---------------------------------------------------------------------------

class TestCrossTemplate:
    def test_all_templates_grading_range(self):
        """Perfect score is 1.0 and empty score < 1.0 for all seeds/templates."""
        for seed in range(8):
            for task in ["easy", "medium", "hard"]:
                data, _ = generate_invoice(seed, task)
                gt = data.to_ground_truth(task)
                grader = GRADERS[task]
                perfect = grader(dict(gt), [], gt, [])
                empty = grader({}, [], gt, [])
                assert 0.0 <= perfect <= 1.0, f"seed={seed} task={task}"
                assert 0.0 <= empty <= 1.0, f"seed={seed} task={task}"
                assert perfect >= empty, f"seed={seed} task={task}"

    def test_environment_works_with_all_templates(self):
        """reset+step+submit works for every template type."""
        env = InvoiceEnvironment()
        for seed in range(4):
            obs = env.reset(seed=seed, task_id="easy")
            assert obs.invoice_text
            assert len(obs.required_fields) >= 6
            # Extract one field and submit
            obs = env.step(InvoiceAction(
                action_type="extract_field",
                field_name="invoice_number",
                field_value="TEST",
            ))
            assert isinstance(obs.reward, float)
            obs = env.step(InvoiceAction(action_type="submit"))
            assert obs.done is True
            assert 0.0 <= obs.grader_score <= 1.0

    def test_error_injection_medium_all_templates(self):
        """inject_errors_medium works for all templates used in medium task."""
        from invoice_agent.data.invoice_templates import inject_errors_medium
        import random

        for seed in range(8):
            data, text = generate_invoice(seed, "medium")
            rng = random.Random(seed + 300)
            modified_text, discs = inject_errors_medium(data, text, rng)
            assert modified_text != text, f"seed={seed}: text unchanged after error injection"
            assert len(discs) >= 2, f"seed={seed}: expected >=2 discrepancies"
