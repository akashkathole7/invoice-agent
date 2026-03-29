"""Tests for InvoiceAgent environment logic (reset / step / done / state)."""
from __future__ import annotations

import pytest

from invoice_agent.models import InvoiceAction, InvoiceObservation
from invoice_agent.server.invoice_environment import InvoiceEnvironment


@pytest.fixture
def env():
    return InvoiceEnvironment()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestResetObservation:
    def test_reset_returns_observation(self, env):
        obs = env.reset(seed=42, task_id="easy")
        assert isinstance(obs, InvoiceObservation)
        assert obs.invoice_text
        assert len(obs.invoice_text) > 10

    def test_reset_has_invoice_number_in_text(self, env):
        obs = env.reset(seed=42, task_id="easy")
        assert "INVOICE" in obs.invoice_text or "Invoice" in obs.invoice_text

    def test_reset_has_required_fields(self, env):
        obs = env.reset(seed=42, task_id="easy")
        assert len(obs.required_fields) > 0
        assert "invoice_number" in obs.required_fields
        assert "total_amount" in obs.required_fields

    def test_reset_starts_not_done(self, env):
        obs = env.reset(seed=42, task_id="easy")
        assert obs.done is False

    def test_reset_step_zero(self, env):
        obs = env.reset(seed=42, task_id="easy")
        assert obs.current_step == 0


class TestResetClearsState:
    def test_reset_clears_extracted_fields(self, env):
        env.reset(seed=42, task_id="easy")
        # Extract a field in first episode.
        env.step(InvoiceAction(
            action_type="extract_field",
            field_name="invoice_number",
            field_value="TEST-123",
        ))
        # Second reset should clear everything.
        obs = env.reset(seed=42, task_id="easy")
        assert obs.extracted_fields == {}, (
            f"Expected empty after reset, got {obs.extracted_fields}"
        )

    def test_reset_with_different_task(self, env):
        obs_easy = env.reset(seed=42, task_id="easy")
        obs_hard = env.reset(seed=42, task_id="hard")
        assert len(obs_hard.required_fields) > len(obs_easy.required_fields)


# ---------------------------------------------------------------------------
# step() — basic reward
# ---------------------------------------------------------------------------

class TestStepReturnsReward:
    def test_step_returns_float_reward(self, env):
        env.reset(seed=42, task_id="easy")
        action = InvoiceAction(
            action_type="extract_field",
            field_name="invoice_number",
            field_value="INV-TEST-0001",
        )
        obs = env.step(action)
        assert isinstance(obs.reward, float)

    def test_step_stores_extracted_field(self, env):
        env.reset(seed=42, task_id="easy")
        env.step(InvoiceAction(
            action_type="extract_field",
            field_name="invoice_number",
            field_value="some_value",
        ))
        # Re-read via another step to see accumulated state
        obs = env.step(InvoiceAction(action_type="validate"))
        assert "invoice_number" in obs.extracted_fields

    def test_correct_extraction_gives_positive_reward(self, env):
        """Extracting a field with the exact ground-truth value gives reward 0.10."""
        from invoice_agent.data.invoice_templates import generate_invoice

        data, _ = generate_invoice(42, "easy")
        gt = data.to_ground_truth("easy")

        env.reset(seed=42, task_id="easy")
        obs = env.step(InvoiceAction(
            action_type="extract_field",
            field_name="invoice_number",
            field_value=gt["invoice_number"],
        ))
        assert obs.reward > 0, f"Correct extraction should give positive reward, got {obs.reward}"

    def test_lookup_vendor_returns_reward(self, env):
        env.reset(seed=42, task_id="easy")
        obs = env.step(InvoiceAction(
            action_type="lookup_vendor",
            vendor_query="Acme",
        ))
        assert isinstance(obs.reward, float)

    def test_lookup_po_returns_reward(self, env):
        from invoice_agent.data.invoice_templates import generate_invoice

        data, _ = generate_invoice(42, "easy")
        env.reset(seed=42, task_id="easy")
        obs = env.step(InvoiceAction(
            action_type="lookup_purchase_order",
            po_number=data.po_number,
        ))
        assert isinstance(obs.reward, float)


# ---------------------------------------------------------------------------
# submit → done
# ---------------------------------------------------------------------------

class TestSubmitEndsEpisode:
    def test_submit_ends_episode(self, env):
        env.reset(seed=42, task_id="easy")
        obs = env.step(InvoiceAction(action_type="submit"))
        assert obs.done is True

    def test_submit_returns_grader_score(self, env):
        env.reset(seed=42, task_id="easy")
        obs = env.step(InvoiceAction(action_type="submit"))
        assert isinstance(obs.grader_score, float)
        assert 0.0 <= obs.grader_score <= 1.0

    def test_after_submit_noop(self, env):
        """Steps after submit return done=True without raising."""
        env.reset(seed=42, task_id="easy")
        env.step(InvoiceAction(action_type="submit"))
        obs = env.step(InvoiceAction(action_type="validate"))
        assert obs.done is True


# ---------------------------------------------------------------------------
# Different seeds → different invoices
# ---------------------------------------------------------------------------

class TestDifferentSeeds:
    def test_different_seeds_produce_different_invoices(self, env):
        obs_42 = env.reset(seed=42, task_id="easy")
        text_42 = obs_42.invoice_text

        obs_123 = env.reset(seed=123, task_id="easy")
        text_123 = obs_123.invoice_text

        assert text_42 != text_123, "Seeds 42 and 123 should produce different invoices"

    def test_same_seed_is_reproducible(self, env):
        obs_first = env.reset(seed=42, task_id="easy")
        obs_second = env.reset(seed=42, task_id="easy")
        assert obs_first.invoice_text == obs_second.invoice_text


# ---------------------------------------------------------------------------
# max_steps termination
# ---------------------------------------------------------------------------

class TestMaxStepsTerminates:
    def test_max_steps_terminates(self, env):
        """After max_steps actions the episode auto-ends."""
        obs = env.reset(seed=42, task_id="easy")
        max_s = obs.max_steps

        for _ in range(max_s + 1):
            if obs.done:
                break
            obs = env.step(InvoiceAction(action_type="validate"))

        assert obs.done is True, "Episode should have terminated after max_steps"

    def test_max_steps_is_positive(self, env):
        obs = env.reset(seed=42, task_id="easy")
        assert obs.max_steps > 0
