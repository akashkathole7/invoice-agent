"""Tests for Phase 4: Confidence scoring."""
from __future__ import annotations

import pytest

from invoice_agent.graders import compute_calibration, grade_medium, grade_hard
from invoice_agent.models import InvoiceAction
from invoice_agent.server.invoice_environment import InvoiceEnvironment


# ---------------------------------------------------------------------------
# compute_calibration unit tests
# ---------------------------------------------------------------------------

class TestComputeCalibration:
    def test_empty_records(self):
        """No records → perfect calibration (no penalty)."""
        assert compute_calibration([]) == 1.0

    def test_perfectly_calibrated(self):
        """Agent says 0.9 and is correct 90% of the time → high score."""
        records = [{"confidence": 0.9, "correct": True}] * 9 + [
            {"confidence": 0.9, "correct": False}
        ]
        score = compute_calibration(records)
        assert score > 0.8

    def test_overconfident(self):
        """Agent says 0.9 but is always wrong → low score."""
        records = [{"confidence": 0.9, "correct": False}] * 10
        score = compute_calibration(records)
        assert score < 0.5

    def test_underconfident(self):
        """Agent says 0.1 but is always correct → lower score."""
        records = [{"confidence": 0.1, "correct": True}] * 10
        score = compute_calibration(records)
        assert score < 0.5

    def test_score_range(self):
        """Calibration score is always in [0.0, 1.0]."""
        test_cases = [
            [{"confidence": 0.5, "correct": True}] * 5,
            [{"confidence": 0.0, "correct": False}] * 5,
            [{"confidence": 1.0, "correct": True}] * 5,
            [{"confidence": 1.0, "correct": False}] * 5,
        ]
        for records in test_cases:
            score = compute_calibration(records)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {records}"


# ---------------------------------------------------------------------------
# Confidence-based reward shaping in environment
# ---------------------------------------------------------------------------

class TestConfidenceRewards:
    def test_high_confidence_correct_higher_reward(self):
        """High confidence + correct field → higher reward than no confidence."""
        env = InvoiceEnvironment()
        obs = env.reset(seed=0, task_id="easy")

        # Get first required field and its ground truth
        gt = env._state.ground_truth_fields
        field = list(gt.keys())[0]
        value = gt[field]

        # With high confidence
        obs = env.step(InvoiceAction(
            action_type="extract_field",
            field_name=field,
            field_value=value,
            confidence=0.9,
        ))
        reward_with_conf = obs.reward

        # Without confidence (new env, same field)
        env2 = InvoiceEnvironment()
        env2.reset(seed=0, task_id="easy")
        obs2 = env2.step(InvoiceAction(
            action_type="extract_field",
            field_name=field,
            field_value=value,
        ))
        reward_without_conf = obs2.reward

        # 0.05 + 0.15*0.9 = 0.185 vs 0.10
        assert reward_with_conf > reward_without_conf

    def test_high_confidence_wrong_bigger_penalty(self):
        """High confidence + wrong field → bigger penalty than no confidence."""
        env = InvoiceEnvironment()
        env.reset(seed=0, task_id="easy")
        gt = env._state.ground_truth_fields
        field = list(gt.keys())[0]

        obs = env.step(InvoiceAction(
            action_type="extract_field",
            field_name=field,
            field_value="COMPLETELY_WRONG_VALUE_XYZ",
            confidence=0.9,
        ))
        reward_with_conf = obs.reward

        env2 = InvoiceEnvironment()
        env2.reset(seed=0, task_id="easy")
        obs2 = env2.step(InvoiceAction(
            action_type="extract_field",
            field_name=field,
            field_value="COMPLETELY_WRONG_VALUE_XYZ",
        ))
        reward_without_conf = obs2.reward

        # -0.03 - 0.12*0.9 = -0.138 vs -0.05
        assert reward_with_conf < reward_without_conf

    def test_zero_confidence_correct(self):
        """Zero confidence + correct → small positive reward."""
        env = InvoiceEnvironment()
        env.reset(seed=0, task_id="easy")
        gt = env._state.ground_truth_fields
        field = list(gt.keys())[0]
        value = gt[field]

        obs = env.step(InvoiceAction(
            action_type="extract_field",
            field_name=field,
            field_value=value,
            confidence=0.0,
        ))
        # 0.05 + 0.15*0.0 = 0.05
        assert abs(obs.reward - 0.05) < 0.001

    def test_no_confidence_backward_compatible(self):
        """Without confidence field, old rewards apply."""
        env = InvoiceEnvironment()
        env.reset(seed=0, task_id="easy")
        gt = env._state.ground_truth_fields
        field = list(gt.keys())[0]
        value = gt[field]

        obs = env.step(InvoiceAction(
            action_type="extract_field",
            field_name=field,
            field_value=value,
        ))
        assert abs(obs.reward - 0.10) < 0.001

    def test_confidence_records_tracked(self):
        """Confidence records are stored in state."""
        env = InvoiceEnvironment()
        env.reset(seed=0, task_id="easy")
        gt = env._state.ground_truth_fields
        field = list(gt.keys())[0]
        value = gt[field]

        env.step(InvoiceAction(
            action_type="extract_field",
            field_name=field,
            field_value=value,
            confidence=0.8,
        ))
        assert len(env._state.confidence_records) == 1
        assert env._state.confidence_records[0]["confidence"] == 0.8
        assert env._state.confidence_records[0]["correct"] is True

    def test_no_confidence_no_record(self):
        """Without confidence, no record is stored."""
        env = InvoiceEnvironment()
        env.reset(seed=0, task_id="easy")
        gt = env._state.ground_truth_fields
        field = list(gt.keys())[0]
        value = gt[field]

        env.step(InvoiceAction(
            action_type="extract_field",
            field_name=field,
            field_value=value,
        ))
        assert len(env._state.confidence_records) == 0


# ---------------------------------------------------------------------------
# Calibration bonus in graders
# ---------------------------------------------------------------------------

class TestCalibrationBonus:
    def test_medium_grader_with_calibration(self):
        """Medium grader gives bonus for well-calibrated confidence."""
        gt_fields = {"a": "1", "b": "2"}
        records = [{"confidence": 0.9, "correct": True}] * 5
        score_with = grade_medium({}, [], gt_fields, [], confidence_records=records)
        score_without = grade_medium({}, [], gt_fields, [])
        assert score_with >= score_without

    def test_hard_grader_with_calibration(self):
        """Hard grader gives bonus for well-calibrated confidence."""
        gt_fields = {"a": "1", "b": "2"}
        records = [{"confidence": 0.9, "correct": True}] * 5
        score_with = grade_hard({}, [], gt_fields, [], confidence_records=records)
        score_without = grade_hard({}, [], gt_fields, [])
        assert score_with >= score_without

    def test_calibration_bonus_capped(self):
        """Calibration bonus never exceeds 0.05."""
        gt_fields = {"a": "1"}
        records = [{"confidence": 1.0, "correct": True}] * 100
        score_with = grade_medium({"a": "1"}, [], gt_fields, [], confidence_records=records)
        score_without = grade_medium({"a": "1"}, [], gt_fields, [])
        diff = score_with - score_without
        assert diff <= 0.051  # small epsilon for float rounding

    def test_graders_backward_compatible(self):
        """Graders work without confidence_records (default None)."""
        gt = {"a": "1"}
        score_e = grade_medium({}, [], gt, [])
        score_h = grade_hard({}, [], gt, [])
        assert isinstance(score_e, float)
        assert isinstance(score_h, float)


# ---------------------------------------------------------------------------
# Confidence in /step endpoint (integration)
# ---------------------------------------------------------------------------

class TestConfidenceEndpoint:
    def test_confidence_field_accepted(self):
        """InvoiceAction accepts confidence field."""
        action = InvoiceAction(
            action_type="extract_field",
            field_name="invoice_number",
            field_value="INV-001",
            confidence=0.85,
        )
        assert action.confidence == 0.85

    def test_confidence_none_by_default(self):
        """Confidence defaults to None."""
        action = InvoiceAction(
            action_type="extract_field",
            field_name="invoice_number",
            field_value="INV-001",
        )
        assert action.confidence is None

    def test_confidence_validation_range(self):
        """Confidence must be 0.0-1.0."""
        with pytest.raises(Exception):
            InvoiceAction(
                action_type="extract_field",
                field_name="test",
                field_value="test",
                confidence=1.5,
            )
        with pytest.raises(Exception):
            InvoiceAction(
                action_type="extract_field",
                field_name="test",
                field_value="test",
                confidence=-0.1,
            )
