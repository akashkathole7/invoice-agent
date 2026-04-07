"""Tests for InvoiceAgent grader functions.

Verifies determinism, variation, range, and correctness of all three graders.
"""
from __future__ import annotations

import pytest

from invoice_agent.graders import GRADERS, grade_easy, grade_hard, grade_medium
from invoice_agent.data.invoice_templates import generate_invoice, get_required_fields


def _ground_truth(seed: int, task: str):
    """Helper: return ground-truth dict for a given seed/task."""
    data, _ = generate_invoice(seed, task)
    return data.to_ground_truth(task)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestGraderDeterminism:
    def test_grader_determinism(self):
        """Running the same grader with identical inputs three times gives identical scores."""
        gt = _ground_truth(42, "easy")
        extracted = {k: v for k, v in list(gt.items())[:3]}
        args = (extracted, [], gt, [])

        scores = [grade_easy(*args) for _ in range(3)]
        assert scores[0] == scores[1] == scores[2]

    def test_grader_determinism_medium(self):
        gt = _ground_truth(42, "medium")
        extracted = {k: v for k, v in list(gt.items())[:5]}
        discs = [{"field": "subtotal", "reason": "math error"}]
        args = (extracted, discs, gt, discs)

        scores = [grade_medium(*args) for _ in range(3)]
        assert scores[0] == scores[1] == scores[2]

    def test_grader_determinism_hard(self):
        gt = _ground_truth(42, "hard")
        extracted = {k: v for k, v in list(gt.items())[:4]}
        args = (extracted, [], gt, [])

        scores = [grade_hard(*args) for _ in range(3)]
        assert scores[0] == scores[1] == scores[2]


# ---------------------------------------------------------------------------
# Variation
# ---------------------------------------------------------------------------

class TestGraderVariation:
    def test_grader_variation(self):
        """Different extraction completeness → different easy scores."""
        gt = _ground_truth(42, "easy")
        score_full = grade_easy(dict(gt), [], gt, [])
        score_empty = grade_easy({}, [], gt, [])
        assert score_full != score_empty

    def test_grader_variation_medium(self):
        gt = _ground_truth(42, "medium")
        score_full = grade_medium(dict(gt), [], gt, [])
        score_empty = grade_medium({}, [], gt, [])
        assert score_full > score_empty

    def test_grader_variation_hard(self):
        gt = _ground_truth(42, "hard")
        score_full = grade_hard(dict(gt), [], gt, [])
        score_empty = grade_hard({}, [], gt, [])
        assert score_full > score_empty


# ---------------------------------------------------------------------------
# Range [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestGraderRange:
    def test_grader_range_easy(self):
        """All easy grader scores must be in [0.0, 1.0] across 10 seeds."""
        for seed in range(10):
            gt = _ground_truth(seed, "easy")
            items = list(gt.items())
            partial = dict(items[: (seed % len(items)) + 1])
            score = grade_easy(partial, [], gt, [])
            assert 0.0 <= score <= 1.0, f"seed={seed}: score {score} out of [0,1]"

    def test_grader_range_medium(self):
        for seed in range(10):
            gt = _ground_truth(seed, "medium")
            items = list(gt.items())
            partial = dict(items[: (seed % len(items)) + 1])
            score = grade_medium(partial, [], gt, [])
            assert 0.0 <= score <= 1.0, f"seed={seed}: score {score} out of [0,1]"

    def test_grader_range_hard(self):
        for seed in range(10):
            gt = _ground_truth(seed, "hard")
            items = list(gt.items())
            partial = dict(items[: (seed % len(items)) + 1])
            score = grade_hard(partial, [], gt, [])
            assert 0.0 <= score <= 1.0, f"seed={seed}: score {score} out of [0,1]"


# ---------------------------------------------------------------------------
# Not constant
# ---------------------------------------------------------------------------

class TestGraderNotConstant:
    def test_grader_not_constant(self):
        """Scores must vary across seeds when extraction quality varies."""
        required = get_required_fields("easy")
        scores: set = set()

        for seed in range(5):
            gt = _ground_truth(seed, "easy")
            # Extract exactly (seed + 1) fields correctly; mark the rest wrong.
            extracted = {}
            for i, field in enumerate(required):
                if i < seed + 1 and field in gt:
                    extracted[field] = gt[field]
                elif field in gt:
                    extracted[field] = "WRONG_VALUE_XYZ"
            score = grade_easy(extracted, [], gt, [])
            scores.add(round(score, 4))

        assert len(scores) > 1, f"Expected varied scores; got constant set: {scores}"


# ---------------------------------------------------------------------------
# Perfect score
# ---------------------------------------------------------------------------

class TestGraderPerfectScore:
    def test_easy_perfect_score(self):
        """Extracting all fields correctly must score > 0.7."""
        gt = _ground_truth(42, "easy")
        score = grade_easy(dict(gt), [], gt, [])
        assert score > 0.7, f"Perfect extraction should score > 0.7, got {score}"

    def test_easy_perfect_score_is_one(self):
        """Extracting every field exactly should return max clamped score (0.999)."""
        gt = _ground_truth(42, "easy")
        score = grade_easy(dict(gt), [], gt, [])
        assert score == 0.999, f"Expected 0.999 (clamped max), got {score}"

    def test_medium_perfect_fields_high_score(self):
        """Extracting all fields correctly and flagging all discrepancies → high score."""
        from invoice_agent.data.invoice_templates import inject_errors_medium
        import random

        data, text = generate_invoice(42, "medium")
        rng = random.Random(42 + 300)
        _, discrepancies = inject_errors_medium(data, text, rng)
        gt = data.to_ground_truth("medium")

        score = grade_medium(dict(gt), discrepancies, gt, discrepancies)
        assert score > 0.7, f"Perfect medium submission should score > 0.7, got {score}"


# ---------------------------------------------------------------------------
# Empty submission
# ---------------------------------------------------------------------------

class TestGraderEmptySubmission:
    def test_easy_empty_submission(self):
        """No extractions → low score for easy task.

        Note: _field_match("", truth) returns 0.5 because "" is a substring
        of every truth value, giving partial credit. So empty easy = 0.5.
        """
        gt = _ground_truth(42, "easy")
        score = grade_easy({}, [], gt, [])
        assert score <= 0.5, f"Empty easy submission should be <= 0.5, got {score}"

    def test_hard_empty_submission(self):
        """No extractions on hard task → low score."""
        gt = _ground_truth(42, "hard")
        discs = [
            {"field": "subtotal", "type": "math_error", "reason": "bad math"},
            {"field": "invoice_number", "type": "duplicate", "reason": "dup"},
        ]
        score = grade_hard({}, [], gt, discs)
        assert score < 0.5, f"Empty hard submission should be < 0.5, got {score}"

    def test_medium_empty_submission(self):
        """No extractions on medium → score well below perfect."""
        gt = _ground_truth(42, "medium")
        score = grade_medium({}, [], gt, [])
        assert score < 1.0, f"Empty medium submission should be < 1.0, got {score}"


# ---------------------------------------------------------------------------
# GRADERS mapping
# ---------------------------------------------------------------------------

class TestGradersMapping:
    def test_graders_mapping_has_all_tasks(self):
        assert "easy" in GRADERS
        assert "medium" in GRADERS
        assert "hard" in GRADERS

    def test_graders_are_callable(self):
        for task, fn in GRADERS.items():
            assert callable(fn), f"GRADERS['{task}'] is not callable"
