"""Deterministic graders for InvoiceAgent tasks.

Each grader takes the agent's extracted fields and flagged discrepancies,
compares them against ground truth, and returns a float score in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _normalize(val: str) -> str:
    """Normalize a value for comparison: strip, lowercase, remove $ and commas."""
    return val.strip().lower().replace("$", "").replace(",", "").replace(" ", "")


def _field_match(extracted: str, truth: str) -> float:
    """Compare an extracted field against ground truth. Returns 0.0, 0.5, or 1.0."""
    e = _normalize(extracted)
    t = _normalize(truth)
    if e == t:
        return 1.0
    # Partial credit: one is substring of the other
    if e in t or t in e:
        return 0.5
    return 0.0


def grade_easy(
    extracted_fields: Dict[str, str],
    flagged_discrepancies: List[Dict[str, str]],
    ground_truth_fields: Dict[str, str],
    ground_truth_discrepancies: List[Dict[str, str]],
) -> float:
    """Grade Task 1 (Easy): Clean invoice extraction.
    
    Score = average field accuracy across all required fields.
    """
    if not ground_truth_fields:
        return 0.0

    total = 0.0
    count = len(ground_truth_fields)

    for field_name, truth_value in ground_truth_fields.items():
        extracted_value = extracted_fields.get(field_name, "")
        total += _field_match(extracted_value, truth_value)

    score = total / count if count > 0 else 0.0
    return round(min(max(score, 0.0), 1.0), 4)


def grade_medium(
    extracted_fields: Dict[str, str],
    flagged_discrepancies: List[Dict[str, str]],
    ground_truth_fields: Dict[str, str],
    ground_truth_discrepancies: List[Dict[str, str]],
    confidence_records: Optional[List[Dict]] = None,
) -> float:
    """Grade Task 2 (Medium): Invoice with validation errors.

    Score = 0.50 * field_accuracy + 0.40 * discrepancy_f1 + 0.10 * efficiency
           + calibration_bonus (up to 0.05)
    """
    if not ground_truth_fields:
        return 0.0

    # --- Field accuracy (50%) ---
    field_total = 0.0
    field_count = len(ground_truth_fields)
    for field_name, truth_value in ground_truth_fields.items():
        extracted_value = extracted_fields.get(field_name, "")
        field_total += _field_match(extracted_value, truth_value)
    field_accuracy = field_total / field_count if field_count > 0 else 0.0

    # --- Discrepancy detection F1 (40%) ---
    disc_f1 = _compute_discrepancy_f1(flagged_discrepancies, ground_truth_discrepancies)

    # --- Efficiency (10%) --- based on field extraction completeness
    fields_attempted = sum(1 for f in ground_truth_fields if f in extracted_fields)
    efficiency = fields_attempted / field_count if field_count > 0 else 0.0

    # --- Calibration bonus (up to 0.05) ---
    cal_bonus = 0.0
    if confidence_records:
        cal_score = compute_calibration(confidence_records)
        cal_bonus = 0.05 * cal_score

    score = 0.50 * field_accuracy + 0.40 * disc_f1 + 0.10 * efficiency + cal_bonus
    return round(min(max(score, 0.0), 1.0), 4)


def grade_hard(
    extracted_fields: Dict[str, str],
    flagged_discrepancies: List[Dict[str, str]],
    ground_truth_fields: Dict[str, str],
    ground_truth_discrepancies: List[Dict[str, str]],
    confidence_records: Optional[List[Dict]] = None,
) -> float:
    """Grade Task 3 (Hard): Multi-document reconciliation with 3-way matching.

    Score = 0.25 * field_accuracy + 0.30 * discrepancy_f1
          + 0.25 * three_way_match + 0.20 * critical_flags - fp_penalty
          + calibration_bonus (up to 0.05)

    Three-way match: measures whether agent detected GR-related discrepancies
    (quantity shortfalls, damaged goods, unreceived items).
    """
    if not ground_truth_fields:
        return 0.0

    # --- Field accuracy (25%) ---
    field_total = 0.0
    field_count = len(ground_truth_fields)
    for field_name, truth_value in ground_truth_fields.items():
        extracted_value = extracted_fields.get(field_name, "")
        field_total += _field_match(extracted_value, truth_value)
    field_accuracy = field_total / field_count if field_count > 0 else 0.0

    # --- Discrepancy F1 (30%) ---
    disc_f1 = _compute_discrepancy_f1(flagged_discrepancies, ground_truth_discrepancies)

    # --- Three-way match (25%) --- GR-related discrepancy detection ---
    gr_types = {"gr_quantity_mismatch", "gr_damaged_goods", "quantity_shortfall", "not_received"}
    gt_gr = [d for d in ground_truth_discrepancies if d.get("type") in gr_types]
    caught_gr = 0
    for gt_disc in gt_gr:
        for flagged in flagged_discrepancies:
            if _discrepancy_matches(flagged, gt_disc):
                caught_gr += 1
                break
    three_way_score = caught_gr / len(gt_gr) if gt_gr else 1.0

    # --- Critical flags (20%) --- did agent catch the most important issues?
    critical_types = {"unauthorized_charge", "duplicate", "po_mismatch"}
    gt_critical = [d for d in ground_truth_discrepancies if d.get("type") in critical_types]
    caught_critical = 0
    for gt_disc in gt_critical:
        for flagged in flagged_discrepancies:
            if _discrepancy_matches(flagged, gt_disc):
                caught_critical += 1
                break
    critical_score = caught_critical / len(gt_critical) if gt_critical else 1.0

    # False positive penalty
    false_positives = _count_false_positives(flagged_discrepancies, ground_truth_discrepancies)
    fp_penalty = min(false_positives * 0.05, 0.20)

    # --- Calibration bonus (up to 0.05) ---
    cal_bonus = 0.0
    if confidence_records:
        cal_score = compute_calibration(confidence_records)
        cal_bonus = 0.05 * cal_score

    score = (
        0.25 * field_accuracy
        + 0.30 * disc_f1
        + 0.25 * three_way_score
        + 0.20 * critical_score
        - fp_penalty
        + cal_bonus
    )
    return round(min(max(score, 0.0), 1.0), 4)


def compute_calibration(confidence_records: List[Dict]) -> float:
    """Compute calibration score from confidence records.

    Measures how well-calibrated the agent's confidence is:
    a perfect calibrator says 0.9 confidence and is correct 90% of the time.

    Returns a score in [0.0, 1.0] where 1.0 = perfectly calibrated.
    """
    if not confidence_records:
        return 1.0  # No confidence data — no penalty

    # Group into bins and measure calibration error
    bins: Dict[int, List[bool]] = {}
    for rec in confidence_records:
        conf = rec["confidence"]
        correct = rec["correct"]
        # Bin by decile (0-9)
        bucket = min(int(conf * 10), 9)
        bins.setdefault(bucket, []).append(correct)

    total_error = 0.0
    total_count = 0
    for bucket, outcomes in bins.items():
        expected_accuracy = (bucket + 0.5) / 10.0  # midpoint of bin
        actual_accuracy = sum(outcomes) / len(outcomes)
        total_error += abs(expected_accuracy - actual_accuracy) * len(outcomes)
        total_count += len(outcomes)

    if total_count == 0:
        return 1.0

    avg_error = total_error / total_count
    # Convert error to score: 0 error = 1.0, 0.5 error = 0.0
    return max(0.0, 1.0 - 2.0 * avg_error)


def _compute_discrepancy_f1(
    flagged: List[Dict[str, str]], ground_truth: List[Dict[str, str]]
) -> float:
    """Compute F1 score for discrepancy detection."""
    if not ground_truth and not flagged:
        return 1.0
    if not ground_truth:
        return 0.0 if flagged else 1.0
    if not flagged:
        return 0.0

    true_positives = 0
    matched_gt: set = set()
    for f_disc in flagged:
        for i, gt_disc in enumerate(ground_truth):
            if i not in matched_gt and _discrepancy_matches(f_disc, gt_disc):
                true_positives += 1
                matched_gt.add(i)
                break

    precision = true_positives / len(flagged) if flagged else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _discrepancy_matches(flagged: Dict[str, str], ground_truth: Dict[str, str]) -> bool:
    """Check if a flagged discrepancy matches a ground truth discrepancy."""
    flagged_field = _normalize(flagged.get("field", flagged.get("flag_field", "")))
    gt_field = _normalize(ground_truth.get("field", ""))

    if not flagged_field or not gt_field:
        return False

    # Field name match (exact or partial)
    if flagged_field == gt_field or flagged_field in gt_field or gt_field in flagged_field:
        return True

    # Type-based match as fallback
    gt_type = ground_truth.get("type", "")
    flagged_reason = _normalize(flagged.get("reason", flagged.get("flag_reason", "")))
    if gt_type and gt_type.replace("_", "") in flagged_reason:
        return True

    return False


def _count_false_positives(
    flagged: List[Dict[str, str]], ground_truth: List[Dict[str, str]]
) -> int:
    """Count flagged discrepancies that don't match any ground truth."""
    false_pos = 0
    for f_disc in flagged:
        matched = any(_discrepancy_matches(f_disc, gt) for gt in ground_truth)
        if not matched:
            false_pos += 1
    return false_pos


# Map task IDs to grader functions
GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
