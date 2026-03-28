"""InvoiceAgent environment server implementation."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from invoice_agent.data.invoice_templates import (
    generate_invoice,
    get_required_fields,
    inject_errors_hard,
    inject_errors_medium,
)
from invoice_agent.data.purchase_orders import (
    check_duplicate_invoice,
    generate_purchase_order,
    lookup_po,
)
from invoice_agent.data.vendor_database import generate_vendor_db, search_vendors
from invoice_agent.graders import GRADERS
from invoice_agent.models import InvoiceAction, InvoiceObservation, InvoiceState


class InvoiceEnvironment:
    """OpenEnv-compliant invoice processing environment."""

    MAX_STEPS = 25

    def __init__(self) -> None:
        self._state: Optional[InvoiceState] = None

    def reset(self, task_id: str = "easy", seed: int = 42) -> Tuple[InvoiceObservation, float, bool, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        episode_id = str(uuid.uuid4())[:8]

        # Generate invoice data
        invoice_data, invoice_text = generate_invoice(seed, task_id)
        ground_truth = invoice_data.to_ground_truth(task_id)
        discrepancies: List[Dict[str, str]] = []

        # Inject errors for medium/hard
        import random
        rng = random.Random(seed + 300)

        if task_id == "medium":
            invoice_text, discrepancies = inject_errors_medium(invoice_data, invoice_text, rng)
        elif task_id == "hard":
            invoice_text, discrepancies = inject_errors_hard(invoice_data, invoice_text, rng)

        # Generate supporting data
        vendor_db = generate_vendor_db(seed, invoice_data.vendor_name, task_id)
        purchase_orders = generate_purchase_order(seed, invoice_data, task_id)

        self._state = InvoiceState(
            task_id=task_id,
            episode_id=episode_id,
            current_step=0,
            max_steps=self.MAX_STEPS,
            done=False,
            seed=seed,
            ground_truth_fields=ground_truth,
            ground_truth_discrepancies=discrepancies,
            invoice_text=invoice_text,
            vendor_database=vendor_db,
            purchase_orders=purchase_orders,
            extracted_fields={},
            flagged_discrepancies=[],
            cumulative_reward=0.0,
            actions_taken=[],
            consecutive_invalid=0,
        )

        obs = self._make_observation("Environment reset. Begin processing the invoice.", "reset")
        return obs, 0.0, False, {"episode_id": episode_id, "task_id": task_id}

    def step(self, action: InvoiceAction) -> Tuple[InvoiceObservation, float, bool, Dict[str, Any]]:
        """Execute one action and return observation, reward, done, info."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.current_step += 1
        self._state.actions_taken.append(action.action_type)
        reward = 0.0
        info: Dict[str, Any] = {}
        result_msg = ""

        try:
            if action.action_type == "extract_field":
                reward, result_msg = self._handle_extract(action)
            elif action.action_type == "lookup_vendor":
                reward, result_msg = self._handle_lookup_vendor(action)
            elif action.action_type == "lookup_purchase_order":
                reward, result_msg = self._handle_lookup_po(action)
            elif action.action_type == "flag_discrepancy":
                reward, result_msg = self._handle_flag(action)
            elif action.action_type == "validate":
                reward, result_msg = self._handle_validate()
            elif action.action_type == "submit":
                reward, result_msg, info = self._handle_submit()
            else:
                reward = -0.05
                result_msg = f"Unknown action type: {action.action_type}"
                self._state.consecutive_invalid += 1
        except Exception as e:
            reward = -0.05
            result_msg = f"Invalid action: {str(e)}"
            self._state.consecutive_invalid += 1

        # Reset consecutive invalid counter on valid action
        if reward >= 0:
            self._state.consecutive_invalid = 0

        # Check termination conditions
        if self._state.consecutive_invalid >= 3:
            self._state.done = True
            result_msg += " Episode ended: 3 consecutive invalid actions."
            info["termination"] = "invalid_actions"

        if self._state.current_step >= self._state.max_steps and not self._state.done:
            self._state.done = True
            reward -= 0.15  # Timeout penalty
            result_msg += " Episode ended: max steps reached."
            info["termination"] = "timeout"
            # Auto-grade on timeout
            grader_score = self._run_grader()
            info["grader_score"] = grader_score

        self._state.cumulative_reward += reward
        obs = self._make_observation(result_msg, action.action_type)
        return obs, reward, self._state.done, info

    def state(self) -> Dict[str, Any]:
        """Return the current environment state."""
        if self._state is None:
            return {"status": "not_initialized"}
        return {
            "task_id": self._state.task_id,
            "episode_id": self._state.episode_id,
            "current_step": self._state.current_step,
            "max_steps": self._state.max_steps,
            "done": self._state.done,
            "cumulative_reward": self._state.cumulative_reward,
            "fields_extracted": len(self._state.extracted_fields),
            "discrepancies_flagged": len(self._state.flagged_discrepancies),
        }

    # --- Action Handlers ---

    def _handle_extract(self, action: InvoiceAction) -> Tuple[float, str]:
        if not action.field_name or not action.field_value:
            return -0.05, "extract_field requires 'field_name' and 'field_value'."

        fname = action.field_name.strip()
        fval = action.field_value.strip()

        # Check for duplicate extraction
        if fname in self._state.extracted_fields:
            return -0.02, f"Field '{fname}' was already extracted. Use a different field."

        self._state.extracted_fields[fname] = fval

        # Check accuracy against ground truth
        gt_val = self._state.ground_truth_fields.get(fname)
        if gt_val is None:
            # Field not in ground truth — might be extra but not penalized heavily
            return 0.01, f"Field '{fname}' extracted (not a required field)."

        from invoice_agent.graders import _normalize
        if _normalize(fval) == _normalize(gt_val):
            return 0.10, f"✓ Field '{fname}' extracted correctly."
        elif _normalize(gt_val) in _normalize(fval) or _normalize(fval) in _normalize(gt_val):
            return 0.03, f"~ Field '{fname}' extracted with partial match."
        else:
            return -0.05, f"✗ Field '{fname}' extracted but value may be incorrect."

    def _handle_lookup_vendor(self, action: InvoiceAction) -> Tuple[float, str]:
        if not action.vendor_query:
            return -0.05, "lookup_vendor requires 'vendor_query'."

        results = search_vendors(self._state.vendor_database, action.vendor_query)
        self._last_vendor_result = results

        if results:
            return 0.05, f"Found {len(results)} vendor match(es)."
        else:
            return 0.02, "No vendor matches found."

    def _handle_lookup_po(self, action: InvoiceAction) -> Tuple[float, str]:
        if not action.po_number:
            return -0.05, "lookup_purchase_order requires 'po_number'."

        result = lookup_po(self._state.purchase_orders, action.po_number)
        self._last_po_result = result

        if result:
            return 0.05, f"Purchase order {action.po_number} found."
        else:
            return 0.01, f"Purchase order {action.po_number} not found."

    def _handle_flag(self, action: InvoiceAction) -> Tuple[float, str]:
        if not action.flag_field or not action.flag_reason:
            return -0.05, "flag_discrepancy requires 'flag_field' and 'flag_reason'."

        flag_entry = {
            "field": action.flag_field.strip(),
            "reason": action.flag_reason.strip(),
        }
        self._state.flagged_discrepancies.append(flag_entry)

        # Check if this matches a real discrepancy
        from invoice_agent.graders import _discrepancy_matches
        matched = any(
            _discrepancy_matches(flag_entry, gt)
            for gt in self._state.ground_truth_discrepancies
        )

        if matched:
            return 0.15, f"✓ Valid discrepancy flagged for '{action.flag_field}'."
        else:
            return -0.10, f"✗ No known discrepancy for '{action.flag_field}'."

    def _handle_validate(self) -> Tuple[float, str]:
        errors: List[str] = []
        warnings: List[str] = []

        # Check if subtotal matches line items (basic math check)
        gt_subtotal = self._state.ground_truth_fields.get("subtotal", "")
        ext_subtotal = self._state.extracted_fields.get("subtotal", "")

        if ext_subtotal and gt_subtotal:
            from invoice_agent.graders import _normalize
            if _normalize(ext_subtotal) != _normalize(gt_subtotal):
                errors.append(
                    f"Subtotal value '{ext_subtotal}' may not match line item sum."
                )

        # Check required fields completeness
        required = get_required_fields(self._state.task_id)
        missing = [f for f in required if f not in self._state.extracted_fields]
        if missing:
            warnings.append(f"Missing required fields: {', '.join(missing)}")

        self._last_validation_errors = errors
        self._last_validation_warnings = warnings

        if errors:
            return 0.05, f"Validation found {len(errors)} error(s) and {len(warnings)} warning(s)."
        elif warnings:
            return 0.02, f"Validation found {len(warnings)} warning(s), no errors."
        else:
            return -0.01, "Validation passed with no issues."

    def _handle_submit(self) -> Tuple[float, str, Dict[str, Any]]:
        self._state.done = True
        grader_score = self._run_grader()

        # Bonus/penalty based on accuracy
        if grader_score >= 0.8:
            reward = 0.20
            msg = f"✓ Submission accepted. Grader score: {grader_score:.4f} (excellent)."
        elif grader_score >= 0.5:
            reward = 0.05
            msg = f"✓ Submission accepted. Grader score: {grader_score:.4f} (good)."
        else:
            reward = -0.10
            msg = f"✓ Submission accepted. Grader score: {grader_score:.4f} (needs improvement)."

        info = {
            "grader_score": grader_score,
            "termination": "submitted",
        }
        return reward, msg, info

    def _run_grader(self) -> float:
        grader_fn = GRADERS.get(self._state.task_id, GRADERS["easy"])
        return grader_fn(
            self._state.extracted_fields,
            self._state.flagged_discrepancies,
            self._state.ground_truth_fields,
            self._state.ground_truth_discrepancies,
        )

    # --- Observation Builder ---

    def _make_observation(self, result_msg: str, action_type: str) -> InvoiceObservation:
        required = get_required_fields(self._state.task_id)
        extracted_count = sum(1 for f in required if f in self._state.extracted_fields)

        obs = InvoiceObservation(
            invoice_text=self._state.invoice_text,
            extracted_fields=dict(self._state.extracted_fields),
            required_fields=required,
            last_action_result=result_msg,
            last_action_type=action_type,
            vendor_lookup_result=(
                {"matches": getattr(self, "_last_vendor_result", None)}
                if hasattr(self, "_last_vendor_result") and self._last_vendor_result
                else None
            ),
            po_lookup_result=(
                getattr(self, "_last_po_result", None)
                if hasattr(self, "_last_po_result") and self._last_po_result
                else None
            ),
            validation_errors=getattr(self, "_last_validation_errors", None),
            validation_warnings=getattr(self, "_last_validation_warnings", None),
            flagged_discrepancies=[dict(d) for d in self._state.flagged_discrepancies],
            fields_extracted=extracted_count,
            fields_remaining=len(required) - extracted_count,
            current_step=self._state.current_step,
            max_steps=self._state.max_steps,
        )

        # Clear transient results
        self._last_vendor_result = None
        self._last_po_result = None
        self._last_validation_errors = None
        self._last_validation_warnings = None

        return obs
