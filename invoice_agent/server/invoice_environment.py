"""InvoiceAgent environment server implementation."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

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

# Global session store: episode_id -> InvoiceEnvironment instance
# Used by the custom /step/{session_id} endpoint to maintain stateful sessions.
_SESSIONS: Dict[str, "InvoiceEnvironment"] = {}


class InvoiceEnvironment(Environment):
    """OpenEnv-compliant invoice processing environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 25

    def __init__(self) -> None:
        super().__init__()
        self._state: Optional[InvoiceState] = None
        self._last_vendor_result = None
        self._last_po_result = None
        self._last_validation_errors = None
        self._last_validation_warnings = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InvoiceObservation:
        """Reset the environment for a new episode."""
        task_id: str = kwargs.get("task_id", "easy")
        resolved_seed: int = seed if seed is not None else 42
        resolved_episode_id: str = episode_id or str(uuid.uuid4())[:8]

        # Generate invoice data
        invoice_data, invoice_text = generate_invoice(resolved_seed, task_id)
        ground_truth = invoice_data.to_ground_truth(task_id)
        discrepancies: List[Dict[str, str]] = []

        import random
        rng = random.Random(resolved_seed + 300)

        if task_id == "medium":
            invoice_text, discrepancies = inject_errors_medium(invoice_data, invoice_text, rng)
        elif task_id == "hard":
            invoice_text, discrepancies = inject_errors_hard(invoice_data, invoice_text, rng)

        vendor_db = generate_vendor_db(resolved_seed, invoice_data.vendor_name, task_id)
        purchase_orders = generate_purchase_order(resolved_seed, invoice_data, task_id)

        self._state = InvoiceState(
            task_id=task_id,
            episode_id=resolved_episode_id,
            current_step=0,
            max_steps=self.MAX_STEPS,
            done=False,
            seed=resolved_seed,
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

        # Store in global sessions for stateful HTTP step calls
        _SESSIONS[resolved_episode_id] = self

        obs = self._make_observation("Environment reset. Begin processing the invoice.", "reset")
        obs.reward = 0.0
        obs.done = False
        obs.session_id = resolved_episode_id
        return obs

    def step(
        self,
        action: InvoiceAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InvoiceObservation:
        """Execute one action and return observation."""
        if self._state is None:
            # Auto-reset with defaults for stateless step calls
            self.reset()

        if self._state.done:
            obs = self._make_observation("Episode is done. Call reset() to start new.", "noop")
            obs.reward = 0.0
            obs.done = True
            obs.session_id = self._state.episode_id
            return obs

        self._state.current_step += 1
        self._state.actions_taken.append(action.action_type)
        reward = 0.0
        grader_score = 0.0
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
                reward, result_msg, grader_score = self._handle_submit()
            else:
                reward = -0.05
                result_msg = f"Unknown action type: {action.action_type}"
                self._state.consecutive_invalid += 1
        except Exception as e:
            reward = -0.05
            result_msg = f"Invalid action: {str(e)}"
            self._state.consecutive_invalid += 1

        if reward >= 0:
            self._state.consecutive_invalid = 0

        if self._state.consecutive_invalid >= 3:
            self._state.done = True
            result_msg += " Episode ended: 3 consecutive invalid actions."

        if self._state.current_step >= self._state.max_steps and not self._state.done:
            self._state.done = True
            reward -= 0.15
            result_msg += " Episode ended: max steps reached."
            grader_score = self._run_grader()

        self._state.cumulative_reward += reward
        obs = self._make_observation(result_msg, action.action_type)
        obs.reward = reward
        obs.done = self._state.done
        obs.session_id = self._state.episode_id
        obs.grader_score = grader_score
        return obs

    @property
    def state(self) -> State:
        """Return current environment state."""
        if self._state is None:
            return State(episode_id=None, step_count=0)
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.current_step,
        )

    def close(self) -> None:
        """No-op: keep state alive for session-based step calls."""
        pass

    # --- Action Handlers ---

    def _handle_extract(self, action: InvoiceAction) -> Tuple[float, str]:
        if not action.field_name or not action.field_value:
            return -0.05, "extract_field requires 'field_name' and 'field_value'."

        fname = action.field_name.strip()
        fval = action.field_value.strip()

        if fname in self._state.extracted_fields:
            return -0.02, f"Field '{fname}' was already extracted. Use a different field."

        self._state.extracted_fields[fname] = fval

        gt_val = self._state.ground_truth_fields.get(fname)
        if gt_val is None:
            return 0.01, f"Field '{fname}' extracted (not a required field)."

        from invoice_agent.graders import _normalize
        if _normalize(fval) == _normalize(gt_val):
            return 0.10, f"Field '{fname}' extracted correctly."
        elif _normalize(gt_val) in _normalize(fval) or _normalize(fval) in _normalize(gt_val):
            return 0.03, f"Field '{fname}' extracted with partial match."
        else:
            return -0.05, f"Field '{fname}' extracted but value may be incorrect."

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

        from invoice_agent.graders import _discrepancy_matches
        matched = any(
            _discrepancy_matches(flag_entry, gt)
            for gt in self._state.ground_truth_discrepancies
        )

        if matched:
            return 0.15, f"Valid discrepancy flagged for '{action.flag_field}'."
        else:
            return -0.10, f"No known discrepancy for '{action.flag_field}'."

    def _handle_validate(self) -> Tuple[float, str]:
        errors: List[str] = []
        warnings: List[str] = []

        gt_subtotal = self._state.ground_truth_fields.get("subtotal", "")
        ext_subtotal = self._state.extracted_fields.get("subtotal", "")

        if ext_subtotal and gt_subtotal:
            from invoice_agent.graders import _normalize
            if _normalize(ext_subtotal) != _normalize(gt_subtotal):
                errors.append(
                    f"Subtotal value '{ext_subtotal}' may not match line item sum."
                )

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

    def _handle_submit(self) -> Tuple[float, str, float]:
        self._state.done = True
        grader_score = self._run_grader()

        if grader_score >= 0.8:
            reward = 0.20
            msg = f"Submission accepted. Grader score: {grader_score:.4f} (excellent)."
        elif grader_score >= 0.5:
            reward = 0.05
            msg = f"Submission accepted. Grader score: {grader_score:.4f} (good)."
        else:
            reward = -0.10
            msg = f"Submission accepted. Grader score: {grader_score:.4f} (needs improvement)."

        return reward, msg, grader_score

    def _run_grader(self) -> float:
        grader_fn = GRADERS.get(self._state.task_id, GRADERS["easy"])
        return grader_fn(
            self._state.extracted_fields,
            self._state.flagged_discrepancies,
            self._state.ground_truth_fields,
            self._state.ground_truth_discrepancies,
        )

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
                {"matches": self._last_vendor_result}
                if self._last_vendor_result
                else None
            ),
            po_lookup_result=(
                self._last_po_result if self._last_po_result else None
            ),
            validation_errors=self._last_validation_errors,
            validation_warnings=self._last_validation_warnings,
            flagged_discrepancies=[dict(d) for d in self._state.flagged_discrepancies],
            fields_extracted=extracted_count,
            fields_remaining=len(required) - extracted_count,
            current_step=self._state.current_step,
            max_steps=self._state.max_steps,
        )

        self._last_vendor_result = None
        self._last_po_result = None
        self._last_validation_errors = None
        self._last_validation_warnings = None

        return obs
