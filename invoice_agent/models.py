"""Pydantic models for InvoiceAgent OpenEnv environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class InvoiceAction(BaseModel):
    """An action the agent can take during invoice processing."""

    action_type: Literal[
        "extract_field",
        "lookup_vendor",
        "lookup_purchase_order",
        "flag_discrepancy",
        "validate",
        "submit",
    ] = Field(..., description="The type of action to perform.")

    # For extract_field
    field_name: Optional[str] = Field(
        None, description="Name of the field to extract, e.g. 'invoice_number', 'total_amount'."
    )
    field_value: Optional[str] = Field(
        None, description="The extracted value for the field."
    )

    # For lookup_vendor
    vendor_query: Optional[str] = Field(
        None, description="Search string to query the vendor database."
    )

    # For lookup_purchase_order
    po_number: Optional[str] = Field(
        None, description="Purchase order number to look up."
    )

    # For flag_discrepancy
    flag_field: Optional[str] = Field(
        None, description="The field that has a discrepancy."
    )
    flag_reason: Optional[str] = Field(
        None, description="Description of the discrepancy found."
    )


class InvoiceObservation(BaseModel):
    """What the agent observes after each step."""

    invoice_text: str = Field(..., description="The raw invoice text to process.")
    extracted_fields: Dict[str, str] = Field(
        default_factory=dict, description="Fields extracted so far."
    )
    required_fields: List[str] = Field(
        default_factory=list, description="List of required field names for this task."
    )
    last_action_result: str = Field(
        "", description="Result message from the last action."
    )
    last_action_type: str = Field("", description="Type of the last action taken.")

    vendor_lookup_result: Optional[Dict[str, Any]] = Field(
        None, description="Result from the most recent vendor lookup."
    )
    po_lookup_result: Optional[Dict[str, Any]] = Field(
        None, description="Result from the most recent PO lookup."
    )
    validation_errors: Optional[List[str]] = Field(
        None, description="Errors found during validation."
    )
    validation_warnings: Optional[List[str]] = Field(
        None, description="Warnings found during validation."
    )
    flagged_discrepancies: List[Dict[str, str]] = Field(
        default_factory=list, description="Discrepancies flagged so far."
    )

    fields_extracted: int = Field(0, description="Number of fields extracted.")
    fields_remaining: int = Field(0, description="Number of required fields remaining.")
    current_step: int = Field(0, description="Current step number.")
    max_steps: int = Field(25, description="Maximum steps allowed.")
    available_actions: List[str] = Field(
        default_factory=lambda: [
            "extract_field",
            "lookup_vendor",
            "lookup_purchase_order",
            "flag_discrepancy",
            "validate",
            "submit",
        ],
        description="Actions available to the agent.",
    )


class InvoiceState(BaseModel):
    """Full internal state of the environment."""

    task_id: str = Field(..., description="Task identifier: easy, medium, or hard.")
    episode_id: str = Field(..., description="Unique episode identifier.")
    current_step: int = Field(0)
    max_steps: int = Field(25)
    done: bool = Field(False)
    seed: int = Field(0, description="Random seed for reproducibility.")

    # Ground truth (hidden from agent)
    ground_truth_fields: Dict[str, str] = Field(default_factory=dict)
    ground_truth_discrepancies: List[Dict[str, str]] = Field(default_factory=list)

    # Invoice data
    invoice_text: str = Field("")
    vendor_database: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    purchase_orders: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Agent progress
    extracted_fields: Dict[str, str] = Field(default_factory=dict)
    flagged_discrepancies: List[Dict[str, str]] = Field(default_factory=list)

    # Scoring
    cumulative_reward: float = Field(0.0)
    actions_taken: List[str] = Field(default_factory=list)
    consecutive_invalid: int = Field(0)
