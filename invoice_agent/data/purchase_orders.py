"""Purchase order generator for InvoiceAgent environment."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from .invoice_templates import InvoiceData


def generate_purchase_order(
    seed: int, invoice_data: InvoiceData, task: str
) -> Dict[str, Dict[str, Any]]:
    """Generate purchase orders that correspond to the invoice."""
    rng = random.Random(seed + 200)
    pos: Dict[str, Dict[str, Any]] = {}

    # Build the PO that matches the invoice
    po_items: List[Dict[str, Any]] = []
    for item in invoice_data.line_items:
        po_qty = item.quantity
        po_price = item.unit_price
        if task == "hard" and rng.random() < 0.3:
            # Introduce quantity mismatch for hard task
            po_qty = item.quantity + rng.choice([-5, -10, 5, 10])
        po_items.append({
            "description": item.description,
            "quantity": po_qty,
            "unit_price": po_price,
        })

    # For hard tasks, remove one item from PO to simulate unauthorized charge detection
    if task == "hard" and invoice_data.has_unauthorized_item:
        # The unauthorized item won't be in the PO
        pass  # It's added to invoice but not here

    po_total = round(sum(it["quantity"] * it["unit_price"] for it in po_items), 2)

    pos[invoice_data.po_number] = {
        "po_number": invoice_data.po_number,
        "vendor_name": invoice_data.vendor_name,
        "date": invoice_data.invoice_date,
        "items": po_items,
        "total": po_total,
        "status": "approved",
    }

    # For hard task, add a previously submitted invoice number for duplicate detection
    if task == "hard" and invoice_data.is_duplicate:
        pos["_submitted_invoices"] = {
            "invoice_numbers": [invoice_data.invoice_number],
        }

    # Add a few unrelated POs for noise
    for _ in range(rng.randint(1, 3)):
        fake_po = f"PO-2026-{rng.randint(1000, 9999)}"
        if fake_po not in pos:
            pos[fake_po] = {
                "po_number": fake_po,
                "vendor_name": rng.choice(["Other Vendor A", "Other Vendor B"]),
                "date": "January 15, 2026",
                "items": [{"description": "Misc Item", "quantity": 10, "unit_price": 5.0}],
                "total": 50.0,
                "status": "approved",
            }

    return pos


def lookup_po(pos: Dict[str, Dict[str, Any]], po_number: str) -> Dict[str, Any] | None:
    """Look up a purchase order by number."""
    return pos.get(po_number)


def check_duplicate_invoice(
    pos: Dict[str, Dict[str, Any]], invoice_number: str
) -> bool:
    """Check if an invoice number has been previously submitted."""
    submitted = pos.get("_submitted_invoices", {})
    if isinstance(submitted, dict):
        return invoice_number in submitted.get("invoice_numbers", [])
    return False
