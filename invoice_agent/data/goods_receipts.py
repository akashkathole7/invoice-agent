"""Goods receipt generator for InvoiceAgent environment.

Generates goods receipt (GR) records that PARTIALLY match the PO.
For hard tasks: some items received in different quantities than ordered,
and one item may not yet have been received at all.
Used for 3-way matching: Invoice qty vs PO qty vs GR qty.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .invoice_templates import InvoiceData


def generate_goods_receipts(
    seed: int, invoice_data: InvoiceData, task: str
) -> Dict[str, Dict[str, Any]]:
    """Generate goods receipt records for the invoice's line items.

    Returns a dict keyed by PO number containing received items with quantities.
    For hard tasks, introduces realistic discrepancies:
    - Some items received in different quantities
    - One item not yet received at all
    - Receiving dates spread over time
    """
    rng = random.Random(seed + 400)
    receipts: Dict[str, Dict[str, Any]] = {}

    if task != "hard":
        # Easy/medium: no goods receipts needed
        return receipts

    gr_items: List[Dict[str, Any]] = []
    discrepancy_items: List[Dict[str, str]] = []

    for i, item in enumerate(invoice_data.line_items):
        received_qty = item.quantity

        # ~30% chance of quantity mismatch
        if rng.random() < 0.3:
            shortfall = rng.randint(5, max(10, item.quantity // 4))
            received_qty = max(0, item.quantity - shortfall)
            discrepancy_items.append({
                "description": item.description,
                "invoice_qty": str(item.quantity),
                "received_qty": str(received_qty),
                "type": "quantity_shortfall",
            })

        # One item completely unreceived
        if i == len(invoice_data.line_items) - 1 and not discrepancy_items:
            received_qty = 0
            discrepancy_items.append({
                "description": item.description,
                "invoice_qty": str(item.quantity),
                "received_qty": "0",
                "type": "not_received",
            })

        receive_day = rng.randint(1, 28)
        receive_month = rng.randint(1, 12)

        gr_items.append({
            "description": item.description,
            "ordered_qty": item.quantity,
            "received_qty": received_qty,
            "unit_price": item.unit_price,
            "receive_date": f"{receive_month:02d}/{receive_day:02d}/2026",
            "condition": rng.choice(["good", "good", "good", "damaged"]),
        })

    receipts[invoice_data.po_number] = {
        "po_number": invoice_data.po_number,
        "vendor_name": invoice_data.vendor_name,
        "items": gr_items,
        "total_received": sum(it["received_qty"] for it in gr_items),
        "total_ordered": sum(it["ordered_qty"] for it in gr_items),
        "receiving_complete": all(
            it["received_qty"] >= it["ordered_qty"] for it in gr_items
        ),
        "discrepancy_items": discrepancy_items,
    }

    return receipts


def lookup_goods_receipt(
    receipts: Dict[str, Dict[str, Any]], po_number: str
) -> Optional[Dict[str, Any]]:
    """Look up goods receipt records by PO number."""
    return receipts.get(po_number)


def get_gr_discrepancies(
    receipts: Dict[str, Dict[str, Any]], po_number: str
) -> List[Dict[str, str]]:
    """Extract discrepancy details from goods receipts for a given PO."""
    gr = receipts.get(po_number)
    if not gr:
        return []

    discrepancies = []
    for item in gr.get("items", []):
        if item["received_qty"] < item["ordered_qty"]:
            discrepancies.append({
                "field": "goods_receipt_qty",
                "type": "gr_quantity_mismatch",
                "reason": (
                    f"{item['description']}: received {item['received_qty']} "
                    f"of {item['ordered_qty']} ordered."
                ),
            })
        if item.get("condition") == "damaged":
            discrepancies.append({
                "field": "goods_receipt_condition",
                "type": "gr_damaged_goods",
                "reason": f"{item['description']}: received in damaged condition.",
            })
    return discrepancies
