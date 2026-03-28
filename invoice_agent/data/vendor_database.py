"""Vendor database generator for InvoiceAgent environment."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from .invoice_templates import VENDOR_NAMES, ADDRESSES


def generate_vendor_db(
    seed: int, target_vendor: str, task: str
) -> Dict[str, Dict[str, Any]]:
    """Generate a vendor database. For hard tasks, adds similar-named vendors."""
    rng = random.Random(seed + 100)
    db: Dict[str, Dict[str, Any]] = {}

    used_names = set()

    # Always include the target vendor
    vid = f"V-{rng.randint(1000, 9999)}"
    db[vid] = {
        "vendor_id": vid,
        "name": target_vendor,
        "address": rng.choice(ADDRESSES),
        "tax_id": f"{rng.randint(10,99)}-{rng.randint(1000000,9999999)}",
        "status": "approved",
        "payment_terms": rng.choice(["Net 30", "Net 60", "Due on Receipt"]),
    }
    used_names.add(target_vendor)

    # Add confusable vendors for hard task
    if task == "hard":
        parts = target_vendor.split()
        if len(parts) >= 2:
            confusable = parts[0] + " " + parts[1] + " Corp"
            vid2 = f"V-{rng.randint(1000, 9999)}"
            db[vid2] = {
                "vendor_id": vid2,
                "name": confusable,
                "address": rng.choice(ADDRESSES),
                "tax_id": f"{rng.randint(10,99)}-{rng.randint(1000000,9999999)}",
                "status": "suspended",
                "payment_terms": "Net 30",
            }
            used_names.add(confusable)

    # Fill with other vendors
    others = [v for v in VENDOR_NAMES if v not in used_names]
    rng.shuffle(others)
    for name in others[: rng.randint(5, 10)]:
        vid_other = f"V-{rng.randint(1000, 9999)}"
        db[vid_other] = {
            "vendor_id": vid_other,
            "name": name,
            "address": rng.choice(ADDRESSES),
            "tax_id": f"{rng.randint(10,99)}-{rng.randint(1000000,9999999)}",
            "status": rng.choice(["approved", "approved", "approved", "pending"]),
            "payment_terms": rng.choice(["Net 30", "Net 60", "Due on Receipt"]),
        }

    return db


def search_vendors(db: Dict[str, Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Search vendor database by name (case-insensitive partial match)."""
    query_lower = query.lower().strip()
    results = []
    for vid, vendor in db.items():
        if query_lower in vendor["name"].lower():
            results.append(vendor)
    return results
