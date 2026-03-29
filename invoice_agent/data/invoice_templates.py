"""Procedural invoice generation for InvoiceAgent environment."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Product & Service Catalogs
# ---------------------------------------------------------------------------

PRODUCT_CATALOG = [
    ("Industrial Bearings 3x5", 12.50),
    ("Hydraulic Fluid 5L", 34.99),
    ("Safety Gloves Box-50", 8.75),
    ("Steel Bolts M10 (100pk)", 15.20),
    ("Copper Wire 14AWG 50m", 42.00),
    ("LED Panel Light 60W", 28.50),
    ("Thermal Paste 10g", 6.99),
    ("Cable Ties 200pk", 4.25),
    ("Rubber Gasket Set", 19.90),
    ("Welding Rods 2.5mm 5kg", 32.00),
    ("Air Filter Cartridge", 11.75),
    ("PVC Pipe 2in 3m", 9.50),
    ("Drill Bit Set HSS 13pc", 24.99),
    ("Lubricant Spray 400ml", 7.80),
    ("Electrical Tape 10pk", 12.30),
    ("Stainless Sheet 1mm 1x2m", 68.00),
    ("Pressure Gauge 0-100psi", 22.50),
    ("Safety Goggles Anti-Fog", 14.99),
    ("Sandpaper Assorted 50pk", 16.40),
    ("Circuit Breaker 20A", 18.75),
]

SERVICE_CATALOG = [
    ("Software Development", 150.00),
    ("UI/UX Design", 125.00),
    ("Data Analysis", 135.00),
    ("Project Management", 110.00),
    ("Quality Assurance Testing", 95.00),
    ("Technical Documentation", 85.00),
    ("DevOps Engineering", 160.00),
    ("Security Audit & Consulting", 175.00),
    ("Database Administration", 140.00),
    ("Cloud Architecture Review", 180.00),
]


# ---------------------------------------------------------------------------
# Vendor & Address Data
# ---------------------------------------------------------------------------

VENDOR_NAMES = [
    "Acme Industrial Supplies LLC",
    "Pacific Coast Hardware Inc",
    "Summit Manufacturing Co",
    "Delta Power Solutions Ltd",
    "Greenfield Electronics Corp",
    "Atlas Building Materials",
    "Pinnacle Tools & Equipment",
    "Riverside Chemical Supply",
    "Northern Steel Fabrication",
    "Westfield Logistics Group",
    "Apex Safety Products Inc",
    "CoreTech Components Ltd",
    "Frontier Electrical Supply",
    "Granite Stone Industries",
    "Harbor Marine Equipment",
]

_VENDOR_ABBREVIATIONS = {
    "Acme Industrial Supplies LLC": "Acme Ind. Sup.",
    "Pacific Coast Hardware Inc": "Pac. Coast Hdw.",
    "Summit Manufacturing Co": "Summit Mfg.",
    "Delta Power Solutions Ltd": "Delta Pwr. Sol.",
    "Greenfield Electronics Corp": "Greenfield Elec.",
    "Atlas Building Materials": "Atlas Bldg. Mat.",
    "Pinnacle Tools & Equipment": "Pinnacle T&E",
    "Riverside Chemical Supply": "Riverside Chem.",
    "Northern Steel Fabrication": "N. Steel Fab.",
    "Westfield Logistics Group": "Westfield Log.",
    "Apex Safety Products Inc": "Apex Safety Prod.",
    "CoreTech Components Ltd": "CoreTech Comp.",
    "Frontier Electrical Supply": "Frontier Elec.",
    "Granite Stone Industries": "Granite Stone Ind.",
    "Harbor Marine Equipment": "Harbor Marine Eq.",
}

ADDRESSES = [
    "123 Commerce Drive, Suite 400, Portland, OR 97201",
    "456 Industrial Blvd, Unit 12, Houston, TX 77001",
    "789 Manufacturing Way, Chicago, IL 60601",
    "321 Enterprise Pkwy, Atlanta, GA 30301",
    "654 Trade Center Rd, Denver, CO 80201",
    "987 Supply Chain Ave, Seattle, WA 98101",
    "111 Factory Lane, Detroit, MI 48201",
    "222 Warehouse Dr, Phoenix, AZ 85001",
    "333 Distribution St, Dallas, TX 75201",
    "444 Logistics Blvd, Miami, FL 33101",
]

BILL_TO_NAMES = [
    "TechCorp Inc",
    "MegaBuild Solutions",
    "Precision Engineering Ltd",
    "GlobalTech Industries",
    "SmartFactory Systems",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class LineItem:
    description: str
    quantity: int
    unit_price: float
    taxable: bool = True
    discount_pct: float = 0.0

    @property
    def amount(self) -> float:
        return round(self.quantity * self.unit_price, 2)

    @property
    def discount_amount(self) -> float:
        return round(self.amount * self.discount_pct / 100, 2)

    @property
    def net_amount(self) -> float:
        return round(self.amount - self.discount_amount, 2)


@dataclass
class InvoiceData:
    invoice_number: str
    invoice_date: str
    vendor_name: str
    vendor_address: str
    bill_to: str
    po_number: str
    line_items: List[LineItem]
    subtotal: float
    tax_rate: float
    tax_amount: float
    total_amount: float
    payment_terms: str
    due_date: str
    template_type: str = "standard"
    # For hard tasks
    is_duplicate: bool = False
    has_unauthorized_item: bool = False
    # For noisy template — display values that differ from canonical
    noisy_vendor_name: str = ""
    noisy_invoice_date: str = ""
    # For detailed template
    discount_total: float = 0.0
    shipping_cost: float = 0.0

    def to_ground_truth(self, task: str) -> Dict[str, str]:
        vendor_name = self.noisy_vendor_name or self.vendor_name
        invoice_date = self.noisy_invoice_date or self.invoice_date

        gt: Dict[str, str] = {
            "invoice_number": self.invoice_number,
            "invoice_date": invoice_date,
            "vendor_name": vendor_name,
            "subtotal": f"${self.subtotal:,.2f}",
            "tax_amount": f"${self.tax_amount:,.2f}",
            "total_amount": f"${self.total_amount:,.2f}",
        }
        if self.template_type != "consulting":
            gt["po_number"] = self.po_number
        if task in ("medium", "hard"):
            gt["vendor_address"] = self.vendor_address
            gt["payment_terms"] = self.payment_terms
            gt["due_date"] = self.due_date
            gt["bill_to"] = self.bill_to
        if task == "hard":
            gt["line_item_count"] = str(len(self.line_items))
        if self.template_type == "detailed":
            gt["discount_total"] = f"${self.discount_total:,.2f}"
            if task in ("medium", "hard"):
                gt["shipping_cost"] = f"${self.shipping_cost:,.2f}"
                net = round(self.subtotal - self.discount_total + self.shipping_cost, 2)
                gt["net_amount"] = f"${net:,.2f}"
        return gt


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _generate_date(rng: random.Random) -> str:
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{MONTHS[month - 1]} {day}, 2026"


def _due_date_from(invoice_date: str, terms: str) -> str:
    parts = invoice_date.replace(",", "").split()
    month_name, day_str, year = parts[0], parts[1], parts[2]
    month_idx = MONTHS.index(month_name)
    day = int(day_str)
    if "Net 30" in terms:
        day += 30
    elif "Net 60" in terms:
        day += 60
    elif "upon receipt" in terms.lower():
        pass  # Due same day as invoice
    else:
        day += 15
    while day > 28:
        day -= 28
        month_idx = (month_idx + 1) % 12
    return f"{MONTHS[month_idx]} {day}, {year}"


def _numeric_date(date_str: str) -> str:
    """Convert 'March 15, 2026' to '03/15/2026'."""
    parts = date_str.replace(",", "").split()
    month_name, day_str, year = parts[0], parts[1], parts[2]
    month_num = MONTHS.index(month_name) + 1
    return f"{month_num:02d}/{int(day_str):02d}/{year}"


# ---------------------------------------------------------------------------
# Main Generation
# ---------------------------------------------------------------------------

def generate_invoice(seed: int, task: str) -> Tuple[InvoiceData, str]:
    """Generate an invoice and return (data, formatted_text)."""
    rng = random.Random(seed)

    # Template selection — consulting not suitable for hard tasks (needs PO matching)
    templates = ["standard", "consulting", "noisy", "detailed"]
    template_type = templates[seed % 4]
    if task == "hard" and template_type == "consulting":
        template_type = "standard"

    # --- Common header data ---
    vendor_name = rng.choice(VENDOR_NAMES)
    vendor_address = rng.choice(ADDRESSES)
    bill_to = rng.choice(BILL_TO_NAMES)
    invoice_number = f"INV-2026-{rng.randint(1000, 9999)}"
    invoice_date = _generate_date(rng)
    po_number = f"PO-2026-{rng.randint(1000, 9999)}"

    if template_type == "consulting":
        payment_terms = rng.choice(["Due upon receipt", "Net 15"])
    else:
        payment_terms = rng.choice(["Net 30", "Net 60", "Due on Receipt"])
    due_date = _due_date_from(invoice_date, payment_terms)

    # --- Line items (template-specific) ---
    if task == "easy":
        n_items = rng.randint(2, 3)
    elif task == "medium":
        n_items = rng.randint(3, 5)
    else:
        n_items = rng.randint(5, 8)

    line_items: List[LineItem] = []

    if template_type == "consulting":
        services = rng.sample(SERVICE_CATALOG, min(n_items, len(SERVICE_CATALOG)))
        for desc, base_rate in services:
            hours = rng.randint(8, 80)
            rate = round(base_rate * rng.uniform(0.9, 1.1), 2)
            line_items.append(LineItem(desc, hours, rate, taxable=True))
    else:
        products = rng.sample(PRODUCT_CATALOG, min(n_items, len(PRODUCT_CATALOG)))
        for desc, base_price in products:
            qty = rng.randint(5, 200)
            price = round(base_price * rng.uniform(0.9, 1.1), 2)
            taxable = True if task == "easy" else rng.random() > 0.2
            discount_pct = 0.0
            if template_type == "detailed":
                discount_pct = float(rng.choice([0, 0, 0, 5, 10, 15]))
            line_items.append(LineItem(desc, qty, price, taxable, discount_pct))

    subtotal = round(sum(item.amount for item in line_items), 2)
    tax_rate = rng.choice([0.06, 0.065, 0.07, 0.075, 0.08, 0.085])

    # Detailed template: compute discounts and shipping
    discount_total = 0.0
    shipping_cost = 0.0
    if template_type == "detailed":
        discount_total = round(sum(item.discount_amount for item in line_items), 2)
        shipping_cost = round(rng.uniform(15.0, 75.0), 2)

    # Tax calculation
    if template_type == "detailed":
        if task == "easy":
            taxable_amount = round(subtotal - discount_total, 2)
        else:
            taxable_amount = round(
                sum(item.net_amount for item in line_items if item.taxable), 2
            )
    elif task == "easy" or template_type == "consulting":
        taxable_amount = subtotal
    else:
        taxable_amount = round(sum(item.amount for item in line_items if item.taxable), 2)

    tax_amount = round(taxable_amount * tax_rate, 2)

    if template_type == "detailed":
        total_amount = round(subtotal - discount_total + shipping_cost + tax_amount, 2)
    else:
        total_amount = round(subtotal + tax_amount, 2)

    # Noisy display values
    noisy_vendor_name = ""
    noisy_invoice_date = ""
    if template_type == "noisy":
        noisy_vendor_name = _VENDOR_ABBREVIATIONS.get(vendor_name, vendor_name)
        noisy_invoice_date = _numeric_date(invoice_date)

    data = InvoiceData(
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        vendor_name=vendor_name,
        vendor_address=vendor_address,
        bill_to=bill_to,
        po_number=po_number,
        line_items=line_items,
        subtotal=subtotal,
        tax_rate=tax_rate,
        tax_amount=tax_amount,
        total_amount=total_amount,
        payment_terms=payment_terms,
        due_date=due_date,
        template_type=template_type,
        noisy_vendor_name=noisy_vendor_name,
        noisy_invoice_date=noisy_invoice_date,
        discount_total=discount_total,
        shipping_cost=shipping_cost,
    )

    formatters = {
        "standard": _format_invoice_standard,
        "consulting": _format_invoice_consulting,
        "noisy": _format_invoice_noisy,
        "detailed": _format_invoice_detailed,
    }
    text = formatters[template_type](data, rng, task)
    return data, text


# ---------------------------------------------------------------------------
# Format: Standard (original template)
# ---------------------------------------------------------------------------

def _format_invoice_standard(data: InvoiceData, rng: random.Random, task: str) -> str:
    sep = "─" * 55
    lines = [
        f"INVOICE #{data.invoice_number}",
        f"Date: {data.invoice_date}",
        "",
        f"From: {data.vendor_name}",
        f"      {data.vendor_address}",
        "",
        f"Bill To: {data.bill_to}",
        "",
        f"PO Reference: {data.po_number}",
        "",
        f"{'Item Description':<30} {'Qty':>5} {'Unit Price':>12} {'Amount':>12}",
        sep,
    ]
    for item in data.line_items:
        tax_mark = "" if item.taxable or task == "easy" else " *"
        lines.append(
            f"{item.description:<30} {item.quantity:>5} "
            f"${item.unit_price:>10,.2f} ${item.amount:>10,.2f}{tax_mark}"
        )
    lines.append(sep)

    if task != "easy" and any(not it.taxable for it in data.line_items):
        lines.append("  * Non-taxable items")
        lines.append("")

    lines.append(f"{'Subtotal:':>48} ${data.subtotal:>10,.2f}")
    lines.append(f"{'Tax (' + f'{data.tax_rate*100:.1f}%):':>48} ${data.tax_amount:>10,.2f}")
    lines.append(f"{'TOTAL DUE:':>48} ${data.total_amount:>10,.2f}")
    lines.append("")
    lines.append(f"Payment Terms: {data.payment_terms}")
    lines.append(f"Due Date: {data.due_date}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Format: Consulting / Services Invoice
# ---------------------------------------------------------------------------

def _format_invoice_consulting(data: InvoiceData, rng: random.Random, task: str) -> str:
    border = "═" * 60
    sep = "─" * 60
    lines = [
        border,
        f"{'CONSULTING SERVICES INVOICE':^60}",
        border,
        "",
        f"INVOICE #{data.invoice_number}",
        f"Date: {data.invoice_date}",
        "",
        f"From: {data.vendor_name}",
        f"      {data.vendor_address}",
        "",
        f"Bill To: {data.bill_to}",
        "",
        f"{'Service Description':<30} {'Hours':>6} {'Rate/Hr':>12} {'Amount':>12}",
        sep,
    ]
    for item in data.line_items:
        lines.append(
            f"{item.description:<30} {item.quantity:>6} "
            f"${item.unit_price:>10,.2f} ${item.amount:>10,.2f}"
        )
    lines.append(sep)
    lines.append("")
    lines.append(f"{'Subtotal:':>48} ${data.subtotal:>10,.2f}")
    lines.append(f"{'Tax (' + f'{data.tax_rate*100:.1f}%):':>48} ${data.tax_amount:>10,.2f}")
    lines.append(f"{'TOTAL DUE:':>48} ${data.total_amount:>10,.2f}")
    lines.append("")
    lines.append(f"Payment Terms: {data.payment_terms}")
    lines.append(f"Due Date: {data.due_date}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Format: Noisy / Messy Invoice
# ---------------------------------------------------------------------------

def _format_invoice_noisy(data: InvoiceData, rng: random.Random, task: str) -> str:
    display_vendor = data.noisy_vendor_name or data.vendor_name
    display_date = data.noisy_invoice_date or data.invoice_date

    sep = "─" * 55
    lines = [
        f"INVOICE  # {data.invoice_number}     << RUSH ORDER - EXPEDITE >>",
        f"Date:  {display_date}",
        "",
        f"From:  {display_vendor}",
        f"       {data.vendor_address}",
        "",
        f"Bill To : {data.bill_to}",
        "",
        f"PO Reference:  {data.po_number}",
        "",
        "--- NOTE: All prices in USD, subject to final review ---",
        "",
        f"{'Item Description':<30} {'Qty':>5} {'Unit Price':>12} {'Amount':>12}",
        sep,
    ]
    for item in data.line_items:
        tax_mark = "" if item.taxable or task == "easy" else " *"
        lines.append(
            f"{item.description:<30} {item.quantity:>5} "
            f"${item.unit_price:>10,.2f} ${item.amount:>10,.2f}{tax_mark}"
        )
    lines.append(sep)

    if task != "easy" and any(not it.taxable for it in data.line_items):
        lines.append("  * Non-taxable items")
        lines.append("")

    lines.append(f"{'Subtotal:':>48} ${data.subtotal:>10,.2f}")
    lines.append(f"{'Tax (' + f'{data.tax_rate*100:.1f}%):':>48} ${data.tax_amount:>10,.2f}")
    lines.append(f"{'TOTAL DUE:':>48} ${data.total_amount:>10,.2f}")
    lines.append("")
    lines.append(f"Payment Terms:  {data.payment_terms}")
    lines.append(f"Due Date:  {data.due_date}")
    lines.append("")
    lines.append(f"** REMIT PAYMENT TO: {data.vendor_name} **")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Format: Detailed Invoice with Discounts
# ---------------------------------------------------------------------------

def _format_invoice_detailed(data: InvoiceData, rng: random.Random, task: str) -> str:
    sep = "─" * 70
    lines = [
        f"INVOICE #{data.invoice_number}",
        f"Date: {data.invoice_date}",
        "",
        f"From: {data.vendor_name}",
        f"      {data.vendor_address}",
        "",
        f"Bill To: {data.bill_to}",
        "",
        f"PO Reference: {data.po_number}",
        "",
        f"{'Item Description':<28} {'Qty':>5} {'Unit Price':>12} {'Disc':>6} {'Amount':>12}",
        sep,
    ]
    for item in data.line_items:
        tax_mark = "" if item.taxable or task == "easy" else " *"
        disc_str = f"{int(item.discount_pct)}%" if item.discount_pct > 0 else "0%"
        lines.append(
            f"{item.description:<28} {item.quantity:>5} "
            f"${item.unit_price:>10,.2f} {disc_str:>5} "
            f"${item.amount:>10,.2f}{tax_mark}"
        )
    lines.append(sep)

    if task != "easy" and any(not it.taxable for it in data.line_items):
        lines.append("  * Non-taxable items")
        lines.append("")

    net = round(data.subtotal - data.discount_total + data.shipping_cost, 2)
    lines.append(f"{'Subtotal:':>48} ${data.subtotal:>10,.2f}")
    lines.append(f"{'Discount Total:':>48} ${data.discount_total:>10,.2f}")
    lines.append(f"{'Shipping & Handling:':>48} ${data.shipping_cost:>10,.2f}")
    lines.append(f"{'Net Amount:':>48} ${net:>10,.2f}")
    lines.append(f"{'Tax (' + f'{data.tax_rate*100:.1f}%):':>48} ${data.tax_amount:>10,.2f}")
    lines.append(f"{'TOTAL DUE:':>48} ${data.total_amount:>10,.2f}")
    lines.append("")
    lines.append(f"Payment Terms: {data.payment_terms}")
    lines.append(f"Due Date: {data.due_date}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Error Injection
# ---------------------------------------------------------------------------

def inject_errors_medium(
    data: InvoiceData, text: str, rng: random.Random
) -> Tuple[str, List[Dict[str, str]]]:
    """Inject 2-3 errors into a medium-difficulty invoice. Returns modified text and discrepancy list."""
    discrepancies: List[Dict[str, str]] = []

    # Error 1: Math error — modify displayed subtotal to be wrong
    wrong_subtotal = round(data.subtotal + rng.uniform(5.0, 50.0), 2)
    text = text.replace(f"${data.subtotal:>10,.2f}", f"${wrong_subtotal:>10,.2f}", 1)
    discrepancies.append({
        "field": "subtotal",
        "type": "math_error",
        "reason": f"Subtotal ${wrong_subtotal:,.2f} does not match sum of line items ${data.subtotal:,.2f}.",
    })

    # Error 2: Tax rate mismatch
    wrong_tax = round(data.tax_amount + rng.uniform(2.0, 20.0), 2)
    old_tax_str = f"${data.tax_amount:>10,.2f}"
    new_tax_str = f"${wrong_tax:>10,.2f}"
    # Replace only the tax line occurrence
    text = text.replace(old_tax_str, new_tax_str, 1)
    recalc_total = round(wrong_subtotal + wrong_tax, 2)
    text = text.replace(
        f"${data.total_amount:>10,.2f}",
        f"${recalc_total:>10,.2f}",
        1,
    )
    discrepancies.append({
        "field": "tax_amount",
        "type": "calculation_error",
        "reason": f"Tax amount ${wrong_tax:,.2f} does not match {data.tax_rate*100:.1f}% of taxable subtotal.",
    })

    return text, discrepancies


def inject_errors_hard(
    data: InvoiceData, text: str, rng: random.Random
) -> Tuple[str, List[Dict[str, str]]]:
    """Inject multiple subtle errors for hard task."""
    discrepancies: List[Dict[str, str]] = []

    # Error 1: One line item quantity different from PO (will be checked during PO lookup)
    discrepancies.append({
        "field": "line_item_quantity",
        "type": "po_mismatch",
        "reason": "Line item quantity does not match purchase order.",
    })

    # Error 2: Unauthorized item — extra line item not in PO
    data.has_unauthorized_item = True
    extra_price = round(rng.uniform(25.0, 75.0), 2)
    extra_line = (
        f"{'Express Shipping Surcharge':<30} {'1':>5} "
        f"${extra_price:>10,.2f} ${extra_price:>10,.2f}"
    )
    # Insert the extra line just before the Subtotal line
    subtotal_marker = f"{'Subtotal:':>48}"
    text = text.replace(subtotal_marker, extra_line + "\n\n" + subtotal_marker, 1)
    discrepancies.append({
        "field": "line_items",
        "type": "unauthorized_charge",
        "reason": "Express Shipping Surcharge not found in purchase order.",
    })

    # Error 3: Duplicate invoice number hint
    data.is_duplicate = True
    discrepancies.append({
        "field": "invoice_number",
        "type": "duplicate",
        "reason": "This invoice number has been previously submitted.",
    })

    return text, discrepancies


# ---------------------------------------------------------------------------
# Required Fields
# ---------------------------------------------------------------------------

def get_required_fields(task: str, template_type: str = "standard") -> List[str]:
    """Return the list of required fields for a given task and template type."""
    base = [
        "invoice_number",
        "invoice_date",
        "vendor_name",
        "subtotal",
        "tax_amount",
        "total_amount",
    ]
    if template_type != "consulting":
        base.append("po_number")
    if task in ("medium", "hard"):
        base.extend(["vendor_address", "payment_terms", "due_date", "bill_to"])
    if task == "hard":
        base.append("line_item_count")
    if template_type == "detailed":
        base.append("discount_total")
        if task in ("medium", "hard"):
            base.extend(["shipping_cost", "net_amount"])
    return base
