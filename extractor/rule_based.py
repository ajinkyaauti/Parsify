import re
from ocr.layout import analyze_layout


# ── Helpers ────────────────────────────────────────────────────────────────

def clean_amount(text):
    """
    Turn a money string like '$1,564.00' into a float 1564.0.
    Returns None if the text doesn't look like a number.
    """
    cleaned = re.sub(r'[^\d.]', '', text)   # strip $, commas, spaces
    try:
        return float(cleaned)
    except ValueError:
        return None


def find_pair(pairs, *possible_keys):
    """
    Look up a field by trying multiple possible label names.
    Invoices use different labels: 'Invoice No', 'Invoice Number', 'Ref No' etc.
    Returns the first match found, or None.
    """
    for key in possible_keys:
        for pair_key, pair_val in pairs.items():
            if key.lower() in pair_key.lower():
                return pair_val["value"]
    return None


# ── Line item parsing ──────────────────────────────────────────────────────

def parse_line_items(tables):
    """
    From the table rows detected by layout.py, find the invoice line items.

    A line item row looks like:
      | 1 Desktop furniture | 1 | $232.00 | $232.00 |

    We detect it by checking: does the first cell start with a number?
    """
    line_items = []

    for row in tables:
        if len(row) < 3:
            continue

        first_cell = row[0].strip()

        # Skip empty rows
        if not first_cell:
            continue

        # Skip header rows
        header_words = {"description", "item", "particulars", "details", "service", "si.", "sl.", "no."}
        if first_cell.lower().strip() in header_words:
            continue

        # Must have a valid numeric amount in the last cell
        if not clean_amount(row[-1]):
            continue

        # At least 2 of the non-first cells must be valid numbers (qty, unit price, amount)
        numeric_count = sum(1 for cell in row[1:] if clean_amount(cell) is not None)
        if numeric_count < 2:
            continue

        description = first_cell.strip()

        # Last cell is the amount, second-to-last is unit price, third-to-last is qty
        amount     = clean_amount(row[-1])
        unit_price = clean_amount(row[-2]) if len(row) >= 4 else None
        qty_text   = row[-3] if len(row) >= 4 else row[1]

        try:
            qty = int(re.sub(r'[^\d]', '', qty_text))
        except (ValueError, TypeError):
            qty = None

        line_items.append({
            "description": description,
            "qty":         qty,
            "unit_price":  unit_price,
            "amount":      amount,
        })

    return line_items


# ── Main extractor ─────────────────────────────────────────────────────────

def extract(image_path):
    """
    Full extraction pipeline:
      1. Run OCR + layout analysis
      2. Pull named fields from key-value pairs
      3. Parse line items from table rows
      4. Return a clean structured dict
    """
    layout = analyze_layout(image_path)
    pairs  = layout["pairs"]
    tables = layout["tables"]

    invoice_no = find_pair(pairs, "invoice no", "invoice number", "ref no", "reference")
    date       = find_pair(pairs, "invoice date", "date", "invoice dt")
    due_date   = find_pair(pairs, "due date", "payment due", "due by")
    vendor     = find_pair(pairs, "from", "vendor", "seller", "billed by")

    raw_total    = find_pair(pairs, "total", "grand total", "amount due", "net total")
    raw_subtotal = find_pair(pairs, "subtotal", "sub total")
    raw_balance  = find_pair(pairs, "balance due", "balance", "amount payable")

    total    = clean_amount(raw_total)    if raw_total    else None
    subtotal = clean_amount(raw_subtotal) if raw_subtotal else None
    balance  = clean_amount(raw_balance)  if raw_balance  else None

    line_items = parse_line_items(tables)

    # Flag for human review if key fields are missing or low confidence
    needs_review = not invoice_no or not date or not total or not line_items

    return {
        "invoice_no":   invoice_no,
        "date":         date,
        "due_date":     due_date,
        "vendor":       vendor,
        "subtotal":     subtotal,
        "total":        total,
        "balance_due":  balance,
        "line_items":   line_items,
        "needs_review": needs_review,
    }


# ── Test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    result = extract("samples/iv.png")
    print(json.dumps(result, indent=2))
