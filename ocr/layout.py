from ocr.reader import read_image


import numpy as np


def _is_flat_rect(position):
    """True if position is [x_min, y_min, x_max, y_max] (flat 4-number array)."""
    arr = np.array(position)
    return arr.ndim == 1 and arr.shape[0] == 4


def get_y_center(position):
    """
    Return the vertical center of a bounding box.
    Handles two formats:
      - 4-corner points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
      - Flat rect:       [x_min, y_min, x_max, y_max]
    """
    arr = np.array(position, dtype=float)
    if _is_flat_rect(arr):
        return (arr[1] + arr[3]) / 2
    return (arr[:, 1].min() + arr[:, 1].max()) / 2


def get_x_left(position):
    """Return the leftmost X coordinate, handling both box formats."""
    arr = np.array(position, dtype=float)
    if _is_flat_rect(arr):
        return float(arr[0])
    return float(arr[:, 0].min())


def get_x_right(position):
    """Return the rightmost X coordinate, handling both box formats."""
    arr = np.array(position, dtype=float)
    if _is_flat_rect(arr):
        return float(arr[2])
    return float(arr[:, 0].max())


def group_into_rows(blocks, y_tolerance=10):
    """
    Group text blocks that share roughly the same vertical position into rows.

    y_tolerance: how many pixels apart two blocks can be and still be
    considered the same row. 10px works well for standard invoices.
    """
    # Sort all blocks top-to-bottom by their vertical center
    sorted_blocks = sorted(blocks, key=lambda b: get_y_center(b["position"]))

    rows = []
    current_row = []

    for block in sorted_blocks:
        y = get_y_center(block["position"])

        if not current_row:
            # First block — start a new row
            current_row.append(block)
        else:
            last_y = get_y_center(current_row[-1]["position"])

            if abs(y - last_y) <= y_tolerance:
                # Close enough vertically — same row
                current_row.append(block)
            else:
                # Too far apart — save current row, start a new one
                rows.append(current_row)
                current_row = [block]

    if current_row:
        rows.append(current_row)

    # Within each row, sort blocks left-to-right
    for row in rows:
        row.sort(key=lambda b: get_x_left(b["position"]))

    return rows


import re


def looks_like_label(text):
    """
    Generic label detection — works across invoices, forms, receipts,
    purchase orders, medical bills, etc.

    A piece of text is treated as a label if it matches any of these patterns:
      1. Ends with ':' or '#'            →  "Invoice No :"  "Order #"
      2. Short (1-4 words)               →  labels are rarely long sentences
      3. All uppercase                   →  "TOTAL"  "DATE"  "FROM"
      4. Title case                      →  "Due Date"  "Bill To"
      5. Ends with 'No', 'No.', 'Num'   →  "PO No"  "Check No."
    """
    t = text.strip()
    if not t:
        return False

    word_count = len(t.split())

    # Rule 1 — ends with colon or hash
    if t.endswith(":") or t.endswith("#"):
        return True

    # Rule 2 — too long to be a label (more than 4 words = likely a value)
    if word_count > 4:
        return False

    # Rule 3 — all uppercase short phrase (but not codes like INV-005 or 3A2B)
    if t.isupper() and not re.search(r'[\d\-_/]', t):
        return True

    # Rule 4 — title case, but not dates/values that happen to be title case
    if t.istitle() and not re.search(r'\d', t):
        return True

    # Rule 5 — ends with No / No. / Num / Number / #
    if re.search(r'\b(No\.?|Num\.?|Number|#)$', t, re.IGNORECASE):
        return True

    return False


def extract_key_value_pairs(rows, x_gap_threshold=5):
    """
    On each row, if there are exactly 2 blocks, the gap is large enough,
    AND the left block looks like a label — treat them as a key-value pair.
    """
    pairs = {}

    for row in rows:
        # Scan consecutive block pairs within the row
        i = 0
        while i < len(row) - 1:
            left_block  = row[i]
            right_block = row[i + 1]

            if looks_like_label(left_block["text"]) and not looks_like_label(right_block["text"]):
                left_x_end    = get_x_right(left_block["position"])
                right_x_start = get_x_left(right_block["position"])
                gap           = right_x_start - left_x_end

                if gap >= x_gap_threshold:
                    key   = left_block["text"].strip().rstrip(":")
                    value = right_block["text"].strip()
                    pairs[key] = {
                        "value":      value,
                        "confidence": min(
                            left_block["confidence"],
                            right_block["confidence"]
                        ),
                    }
                    i += 2   # skip the value block, it's consumed
                    continue

            i += 1

    return pairs


def extract_tables(rows, min_cols=3):
    """
    Find rows that look like table rows — they have 3 or more columns
    aligned at roughly the same X positions across multiple rows.
    Returns a list of table rows, each as a list of cell texts.
    """
    # Collect rows with enough columns to be a table row
    candidate_rows = [row for row in rows if len(row) >= min_cols]

    if len(candidate_rows) < 2:
        return []

    table = []
    for row in candidate_rows:
        table.append([block["text"] for block in row])

    return table


def analyze_layout(image_path):
    """
    Full layout analysis pipeline:
      1. Run OCR to get raw text blocks
      2. Group into rows
      3. Extract key-value pairs
      4. Extract tables
    Returns a dict with all three results.
    """
    blocks = read_image(image_path)
    rows   = group_into_rows(blocks)
    pairs  = extract_key_value_pairs(rows)
    tables = extract_tables(rows)

    return {
        "rows":   rows,
        "pairs":  pairs,
        "tables": tables,
    }


# ---- TEST IT ----
if __name__ == "__main__":
    result = analyze_layout("samples/iv.png")

    print("\n=== KEY-VALUE PAIRS ===")
    for key, val in result["pairs"].items():
        print(f"  {key!r:30} → {val['value']!r}  (conf: {val['confidence']})")

    print("\n=== TABLE ROWS ===")
    for row in result["tables"]:
        print("  |", " | ".join(row), "|")

    print(f"\n=== TOTAL ROWS DETECTED: {len(result['rows'])} ===")
