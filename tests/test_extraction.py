"""
Test suite for SmartDoc extraction pipeline.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_extraction.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor import rule_based, llm_extractor


# ── Sample invoice ground truth ────────────────────────────────────────────
# These are the correct values we expect from samples/iv.png
# When you add more sample invoices, add more entries here.

SAMPLE_INVOICE = "samples/iv.png"

EXPECTED = {
    "invoice_no": "INV-005",
    "date":       "Jun 22, 2021",
    "due_date":   "Jun 27, 2021",
    "total":      1564.0,
    "balance_due": 1332.0,
    "line_items_count": 3,
}


# ── Helper ─────────────────────────────────────────────────────────────────

def check_field(result, field, expected_value, results_log):
    """
    Check one field. Adds pass/fail to results_log.
    Returns True if passed.
    """
    actual = result.get(field)

    # Normalize strings for comparison (strip extra spaces)
    if isinstance(expected_value, str) and isinstance(actual, str):
        passed = actual.strip().lower() == expected_value.strip().lower()
    else:
        passed = actual == expected_value

    status = "✅ PASS" if passed else "❌ FAIL"
    results_log.append({
        "field":    field,
        "expected": expected_value,
        "actual":   actual,
        "passed":   passed,
    })
    print(f"  {status}  {field}: expected={expected_value!r}  got={actual!r}")
    return passed


# ── Tests ──────────────────────────────────────────────────────────────────

def test_rule_based():
    print("\n── Rule-Based Extractor ──────────────────────────────")
    result = rule_based.extract(SAMPLE_INVOICE)
    log    = []

    check_field(result, "invoice_no",   EXPECTED["invoice_no"],   log)
    check_field(result, "date",         EXPECTED["date"],         log)
    check_field(result, "due_date",     EXPECTED["due_date"],     log)
    check_field(result, "total",        EXPECTED["total"],        log)
    check_field(result, "balance_due",  EXPECTED["balance_due"],  log)

    # Check line item count separately
    actual_count = len(result.get("line_items") or [])
    expected_count = EXPECTED["line_items_count"]
    passed = actual_count == expected_count
    print(f"  {'✅ PASS' if passed else '❌ FAIL'}  line_items_count: expected={expected_count}  got={actual_count}")
    log.append({"field": "line_items_count", "passed": passed})

    passed_count = sum(1 for r in log if r["passed"])
    total        = len(log)
    accuracy     = round(passed_count / total * 100, 1)
    print(f"\n  Accuracy: {passed_count}/{total} fields correct ({accuracy}%)")
    return accuracy


def test_llm():
    print("\n── LLM Extractor (Gemini) ────────────────────────────")
    result = llm_extractor.extract(SAMPLE_INVOICE)
    log    = []

    # LLM may use different key names — check common variations
    invoice_no = (
        result.get("invoice_no") or
        result.get("invoice_number") or
        result.get("invoice_num")
    )
    date = (
        result.get("date") or
        result.get("invoice_date") or
        result.get("invoice_dt")
    )
    total = result.get("total") or result.get("total_amount")
    balance = result.get("balance_due") or result.get("balance")

    normalized = {
        "invoice_no":        invoice_no,
        "date":              date,
        "due_date":          result.get("due_date"),
        "total":             float(total) if total else None,
        "balance_due":       float(balance) if balance else None,
        "line_items_count":  len(result.get("line_items") or []),
    }

    for field, expected in EXPECTED.items():
        check_field(normalized, field, expected, log)

    passed_count = sum(1 for r in log if r["passed"])
    total_fields = len(log)
    accuracy     = round(passed_count / total_fields * 100, 1)
    print(f"\n  Accuracy: {passed_count}/{total_fields} fields correct ({accuracy}%)")
    return accuracy


# ── Run all tests ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 54)
    print("  SmartDoc Extraction Test Suite")
    print("=" * 54)

    rb_accuracy  = test_rule_based()
    llm_accuracy = test_llm()

    print("\n── Summary ───────────────────────────────────────────")
    print(f"  Rule-based accuracy : {rb_accuracy}%")
    print(f"  LLM accuracy        : {llm_accuracy}%")
    print("=" * 54)
