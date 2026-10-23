#!/usr/bin/env python3
"""Authored CPU tests for the frozen quality scorer.

These tests are written without any GSM8K heldout answer and are run and
committed before heldout generation.  They pin the exact normalisation and
candidate-selection behaviour described in the quality task.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_scorer as qs  # noqa: E402


PREDICTION_CASES = [
    # name, text, expected canonical or None, expected method
    ("integer_boxed", r"\boxed{18}", "18", "boxed"),
    ("negative_integer", r"\boxed{-7}", "-7", "boxed"),
    ("positive_decimal", r"\boxed{+3.0}", "3", "boxed"),
    ("grouped_comma", r"\boxed{1,234}", "1234", "boxed"),
    ("plain_same_value", r"\boxed{1234}", "1234", "boxed"),
    ("decimal_half", r"\boxed{0.50}", "1/2", "boxed"),
    ("bare_decimal", r"\boxed{.5}", "1/2", "boxed"),
    ("scientific", r"\boxed{1e3}", "1000", "boxed"),
    ("scientific_small", r"\boxed{1.5e-2}", "3/200", "boxed"),
    ("simple_fraction", r"\boxed{1/2}", "1/2", "boxed"),
    ("latex_frac", r"\boxed{\frac{1}{2}}", "1/2", "boxed"),
    ("latex_dfrac_negative", r"\boxed{\dfrac{-3}{4}}", "-3/4", "boxed"),
    ("latex_frac_negative", r"\boxed{-\frac{3}{4}}", "-3/4", "boxed"),
    ("fbox", r"\fbox{7}", "7", "boxed"),
    ("two_boxes_choose_last", r"\boxed{2}\boxed{3}", "3", "boxed"),
    ("last_complete_before_unclosed", r"\boxed{2}\boxed{3", "2", "boxed"),
    ("unclosed_box_fallback", r"\boxed{5", "5", "last_number_fallback"),
    ("invalid_last_box_fails", r"\boxed{5}\boxed{2+3}", None, "boxed"),
    ("expression_rejected", r"\boxed{2+3}", None, "boxed"),
    ("zero_denominator", r"\boxed{1/0}", None, "boxed"),
    ("bad_grouping", r"\boxed{1,23}", None, "boxed"),
    ("units_rejected", r"\boxed{5\text{ cm}}", None, "boxed"),
    ("plain_units_rejected", r"\boxed{5 cm}", None, "boxed"),
    ("empty_output", "", None, "last_number_fallback"),
    ("nonnumeric_output", "I cannot solve this.", None, "last_number_fallback"),
    ("currency_stripped", r"\boxed{\$1,200}", "1200", "boxed"),
    ("dollar_prefix", r"\boxed{$5}", "5", "boxed"),
    ("signed_grouped", r"\boxed{-1,234}", "-1234", "boxed"),
    ("percent_plain", r"\boxed{5\%}", "5", "boxed"),
    ("percent_bare", r"\boxed{5%}", "5", "boxed"),
    ("negative_zero", r"\boxed{-0}", "0", "boxed"),
    ("trailing_period", r"\boxed{18.}", "18", "boxed"),
    ("fallback_last_number", "first 2 then 7", "7", "last_number_fallback"),
    ("fallback_negative", "the answer is -12.", "-12", "last_number_fallback"),
    ("fallback_scientific", "value 1.5e2", "150", "last_number_fallback"),
    ("fallback_fraction_last", "ratio 3/4", "3/4", "last_number_fallback"),
    ("fallback_latex_frac_last", "ratio \\frac{3}{4}", "3/4", "last_number_fallback"),
    # GPT pre-generation scorer review edge cases.
    ("latex_frac_signed_num", r"\boxed{\frac{-1}{2}}", "-1/2", "boxed"),
    ("latex_frac_signed_den", r"\boxed{\frac{1}{-2}}", "-1/2", "boxed"),
    ("simple_frac_signed_den", r"\boxed{1/-2}", "-1/2", "boxed"),
    ("simple_frac_double_negative", r"\boxed{-1/-2}", "1/2", "boxed"),
    ("external_minus_latex_frac", r"\boxed{-\frac{1}{2}}", "-1/2", "boxed"),
    ("simple_frac_spaces", r"\boxed{1 / 2}", "1/2", "boxed"),
    ("latex_frac_spaces", r"\boxed{\frac {1} {2}}", "1/2", "boxed"),
    ("separate_digits_rejected", r"\boxed{1 2}", None, "boxed"),
    ("huge_exponent_rejected", r"\boxed{1e1001}", None, "boxed"),
    ("valid_negative_exponent", r"\boxed{1e-3}", "1/1000", "boxed"),
]

GOLD_CASES = [
    ("gold_plain", "reasoning\n#### 42", "42"),
    ("gold_negative_fraction", "chain\n#### -3/4", "-3/4"),
    ("gold_grouped", "chain\n#### 1,234", "1234"),
    ("gold_decimal", "chain\n#### 0.50", "1/2"),
    ("gold_percent", "chain\n#### 5%", "5"),
]

GOLD_INVALID_CASES = [
    ("gold_no_marker", "42"),
    ("gold_nonnumeric", "chain\n#### abc"),
    ("gold_zero_denominator", "chain\n#### 1/0"),
    ("gold_units", "chain\n#### 5 cm"),
]

CANONICAL_CASES = [
    (Fraction(1, 2), "1/2"),
    (Fraction(-3, 4), "-3/4"),
    (Fraction(3, 1), "3"),
    (Fraction(0, 1), "0"),
    (Fraction(2, 4), "1/2"),
]


def run_tests():
    results = []
    passed = failed = 0
    for name, text, expected, method in PREDICTION_CASES:
        out = qs.extract_prediction(text)
        got = qs.canonical_fraction(out["value"])
        ok = got == expected and out["method"] == method
        results.append({
            "kind": "prediction", "name": name, "input": text,
            "expected_canonical": expected, "got_canonical": got,
            "expected_method": method, "got_method": out["method"],
            "form": out["form"], "failure_reason": out["failure_reason"],
            "pass": ok,
        })
        passed += int(ok)
        failed += int(not ok)

    for name, answer, expected in GOLD_CASES:
        try:
            got = qs.canonical_fraction(qs.extract_gold(answer))
            ok = got == expected
            results.append({"kind": "gold", "name": name, "input": answer,
                            "expected_canonical": expected, "got_canonical": got,
                            "pass": ok})
        except Exception as exc:  # noqa: BLE001
            results.append({"kind": "gold", "name": name, "input": answer,
                            "expected_canonical": expected, "got_canonical": None,
                            "error": repr(exc), "pass": False})
        passed += int(results[-1]["pass"])
        failed += int(not results[-1]["pass"])

    for name, answer in GOLD_INVALID_CASES:
        try:
            qs.extract_gold(answer)
            raised = False
        except ValueError:
            raised = True
        results.append({"kind": "gold_invalid", "name": name, "input": answer,
                        "expected": "ValueError", "raised_value_error": raised,
                        "pass": raised})
        passed += int(raised)
        failed += int(not raised)

    for value, expected in CANONICAL_CASES:
        got = qs.canonical_fraction(value)
        ok = got == expected
        results.append({"kind": "canonical", "name": f"canonical_{expected}",
                        "input": str(value), "expected_canonical": expected,
                        "got_canonical": got, "pass": ok})
        passed += int(ok)
        failed += int(not ok)

    return {
        "schema_version": 1,
        "scorer_sha256": _sha256(Path(qs.__file__)),
        "tests_sha256": _sha256(Path(__file__)),
        "total_cases": len(results),
        "passed": passed,
        "failed": failed,
        "status": "PASSED" if failed == 0 else "FAILED",
        "cases": results,
    }


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = run_tests()
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
