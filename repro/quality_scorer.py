#!/usr/bin/env python3
"""Frozen deterministic exact-rational numeric scorer for the quality stage.

This module is the *only* primary numeric exact-match scorer for
flashblock-20261023-quality.  It is authored and committed before any heldout
generation.  It never consults the gold answer when choosing a prediction
candidate, never uses eval(), and never solves expressions.

Prediction rule (frozen):
  1. Find every `\\boxed{...}` / `\\fbox{...}` occurrence and keep only
     complete, balanced-brace occurrences.
  2. If at least one complete box exists, take the LAST complete box.  If its
     content cannot be normalized to an exact rational number, this is an
     extraction failure.  Do NOT fall back to an earlier box.
  3. If no complete box exists, fall back to the last numeric literal in the
     text (explicitly flagged as permissive).

Accepted candidate syntax after normalisation: optional surrounding math `$`,
optional leading sign, optional currency `$`, thousands comma grouping,
integers/decimals/scientific notation, simple `a/b`, and simple
`\\frac{a}{b}` / `\\dfrac{a}{b}` with numeric arguments.  A trailing `%` is
reported as its numeric value (5% -> 5); fractions are never rescaled by
percent.  Units, equations, nested expressions and any extraneous text make a
boxed candidate invalid.
"""
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

MAX_CANDIDATE_CHARS = 200
MAX_DECIMAL_ADJUSTED = 1000
MAX_INT_DIGITS = 2000

_GROUPED = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_NUM_BODY = rf"(?:{_GROUPED}(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUM_RE = re.compile(rf"^[+-]?{_NUM_BODY}$")
_SIMPLE_FRAC_RE = re.compile(
    rf"^({_NUM_BODY})\s*/\s*([+-]?{_NUM_BODY})$"
)
_LATEX_FRAC_RE = re.compile(
    r"^\\(?:d?frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}$"
)

_BOX_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")

# Fallback scanning patterns.  The fraction patterns are consumed first and
# their spans are excluded from the plain-number scan, so the last *literal*
# is selected rather than a denominator digit inside a fraction.
_LATEX_FRAC_SPAN_RE = re.compile(
    r"\\(?:d?frac)\s*\{[^{}]*\}\s*\{[^{}]*\}"
)
_SIMPLE_FRAC_SPAN_RE = re.compile(
    rf"(?<![A-Za-z0-9_])([+-]?)({_NUM_BODY})\s*/\s*([+-]?{_NUM_BODY})(?![A-Za-z0-9_])"
)
_PLAIN_SPAN_RE = re.compile(
    rf"(?<![A-Za-z0-9_])[+-]?(?:\$)?(?:{_NUM_BODY})%?(?![A-Za-z0-9_])"
)


def _parse_decimal_number(s):
    s = s.strip()
    if not s or len(s) > MAX_CANDIDATE_CHARS:
        return None
    if not _NUM_RE.fullmatch(s):
        return None
    try:
        d = Decimal(s.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    # Bound pathological exponents without changing ordinary GSM8K values.
    if d != 0 and abs(d.adjusted()) > MAX_DECIMAL_ADJUSTED:
        return None
    try:
        value = Fraction(d)
    except (ValueError, OverflowError):
        return None
    # Conservative frozen size bound: Python 3.10 refuses int->str beyond
    # ~4300 digits, so never let a pathological candidate reach canonical_fraction.
    if (len(str(abs(value.numerator))) > MAX_INT_DIGITS
            or len(str(value.denominator)) > MAX_INT_DIGITS):
        return None
    return value


def normalize_candidate(text):
    """Return (Fraction or None, form or None); never raises on candidate text."""
    try:
        return _normalize_candidate_impl(text)
    except Exception:  # noqa: BLE001 - a model prediction must never crash scoring
        return None, None


def _normalize_candidate_impl(text):
    """Internal strict normalizer; callers use normalize_candidate()."""
    if text is None:
        return None, None
    s = str(text).strip()
    if not s or len(s) > MAX_CANDIDATE_CHARS:
        return None, None

    # Strip a surrounding math-mode pair `$...$`.
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    # Leading sign (kept aside so `-$5`, `-1/2`, `-\frac{1}{2}` work).
    sign = 1
    m = re.match(r"^([+-])(.*)$", s, re.S)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        s = m.group(2).strip()

    # Optional currency symbol after the sign.
    if s.startswith("\\$"):
        s = s[2:].strip()
    elif s.startswith("$"):
        s = s[1:].strip()

    # Trailing percent: numeric value is reported unchanged.
    if s.endswith("\\%"):
        s = s[:-2].strip()
    elif s.endswith("%"):
        s = s[:-1].strip()

    if not s or len(s) > MAX_CANDIDATE_CHARS:
        return None, None

    form = None
    value = None

    m = _LATEX_FRAC_RE.fullmatch(s)
    if m:
        num = _parse_decimal_number(m.group(1))
        den = _parse_decimal_number(m.group(2))
        if num is not None and den is not None and den != 0:
            value = num / den
            form = "latex_fraction"

    if value is None:
        m = _SIMPLE_FRAC_RE.fullmatch(s)
        if m:
            num = _parse_decimal_number(m.group(1))
            den = _parse_decimal_number(m.group(2))
            if num is not None and den is not None and den != 0:
                value = num / den
                form = "fraction"

    if value is None:
        num = _parse_decimal_number(s)
        if num is not None:
            value = num
            form = "number"

    if value is None:
        return None, None
    return sign * value, form


def _find_boxes(text):
    """Return all boxed/fbox occurrences with balanced-brace completeness."""
    boxes = []
    for m in _BOX_RE.finditer(text):
        start = m.start()
        i = m.end()
        depth = 1
        while i < len(text):
            ch = text[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if i < len(text):
            boxes.append({"start": start, "content": text[m.end():i],
                          "complete": True, "end": i + 1})
        else:
            boxes.append({"start": start, "content": text[m.end():],
                          "complete": False, "end": len(text)})
    return boxes


def _last_number(text):
    """Permissive last-numeric-literal fallback.  Returns (value, form, preview)."""
    spans = []
    candidates = []

    for m in _LATEX_FRAC_SPAN_RE.finditer(text):
        spans.append((m.start(), m.end()))
        value, form = normalize_candidate(m.group(0))
        if value is not None:
            candidates.append((m.start(), value, form, m.group(0)))

    for m in _SIMPLE_FRAC_SPAN_RE.finditer(text):
        span = (m.start(), m.end())
        if any(a <= m.start() < b for a, b in spans):
            continue
        spans.append(span)
        value, form = normalize_candidate(m.group(0))
        if value is not None:
            candidates.append((m.start(), value, form, m.group(0)))

    for m in _PLAIN_SPAN_RE.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        value, form = normalize_candidate(m.group(0))
        if value is not None:
            candidates.append((m.start(), value, form, m.group(0)))

    if not candidates:
        return None, None, None
    _, value, form, preview = max(candidates, key=lambda x: x[0])
    return value, form, preview


def extract_prediction(text):
    """Frozen prediction extraction.  Never looks at gold."""
    if text is None:
        text = ""
    boxes = _find_boxes(text)
    complete = [b for b in boxes if b["complete"]]
    result = {
        "value": None,
        "method": "last_number_fallback",
        "form": None,
        "complete_box_count": len(complete),
        "incomplete_box_count": len(boxes) - len(complete),
        "candidate_preview": None,
        "failure_reason": None,
    }
    if complete:
        last = complete[-1]
        content = last["content"]
        result["candidate_preview"] = content[:80]
        value, form = normalize_candidate(content)
        result["method"] = "boxed"
        if value is None:
            result["failure_reason"] = "last_complete_box_not_numeric"
            return result
        result["value"] = value
        result["form"] = form
        return result

    value, form, preview = _last_number(text)
    result["candidate_preview"] = (preview or "")[:80] or None
    if value is None:
        result["failure_reason"] = "no_numeric_literal"
        return result
    result["value"] = value
    result["form"] = form
    return result


def extract_gold(answer):
    """Extract and validate the GSM8K `####` gold suffix as an exact rational."""
    if not isinstance(answer, str) or "####" not in answer:
        raise ValueError("gold answer has no #### suffix")
    suffix = answer.rsplit("####", 1)[1].strip()
    value, _ = normalize_candidate(suffix)
    if value is None:
        raise ValueError(f"gold suffix is not a strict number: {suffix[:80]!r}")
    return value


def canonical_fraction(value):
    """Canonical rational string: integer alone, else reduced n/d with sign in n."""
    if value is None:
        return None
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
