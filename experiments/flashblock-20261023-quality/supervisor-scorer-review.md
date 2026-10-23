# GPT scorer static review before heldout generation

Current scorer generally matches frozen numeric EM policy. Resolve these authored edge cases before freezing/any heldout generation; no heldout outputs have been inspected by GPT or used for this review.

1. Signed numeric fraction arguments are currently rejected because _parse_decimal_number/_SIMPLE_FRAC_RE omit internal signs. Require exact normalization of boxed{\\frac{-1}{2}}, boxed{\\frac{1}{-2}}, boxed{1/-2}, boxed{-1/-2}, and externalminus-latexfrac; expected -1/2,-1/2,-1/2,+1/2 respectively (externalminus example choose explicit test).
2. Whitespace around slash and LaTex fraction braces should be explicitly normalized without joining separate digit tokens: accept boxed{1 / 2} and boxed{\\frac {1} {2}}, reject boxed{1 2}. Add authored tests.
3. MAX_DECIMAL_ADJUSTED=10000 can normalize a huge rational then canonical_fraction str(numerator) fails Python3.10 integer-string digit limit (~4300), turning a bad output into scorer exception. Choose a conservative frozen exponent bound (e.g absadjusted<=1000, numerator/denominator expansion<=~2000digits) and test exponent rejection/validscientificnotation. Never raise on a model prediction; invalid candidate ->None with reason. Goldmalformed mayraise aspreflight invariant. No eval/symbolics.
4. Keep strict boxed units/equations rejection and lastcompletebox rule unchanged; no answer-aware fallback. Record that lastnumberfallback can select intermediate numbers and is separately reported. Do not broaden parsing based on test answers later.

Explicitlystage this review plus fixes/tests/config before GPU probes. Afterchanges run authoredtests, then train-only ordinaryEOS smoke. Once allpass, proceedfixedtest as authorized, no extra approval needed.
