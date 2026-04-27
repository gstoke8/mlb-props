"""
Tests for hits model probability conversion.

Regression guard for the bug where HitsModel's P(1+ hit) output was used
directly as P(over line) for all line values, causing wrong-direction bets
on hits Over 1.5 lines.

Run with: python3 test_hits_model_prob.py
"""
from scipy.stats import binom

_MLB_PA_PER_GAME = 4


def _p_over_from_p1plus(p1plus: float, line: float) -> float:
    """Convert HitsModel P(1+ hit) to P(hits > line) via Binomial."""
    p_pa = 1.0 - max(0.0, 1.0 - p1plus) ** (1.0 / _MLB_PA_PER_GAME)
    p_pa = min(max(p_pa, 0.001), 0.999)
    return float(1.0 - binom.cdf(int(line), n=_MLB_PA_PER_GAME, p=p_pa))


def test_line_half_unchanged():
    """For line=0.5, P(over) == P(1+ hit) — model output is used directly."""
    for p1plus in [0.30, 0.41, 0.58, 0.75]:
        result = _p_over_from_p1plus(p1plus, line=0.5)
        assert abs(result - p1plus) < 1e-9, (
            f"line=0.5: expected P(over)={p1plus}, got {result}"
        )


def test_line_1_5_strictly_less_than_p1plus():
    """For line=1.5, P(2+ hits) must be strictly less than P(1+ hit)."""
    for p1plus in [0.30, 0.41, 0.58, 0.75]:
        result = _p_over_from_p1plus(p1plus, line=1.5)
        assert result < p1plus, (
            f"line=1.5: P(2+hits)={result:.4f} should be < P(1+hit)={p1plus}"
        )


def test_line_1_5_specific_values():
    """Spot-check values from the original bug: model said ~41%, correct is ~8%."""
    # Sam Haggerty / Randy Arozarena / Cole Young were at ~0.41 model_prob
    # Market no-vig was ~0.21 — so edge should be negative (Under, not Over)
    for p1plus in [0.4106, 0.4082, 0.4010]:
        result = _p_over_from_p1plus(p1plus, line=1.5)
        assert result < 0.12, (
            f"P(2+hits) should be < 12% when P(1+hit)={p1plus:.4f}, got {result:.4f}"
        )
        # Correct edge vs market (21%) is negative — would NOT bet Over
        market_novig = 0.211
        edge = result - market_novig
        assert edge < 0, (
            f"Edge should be negative (bet Under, not Over) when P(2+hits)={result:.4f} vs market={market_novig}"
        )


def test_probability_monotone_in_line():
    """P(hits > line) decreases as line increases for a fixed p1plus."""
    p1plus = 0.60
    prev = _p_over_from_p1plus(p1plus, line=0.5)
    for line in [1.5, 2.5]:
        curr = _p_over_from_p1plus(p1plus, line=line)
        assert curr < prev, (
            f"Expected P(over {line}) < P(over {line-1}) but {curr:.4f} >= {prev:.4f}"
        )
        prev = curr


def test_edge_clamped():
    """p_pa is clamped to [0.001, 0.999] — no division by zero or invalid Binomial inputs."""
    _p_over_from_p1plus(0.0, line=0.5)   # p_pa approaches 0
    _p_over_from_p1plus(1.0, line=0.5)   # p_pa approaches 1


if __name__ == "__main__":
    tests = [
        test_line_half_unchanged,
        test_line_1_5_strictly_less_than_p1plus,
        test_line_1_5_specific_values,
        test_probability_monotone_in_line,
        test_edge_clamped,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed < len(tests):
        raise SystemExit(1)
