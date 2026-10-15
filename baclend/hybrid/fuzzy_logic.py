"""
Pure-Python Mamdani-style fuzzy inference system (no skfuzzy dependency)
that turns three crisp inputs into a single crisp "priority" score.

Inputs:
    risk      - fault-detection probability from the neural net (0-1)
    exec_time - normalized test execution time (0-1, 0 = fast, 1 = slow)
    coverage  - normalized code coverage of changed lines (0-1)

Output:
    priority  - crisp priority score (0-100), higher = run sooner
"""

from dataclasses import dataclass


def _tri(x, a, b, c):
    """Triangular membership function with vertices a <= b <= c."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _low(x):
    return _tri(x, -0.25, 0.0, 0.5)


def _med(x):
    return _tri(x, 0.15, 0.5, 0.85)


def _high(x):
    return _tri(x, 0.5, 1.0, 1.25)


@dataclass
class FuzzyResult:
    priority: float
    label: str
    memberships: dict


# Output singleton positions used for a simplified (Sugeno-style) weighted
# defuzzification -- keeps the implementation dependency-free while still
# being a genuine fuzzy rule base with linguistic terms and rule weights.
_OUTPUT_SINGLETONS = {"low": 15.0, "medium": 50.0, "high": 90.0}


def infer_priority(risk: float, exec_time: float, coverage: float) -> FuzzyResult:
    r_low, r_med, r_high = _low(risk), _med(risk), _high(risk)
    t_low, t_med, t_high = _low(exec_time), _med(exec_time), _high(exec_time)
    c_low, c_med, c_high = _low(coverage), _med(coverage), _high(coverage)

    # Rule base (Mamdani, min for AND, standard test-prioritization heuristics):
    rules = [
        # (strength, output_term)
        (min(r_high, c_high), "high"),
        (min(r_high, t_low), "high"),
        (min(r_med, c_med), "medium"),
        (min(r_low, t_high), "low"),
        (min(r_low, c_low), "low"),
        (min(r_med, t_high, c_low), "medium"),
        (min(r_high, t_high), "medium"),  # slow but risky -> still important, not top
        (min(r_med, t_low, c_med), "high"),
    ]

    numerator = 0.0
    denominator = 0.0
    agg = {"low": 0.0, "medium": 0.0, "high": 0.0}
    for strength, term in rules:
        if strength <= 0:
            continue
        agg[term] = max(agg[term], strength)

    for term, strength in agg.items():
        numerator += strength * _OUTPUT_SINGLETONS[term]
        denominator += strength

    if denominator == 0:
        priority = 50.0  # fallback: no rule fired strongly, treat as medium
    else:
        priority = numerator / denominator

    label = max(agg, key=agg.get) if any(agg.values()) else "medium"
    return FuzzyResult(priority=round(priority, 2), label=label, memberships=agg)
