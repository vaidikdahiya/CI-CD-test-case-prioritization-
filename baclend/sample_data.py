"""Generates a plausible synthetic test-suite dataset, standing in for
what you'd normally pull from CI history (JUnit reports, git churn, a
coverage tool like Istanbul/Coverage.py, etc.) in a real repo.

Besides the original columns, each test also carries the four extra inputs
the trained TCP-CI model needs:
    recent_fail_rate   failure rate over the recent window (0-1)
    last_failure_age   builds since the test last failed (-1 = never failed)
    change_coverage    how strongly the test covers the changed code
    lines_added        lines added to the test code recently
"""

import numpy as np

MODULES = [
    "auth", "payments", "checkout", "inventory", "search", "notifications",
    "user_profile", "cart", "shipping", "recommendations", "reporting", "admin",
]

TEST_KINDS = ["unit", "integration", "e2e"]


def generate_sample_suite(n: int = 40, seed: int = 1):
    rng = np.random.default_rng(seed)
    rng_extra = np.random.default_rng(seed + 1000)   # separate stream: original values unchanged
    tests = []
    for i in range(n):
        module = rng.choice(MODULES)
        kind = rng.choice(TEST_KINDS, p=[0.55, 0.30, 0.15])
        base_time = {"unit": 0.3, "integration": 2.5, "e2e": 12.0}[kind]
        exec_time = round(float(max(0.05, rng.normal(base_time, base_time * 0.4))), 2)

        historical_fail_rate = round(float(rng.beta(1.5, 6)), 3)
        code_churn = round(float(rng.beta(2, 4)), 3)
        coverage = round(float(rng.beta(3, 2)), 3)
        days_since_last_run = int(rng.integers(0, 30))

        recent_fail_rate = round(float(min(1.0, historical_fail_rate * rng_extra.uniform(0.0, 2.0))), 3)
        last_failure_age = int(rng_extra.integers(0, 150)) if historical_fail_rate > 0.1 else -1
        change_coverage = round(float(rng_extra.gamma(1.2, 1.0) * code_churn * coverage * 3), 3)
        lines_added = int(rng_extra.integers(0, 150)) if rng_extra.random() < 0.3 else 0

        tests.append(
            {
                "name": f"test_{module}_{kind}_{i:03d}",
                "module": module,
                "kind": kind,
                "exec_time": exec_time,
                "historical_fail_rate": historical_fail_rate,
                "code_churn": code_churn,
                "coverage": coverage,
                "days_since_last_run": days_since_last_run,
                "recent_fail_rate": recent_fail_rate,
                "last_failure_age": last_failure_age,
                "change_coverage": change_coverage,
                "lines_added": lines_added,
            }
        )
    return tests
