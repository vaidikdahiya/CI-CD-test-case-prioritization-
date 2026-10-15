"""
Orchestrates the full Neuro-Fuzzy-GA hybrid:

    1. Neural Net   -> per-test fault-detection risk score (learned)
    2. Fuzzy Logic  -> turns (risk, exec_time, coverage) into a crisp,
                       human-interpretable priority score per test
    3. Genetic Algo -> searches for the run ORDER of the whole suite that
                       maximizes APFD (front-loads high-value tests)

Stage 1 uses the neuro-fuzzy model trained on the TCP-CI dataset (model.json)
whenever every test carries its four extra inputs (recent_fail_rate,
last_failure_age, change_coverage, lines_added). Otherwise it falls back to
the original demo network, so older CSVs keep working.

Also computes a simple "before / after" CI-time-to-first-likely-failure
comparison so the effect of prioritization is visible.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .neural_net import get_model
from .fuzzy_logic import infer_priority
from .genetic_algorithm import optimize_order, _apfd
from .trained_model import TrainedModel


MODEL_INPUTS = ["recent_fail_rate", "last_failure_age", "change_coverage", "lines_added"]


@lru_cache(maxsize=1)
def _get_trained_model() -> TrainedModel:
    return TrainedModel(os.path.join(os.path.dirname(__file__), "model.json"))


def _has_trained_inputs(tests: List[Dict[str, Any]]) -> bool:
    return all(t.get(c) is not None for t in tests for c in MODEL_INPUTS)


def _trained_risk(tests, hist_fail: np.ndarray, exec_times: np.ndarray) -> np.ndarray:
    """Score tests with the TCP-CI-trained model, rescaled to 0-1 within the suite."""
    df = pd.DataFrame({
        "REC_RecentFailRate": [float(t["recent_fail_rate"]) for t in tests],
        "REC_TotalFailRate": hist_fail,
        "REC_LastFailureAge": [float(t["last_failure_age"]) for t in tests],   # -1 = never failed
        "REC_TotalAvgExeTime": exec_times * 1000.0,                            # seconds -> milliseconds
        "COV_ImpScoreSum": [float(t["change_coverage"]) for t in tests],
        "TES_CHN_LinesAdded": [float(t["lines_added"]) for t in tests],
    })
    return _normalize(_get_trained_model().score(df))


def _normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


@dataclass
class PrioritizedTest:
    name: str
    exec_time: float
    risk_score: float
    priority_score: float
    priority_label: str
    original_rank: int
    new_rank: int = 0


def run_pipeline(tests: List[Dict[str, Any]], generations: int = 150, population_size: int = 60):
    """
    tests: list of dicts, each with keys:
        name (str)
        exec_time (float, seconds)
        historical_fail_rate (float 0-1)
        code_churn (float 0-1)
        coverage (float 0-1)
        days_since_last_run (float, will be normalized internally)
    optional (all four needed to use the trained model):
        recent_fail_rate (float 0-1)
        last_failure_age (builds since last failure, -1 = never)
        change_coverage (float >= 0)
        lines_added (float >= 0)

    Returns a dict with the prioritized list plus before/after metrics.
    """
    n = len(tests)
    if n == 0:
        return {"tests": [], "metrics": {}}

    exec_times = np.array([float(t.get("exec_time", 1.0)) for t in tests])
    hist_fail = np.array([float(t.get("historical_fail_rate", 0.0)) for t in tests])
    churn = np.array([float(t.get("code_churn", 0.0)) for t in tests])
    coverage = np.array([float(t.get("coverage", 0.0)) for t in tests])
    staleness_raw = np.array([float(t.get("days_since_last_run", 0.0)) for t in tests])

    norm_exec = _normalize(exec_times)
    norm_stale = _normalize(staleness_raw)

    # ---- Stage 1: risk scores ----
    use_trained = _has_trained_inputs(tests)
    if use_trained:
        risk_scores = _trained_risk(tests, hist_fail, exec_times)
    else:
        features = np.column_stack([hist_fail, churn, coverage, norm_stale, norm_exec])
        model = get_model()
        risk_scores = model.predict(features)

    # ---- Stage 2: Fuzzy priority scores ----
    priority_scores = np.zeros(n)
    priority_labels = []
    for i in range(n):
        result = infer_priority(risk=risk_scores[i], exec_time=norm_exec[i], coverage=coverage[i])
        priority_scores[i] = result.priority
        priority_labels.append(result.label)

    # ---- Stage 3: GA-optimized run order ----
    best_order, best_apfd, history = optimize_order(
        priority_scores, population_size=population_size, generations=generations
    )

    # ---- Baseline (unordered / original / alphabetical-ish) comparison ----
    original_order = np.arange(n)
    baseline_apfd = _apfd(original_order, priority_scores)

    prioritized: List[PrioritizedTest] = []
    for new_rank, idx in enumerate(best_order):
        prioritized.append(
            PrioritizedTest(
                name=tests[idx]["name"],
                exec_time=float(exec_times[idx]),
                risk_score=round(float(risk_scores[idx]), 4),
                priority_score=float(priority_scores[idx]),
                priority_label=priority_labels[idx],
                original_rank=int(idx) + 1,
                new_rank=new_rank + 1,
            )
        )

    total_time = float(exec_times.sum())

    # Time-to-80%-cumulative-priority-value, before vs after -- a proxy for
    # "how long before CI has run the tests that matter most".
    def time_to_threshold(order, threshold=0.8):
        weights = priority_scores[order]
        total_w = weights.sum()
        if total_w == 0:
            return total_time
        cum_w = 0.0
        cum_t = 0.0
        for idx2 in order:
            cum_w += priority_scores[idx2]
            cum_t += exec_times[idx2]
            if cum_w / total_w >= threshold:
                return cum_t
        return total_time

    before_time = time_to_threshold(original_order)
    after_time = time_to_threshold(best_order)
    improvement_pct = 0.0 if before_time == 0 else round((before_time - after_time) / before_time * 100, 1)

    metrics = {
        "risk_model": "trained TCP-CI neuro-fuzzy" if use_trained else "legacy demo network",
        "total_tests": n,
        "total_suite_time_sec": round(total_time, 2),
        "baseline_apfd": round(baseline_apfd, 4),
        "optimized_apfd": round(best_apfd, 4),
        "apfd_improvement_pct": round((best_apfd - baseline_apfd) * 100, 2),
        "time_to_80pct_value_before_sec": round(before_time, 2),
        "time_to_80pct_value_after_sec": round(after_time, 2),
        "ci_time_saved_pct": improvement_pct,
        "ga_convergence": [round(h, 4) for h in history[:: max(1, len(history) // 40)]],
    }

    return {
        "tests": [vars(p) for p in prioritized],
        "metrics": metrics,
    }
