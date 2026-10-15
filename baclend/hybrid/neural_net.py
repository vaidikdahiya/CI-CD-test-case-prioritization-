"""
Small feed-forward neural network (pure NumPy, no ML framework needed)
that learns to predict the probability a given test will catch a fault,
based on historical/static features of the test.

Features (all normalized 0-1 before being fed in):
    0. historical_fail_rate   - how often this test has failed historically
    1. code_churn             - how much the code it covers has changed recently
    2. coverage               - fraction of the changed lines this test covers
    3. days_since_last_run    - staleness of the test (normalized)
    4. exec_time              - normalized execution time (used only as a soft signal)

This is intentionally small and dependency-free so the whole project
runs with nothing heavier than numpy installed.
"""

import numpy as np

RNG = np.random.default_rng(42)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_deriv(a):
    return a * (1.0 - a)


class FaultRiskNet:
    """A tiny 5-in -> 8-hidden -> 1-out MLP trained on synthetic labels
    that encode a reasonable domain heuristic (higher churn + higher
    historical fail rate + higher coverage => higher fault-detection
    probability). Training happens once, at construction time, so the
    API has a "ready-to-use" model with no external dataset required.
    """

    def __init__(self, n_features: int = 5, n_hidden: int = 8, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.w1 = rng.normal(0, 0.6, size=(n_features, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.w2 = rng.normal(0, 0.6, size=(n_hidden, 1))
        self.b2 = np.zeros((1, 1))
        self._trained = False

    def forward(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        a1 = _sigmoid(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = _sigmoid(z2)
        return a1, a2

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, a2 = self.forward(np.atleast_2d(x))
        return a2.ravel()

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 800, lr: float = 0.4):
        y = y.reshape(-1, 1)
        for _ in range(epochs):
            a1, a2 = self.forward(X)

            error = a2 - y
            d2 = error * _sigmoid_deriv(a2)
            grad_w2 = a1.T @ d2 / len(X)
            grad_b2 = d2.mean(axis=0, keepdims=True)

            d1 = (d2 @ self.w2.T) * _sigmoid_deriv(a1)
            grad_w1 = X.T @ d1 / len(X)
            grad_b1 = d1.mean(axis=0, keepdims=True)

            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1
        self._trained = True


def _synthetic_training_set(n=1500, seed=7):
    """Generates synthetic (feature, label) pairs from a domain heuristic
    so the network has something sensible to learn from at startup.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 5))
    hist_fail, churn, coverage, staleness, exec_time = X.T

    # Heuristic ground truth: faults are more likely to be caught by tests
    # covering recently-changed, historically-flaky code, slightly
    # discounted by staleness, with a bit of noise.
    score = (
        0.45 * hist_fail
        + 0.35 * churn
        + 0.25 * coverage
        + 0.10 * staleness
        - 0.05 * exec_time
    )
    score += rng.normal(0, 0.05, size=n)
    y = 1.0 / (1.0 + np.exp(-6 * (score - score.mean())))
    return X, y


_MODEL: FaultRiskNet | None = None


def get_model() -> FaultRiskNet:
    """Lazily builds + trains the model once per process."""
    global _MODEL
    if _MODEL is None:
        model = FaultRiskNet()
        X, y = _synthetic_training_set()
        model.train(X, y)
        _MODEL = model
    return _MODEL
