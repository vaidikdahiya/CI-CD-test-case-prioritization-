"""
Genetic algorithm that takes the fuzzy priority scores + execution times
for every test and searches for a running ORDER of the full test suite
that front-loads fault-detection value as early as possible in the CI run
(this is what "adaptive test suite prioritization" optimizes for: you
rarely get to skip tests entirely in CI, but you want the tests most
likely to catch a fault to run first, so a build can fail fast).

Fitness = Average Percentage of Faults Detected (APFD-style metric):
a classic test-prioritization objective. Higher is better.
"""

import numpy as np

RNG = np.random.default_rng(123)


def _apfd(order: np.ndarray, priorities: np.ndarray) -> float:
    """Average Percentage of Faults Detected, using priority score as a
    proxy weight for "fault detected by this test". order is an array of
    test indices (a permutation). Higher APFD = risky tests run earlier.
    """
    n = len(order)
    if n == 0:
        return 0.0
    weights = priorities[order]
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    # position (1-indexed) of each test in the run
    positions = np.arange(1, n + 1)
    apfd = 1.0 - (np.sum(weights * positions) / (total_weight * n)) + (1.0 / (2 * n))
    return float(apfd)


def _tournament_select(pop, fitness, k=3):
    idx = RNG.integers(0, len(pop), size=k)
    best = idx[np.argmax(fitness[idx])]
    return pop[best].copy()


def _order_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    n = len(p1)
    a, b = sorted(RNG.integers(0, n, size=2))
    child = -np.ones(n, dtype=int)
    child[a:b] = p1[a:b]
    fill = [g for g in p2 if g not in child[a:b]]
    j = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[j]
            j += 1
    return child


def _mutate(order: np.ndarray, rate: float = 0.05):
    n = len(order)
    for i in range(n):
        if RNG.random() < rate:
            j = RNG.integers(0, n)
            order[i], order[j] = order[j], order[i]
    return order


def optimize_order(
    priorities: np.ndarray,
    population_size: int = 60,
    generations: int = 150,
    elitism: int = 4,
    mutation_rate: float = 0.05,
):
    """Runs the GA and returns (best_order, best_apfd, history) where
    history is the best fitness per generation (useful for showing
    convergence in the UI).
    """
    n = len(priorities)
    if n <= 1:
        order = np.arange(n)
        return order, _apfd(order, priorities), [1.0] * max(generations, 1)

    population = np.array([RNG.permutation(n) for _ in range(population_size)])
    history = []

    best_order = population[0]
    best_fit = -1.0

    for _ in range(generations):
        fitness = np.array([_apfd(ind, priorities) for ind in population])

        gen_best_idx = np.argmax(fitness)
        if fitness[gen_best_idx] > best_fit:
            best_fit = float(fitness[gen_best_idx])
            best_order = population[gen_best_idx].copy()
        history.append(best_fit)

        # Elitism
        elite_idx = np.argsort(fitness)[-elitism:]
        new_population = [population[i].copy() for i in elite_idx]

        while len(new_population) < population_size:
            parent1 = _tournament_select(population, fitness)
            parent2 = _tournament_select(population, fitness)
            child = _order_crossover(parent1, parent2)
            child = _mutate(child, rate=mutation_rate)
            new_population.append(child)

        population = np.array(new_population)

    return best_order, best_fit, history
