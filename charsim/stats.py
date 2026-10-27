"""Statistics without dependencies: Krippendorff's alpha, bootstrap intervals, permutation
tests and the Holm correction."""

import collections
import itertools
import random


def krippendorff_alpha(units, metric="nominal"):
    """units: list of lists of the values given to one unit by the coders (None = missing)."""
    units = [[v for v in u if v is not None] for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    if metric == "nominal":
        def d(a, b):
            return float(a != b)
    else:
        def d(a, b):
            return (float(a) - float(b)) ** 2
    n = sum(len(u) for u in units)
    do = sum(sum(d(a, b) for a, b in itertools.permutations(u, 2)) / (len(u) - 1) for u in units) / n
    values = [v for u in units for v in u]
    counts = collections.Counter(values)
    de = sum(counts[a] * counts[b] * d(a, b) for a in counts for b in counts) / (n * (n - 1))
    return 1.0 - do / de if de > 0 else 1.0


def bootstrap_ci(values, stat=lambda xs: sum(xs) / len(xs), n_boot=5000, alpha=0.05, seed=12345):
    if not values:
        return None, None
    rng = random.Random(seed)
    boots = sorted(stat([rng.choice(values) for _ in values]) for _ in range(n_boot))
    return boots[int(alpha / 2 * n_boot)], boots[int((1 - alpha / 2) * n_boot) - 1]


def permutation_test(a, b, n_perm=10000, seed=12345):
    """Two-sided permutation test of the difference of the means of two samples."""
    if not a or not b:
        return None
    rng = random.Random(seed)
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    pooled = list(a) + list(b)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        x, y = pooled[:len(a)], pooled[len(a):]
        hits += abs(sum(x) / len(x) - sum(y) / len(y)) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def holm(pvals):
    """Holm-adjusted p-values in the original order (None stays None)."""
    idx = sorted((p, i) for i, p in enumerate(pvals) if p is not None)
    out, running = [None] * len(pvals), 0.0
    for rank, (p, i) in enumerate(idx):
        running = max(running, min(1.0, (len(idx) - rank) * p))
        out[i] = running
    return out
