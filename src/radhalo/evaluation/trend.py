"""Cochran-Armitage test for a linear trend in proportions."""

import numpy as np
from scipy import stats


def cochran_armitage(scores, positives, totals) -> tuple[float, float]:
    """Return (z, two-sided p) for a 2 x k table with ordinal column scores."""
    scores, positives, totals = (np.asarray(x, dtype=float) for x in (scores, positives, totals))
    n = totals.sum()
    r = positives.sum()
    t = (scores * positives).sum()
    mean = r * (scores * totals).sum() / n
    var = r * (n - r) / (n * (n - 1)) * (n * (scores**2 * totals).sum() - (scores * totals).sum() ** 2) / n
    z = (t - mean) / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))
