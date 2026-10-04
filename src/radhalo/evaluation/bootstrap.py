"""Report-level (cluster) bootstrap.

Sentences of one report share context, so resampling is done over reports: a
report drawn k times contributes k copies of its sentences.
"""

import numpy as np


def resample_report_ids(report_ids: list[int], n_bootstrap: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.array(report_ids)
    n = len(ids)
    return [ids[rng.integers(0, n, size=n)] for _ in range(n_bootstrap)]


def percentile_ci(values: np.ndarray, confidence_level: float = 0.95) -> tuple:
    """Percentile interval over the non-NaN replicates, or (None, None)."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return None, None
    alpha = (1 - confidence_level) / 2
    return float(np.percentile(valid, 100 * alpha)), float(np.percentile(valid, 100 * (1 - alpha)))
