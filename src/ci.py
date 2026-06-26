"""Confidence intervals over seeds/draws. Every cross-method comparison reports a
95% CI (autonomy rule 5). Uses the t-interval for small sample sizes (>=5 seeds)."""
from __future__ import annotations

import numpy as np


def mean_ci(values, ci: float = 95.0):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))],
                   dtype=float)
    n = len(v)
    mean = float(np.mean(v)) if n else float("nan")
    if n < 2:
        return {"mean": mean, "lo": mean, "hi": mean, "n": n, "sd": 0.0, "half_width": 0.0}
    sd = float(np.std(v, ddof=1))
    se = sd / np.sqrt(n)
    from scipy import stats
    t = stats.t.ppf(0.5 + ci / 200.0, df=n - 1)
    hw = t * se
    return {"mean": mean, "lo": mean - hw, "hi": mean + hw, "n": n, "sd": sd, "half_width": hw}


def paired_diff_ci(a, b, ci: float = 95.0):
    """CI on the paired difference a-b (same seeds), and whether it excludes 0."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    res = mean_ci(d, ci)
    res["significant"] = bool(res["lo"] > 0 or res["hi"] < 0)
    return res
